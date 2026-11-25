"""Input-drift detection.

At training time we store, per monitored image feature, decile bin edges and the
share of training images in each bin. In production the same features are computed
for every prediction (``preprocessing.compute_stats``), and the Population Stability
Index (PSI) compares the recent window with the reference:

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift. A
two-sample Kolmogorov-Smirnov p-value is reported alongside as a second opinion.
Brightness and stain fraction matter most: the EDA shows slide-level staining and
illumination differences are larger than within-slide ones, so a new clinic, stain
batch or microscope shows up here first.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

DRIFT_FEATURES = ("brightness", "stain_fraction", "mean_r", "mean_g", "mean_b", "width", "background_fraction")
# Features that can raise an alert. Mean red and blue are reported but do not alert: on the held-out test patients
# they show PSI ~0.4-0.5 against the training reference (slide-to-slide hue differences) while the model is still
# 95.8% accurate there, so they would fire on harmless variation. The alerting features stay below 0.13 for random
# 500-cell windows of the same held-out data and move sharply under a real stain/illumination shift.
ALERT_FEATURES = ("brightness", "stain_fraction", "mean_g", "width", "background_fraction")
_EPS = 1e-4


def build_reference(values_by_feature: dict[str, list[float]], n_bins: int = 10, sample: int = 2000,
                    seed: int = 0) -> dict:
    ref = {}
    rng = np.random.default_rng(seed)
    for f, vals in values_by_feature.items():
        v = np.asarray(vals, dtype=float)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
        inner = edges[1:-1]
        counts = np.bincount(np.searchsorted(inner, v, side="right"), minlength=len(inner) + 1)
        ref[f] = {
            "edges": inner.round(6).tolist(),
            "expected": (counts / counts.sum()).round(6).tolist(),
            "mean": float(v.mean()), "std": float(v.std()),
            "sample": rng.choice(v, size=min(sample, len(v)), replace=False).round(5).tolist(),
        }
    return ref


def psi(reference: dict, values: list[float]) -> float:
    inner = np.asarray(reference["edges"])
    e = np.asarray(reference["expected"]) + _EPS
    counts = np.bincount(np.searchsorted(inner, np.asarray(values, float), side="right"), minlength=len(e))
    a = counts / max(counts.sum(), 1) + _EPS
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: dict, recent: dict[str, list[float]], alert: float = 0.2, min_n: int = 30) -> dict:
    out = {}
    for f, ref in reference.items():
        vals = recent.get(f, [])
        if len(vals) < min_n:
            out[f] = {"n": len(vals), "status": "insufficient_data"}
            continue
        score = psi(ref, vals)
        ks = ks_2samp(ref["sample"], vals) if ref.get("sample") else None
        out[f] = {
            "n": len(vals),
            "psi": round(score, 4),
            "ks_pvalue": round(float(ks.pvalue), 6) if ks else None,
            "recent_mean": round(float(np.mean(vals)), 4),
            "reference_mean": round(ref["mean"], 4),
            "status": "drift" if score > alert else "moderate" if score > 0.1 else "stable",
            "alerting": f in ALERT_FEATURES,
        }
        if f not in ALERT_FEATURES and out[f]["status"] != "stable":
            out[f]["status"] = "shifted (info)"
    return out
