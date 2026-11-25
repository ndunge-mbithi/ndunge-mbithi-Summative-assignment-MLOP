"""Operational and model-quality monitoring built from the stored logs.

What can and cannot be measured in production:
* Always measurable: traffic, latency, error rates, model version mix, prediction
  class mix, confidence distribution, low-confidence (referral) rate, input drift.
* Accuracy/precision/recall are only computed over predictions that received a
  ground-truth label via ``POST /api/predictions/{id}/feedback`` (e.g. a microscopist's
  confirmed read). Without labels the API reports ``null`` rather than a guess.

``retraining_signal`` combines these into a recommendation: drift above the PSI alert
level, labelled accuracy below the model's own test accuracy by > 3 points, or enough
new labelled uploads waiting -> suggest retraining.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .config import Settings
from .data_store import Store
from .drift import DRIFT_FEATURES, drift_report


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 2) if values else None


def request_stats(store: Store, window_s: int = 3600) -> dict:
    since = time.time() - window_s
    rows = store.q("SELECT ts, replica, path, status, ms FROM requests WHERE ts >= ?", (since,))
    by_path: dict[str, list] = {}
    for r in rows:
        by_path.setdefault(r["path"], []).append(r)
    endpoints = []
    for path, rs in sorted(by_path.items(), key=lambda kv: -len(kv[1])):
        ms = [r["ms"] for r in rs]
        endpoints.append({"path": path, "count": len(rs), "errors_5xx": sum(r["status"] >= 500 for r in rs),
                          "errors_4xx": sum(400 <= r["status"] < 500 for r in rs), "p50_ms": _pct(ms, 50),
                          "p95_ms": _pct(ms, 95), "p99_ms": _pct(ms, 99)})
    per_min: dict[int, list] = {}
    for r in rows:
        per_min.setdefault(int(r["ts"] // 60) * 60, []).append(r)
    timeline = [{"t": t, "requests": len(v), "errors": sum(x["status"] >= 500 for x in v),
                 "p95_ms": _pct([x["ms"] for x in v if x["path"] == "/api/predict"], 95)}  # prediction latency only
                for t, v in sorted(per_min.items())]
    replicas = {}
    for r in rows:
        replicas[r["replica"]] = replicas.get(r["replica"], 0) + 1
    total = len(rows)
    return {"window_s": window_s, "total": total, "error_rate_5xx": round(sum(r["status"] >= 500 for r in rows) / total, 4)
            if total else 0.0, "endpoints": endpoints, "timeline": timeline, "by_replica": replicas}


def prediction_stats(store: Store, settings: Settings, reference: dict | None, expected_accuracy: float | None) -> dict:
    preds = store.q("SELECT * FROM predictions ORDER BY ts DESC LIMIT ?", (max(settings.drift_window, 2000),))
    total = store.one("SELECT COUNT(*) n FROM predictions")["n"]
    recent = preds[: settings.drift_window]
    class_mix: dict[str, int] = {}
    versions: dict[str, int] = {}
    for p in recent:
        class_mix[p["predicted_class"]] = class_mix.get(p["predicted_class"], 0) + 1
        versions[p["model_version"] or "none"] = versions.get(p["model_version"] or "none", 0) + 1
    conf = [p["confidence"] for p in recent]
    hist = np.histogram(conf, bins=np.linspace(0.5, 1.0, 11))[0].tolist() if conf else [0] * 10
    labelled = store.q("SELECT predicted_class, true_label, model_version FROM predictions WHERE true_label IS NOT NULL")
    quality = None
    if labelled:
        y = np.array([r["true_label"] == "Parasitized" for r in labelled])
        yhat = np.array([r["predicted_class"] == "Parasitized" for r in labelled])
        tp, fp, fn = int((y & yhat).sum()), int((~y & yhat).sum()), int((y & ~yhat).sum())
        quality = {"n_labelled": len(labelled), "accuracy": round(float((y == yhat).mean()), 4),
                   "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                   "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                   "expected_accuracy_from_test": expected_accuracy}
    drift = None
    if reference:
        stats = [json.loads(p["stats_json"]) for p in recent if p["stats_json"]]
        drift = drift_report(reference, {f: [s[f] for s in stats if f in s] for f in DRIFT_FEATURES},
                             alert=settings.drift_psi_alert)
    hourly: dict[int, dict] = {}
    for p in preds:
        h = int(p["ts"] // 3600) * 3600
        d = hourly.setdefault(h, {"t": h, "Parasitized": 0, "Uninfected": 0})
        d[p["predicted_class"]] = d.get(p["predicted_class"], 0) + 1
    return {
        "total_predictions": total,
        "window": len(recent),
        "class_mix": class_mix,
        "model_versions": versions,
        "mean_confidence": round(float(np.mean(conf)), 4) if conf else None,
        "confidence_hist": {"edges": np.linspace(0.5, 1.0, 11).round(2).tolist(), "counts": hist},
        "low_confidence_rate": round(float(np.mean([p["low_confidence"] for p in recent])), 4) if recent else None,
        "mean_latency_ms": round(float(np.mean([p["latency_ms"] for p in recent])), 2) if recent else None,
        "labelled_quality": quality,
        "drift": drift,
        "hourly": sorted(hourly.values(), key=lambda d: d["t"])[-48:],
    }


def retraining_signal(pred_stats: dict, pending_samples: int, settings: Settings) -> dict:
    reasons = []
    drift = pred_stats.get("drift") or {}
    drifted = [f for f, d in drift.items() if d.get("status") == "drift" and d.get("alerting")]
    if drifted:
        reasons.append(f"input drift (PSI > {settings.drift_psi_alert}) on: {', '.join(drifted)}")
    q = pred_stats.get("labelled_quality")
    if q and q["n_labelled"] >= 30 and q.get("expected_accuracy_from_test") and \
            q["accuracy"] < q["expected_accuracy_from_test"] - 0.03:
        reasons.append(f"labelled production accuracy {q['accuracy']:.3f} is >3 points below test accuracy")
    if pending_samples >= settings.retrain_min_new_samples:
        reasons.append(f"{pending_samples} new labelled samples are waiting")
    return {"recommended": bool(reasons), "reasons": reasons}
