"""Summarise the dataset into data/insights.json for the web UI's insights panel.

Everything shown in the UI is computed here from data/manifest.csv - the UI renders
numbers, it does not invent them. Interpretations are kept next to the numbers they
explain and are re-derived (with the numbers interpolated) every time this runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def hist(series: pd.Series, bins: np.ndarray) -> list[int]:
    return np.histogram(series.clip(bins[0], bins[-1]), bins=bins)[0].tolist()


def main(data_dir: Path = ROOT / "data") -> dict:
    m = pd.read_csv(data_dir / "manifest.csv")
    y = (m.label == "Parasitized").astype(int)
    card = json.loads((data_dir / "dataset_card.json").read_text())

    def by_class(col, bins):
        return {"bins": bins.round(4).tolist(),
                "Parasitized": hist(m.loc[y == 1, col], bins), "Uninfected": hist(m.loc[y == 0, col], bins)}

    stain_bins = np.linspace(0, 0.06, 25)
    bright_bins = np.linspace(100, 195, 20)
    size_bins = np.linspace(40, 260, 23)
    auc = {c: round(float(roc_auc_score(y, m[c])), 3) for c in ("stain_fraction", "brightness", "width")}
    stained = m.groupby("label").stain_fraction.apply(lambda s: float((s > 0.01).mean()))
    slide_means = m.groupby("patient_id").brightness.mean()
    within = float(m.groupby("patient_id").brightness.std().mean())
    per_patient = m.groupby("patient_id").size()
    only_uninf = int((m.groupby(["patient_id", "label"]).size().unstack(fill_value=0).Parasitized == 0).sum())

    insights = {
        "card": card,
        "class_counts": m.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "features": [
            {
                "key": "stain_fraction",
                "title": "Purple chromatin (Giemsa stain) fraction",
                "chart": by_class("stain_fraction", stain_bins),
                "x_label": "share of cell pixels that are dark purple",
                "finding": (f"{stained['Parasitized']:.0%} of parasitized cells have more than 1% dark-purple pixels, "
                            f"against {stained['Uninfected']:.1%} of uninfected cells. This single hand-crafted feature "
                            f"separates the classes with ROC-AUC {auc['stain_fraction']}."),
                "why_it_matters": ("The signal the CNN must learn is small, localised and colour-defined, so preprocessing "
                                   "must keep colour and avoid aggressive downsampling. Parasitized cells with almost no "
                                   "stain (early ring forms, out-of-focus) and uninfected cells with stain specks "
                                   "(platelets, stain precipitate) are where errors and label noise concentrate."),
            },
            {
                "key": "brightness",
                "title": "Cell brightness and slide-level staining variation",
                "chart": by_class("brightness", bright_bins),
                "x_label": "mean luminance of cell pixels (0-255)",
                "finding": (f"Parasitized cells are slightly darker (median {m.loc[y == 1, 'brightness'].median():.0f} vs "
                            f"{m.loc[y == 0, 'brightness'].median():.0f}), but the spread of per-slide mean brightness "
                            f"(sd {slide_means.std():.1f}) exceeds the spread within a slide (sd {within:.1f})."),
                "why_it_matters": ("Staining and illumination differ more between slides than between cells on a slide - a "
                                   "batch effect. A random cell-level split would let the model memorise slide appearance, "
                                   "so I split by patient, augment brightness/contrast, and monitor brightness drift in "
                                   "production because a new clinic or stain batch shows up here first."),
            },
            {
                "key": "width",
                "title": "Cell crop size",
                "chart": by_class("width", size_bins),
                "x_label": "crop width in pixels",
                "finding": (f"Crops range from {int(m.width.min())} to {int(m.width.max())} px wide (median {m.width.median():.0f}); "
                            f"size barely differs by class (ROC-AUC {auc['width']})."),
                "why_it_matters": ("Size is not a shortcut the model can exploit, which is good. Variable crop sizes need a "
                                   "fixed input: I resize the long side to 128 px and pad with black, preserving aspect "
                                   "ratio and matching the black background the crops already have."),
            },
            {
                "key": "patients",
                "title": "Cells per patient",
                "chart": {"bins": [], "values": sorted(per_patient.tolist(), reverse=True)},
                "x_label": "patients, sorted by number of cells",
                "finding": (f"{len(per_patient)} patients contribute between {per_patient.min()} and {per_patient.max()} "
                            f"cells each (median {per_patient.median():.0f}); {only_uninf} patients are uninfected-only, "
                            f"and every infected patient also contributes uninfected cells."),
                "why_it_matters": ("A few heavy patients dominate the pool. Patient-grouped splitting keeps any one person's "
                                   "slide out of both training and test, so test scores estimate performance on new patients."),
            },
        ],
    }
    (data_dir / "insights.json").write_text(json.dumps(insights, indent=2))
    return insights


if __name__ == "__main__":
    out = main()
    for f in out["features"]:
        print("-", f["title"], ":", f["finding"])
