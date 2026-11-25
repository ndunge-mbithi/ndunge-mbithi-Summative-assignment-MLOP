"""Select the fixed replay pool used by retraining (data/replay_pool.csv).

3,000 training cells, balanced by class and spread across all training patients
(at most ceil(per-class quota / patients) cells per patient per class), so replayed
data represents the original distribution rather than the few largest slides.
These are the only training images that ship inside the container.
"""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N, SEED = 3000, 42

m = pd.read_csv(ROOT / "data" / "manifest.csv")
tr = m[m.split == "train"]
parts = []
for label, g in tr.groupby("label"):
    quota = N // 2
    cap = math.ceil(quota / g.patient_id.nunique())
    per_pat = g.groupby("patient_id", group_keys=False).apply(lambda d: d.sample(min(len(d), cap), random_state=SEED))
    extra = g.drop(per_pat.index).sample(max(0, quota - len(per_pat)), random_state=SEED)
    parts.append(pd.concat([per_pat, extra]).sample(quota, random_state=SEED))
pool = pd.concat(parts).sort_values("path")[["path", "label", "patient_id"]]
pool.to_csv(ROOT / "data" / "replay_pool.csv", index=False)
print(pool.label.value_counts().to_dict(), "patients:", pool.patient_id.nunique())
