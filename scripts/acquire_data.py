"""Acquire the NIH/NLM malaria cell dataset and build a patient-grouped split.

Source of record: U.S. National Library of Medicine, Lister Hill National Center for
Biomedical Communications (Rajaraman et al., PeerJ 2018, doi:10.7717/peerj.4568).
The official host (lhncbc.nlm.nih.gov) is the canonical download; this script can use
either a local copy of ``cell_images/`` or a git mirror that preserves the original
file names, because the slide/patient identifier is encoded in each file name and is
what makes a leakage-free split possible.

Outputs
  data/manifest.csv            one row per image: path, label, patient_id, split, sha256, stats
  data/{train,val,test}/<cls>/ the images, copied
  data/dataset_card.json       counts, provenance, licence, split seed

Usage
  python scripts/acquire_data.py --source /path/to/cell_images
  python scripts/acquire_data.py --git-mirror https://github.com/TheODDYSEY/Malaria-Detection-Model
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src.preprocessing import compute_stats  # noqa: E402

CLASSES = ("Parasitized", "Uninfected")
ID_RE = re.compile(r"^(.*?)_IMG_", re.IGNORECASE)
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def patient_id(filename: str) -> str:
    """'C39P4thinF_original_IMG_..._cell_9.png' -> 'c39p4thinf'.

    The prefix before ``_IMG_`` identifies the slide; '_original' marks a re-capture of
    the same slide, so it is folded into the same group.
    """
    m = ID_RE.match(filename)
    if not m:
        raise ValueError(f"cannot parse patient id from {filename}")
    return m.group(1).lower().replace("_original", "")


def assign_splits(groups: dict[str, Counter], seed: int, trials: int = 3000) -> dict[str, str]:
    """Patient-level split whose per-class cell shares best match SPLITS.

    Each trial shuffles patients and fills val, then test, then train by cell count.
    The trial with the smallest worst-case deviation from the target share (over both
    classes and all splits) wins. Deterministic for a given seed; no patient spans splits.
    """
    rng = random.Random(seed)
    pids = sorted(groups)
    totals = Counter()
    for c in groups.values():
        totals.update(c)
    grand = sum(totals.values())
    best, best_err = None, float("inf")
    for _ in range(trials):
        rng.shuffle(pids)
        out, filled, i = {}, {s: Counter() for s in SPLITS}, 0
        for s in ("val", "test"):
            while i < len(pids) and sum(filled[s].values()) < SPLITS[s] * grand:
                out[pids[i]] = s
                filled[s].update(groups[pids[i]])
                i += 1
        for p in pids[i:]:
            out[p] = "train"
            filled["train"].update(groups[p])
        err = max(abs(filled[s][c] / totals[c] - SPLITS[s]) for s in SPLITS for c in totals)
        if err < best_err:
            best, best_err = dict(out), err
    print(f"split search: worst per-class share deviation {best_err:.4f}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="local cell_images/ directory")
    ap.add_argument("--git-mirror", help="git URL of a mirror containing cell_images/")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tmp = None
    source = args.source
    if source is None:
        if not args.git_mirror:
            ap.error("give --source or --git-mirror")
        tmp = Path(tempfile.mkdtemp())
        subprocess.run(["git", "clone", "--depth", "1", args.git_mirror, str(tmp / "m")], check=True)
        source = tmp / "m" / "cell_images"

    files = []
    for cls in CLASSES:
        for f in sorted((source / cls).glob("*.png")):
            files.append((f, cls, patient_id(f.name)))
    print(f"found {len(files)} images")

    groups: dict[str, Counter] = defaultdict(Counter)
    for _, cls, pid in files:
        groups[pid][cls] += 1
    split_of = assign_splits(groups, args.seed)

    for s in SPLITS:
        for cls in CLASSES:
            d = args.out / s / cls
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    rows, seen = [], {}
    for f, cls, pid in files:
        data = f.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            print(f"skip duplicate {f.name} == {seen[digest]}")
            continue
        seen[digest] = f.name
        with Image.open(f) as im:
            im.verify()
        with Image.open(f) as im:
            stats = compute_stats(im.convert("RGB"))
        split = split_of[pid]
        dest = args.out / split / cls / f.name
        dest.write_bytes(data)
        rows.append({"path": str(dest.relative_to(args.out)), "label": cls, "patient_id": pid,
                     "split": split, "sha256": digest, **stats.as_dict()})

    with open(args.out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    counts = {s: dict(Counter(r["label"] for r in rows if r["split"] == s)) for s in SPLITS}
    patients = {s: len({r["patient_id"] for r in rows if r["split"] == s}) for s in SPLITS}
    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["patient_id"])
    overlap = sum(len(by_split[a] & by_split[b]) for a in SPLITS for b in SPLITS if a < b)
    assert overlap == 0, "patient leakage between splits"
    card = {
        "name": "NIH/NLM Malaria Cell Images (thin smear, P. falciparum)",
        "provenance": "Chittagong Medical College Hospital, Bangladesh; imaged and annotated by "
                      "Mahidol-Oxford Tropical Medicine Research Unit; published by U.S. NLM/LHNCBC",
        "canonical_url": "https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html",
        "mirror_used": args.git_mirror or str(args.source),
        "citation": "Rajaraman S. et al. Pre-trained convolutional neural networks as feature extractors "
                    "toward improved malaria parasite detection in thin blood smear images. PeerJ 6:e4568 (2018)",
        "licence": "U.S. Government work (NLM) - public domain in the United States; attribution requested",
        "classes": list(CLASSES),
        "n_images": len(rows),
        "n_patients": len(groups),
        "split_seed": args.seed,
        "split_strategy": "patient-grouped (slide id from filename), no patient in more than one split",
        "counts": counts,
        "patients_per_split": patients,
        "patient_overlap_between_splits": overlap,
    }
    (args.out / "dataset_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
