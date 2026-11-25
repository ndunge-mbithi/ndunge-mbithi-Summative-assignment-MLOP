"""Docker build helper: keep only runtime data (gold test set + replay pool images).

Run inside the build's data stage; deletes training images not listed in
data/replay_pool.csv, plus validation images and caches, and fails loudly if any
replay image is missing from the build context.
"""
import csv
import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")
keep = {r["path"] for r in csv.DictReader(open(root / "replay_pool.csv"))}
removed = 0
for p in (root / "train").rglob("*.png"):
    if str(p.relative_to(root)) not in keep:
        p.unlink()
        removed += 1
missing = [k for k in keep if not (root / k).exists()]
if missing:
    sys.exit(f"{len(missing)} replay images missing from build context, e.g. {missing[:3]}")
for d in ("val", "cache"):
    shutil.rmtree(root / d, ignore_errors=True)
n_test = sum(1 for _ in (root / "test").rglob("*.png"))
if n_test == 0:
    sys.exit("gold test set (data/test) missing from build context")
print(f"runtime data: {len(keep)} replay images, {n_test} gold test images ({removed} training images pruned)")
