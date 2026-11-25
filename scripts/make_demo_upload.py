"""Build demo_upload/demo_batch.zip: known-label cells for the bulk-upload demo.

Images come from the validation split: held-out patients that were never used to
train v1 and are not part of the gold test set the acceptance gate scores on, so the
demo neither leaks into evaluation nor re-uploads data the model was trained on.
The generated ZIP is committed, so a clean clone (which has no data/val) can still run
the demo; without data/val the script falls back to rotated replay-pool cells.

    python scripts/make_demo_upload.py --n 40
"""
import argparse
import csv
import io
import random
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="images in total (half per class)")
    ap.add_argument("--out", type=Path, default=ROOT / "demo_upload" / "demo_batch.zip")
    args = ap.parse_args()
    rng = random.Random(7)
    picked: list[tuple[str, str, bytes]] = []
    for cls in ("Parasitized", "Uninfected"):
        val = sorted((ROOT / "data" / "val" / cls).glob("*.png"))
        if val:
            for f in rng.sample(val, args.n // 2):
                picked.append((cls, f.name, f.read_bytes()))
        else:  # clean clone without data/val: use replay-pool cells, rotated so their bytes are new
            rows = [r for r in csv.DictReader(open(ROOT / "data" / "replay_pool.csv")) if r["label"] == cls]
            for r in rng.sample(rows, args.n // 2):
                buf = io.BytesIO()
                Image.open(ROOT / "data" / r["path"]).rotate(90, expand=True).save(buf, format="PNG")
                picked.append((cls, "rot90_" + Path(r["path"]).name, buf.getvalue()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w") as zf:
        for cls, name, data in picked:
            zf.writestr(f"{cls}/{name}", data)
    print(f"wrote {args.out} with {len(picked)} images")


if __name__ == "__main__":
    main()
