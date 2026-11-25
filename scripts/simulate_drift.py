"""Exercise production monitoring with realistic traffic, then a simulated domain shift.

Phase 1 sends unmodified held-out test cells (in-distribution).
Phase 2 sends the same kind of cells darkened and blue-shifted, mimicking a thicker
smear / different Giemsa batch / dimmer microscope lamp - the slide-level variation the
EDA identified as the dominant source of non-biological variance.
After each phase the script prints the drift report and retraining signal from
/api/monitoring and the model's accuracy on that traffic (the true labels are known
here, and are also posted as feedback so the production-quality panel fills in).

    python scripts/simulate_drift.py --host http://localhost:8080 --n 150
"""
import argparse
import io
import json
import random
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]


def post_multipart(url: str, filename: str, data: bytes) -> dict:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def shift(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(0.72)
    r, g, b = img.split()
    img = Image.merge("RGB", (r.point(lambda v: v * 0.9), g, b.point(lambda v: min(255, v * 1.12))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_phase(host: str, files: list, transform, feedback: bool) -> float:
    correct = 0
    for f, label in files:
        data = f.read_bytes()
        r = post_multipart(f"{host}/api/predict", f.name, transform(data) if transform else data)
        correct += r["predicted_class"] == label
        if feedback:
            post_json(f"{host}/api/predictions/{r['prediction_id']}/feedback", {"true_label": label})
    return correct / len(files)


def report(host: str, title: str, acc: float) -> None:
    m = json.load(urllib.request.urlopen(f"{host}/api/monitoring", timeout=30))
    print(f"\n== {title}: accuracy on this traffic {acc:.3f}")
    for f, d in (m["predictions"]["drift"] or {}).items():
        print(f"  {f:20s} PSI={d.get('psi', '-'):>8} status={d['status']}{'' if d.get('alerting', True) else ' (informational)'}")
    print("  retraining signal:", json.dumps(m["retraining_signal"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8080")
    ap.add_argument("--n", type=int, default=500, help="images per phase (500 = one full drift window)")
    ap.add_argument("--feedback", action="store_true", help="also post true labels as feedback")
    args = ap.parse_args()
    files = [(f, f.parent.name) for f in sorted((ROOT / "data" / "test").glob("*/*.png"))]
    random.Random(1).shuffle(files)
    acc = run_phase(args.host, files[: args.n], None, args.feedback)
    report(args.host, "phase 1 - in-distribution traffic", acc)
    acc = run_phase(args.host, files[args.n: 2 * args.n], shift, args.feedback)
    report(args.host, "phase 2 - darker / blue-shifted stain (simulated new clinic)", acc)


if __name__ == "__main__":
    main()
