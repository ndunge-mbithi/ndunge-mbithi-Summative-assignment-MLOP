"""Deploy the Docker app to a Hugging Face Space (free CPU tier: 2 vCPU, 16 GB RAM).

    pip install huggingface_hub==0.26.5
    export HF_TOKEN=hf_...            # a write token, never committed
    python scripts/deploy_hf.py --space <hf-username>/smearscan

The script stages exactly the files the Docker build needs (code, model registry,
gold test set, replay-pool images) and uploads them through the Hub HTTP API, which
handles large files itself - no git-lfs setup. The Space then builds the Dockerfile.

Persistence: on the free tier the container filesystem is reset when the Space
restarts. Enable "Persistent storage" in the Space settings (mounted at /data) to keep
uploads, retrained models and logs across restarts; the entrypoint uses /data either way.
"""
import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["Dockerfile", "requirements.txt", "src", "api", "frontend", "scripts/entrypoint.sh",
        "scripts/prune_build_data.py", "models/registry.json", "data/replay_pool.csv", "data/insights.json",
        "data/dataset_card.json", "data/test"]


def stage(dst: Path) -> None:
    for item in CODE:
        s, d = ROOT / item, dst / item
        d.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if s.is_dir() else shutil.copy2)(s, d, **({"ignore": shutil.ignore_patterns("__pycache__")} if s.is_dir() else {}))
    import json
    active = json.loads((ROOT / "models/registry.json").read_text())["active"]
    shutil.copytree(ROOT / "models" / active, dst / "models" / active)
    if active != "v1":  # Dockerfile seeds models/v1; keep it present
        shutil.copytree(ROOT / "models" / "v1", dst / "models" / "v1")
    for r in csv.DictReader(open(ROOT / "data/replay_pool.csv")):
        (dst / "data" / r["path"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "data" / r["path"], dst / "data" / r["path"])
    shutil.copy2(ROOT / "deploy/huggingface/SPACE_README.md", dst / "README.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<user>/<space-name>")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN to a Hugging Face write token")
    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space",
                          commit_message="Deploy SmearScan")
    user, name = args.space.split("/")
    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{args.space}")
    print(f"App URL once built:  https://{user}-{name}.hf.space".lower().replace("_", "-"))


if __name__ == "__main__":
    main()
