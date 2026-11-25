"""Test environment: an isolated copy of the model registry, a small data directory with
known-label images from held-out patients, and empty storage - so tests exercise the
real artifact and code paths without touching the working tree."""
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="smearscan-test-"))
DATA, MODELS, STORAGE = TMP / "data", TMP / "models", TMP / "storage"
N_TEST_PER_CLASS = 30

os.environ.update({
    "SMEAR_DATA_DIR": str(DATA), "SMEAR_MODELS_DIR": str(MODELS), "SMEAR_STORAGE_DIR": str(STORAGE),
    "SMEAR_RETRAIN_MIN_NEW_SAMPLES": "20", "SMEAR_RETRAIN_REPLAY_MIN": "40", "SMEAR_RETRAIN_MAX_EPOCHS": "2",
    "SMEAR_GATE_EVAL_LIMIT": "0", "SMEAR_LOG_LEVEL": "WARNING",
})


def _build_fixture_tree() -> None:
    src_models = Path(os.environ.get("SMEARSCAN_TEST_MODELS_SRC", ROOT / "models"))
    if not (src_models / "registry.json").exists():
        pytest.exit("models/registry.json missing - run the notebook (or scripts/train.py) first", returncode=2)
    MODELS.mkdir(parents=True)
    active = __import__("json").loads((src_models / "registry.json").read_text())["active"]
    shutil.copytree(src_models / active, MODELS / active)
    reg = {"active": active, "versions": {active: {"status": "active", "created_at": "2026-01-01T00:00:00+00:00",
                                                   "parent": None, "summary": {}}}, "events": []}
    (MODELS / "registry.json").write_text(__import__("json").dumps(reg))
    (MODELS / "pretrained").mkdir()
    for cls in ("Parasitized", "Uninfected"):
        dst = DATA / "test" / cls
        dst.mkdir(parents=True)
        for f in sorted((ROOT / "data" / "test" / cls).glob("*.png"))[::37][:N_TEST_PER_CLASS]:
            shutil.copy(f, dst / f.name)
    with open(ROOT / "data" / "replay_pool.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    picked = [r for r in rows if r["label"] == "Parasitized"][:40] + [r for r in rows if r["label"] == "Uninfected"][:40]
    for r in picked:
        (DATA / r["path"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "data" / r["path"], DATA / r["path"])
    with open(DATA / "replay_pool.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(picked[0]))
        w.writeheader()
        w.writerows(picked)
    shutil.copy(ROOT / "data" / "insights.json", DATA / "insights.json")
    STORAGE.mkdir()


_build_fixture_tree()


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def test_images() -> list[tuple[Path, str]]:
    return [(f, f.parent.name) for f in sorted((DATA / "test").glob("*/*.png"))]


@pytest.fixture(scope="session")
def upload_images() -> list[tuple[Path, str]]:
    """Known-label images that are NOT in the fixture's gold set (used as 'new' uploads)."""
    gold = {f.name for f in (DATA / "test").glob("*/*.png")}
    out = []
    for cls in ("Parasitized", "Uninfected"):
        out += [(f, cls) for f in sorted((ROOT / "data" / "test" / cls).glob("*.png"))[5::41] if f.name not in gold][:20]
    return out


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(TMP, ignore_errors=True)
