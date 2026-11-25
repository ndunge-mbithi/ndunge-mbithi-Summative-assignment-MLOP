"""API tests, including the full upload -> persist -> retrain -> new version workflow."""
import io
import json
import time
import zipfile

import pytest
from PIL import Image

from src.config import get_settings


def test_health_and_ready(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["model_version"]
    assert client.get("/ready").json()["ready"] is True


def test_model_info(client):
    info = client.get("/api/model").json()
    assert info["version"] and info["metrics"]["test"]["f1"] > 0.9
    assert "sample_ids" not in info.get("training_data", {})


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "SmearScan" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_predict_known_examples(client, test_images):
    wrong = 0
    for path, label in test_images[::6]:
        r = client.post("/api/predict", files={"file": (path.name, path.read_bytes(), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["prediction_id"].startswith("pred_") and body["model_version"]
        wrong += body["predicted_class"] != label
    assert wrong <= 1


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40])
def test_predict_rejects_bad_input(client, payload):
    r = client.post("/api/predict", files={"file": ("bad.png", payload, "image/png")})
    assert r.status_code == 422 and "invalid image" in r.json()["detail"]


def test_predict_requires_file(client):
    assert client.post("/api/predict").status_code == 422


def test_feedback_and_monitoring(client, test_images):
    path, label = test_images[0]
    pid = client.post("/api/predict", files={"file": (path.name, path.read_bytes(), "image/png")}).json()["prediction_id"]
    assert client.post(f"/api/predictions/{pid}/feedback", json={"true_label": label}).status_code == 200
    assert client.post(f"/api/predictions/{pid}/feedback", json={"true_label": "Maybe"}).status_code == 422
    assert client.post("/api/predictions/pred_nope/feedback", json={"true_label": label}).status_code == 404
    m = client.get("/api/monitoring").json()
    assert m["predictions"]["total_predictions"] >= 1
    assert m["predictions"]["labelled_quality"]["n_labelled"] >= 1
    assert any(e["path"] == "/api/predict" for e in m["requests"]["endpoints"])
    assert set(m["predictions"]["drift"]) >= {"brightness", "stain_fraction"}


def test_upload_validation(client, upload_images):
    path, label = upload_images[-1]  # an Uninfected image not otherwise uploaded
    assert label == "Uninfected"
    # missing label for loose files
    r = client.post("/api/uploads", files=[("files", (path.name, path.read_bytes(), "image/png"))])
    assert r.status_code == 422
    # unknown label
    r = client.post("/api/uploads", data={"label": "Plasmodium"}, files=[("files", (path.name, path.read_bytes(), "image/png"))])
    assert r.status_code == 422
    # corrupted file is rejected per-file, valid file accepted
    r = client.post("/api/uploads", data={"label": "Uninfected"},
                    files=[("files", ("broken.png", b"garbage", "image/png")), ("files", (path.name, path.read_bytes(), "image/png"))])
    body = r.json()
    assert r.status_code == 200 and body["accepted"] == 1 and body["rejected"] == 1
    assert "not a readable image" in body["rejected_items"][0]["reason"]
    # duplicates are detected by content hash
    r = client.post("/api/uploads", data={"label": "Uninfected"}, files=[("files", ("copy.png", path.read_bytes(), "image/png"))])
    assert r.json()["duplicates"] == 1
    assert client.post("/api/uploads", files=[("archive", ("x.zip", b"nope", "application/zip"))]).status_code == 422


def _zip_of(items):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, label in items:
            zf.writestr(f"cells/{label}/{path.name}", path.read_bytes())
        zf.writestr("__MACOSX/._junk", b"x")
    return buf.getvalue()


def _augmented_copy(path, k):
    """Distinct-bytes variant of a real cell (rotation), so de-dup does not drop it."""
    buf = io.BytesIO()
    Image.open(path).rotate(90 * k, expand=True).save(buf, format="PNG")
    return buf.getvalue()


def test_retrain_rejected_without_enough_data(client):
    r = client.post("/api/retrain", json={})
    assert r.status_code == 409 and "need at least" in r.json()["detail"]


def test_full_upload_to_retraining_to_new_model(client, upload_images):
    s = get_settings()
    before = client.get("/api/models").json()
    parent = before["active"]
    # 1. bulk upload: a ZIP with class folders + loose files with a label
    zipped = upload_images[::2]
    r = client.post("/api/uploads", data={"note": "pytest zip"}, files=[("archive", ("batch.zip", _zip_of(zipped), "application/zip"))])
    assert r.status_code == 200 and r.json()["accepted"] == len(zipped)
    paras = [p for p, lbl in upload_images if lbl == "Parasitized"][:12]
    r = client.post("/api/uploads", data={"label": "Parasitized"},
                    files=[("files", (f"rot_{i}.png", _augmented_copy(p, 1), "image/png")) for i, p in enumerate(paras)])
    assert r.json()["accepted"] == len(paras)
    # 2. persistence: rows, raw files and preprocessed arrays exist
    ds = client.get("/api/datasets").json()
    assert ds["pending_for_training"] >= 20
    items = client.get("/api/samples?limit=500").json()["items"]
    assert all((s.storage_dir / "uploads" / i["batch_id"]).exists() for i in items)
    assert client.get(f"/api/samples/{items[0]['id']}/image").status_code == 200
    # 3. trigger real retraining (separate process) and follow it to completion
    r = client.post("/api/retrain", json={"epochs": 2})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.post("/api/retrain", json={}).status_code == 409  # one job at a time
    deadline = time.time() + 900
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(3)
    assert job["status"] in ("succeeded", "rejected"), json.dumps(job, indent=1)[-3000:]
    assert job["parent_version"] == parent
    assert len(job["epochs"]) >= 1 and job["epochs"][0]["epoch"] == 1
    res = job["result"]
    # 4. a new immutable version exists, with lineage and evaluation, and promotion follows the gate
    after = client.get("/api/models").json()
    new = res["version"]
    assert new in after["versions"] and new != parent
    meta = client.get(f"/api/models/{new}").json()
    assert meta["parent_version"] == parent and meta["training_data"]["new_samples"] >= 20
    assert (s.models_dir / new / "model.keras").exists()
    if res["gate"]["passed"]:
        assert after["active"] == new
        assert client.get("/health").json()["model_version"] == new  # hot reload, no restart
        assert client.get("/api/datasets").json()["pending_for_training"] == 0
        # rollback restores the parent
        assert client.post("/api/models/rollback").json()["active"] == parent
        assert client.get("/health").json()["model_version"] == parent
        client.post(f"/api/models/{new}/activate")
    else:
        assert after["active"] == parent and after["versions"][new]["status"] == "rejected"
