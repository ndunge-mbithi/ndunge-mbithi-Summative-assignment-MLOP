"""SmearScan HTTP API and web UI host.

Run: ``uvicorn api.main:app --host 0.0.0.0 --port 8000``
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.config import CLASS_NAMES, PROJECT_ROOT, get_settings
from src.data_store import AsyncWriter, Store, utcnow
from src.ingestion import Ingestor, iter_zip, normalise_label
from src.monitoring import prediction_stats, request_stats, retraining_signal
from src.prediction import ModelNotReady, Predictor
from src.preprocessing import ImageValidationError
from src.registry import ModelRegistry, RegistryError

from .schemas import FeedbackIn, RetrainIn

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")
REPLICA = os.environ.get("SMEAR_REPLICA_NAME") or socket.gethostname()
STARTED = time.time()

store = Store(settings.db_path)
writer = AsyncWriter(store)
registry = ModelRegistry(settings.models_dir)
predictor = Predictor(settings)
ingestor = Ingestor(settings, store)


@asynccontextmanager
async def lifespan(_: FastAPI):
    stale = store.fail_stale_jobs()
    if stale:
        log.warning("marked %d stale retraining job(s) as failed", stale)
    try:
        predictor.refresh(force=True)
    except Exception:  # the API still starts so health/upload endpoints work; /ready reports not ready
        log.exception("could not load active model at startup")
    yield
    writer.close()


app = FastAPI(title="SmearScan - malaria cell screening API", version="1.0.0", lifespan=lifespan,
              description="Prediction, labelled-data ingestion, retraining and monitoring for the malaria cell classifier.")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins.split(","), allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def record_request(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = request.scope.get("route").path if request.scope.get("route") else request.url.path
        if path.startswith("/api") or path in ("/health", "/ready"):
            writer.put("requests", {"ts": time.time(), "replica": REPLICA, "method": request.method, "path": path,
                                    "status": status, "ms": round((time.perf_counter() - t0) * 1000, 2)})


@app.exception_handler(ImageValidationError)
async def invalid_image(_: Request, exc: ImageValidationError):
    return JSONResponse(status_code=422, content={"detail": f"invalid image: {exc}"})


@app.exception_handler(ModelNotReady)
async def not_ready(_: Request, exc: ModelNotReady):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ------------------------------------------------------------------ health
@app.get("/health", tags=["system"])
def health():
    """Liveness: the process is up. Also reports uptime and the served model."""
    try:
        predictor.refresh()  # one stat() call unless the registry changed
    except Exception:
        log.exception("model refresh failed")
    return {"status": "ok", "replica": REPLICA, "uptime_s": round(time.time() - STARTED, 1),
            "model_loaded": predictor.ready, "model_version": predictor.version, "time": utcnow()}


@app.get("/ready", tags=["system"])
def ready():
    """Readiness: a model is loaded and the database answers."""
    store.one("SELECT 1 AS ok")
    if not predictor.ready:
        raise HTTPException(503, "model not loaded")
    return {"ready": True, "model_version": predictor.version}


# ------------------------------------------------------------------- model
def _model_info(version: str, meta: dict) -> dict:
    keep = ("architecture", "framework", "image_size", "class_names", "positive_class", "preprocessing",
            "hyperparameters", "training_data", "metrics", "baseline_metrics", "parent_version", "trigger",
            "created_at", "gate", "parent_metrics_same_eval", "history")
    info = {k: meta[k] for k in keep if k in meta}
    if "training_data" in info:
        info["training_data"] = {k: v for k, v in info["training_data"].items() if k != "sample_ids"}
    return {"version": version, **info}


@app.get("/api/model", tags=["model"])
def model_info():
    predictor.refresh()
    if not predictor.version:
        raise HTTPException(503, "no active model")
    return {**_model_info(predictor.version, predictor.metadata), "replica": REPLICA,
            "loaded_at": predictor.loaded_at, "decision_threshold": settings.decision_threshold}


@app.get("/api/models", tags=["model"])
def list_models():
    state = registry.state()
    return {"active": state["active"], "versions": state["versions"], "events": state["events"][-100:]}


@app.get("/api/models/{version}", tags=["model"])
def get_model(version: str):
    if version not in registry.state()["versions"]:
        raise HTTPException(404, "unknown version")
    return _model_info(version, registry.metadata(version))


@app.post("/api/models/{version}/activate", tags=["model"])
def activate(version: str):
    """Manually promote a version (e.g. re-activate an archived model)."""
    try:
        registry.promote(version, reason="manual activation via API")
    except RegistryError as exc:
        raise HTTPException(400, str(exc)) from exc
    predictor.refresh()
    return {"active": registry.active_version()}


@app.post("/api/models/rollback", tags=["model"])
def rollback():
    """Return to the version that was active before the current one."""
    try:
        version = registry.rollback()
    except RegistryError as exc:
        raise HTTPException(400, str(exc)) from exc
    predictor.refresh()
    return {"active": version}


# --------------------------------------------------------------- predictions
@app.post("/api/predict", tags=["prediction"])
async def predict(file: UploadFile = File(..., description="single cell image (PNG/JPEG/...)")):
    data = await file.read()
    result = await run_in_threadpool(predictor.predict_bytes, data)
    pred_id = f"pred_{uuid.uuid4().hex[:12]}"
    writer.put("predictions", {
        "id": pred_id, "created_at": utcnow(), "ts": time.time(), "model_version": result["model_version"],
        "predicted_class": result["predicted_class"], "p_parasitized": result["probabilities"]["Parasitized"],
        "confidence": result["confidence"], "low_confidence": int(result["low_confidence"]),
        "latency_ms": result["latency_ms"], "stats_json": json.dumps(result["image_stats"]), "replica": REPLICA})
    return {"prediction_id": pred_id, "filename": file.filename, **result}


@app.post("/api/predictions/{pred_id}/feedback", tags=["prediction"])
def feedback(pred_id: str, body: FeedbackIn):
    """Attach a confirmed ground-truth label to a past prediction (enables production accuracy)."""
    writer.flush()
    try:
        label = normalise_label(body.true_label)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    # Prediction rows are written asynchronously (batched every 0.5 s) by whichever replica served the
    # prediction, which may not be this one. Allow for that lag before declaring the id unknown.
    for attempt in range(6):
        if store.add_feedback(pred_id, label):
            return {"prediction_id": pred_id, "true_label": label}
        time.sleep(0.4 * (attempt + 1) / 3)
    raise HTTPException(404, "unknown prediction id")


@app.get("/api/predictions", tags=["prediction"])
def recent_predictions(limit: int = 50):
    writer.flush()
    rows = store.q("SELECT id, created_at, model_version, predicted_class, p_parasitized, confidence, low_confidence, "
                   "latency_ms, replica, true_label FROM predictions ORDER BY ts DESC LIMIT ?", (min(limit, 500),))
    return {"items": rows}


# ------------------------------------------------------------------- uploads
@app.post("/api/uploads", tags=["data"])
async def upload(files: list[UploadFile] = File(default=[]), label: str | None = Form(default=None),
                 archive: UploadFile | None = File(default=None), note: str = Form(default="")):
    """Bulk labelled upload: many image ``files`` with one ``label``, and/or a class-folder ZIP ``archive``."""
    items: list[tuple[str, str | None, bytes]] = []
    if label is not None and files:
        try:
            label = normalise_label(label)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if len(files) > settings.max_bulk_files:
        raise HTTPException(413, f"too many files ({len(files)} > {settings.max_bulk_files})")
    for f in files:
        items.append((f.filename or "unnamed", label, await f.read()))
    if archive is not None:
        try:
            items.extend(iter_zip(await archive.read()))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if not items:
        raise HTTPException(422, "send image files with a label, or a ZIP archive with class folders")
    if files and label is None and archive is None:
        raise HTTPException(422, f"label is required for loose files; one of {list(CLASS_NAMES)}")
    report = await run_in_threadpool(ingestor.ingest, items, None, note)
    return report.as_dict()


@app.get("/api/datasets", tags=["data"])
def datasets():
    """Stored uploaded data: totals by label/status and recent upload batches."""
    return store.dataset_summary()


@app.get("/api/samples", tags=["data"])
def samples(limit: int = 60, batch_id: str | None = None):
    sql = "SELECT id, label, batch_id, original_name, width, height, status, used_in_version, created_at FROM samples"
    args: tuple = ()
    if batch_id:
        sql += " WHERE batch_id=?"
        args = (batch_id,)
    return {"items": store.q(sql + " ORDER BY id DESC LIMIT ?", (*args, min(limit, 500)))}


@app.get("/api/samples/{sample_id}/image", tags=["data"])
def sample_image(sample_id: int):
    row = store.one("SELECT raw_path FROM samples WHERE id=?", (sample_id,))
    if not row:
        raise HTTPException(404, "unknown sample")
    path = (settings.storage_dir / row["raw_path"]).resolve()
    if settings.storage_dir.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "file missing")
    return FileResponse(path)


@app.get("/api/examples/{kind}", tags=["data"])
def example(kind: str):
    """A random known-label image from the held-out test patients (for demos and smoke tests)."""
    label = {"parasitized": "Parasitized", "uninfected": "Uninfected"}.get(kind.lower())
    if not label:
        raise HTTPException(404, "kind must be parasitized or uninfected")
    files = sorted((settings.data_dir / "test" / label).glob("*.png"))
    if not files:
        raise HTTPException(404, "test images not available")
    f = files[int(time.time() * 1000) % len(files)]
    return FileResponse(f, media_type="image/png", headers={"X-Example-Name": f"{label}__{f.name}"})


# ----------------------------------------------------------------- retraining
@app.post("/api/retrain", tags=["retraining"], status_code=202)
def retrain(body: RetrainIn | None = None):
    """Start a retraining job in a separate worker process. One job at a time."""
    pending = len(store.trainable_samples())
    if pending < settings.retrain_min_new_samples:
        raise HTTPException(409, f"need at least {settings.retrain_min_new_samples} new labelled samples "
                                 f"(have {pending}); upload data first")
    params = (body or RetrainIn()).model_dump(exclude_none=True)
    job_id, busy = store.create_job_exclusive("retrain", params, registry.active_version())
    if job_id is None:
        raise HTTPException(409, f"job {busy['id']} is already {busy['status']}")
    log_dir = settings.storage_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{job_id}.log", "w") as fh:
        proc = subprocess.Popen([sys.executable, "-m", "src.retraining", "--job", job_id], cwd=PROJECT_ROOT,
                                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    store.update_job(job_id, pid=proc.pid)
    return {"job_id": job_id, "status": "queued", "pending_samples": pending}


@app.get("/api/jobs", tags=["retraining"])
def jobs(limit: int = 20):
    return {"items": store.list_jobs(limit)}


@app.get("/api/jobs/{job_id}", tags=["retraining"])
def job(job_id: str):
    j = store.get_job(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return j


# ----------------------------------------------------------------- monitoring
@app.get("/api/monitoring", tags=["monitoring"])
def monitoring(window_s: int = 3600):
    writer.flush()
    predictor.refresh()
    meta = predictor.metadata or {}
    expected = (meta.get("metrics") or {}).get("test", {}).get("accuracy")
    pstats = prediction_stats(store, settings, meta.get("drift_reference"), expected)
    ds = store.dataset_summary()
    state = registry.state()
    return {
        "system": {"replica": REPLICA, "uptime_s": round(time.time() - STARTED, 1), "model_version": predictor.version,
                   "writer_dropped": writer.dropped, "time": utcnow()},
        "requests": request_stats(store, window_s),
        "predictions": pstats,
        "data": {"total_samples": ds["total_samples"], "pending_for_training": ds["pending_for_training"]},
        "retraining_signal": retraining_signal(pstats, ds["pending_for_training"], settings),
        "models": {"active": state["active"], "count": len(state["versions"]),
                   "history": [{"version": v, **info} for v, info in state["versions"].items()]},
        "jobs": store.list_jobs(10),
    }


@app.get("/api/insights", tags=["monitoring"])
def insights():
    path = settings.data_dir / "insights.json"
    if not path.exists():
        raise HTTPException(404, "insights not generated; run scripts/build_insights.py")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------- UI
FRONTEND = PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")
