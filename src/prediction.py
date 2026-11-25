"""Online inference.

``Predictor`` owns the serving model for one process. It reads the registry to find
the active version, and on every call checks (cheaply, by file mtime) whether the
registry changed - so when the retraining worker promotes a new model, or an operator
rolls back, every API replica switches over without a restart.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, POSITIVE_CLASS, Settings, get_settings
from .preprocessing import compute_stats, load_image, scale_uint8, to_uint8_array
from .registry import ModelRegistry

log = logging.getLogger(__name__)


class ModelNotReady(RuntimeError):
    pass


class Predictor:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.registry = ModelRegistry(self.settings.models_dir)
        self._lock = threading.Lock()
        self._model = None
        self._infer = None
        self.version: str | None = None
        self.metadata: dict = {}
        self._seen_mtime = -1.0
        self.loaded_at: float | None = None

    # ---------------------------------------------------------------- lifecycle
    def refresh(self, force: bool = False) -> bool:
        """Load the active model if the registry changed. Returns True on (re)load."""
        mtime = self.registry.mtime()
        if not force and mtime == self._seen_mtime:
            return False
        with self._lock:
            if not force and mtime == self._seen_mtime:
                return False
            active = self.registry.active_version()
            self._seen_mtime = mtime
            if active is None or (active == self.version and not force):
                return False
            import tensorflow as tf

            t0 = time.time()
            model = tf.keras.models.load_model(self.registry.model_path(active), compile=False)
            infer = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
            size = self.settings.image_size
            infer(np.zeros((1, size, size, 3), np.float32))  # warm-up trace
            self._model, self._infer = model, infer
            previous, self.version = self.version, active
            self.metadata = self.registry.metadata(active)
            self.loaded_at = time.time()
            log.info("loaded model %s (previous %s) in %.2fs", active, previous, time.time() - t0)
            return True

    @property
    def ready(self) -> bool:
        return self._infer is not None

    # ---------------------------------------------------------------- inference
    def predict_arrays(self, batch_uint8: np.ndarray) -> np.ndarray:
        self.refresh()
        if not self.ready:
            raise ModelNotReady("no active model in registry")
        with self._lock:  # snapshot the function so a concurrent reload can't swap it mid-call
            infer = self._infer
        return np.asarray(infer(scale_uint8(batch_uint8))).reshape(-1)

    def predict_bytes(self, data: bytes) -> dict:
        s = self.settings
        t0 = time.perf_counter()
        img = load_image(data, max_bytes=s.max_upload_bytes, min_side=s.min_image_side, max_side=s.max_image_side)
        arr = to_uint8_array(img, s.image_size)
        p = float(self.predict_arrays(arr[None])[0])
        return self._format(p, compute_stats(img).as_dict(), (time.perf_counter() - t0) * 1000)

    def _format(self, p_pos: float, stats: dict, latency_ms: float) -> dict:
        s = self.settings
        label = POSITIVE_CLASS if p_pos >= s.decision_threshold else CLASS_NAMES[0]
        probs = {CLASS_NAMES[0]: round(1 - p_pos, 6), CLASS_NAMES[1]: round(p_pos, 6)}
        margin = abs(p_pos - s.decision_threshold)
        return {
            "predicted_class": label,
            "confidence": round(max(probs.values()), 6),
            "probabilities": probs,
            "low_confidence": margin < s.low_confidence_margin,
            "recommendation": ("Refer for microscopist review: low model confidence" if margin < s.low_confidence_margin
                               else "Parasite features detected - confirm by microscopy" if label == POSITIVE_CLASS
                               else "No parasite features detected"),
            "threshold": s.decision_threshold,
            "model_version": self.version,
            "image_stats": stats,
            "latency_ms": round(latency_ms, 2),
        }


def predict_file(path: str | Path, predictor: Predictor | None = None) -> dict:
    """Convenience wrapper used by the notebook and CLI."""
    predictor = predictor or Predictor()
    return predictor.predict_bytes(Path(path).read_bytes())
