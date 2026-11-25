"""Model architectures, data loading, training and evaluation.

Two architectures are defined:

* ``build_baseline_cnn`` - a small convolutional network trained from scratch. It is
  the yardstick that tells us how much transfer learning actually buys.
* ``build_transfer_model`` - ImageNet-pretrained MobileNetV2 (alpha 1.0, 128 px)
  with a new classification head. MobileNetV2 was chosen for its accuracy per FLOP:
  the service runs on CPU-only containers and must keep p95 latency low under load.

Training is two-stage: (1) train the head on a frozen backbone, (2) unfreeze the top
inverted-residual blocks and fine-tune at a low learning rate with cosine decay. Both
stages use early stopping on validation loss with best-weight restoration.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CLASS_NAMES  # noqa: E402
from .preprocessing import augment_batch, load_image, to_uint8_array  # noqa: E402

log = logging.getLogger(__name__)
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


# --------------------------------------------------------------------------- data
def read_manifest(manifest: Path) -> list[dict]:
    with open(manifest, newline="") as fh:
        return list(csv.DictReader(fh))


def load_split_arrays(data_dir: Path, split: str, size: int, cache: bool = True,
                      limit: int = 0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load one split as uint8 images + int labels, caching to ``data/cache``.

    The cache is keyed by split and image size; it holds exactly what
    ``preprocessing.to_uint8_array`` produces, so cached and online paths agree.
    """
    rows = [r for r in read_manifest(data_dir / "manifest.csv") if r["split"] == split]
    rows.sort(key=lambda r: r["path"])
    if limit:
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), size=min(limit, len(rows)), replace=False))]
    cache_file = data_dir / "cache" / f"{split}_{size}{'_l' + str(limit) if limit else ''}.npz"
    if cache and cache_file.exists():
        z = np.load(cache_file)
        if len(z["y"]) == len(rows):
            return z["x"], z["y"], rows
    x = np.empty((len(rows), size, size, 3), dtype=np.uint8)
    for i, r in enumerate(rows):
        x[i] = to_uint8_array(load_image((data_dir / r["path"]).read_bytes()), size)
    y = np.array([LABEL_TO_INDEX[r["label"]] for r in rows], dtype=np.int32)
    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, x=x, y=y)
    return x, y, rows


def make_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, augment: bool = False,
                 shuffle: bool = False, seed: int = 42) -> tf.data.Dataset:
    """Stream uint8 batches by index.

    ``from_tensor_slices`` + a full shuffle buffer would copy the (up to ~1 GB) array
    twice; indexing the NumPy array per batch keeps memory flat. Shuffling permutes
    indices with a seeded generator, re-drawn every epoch.
    """
    y = y.astype(np.float32)
    size = x.shape[1]
    rng = np.random.default_rng(seed)

    def gen():
        idx = rng.permutation(len(x)) if shuffle else np.arange(len(x))
        for i in range(0, len(idx), batch_size):
            b = np.sort(idx[i:i + batch_size])
            yield x[b], y[b]

    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec((None, size, size, 3), tf.uint8), tf.TensorSpec((None,), tf.float32)))
    ds = ds.apply(tf.data.experimental.assert_cardinality(int(np.ceil(len(x) / batch_size))))
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32) / 127.5 - 1.0, b), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda a, b: (augment_batch(a), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------------ architectures
def build_baseline_cnn(image_size: int, dropout: float = 0.3, l2: float = 1e-4) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(l2)
    inp = tf.keras.Input((image_size, image_size, 3), name="image")
    # stride-2 stem: halves resolution up front, ~4x cheaper on CPU, little accuracy cost on 128 px cells
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=reg)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for filters in (32, 64, 128, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="p_parasitized")(x)
    return tf.keras.Model(inp, out, name="baseline_cnn")


def build_transfer_model(image_size: int, weights: str | Path | None, dropout: float = 0.3,
                         l2: float = 1e-4, dense_units: int = 128) -> tf.keras.Model:
    """MobileNetV2 backbone + regularised head. ``weights`` is a local .h5 path."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights=None, alpha=1.0)
    if weights:
        base.load_weights(str(weights))
    base.trainable = False
    reg = tf.keras.regularizers.l2(l2)
    inp = tf.keras.Input((image_size, image_size, 3), name="image")
    x = base(inp, training=False)  # keep BatchNorm in inference mode even when unfrozen
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="drop1")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="fc")(x)
    x = tf.keras.layers.Dropout(dropout, name="drop2")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="p_parasitized")(x)
    return tf.keras.Model(inp, out, name="mobilenetv2_transfer")


def backbone(model: tf.keras.Model) -> tf.keras.Model:
    return next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))


def unfreeze_top(model: tf.keras.Model, from_block: int = 13) -> int:
    """Unfreeze MobileNetV2 layers from ``block_{from_block}`` upward (BN stays frozen)."""
    base = backbone(model)
    base.trainable = True
    start = next(i for i, layer in enumerate(base.layers) if layer.name.startswith(f"block_{from_block}_"))
    n = 0
    for i, layer in enumerate(base.layers):
        trainable = i >= start and not isinstance(layer, tf.keras.layers.BatchNormalization)
        layer.trainable = trainable
        n += int(trainable)
    return n


def compile_model(model: tf.keras.Model, lr: float | tf.keras.optimizers.schedules.LearningRateSchedule,
                  optimizer: str = "adam") -> None:
    opt = {"adam": tf.keras.optimizers.Adam, "sgd": lambda learning_rate: tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)}[optimizer](learning_rate=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])


class ProgressCallback(tf.keras.callbacks.Callback):
    """Forwards epoch metrics to a callable (the retraining worker writes them to the DB)."""

    def __init__(self, fn: Callable[[int, dict], None]):
        super().__init__()
        self.fn = fn

    def on_epoch_end(self, epoch, logs=None):
        self.fn(epoch, {k: float(v) for k, v in (logs or {}).items()})


@dataclass
class FitResult:
    history: dict
    seconds: float
    epochs_run: int


def fit(model: tf.keras.Model, train_ds, val_ds, epochs: int, patience: int,
        extra_callbacks: list | None = None, verbose: int = 2) -> FitResult:
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    cbs += extra_callbacks or []
    t0 = time.time()
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=verbose, shuffle=False)
    return FitResult({k: [float(v) for v in vals] for k, vals in h.history.items()},
                     time.time() - t0, len(h.history["loss"]))


# ---------------------------------------------------------------------- evaluation
def predict_proba(model: tf.keras.Model, x_uint8: np.ndarray, batch_size: int = 128) -> np.ndarray:
    ds = make_dataset(x_uint8, np.zeros(len(x_uint8)), batch_size)
    return model.predict(ds, verbose=0).reshape(-1)


def evaluate_probs(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Full metric set; Parasitized (index 1) is the positive class."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision_score(y, pred, pos_label=idx, zero_division=0)),
            "recall": float(recall_score(y, pred, pos_label=idx, zero_division=0)),
            "f1": float(f1_score(y, pred, pos_label=idx, zero_division=0)),
            "support": int((y == idx).sum()),
        }
    both = len(np.unique(y)) > 1
    return {
        "n": int(len(y)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "precision": per_class["Parasitized"]["precision"],
        "recall": per_class["Parasitized"]["recall"],
        "f1": per_class["Parasitized"]["f1"],
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "confusion_matrix": {"labels": list(CLASS_NAMES), "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]},
        "per_class": per_class,
    }


def load_keras(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, compile=False)


def array_from_bytes(data: bytes, size: int) -> np.ndarray:
    return to_uint8_array(load_image(data), size)
