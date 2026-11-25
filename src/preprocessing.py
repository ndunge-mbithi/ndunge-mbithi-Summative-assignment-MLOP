"""Image validation and preprocessing.

One implementation serves training, the retraining worker and the online API, so
the model never sees differently-prepared pixels at train and serve time.

Contract: RGB, resized with aspect ratio preserved and zero (black) padding to a
square -- the NIH crops already sit on a black background, so padding with black
matches the training distribution -- then float32 scaled to [-1, 1] as expected by
MobileNetV2. Models embed no rescaling layer; scaling happens here only.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"PNG", "JPEG", "BMP", "TIFF", "WEBP"}


class ImageValidationError(ValueError):
    """Raised for unsupported, corrupted or out-of-bounds images."""


@dataclass(frozen=True)
class ImageStats:
    """Cheap per-image descriptors used for EDA, the UI and drift monitoring."""

    width: int
    height: int
    mean_r: float
    mean_g: float
    mean_b: float
    brightness: float  # mean luminance of cell (non-background) pixels, 0-255
    stain_fraction: float  # share of cell pixels that are dark purple (Giemsa-stained chromatin)
    background_fraction: float  # share of near-black padding pixels

    def as_dict(self) -> dict:
        return {k: round(float(v), 5) for k, v in self.__dict__.items()}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_image(data: bytes, *, max_bytes: int = 5 * 1024 * 1024, min_side: int = 16,
               max_side: int = 2048) -> Image.Image:
    """Decode and validate raw bytes, returning an RGB PIL image."""
    if not data:
        raise ImageValidationError("empty file")
    if len(data) > max_bytes:
        raise ImageValidationError(f"file too large ({len(data)} bytes > {max_bytes})")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()  # detects truncated / corrupted files
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(f"not a readable image: {exc.__class__.__name__}") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError(f"unsupported format {fmt}; allowed {sorted(ALLOWED_FORMATS)}")
    w, h = img.size
    if min(w, h) < min_side or max(w, h) > max_side:
        raise ImageValidationError(f"image dimensions {w}x{h} outside [{min_side}, {max_side}]")
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_pad(img: Image.Image, size: int) -> Image.Image:
    """Aspect-preserving resize onto a black square canvas."""
    scale = size / max(img.size)
    if scale != 1.0:  # scale the longest side to `size` (up or down) so the cell fills the canvas
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def to_uint8_array(img: Image.Image, size: int) -> np.ndarray:
    """PIL image -> uint8 (size, size, 3). Compact form used for caches and storage."""
    return np.asarray(resize_pad(img, size), dtype=np.uint8)


def scale_uint8(arr: np.ndarray) -> np.ndarray:
    """uint8 pixels -> float32 in [-1, 1] (MobileNetV2 convention)."""
    return arr.astype(np.float32) / 127.5 - 1.0


def to_model_array(img: Image.Image, size: int) -> np.ndarray:
    """PIL image -> float32 (size, size, 3) in [-1, 1]."""
    return scale_uint8(to_uint8_array(img, size))


def preprocess_bytes(data: bytes, size: int, **limits) -> tuple[np.ndarray, ImageStats]:
    img = load_image(data, **limits)
    return to_model_array(img, size), compute_stats(img)


def compute_stats(img: Image.Image) -> ImageStats:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    background = lum < 12.0
    cell = ~background
    n_cell = max(int(cell.sum()), 1)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    # Giemsa stains parasite chromatin dark purple: noticeably darker than the
    # pink erythrocyte body, with blue >= green and red > green.
    stained = cell & (lum < 0.75 * float(lum[cell].mean() if cell.any() else 0.0)) & (b >= g) & (r > g)
    return ImageStats(
        width=img.width,
        height=img.height,
        mean_r=float(r[cell].mean()) if cell.any() else 0.0,
        mean_g=float(g[cell].mean()) if cell.any() else 0.0,
        mean_b=float(b[cell].mean()) if cell.any() else 0.0,
        brightness=float(lum[cell].mean()) if cell.any() else 0.0,
        stain_fraction=float(stained.sum()) / n_cell,
        background_fraction=float(background.mean()),
    )


def augment_batch(tf_images, seed=None):  # pragma: no cover - thin TF wrapper, exercised in training
    """Label-preserving augmentation for single cells (orientation is meaningless)."""
    import tensorflow as tf

    x = tf.image.random_flip_left_right(tf_images)
    x = tf.image.random_flip_up_down(x)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    # Mild stain/illumination variation between slides and microscopes.
    x = tf.image.random_brightness(x, 0.08)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    return tf.clip_by_value(x, -1.0, 1.0)
