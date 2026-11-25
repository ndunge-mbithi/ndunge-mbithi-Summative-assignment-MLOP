import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (ImageValidationError, compute_stats, load_image, preprocess_bytes, resize_pad,
                               scale_uint8, to_model_array, to_uint8_array)


def png_bytes(w=120, h=90, color=(200, 120, 140), mode="RGB", fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (w, h), color if mode == "RGB" else 128).save(buf, format=fmt)
    return buf.getvalue()


def test_output_contract_shape_dtype_range():
    arr, stats = preprocess_bytes(png_bytes(), 128)
    assert arr.shape == (128, 128, 3) and arr.dtype == np.float32
    assert arr.min() >= -1.0 and arr.max() <= 1.0
    assert stats.width == 120 and stats.height == 90


def test_aspect_ratio_preserved_with_black_padding():
    img = resize_pad(Image.new("RGB", (200, 100), (255, 255, 255)), 128)
    a = np.asarray(img)
    assert a.shape == (128, 128, 3)
    assert a[0, 64].sum() == 0 and a[127, 64].sum() == 0  # padded rows top/bottom
    assert a[64, 64].sum() == 255 * 3  # content in the middle


def test_small_crops_are_upscaled_to_fill():
    a = to_uint8_array(Image.new("RGB", (40, 40), (255, 0, 0)), 128)
    assert (a[..., 0] == 255).mean() > 0.95


def test_uint8_and_float_paths_agree():
    img = load_image(png_bytes())
    assert np.allclose(scale_uint8(to_uint8_array(img, 128)), to_model_array(img, 128))


def test_grayscale_and_jpeg_are_converted_to_rgb():
    arr, _ = preprocess_bytes(png_bytes(mode="L"), 64)
    assert arr.shape == (64, 64, 3)
    arr, _ = preprocess_bytes(png_bytes(fmt="JPEG"), 64)
    assert arr.shape == (64, 64, 3)


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"),
    (b"this is not an image at all", "not a readable image"),
    (png_bytes()[:60], "not a readable image"),  # truncated / corrupted PNG
    (png_bytes(8, 8), "outside"),
])
def test_invalid_inputs_rejected(data, msg):
    with pytest.raises(ImageValidationError, match=msg):
        load_image(data)


def test_oversized_file_rejected():
    with pytest.raises(ImageValidationError, match="too large"):
        load_image(png_bytes(), max_bytes=100)


def test_gif_format_rejected():
    buf = io.BytesIO()
    Image.new("RGB", (50, 50)).save(buf, format="GIF")
    with pytest.raises(ImageValidationError, match="unsupported format"):
        load_image(buf.getvalue())


def test_stain_feature_separates_real_examples(test_images):
    stain = {"Parasitized": [], "Uninfected": []}
    for path, label in test_images:
        stain[label].append(compute_stats(Image.open(path).convert("RGB")).stain_fraction)
    assert np.median(stain["Parasitized"]) > np.median(stain["Uninfected"])
