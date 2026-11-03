import io

import numpy as np
from PIL import Image

from app.vision import preprocess_image


def test_preprocess_image_reduced_jpeg_decode():
    y, x = np.mgrid[0:1200, 0:1600]
    arr = np.stack([127 + 100 * np.sin(x * f + y * f / 2) for f in (0.01, 0.02, 0.005)], -1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG")

    img = Image.open(io.BytesIO(buf.getvalue()))
    out = preprocess_image(img)
    # Large JPEGs are decoded at reduced scale, but stay >= 2x the model input
    assert img.size == (800, 600)
    full = Image.open(io.BytesIO(buf.getvalue()))
    full.load()  # loaded images skip the reduced decode
    ref = preprocess_image(full)
    assert out.shape == ref.shape == (1, 3, 224, 224)
    assert np.abs(out - ref).mean() < 0.01
