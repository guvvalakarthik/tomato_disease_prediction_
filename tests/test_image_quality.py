from __future__ import annotations

import numpy as np
from PIL import Image

from api.images import pre_model_rejection_reason


def test_textured_color_photo_passes_pre_model_guard() -> None:
    y, x = np.mgrid[0:128, 0:128]
    pixels = np.stack(
        (
            35 + (x % 31),
            90 + ((x + y) % 90),
            25 + (y % 37),
        ),
        axis=-1,
    ).astype(np.uint8)
    assert pre_model_rejection_reason(Image.fromarray(pixels)) is None


def test_textured_monochrome_input_is_rejected() -> None:
    y, x = np.mgrid[0:128, 0:128]
    gray = (30 + ((x + y) % 180)).astype(np.uint8)
    pixels = np.repeat(gray[..., None], 3, axis=2)
    reason = pre_model_rejection_reason(Image.fromarray(pixels))
    assert reason is not None
    assert "color information" in reason
