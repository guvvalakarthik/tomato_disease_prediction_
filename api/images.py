from __future__ import annotations

import io

import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import APIError


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
QUALITY_SAMPLE_SIZE = (128, 128)


def pre_model_rejection_reason(image: Image.Image) -> str | None:
    """Reject obviously unusable inputs before a classifier can be overconfident."""
    sample = ImageOps.fit(
        image.convert("RGB"),
        QUALITY_SAMPLE_SIZE,
        method=Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(sample, dtype=np.float32)
    luminance = (
        0.2126 * pixels[..., 0]
        + 0.7152 * pixels[..., 1]
        + 0.0722 * pixels[..., 2]
    )
    mean_luminance = float(luminance.mean())
    if mean_luminance < 8.0:
        return "The image is too dark to evaluate. Retake it in natural, even light."
    if mean_luminance > 247.0:
        return "The image is too bright to evaluate. Avoid glare and retake the photo."

    low, high = np.percentile(luminance, [1.0, 99.0])
    horizontal = np.abs(np.diff(luminance, axis=1)).mean()
    vertical = np.abs(np.diff(luminance, axis=0)).mean()
    mean_gradient = float((horizontal + vertical) / 2.0)
    if float(high - low) < 12.0 or mean_gradient < 1.0:
        return (
            "The image has too little visual detail and may be synthetic, blank, or "
            "out of focus. Upload a sharp photograph of one tomato leaf."
        )

    channel_spread = float((pixels.max(axis=2) - pixels.min(axis=2)).mean())
    if channel_spread < 2.0:
        return (
            "The image does not contain enough color information. Upload an unfiltered "
            "color photograph of a tomato leaf."
        )
    return None


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise APIError(
            415,
            "unsupported_media_type",
            "Only JPEG and PNG images are accepted.",
        )

    payload = await file.read(max_bytes + 1)
    if not payload:
        raise APIError(422, "empty_upload", "The uploaded file is empty.")
    if len(payload) > max_bytes:
        raise APIError(
            413,
            "upload_too_large",
            f"The image exceeds the {max_bytes // (1024 * 1024)} MB limit.",
        )
    return payload


def decode_image(payload: bytes, max_pixels: int) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.format not in ALLOWED_FORMATS:
                raise APIError(
                    415,
                    "unsupported_image_format",
                    "The file contents are not JPEG or PNG.",
                )
            if getattr(opened, "n_frames", 1) != 1:
                raise APIError(
                    422,
                    "animated_image",
                    "Animated images are not supported.",
                )
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise APIError(
                    422,
                    "unsafe_dimensions",
                    "The image dimensions are invalid or too large.",
                )
            opened.verify()

        with Image.open(io.BytesIO(payload)) as reopened:
            image = ImageOps.exif_transpose(reopened).convert("RGB")
            image.load()
            return image
    except APIError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise APIError(
            422,
            "invalid_image",
            "The upload could not be decoded as a safe image.",
        ) from None


def to_model_array(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = ImageOps.fit(image, size, method=Image.Resampling.BILINEAR)
    return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
