from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.config import Settings
from api.inference import ModelMetadata
from api.main import create_app


CLASSES = tuple(
    {"id": f"class_{index}", "label": f"Class {index}"} for index in range(10)
)


class DummyRuntime:
    def __init__(self, probabilities: list[float], heatmap: str | None = None):
        self.metadata = ModelMetadata(
            version="test-1",
            classes=CLASSES,
            input_size=(32, 32),
            temperature=1.0,
            confidence_threshold=0.6,
            calibrated=True,
            disclaimer="Test disclaimer",
        )
        self.sha256 = "a" * 64
        self.probabilities = np.asarray(probabilities)
        self.heatmap = heatmap
        self.last_shape = None

    def predict(self, batch, include_explanation=False):
        self.last_shape = batch.shape
        return self.probabilities, self.heatmap if include_explanation else None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        model_path=tmp_path / "unused.onnx",
        metadata_path=tmp_path / "unused.json",
        allowed_origins=("https://example.test",),
        max_upload_bytes=1024,
        max_image_pixels=1_000_000,
        log_level="WARNING",
    )


def image_bytes(mode="RGB", image_format="PNG") -> bytes:
    color = 128 if mode == "L" else (30, 120, 40, 180) if mode == "RGBA" else (30, 120, 40)
    image = Image.new(mode, (40, 20), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_health_and_accepted_prediction(settings):
    runtime = DummyRuntime([0.8, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.001, 0.001], "YWJj")
    with TestClient(create_app(settings, runtime)) as client:
        assert client.get("/health/live").json() == {
            "status": "ok",
            "model_version": None,
            "model_sha256": None,
            "class_count": None,
            "detail": None,
        }
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["class_count"] == 10

        response = client.post(
            "/v1/predict?explain=true",
            files={"file": ("leaf.png", image_bytes(), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "predicted"
    assert body["prediction"]["class_id"] == "class_0"
    assert body["explanation"]["png_base64"] == "YWJj"
    assert body["model"]["version"] == "test-1"
    assert runtime.last_shape == (1, 32, 32, 3)
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_low_confidence_is_uncertain(settings):
    runtime = DummyRuntime([0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.02])
    with TestClient(create_app(settings, runtime)) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("leaf.jpg", image_bytes(image_format="JPEG"), "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "uncertain"
    assert response.json()["prediction"] is None
    assert len(response.json()["top_predictions"]) == 3


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_grayscale_and_rgba_are_normalized_to_rgb(settings, mode):
    runtime = DummyRuntime([1.0] + [0.0] * 9)
    with TestClient(create_app(settings, runtime)) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("leaf.png", image_bytes(mode=mode), "image/png")},
        )
    assert response.status_code == 200
    assert runtime.last_shape[-1] == 3


def test_rejects_wrong_mime_type(settings):
    runtime = DummyRuntime([1.0] + [0.0] * 9)
    with TestClient(create_app(settings, runtime)) as client:
        response = client.post(
            "/v1/predict", files={"file": ("leaf.gif", b"GIF89a", "image/gif")}
        )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media_type"


def test_rejects_corrupt_and_oversized_images(settings):
    runtime = DummyRuntime([1.0] + [0.0] * 9)
    with TestClient(create_app(settings, runtime)) as client:
        corrupt = client.post(
            "/v1/predict", files={"file": ("leaf.png", b"not-a-png", "image/png")}
        )
        oversized = client.post(
            "/v1/predict", files={"file": ("leaf.png", b"x" * 1025, "image/png")}
        )
    assert corrupt.status_code == 422
    assert corrupt.json()["code"] == "invalid_image"
    assert oversized.status_code == 413
    assert oversized.json()["code"] == "upload_too_large"


def test_not_ready_has_503(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
