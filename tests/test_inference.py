import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from api.inference import (
    ModelRuntime,
    _softmax,
    _temperature_scale,
)


def test_softmax_is_stable_and_normalized():
    probabilities = _softmax(np.asarray([1000.0, 1001.0, 999.0]))
    assert np.isclose(probabilities.sum(), 1.0)
    assert int(np.argmax(probabilities)) == 1


def test_temperature_scaling_preserves_rank_and_softens_distribution():
    original = np.asarray([0.8, 0.15, 0.05])
    scaled = _temperature_scale(original, 2.0)
    assert np.isclose(scaled.sum(), 1.0)
    assert int(np.argmax(scaled)) == int(np.argmax(original))
    assert scaled.max() < original.max()


def metadata_payload():
    return {
        "version": "1.2.3",
        "input_size": [64, 64],
        "classes": [
            {"id": "a", "label": "A"},
            {"id": "b", "label": "B"},
        ],
        "calibration": {"fitted": True, "temperature": 2.0},
        "rejection": {"confidence_threshold": 0.7},
        "disclaimer": "Use carefully",
    }


def test_metadata_and_artifact_identity_are_loaded(tmp_path: Path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload()), encoding="utf-8")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")

    runtime = ModelRuntime(model_path, metadata_path)
    assert runtime.metadata.version == "1.2.3"
    assert runtime.metadata.input_size == (64, 64)
    assert runtime.metadata.calibrated is True
    assert len(runtime.sha256) == 64


def test_unsupported_artifact_fails_explicitly(tmp_path: Path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload()), encoding="utf-8")
    model_path = tmp_path / "model.txt"
    model_path.write_text("bad", encoding="utf-8")
    runtime = ModelRuntime(model_path, metadata_path)
    with pytest.raises(RuntimeError, match="Unsupported model artifact"):
        runtime.load()


def test_predict_requires_loaded_runtime(tmp_path: Path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload()), encoding="utf-8")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    runtime = ModelRuntime(model_path, metadata_path)
    with pytest.raises(RuntimeError, match="has not been loaded"):
        runtime.predict(np.zeros((1, 64, 64, 3), dtype=np.float32))


class FakeInput:
    name = "image"


class FakeOnnxSession:
    def get_inputs(self):
        return [FakeInput()]

    def run(self, _outputs, inputs):
        assert inputs["image"].shape == (1, 4, 4, 3)
        logits = np.asarray([[3.0, 1.0]], dtype=np.float32)
        features = np.ones((1, 2, 2, 3), dtype=np.float32)
        return [logits, features]


def test_onnx_logits_are_calibrated_and_cam_is_encoded(tmp_path: Path):
    metadata_path = tmp_path / "metadata.json"
    payload = metadata_payload()
    payload["input_size"] = [4, 4]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    runtime = ModelRuntime(model_path, metadata_path)
    runtime._runtime = FakeOnnxSession()
    runtime._classifier_weights = np.asarray(
        [[1.0, -1.0], [0.5, 0.2], [0.1, 0.3]], dtype=np.float32
    )

    probabilities, encoded = runtime.predict(
        np.zeros((1, 4, 4, 3), dtype=np.float32), include_explanation=True
    )
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0] > probabilities[1]
    assert encoded


def test_cam_output_is_a_valid_png():
    runtime = object.__new__(ModelRuntime)
    runtime._classifier_weights = np.ones((3, 2), dtype=np.float32)
    encoded = runtime._class_activation_map(np.zeros((2, 2, 3)), 0)
    import base64
    import io

    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.format == "PNG"
        assert image.size == (2, 2)
