from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

from scripts.export_legacy_onnx import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_onnx_export_is_deterministic_and_executable(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "model" / "tomatoes.h5"
    tracked = PROJECT_ROOT / "model" / "tomatoes.onnx"
    generated = tmp_path / "tomatoes.onnx"

    build_model(source, generated)

    assert _sha256(generated) == _sha256(tracked)
    session = ort.InferenceSession(str(generated), providers=["CPUExecutionProvider"])
    batch = np.random.default_rng(42).uniform(
        0.0, 255.0, size=(1, 128, 128, 3)
    ).astype(np.float32)
    probabilities = session.run(None, {"image": batch})[0]

    assert probabilities.shape == (1, 10)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0], atol=1e-6)
