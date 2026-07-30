from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from training.tomato_guard_ml.export_onnx import (
    compare_export_outputs,
    validate_cam_contract,
)


class FakeClassifier:
    def __init__(self, channels: int = 8, classes: int = 3):
        self.values = [
            np.zeros((channels, classes), dtype=np.float32),
            np.zeros(classes, dtype=np.float32),
        ]

    def get_weights(self):
        return self.values


class FakeModel:
    output_shape = (None, 3)

    def __init__(self, channels: int = 8):
        self.features = SimpleNamespace(output=SimpleNamespace(shape=(None, 7, 7, channels)))
        self.classifier = FakeClassifier()

    def get_layer(self, name: str):
        return getattr(self, name)


def test_cam_contract_binds_features_to_classifier_weights() -> None:
    output, weights = validate_cam_contract(FakeModel(), class_count=3)
    assert tuple(output.shape) == (None, 7, 7, 8)
    assert weights.shape == (8, 3)


def test_cam_contract_rejects_channel_mismatch() -> None:
    with pytest.raises(ValueError, match="channels"):
        validate_cam_contract(FakeModel(channels=7), class_count=3)


def test_export_parity_covers_logits_and_features() -> None:
    expected = [np.ones((2, 3)), np.ones((2, 7, 7, 8))]
    actual = [expected[0] + 1e-6, expected[1] - 1e-6]
    report = compare_export_outputs(expected, actual, tolerance=1e-4)
    assert report["passed"] is True
    assert report["output_contract"] == ["logits", "features"]


def test_export_parity_rejects_wrong_output_count_and_shape() -> None:
    with pytest.raises(ValueError, match="exactly 2 outputs"):
        compare_export_outputs([np.ones((1, 3))], [np.ones((1, 3))], 1e-4)
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_export_outputs(
            [np.ones((1, 3)), np.ones((1, 2, 2, 4))],
            [np.ones((1, 4)), np.ones((1, 2, 2, 4))],
            1e-4,
        )
