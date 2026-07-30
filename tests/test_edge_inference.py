from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from edge.raspberry_pi.infer import (
    dequantize_output,
    latency_summary,
    quantize_input,
    softmax,
)
from training.tomato_guard_ml.export_tflite import representative_images


def test_quantized_input_round_trip() -> None:
    detail = {"dtype": np.uint8, "quantization": (0.5, 10)}
    values = np.asarray([[[[0.0, 10.0, 20.0]]]], dtype=np.float32)
    quantized = quantize_input(values, detail)
    assert quantized.dtype == np.uint8
    restored = dequantize_output(quantized, detail)
    assert np.allclose(restored, values, atol=0.25)


def test_invalid_quantization_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid scale"):
        quantize_input(
            np.zeros((1, 1), dtype=np.float32),
            {"dtype": np.int8, "quantization": (0.0, 0)},
        )


def test_temperature_softmax_is_normalized() -> None:
    probabilities = softmax(np.asarray([2.0, 1.0, 0.0]), temperature=2.0)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_latency_summary_uses_warm_percentiles() -> None:
    summary = latency_summary([10.0, 20.0, 30.0, 40.0])
    assert summary["p50_ms"] == pytest.approx(25.0)
    assert summary["p95_ms"] > summary["p50_ms"]
    assert summary["maximum_ms"] == 40.0


def test_representative_images_use_locked_train_rows(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    rows = []
    for index, split in enumerate(("test", "train", "train")):
        image_name = f"leaf-{index}.png"
        Image.new("RGB", (12, 8), color=(index * 30, 100, 20)).save(root / image_name)
        rows.append(
            {
                "relative_path": image_name,
                "split": split,
                "sha256": f"{index:064x}",
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    batches = list(representative_images(manifest, root, (16, 16), samples=1))
    assert len(batches) == 1
    assert batches[0][0].shape == (1, 16, 16, 3)
    assert batches[0][0].dtype == np.float32
