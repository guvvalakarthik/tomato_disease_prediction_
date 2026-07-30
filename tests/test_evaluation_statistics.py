from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from training.tomato_guard_ml.evaluate import (
    EVALUATION_ARTIFACTS,
    classification_bootstrap_intervals,
    write_evidence_manifest,
)


def test_stratified_classification_intervals_are_seeded_and_per_class() -> None:
    labels = np.repeat(np.arange(3), 8)
    predictions = labels.copy()
    predictions[[1, 10, 20]] = [1, 2, 0]
    first = classification_bootstrap_intervals(
        labels, predictions, ["a", "b", "c"], iterations=200, seed=17
    )
    second = classification_bootstrap_intervals(
        labels, predictions, ["a", "b", "c"], iterations=200, seed=17
    )
    assert first == second
    assert first["method"] == "class_stratified_percentile_bootstrap"
    assert set(first["per_class"]) == {"a", "b", "c"}
    assert 0.0 <= first["macro_f1"]["lower_95"] <= first["macro_f1"]["upper_95"] <= 1.0
    assert set(first["per_class"]["a"]) == {"precision", "recall", "f1"}


def test_intervals_reject_missing_class_support() -> None:
    with pytest.raises(ValueError, match="without samples"):
        classification_bootstrap_intervals(
            np.array([0, 0]), np.array([0, 0]), ["a", "b"], iterations=100
        )


def test_evaluation_manifest_hashes_every_required_artifact(tmp_path: Path) -> None:
    for index, name in enumerate(EVALUATION_ARTIFACTS):
        (tmp_path / name).write_bytes(f"artifact-{index}".encode())
    manifest = write_evidence_manifest(tmp_path)
    assert (tmp_path / "evidence_manifest.json").is_file()
    for name in EVALUATION_ARTIFACTS:
        expected = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert manifest["artifacts"][name]["sha256"] == expected


def test_evaluation_manifest_rejects_incomplete_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incomplete"):
        write_evidence_manifest(tmp_path)
