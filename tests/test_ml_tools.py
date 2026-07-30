from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from training.tomato_guard_ml.calibration import (
    choose_rejection_threshold,
    fit_temperature,
    negative_log_likelihood,
    softmax,
)
from training.tomato_guard_ml.manifest import (
    assign_grouped_splits,
    validate_duplicates,
    validate_field_manifest,
)


def test_temperature_scaling_reduces_validation_nll():
    logits = np.asarray([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    labels = np.asarray([0, 1, 1, 1])
    temperature = fit_temperature(logits, labels)
    assert temperature > 1.0
    assert negative_log_likelihood(logits, labels, temperature) < negative_log_likelihood(
        logits, labels, 1.0
    )


def test_rejection_threshold_uses_id_precision_and_ood_acceptance():
    validation = np.asarray(
        [[0.95, 0.05], [0.9, 0.1], [0.4, 0.6], [0.55, 0.45]]
    )
    labels = np.asarray([0, 0, 1, 1])
    ood = np.asarray([[0.52, 0.48], [0.5, 0.5], [0.51, 0.49]])
    result = choose_rejection_threshold(validation, labels, ood)
    assert result["validated"] is True
    assert result["validation_precision"] >= 0.9
    assert result["ood_acceptance"] <= 0.1
    assert np.isclose(softmax(np.log(validation)).sum(axis=1), 1.0).all()


def records_for_split():
    records = []
    for class_id in ("a", "b"):
        for group in range(6):
            records.append(
                {
                    "sample_id": f"{class_id}-{group}",
                    "relative_path": f"{class_id}/{group}.jpg",
                    "class_id": class_id,
                    "sha256": f"{class_id}{group}",
                    "leaf_group": f"{class_id}-leaf-{group}",
                }
            )
    return records


def test_grouped_split_is_deterministic_and_leak_free():
    first = assign_grouped_splits(records_for_split(), seed=42)
    second = assign_grouped_splits(records_for_split(), seed=42)
    assert [record["split"] for record in first] == [
        record["split"] for record in second
    ]
    assert {record["split"] for record in first} == {"train", "validation", "test"}
    group_splits = {}
    for record in first:
        group_splits.setdefault(record["leaf_group"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_conflicting_exact_duplicates_fail():
    records = [
        {"sha256": "same", "class_id": "a"},
        {"sha256": "same", "class_id": "b"},
    ]
    with pytest.raises(ValueError, match="conflicting labels"):
        validate_duplicates(records)


def test_field_manifest_requires_confirmed_consent(tmp_path: Path):
    path = tmp_path / "field.csv"
    fields = [
        "sample_id", "relative_path", "class_id", "expert_reviewer",
        "review_status", "consent_recorded", "capture_date", "device_family",
        "lighting",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "one", "relative_path": "one.jpg", "class_id": "a",
                "expert_reviewer": "reviewer-1", "review_status": "unresolved",
                "consent_recorded": "false", "capture_date": "2026-07-01",
                "device_family": "phone", "lighting": "shade",
            }
        )
    with pytest.raises(ValueError, match="Unconfirmed or unconsented"):
        validate_field_manifest(path)
