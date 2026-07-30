from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from training.tomato_guard_ml.constants import CLASS_IDS
from training.tomato_guard_ml.field_benchmark import validate_field_benchmark


def _manifest(tmp_path: Path, samples_per_class: int = 30) -> Path:
    rows = []
    for class_index, class_id in enumerate(CLASS_IDS):
        for index in range(samples_per_class):
            sample_id = f"field-{class_index:02d}-{index:03d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "relative_path": f"images/{sample_id}.jpg",
                    "sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
                    "class_id": class_id,
                    "expert_reviewer": f"reviewer-{index % 4}",
                    "expert_role": "plant_pathologist",
                    "reviewer_count": "2",
                    "adjudication_status": "agreed",
                    "review_status": "confirmed",
                    "consent_recorded": "true",
                    "capture_date": "2026-07-01",
                    "device_family": "Android phone",
                    "lighting": "natural shade",
                    "background": "field",
                    "location_bucket": "region-01",
                    "dataset_version": "1.0.0",
                }
            )
    path = tmp_path / "field.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _mutate(path: Path, column: str, value: str, row: int = 0) -> None:
    frame = pd.read_csv(path, dtype=str)
    frame.loc[row, column] = value
    frame.to_csv(path, index=False)


def test_valid_field_benchmark_produces_lock_summary(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    frame, summary = validate_field_benchmark(path, class_ids=list(CLASS_IDS))
    assert len(frame) == 300
    assert summary["sample_count"] == 300
    assert summary["dataset_version"] == "1.0.0"
    assert summary["minimum_independent_reviews"] == 2
    assert summary["field_data_used_for_training"] is False


def test_rejects_under_sized_field_benchmark(tmp_path: Path) -> None:
    path = _manifest(tmp_path, samples_per_class=20)
    with pytest.raises(ValueError, match="at least 300"):
        validate_field_benchmark(path, class_ids=list(CLASS_IDS))


def test_rejects_duplicate_image_hash(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    frame = pd.read_csv(path, dtype=str)
    frame.loc[1, "sha256"] = frame.loc[0, "sha256"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate sha256"):
        validate_field_benchmark(path, class_ids=list(CLASS_IDS))


def test_rejects_unconsented_sample(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    _mutate(path, "consent_recorded", "false")
    with pytest.raises(ValueError, match="Unconfirmed or unconsented"):
        validate_field_benchmark(path, class_ids=list(CLASS_IDS))


def test_rejects_single_reviewer(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    _mutate(path, "reviewer_count", "1")
    with pytest.raises(ValueError, match="at least 2 expert reviews"):
        validate_field_benchmark(path, class_ids=list(CLASS_IDS))


def test_rejects_path_traversal(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    _mutate(path, "relative_path", "../private.jpg")
    with pytest.raises(ValueError, match="stay inside field root"):
        validate_field_benchmark(path, class_ids=list(CLASS_IDS))


def test_verifies_image_checksum_when_root_is_provided(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    frame = pd.read_csv(path, dtype=str)
    field_root = tmp_path / "field"
    for row in frame.itertuples(index=False):
        image = field_root / row.relative_path
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(row.sample_id.encode())
        frame.loc[frame["sample_id"] == row.sample_id, "sha256"] = hashlib.sha256(
            row.sample_id.encode()
        ).hexdigest()
    frame.to_csv(path, index=False)
    validate_field_benchmark(path, field_root, list(CLASS_IDS))
    first = field_root / frame.iloc[0]["relative_path"]
    first.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_field_benchmark(path, field_root, list(CLASS_IDS))
