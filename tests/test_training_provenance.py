from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training.tomato_guard_ml.provenance import (
    artifact_hashes,
    dataset_summary,
    json_sha256,
    sha256_file,
)
from training.tomato_guard_ml.validate_run import REQUIRED_ARTIFACTS, validate_run


def test_json_hash_is_order_independent() -> None:
    assert json_sha256({"a": 1, "b": 2}) == json_sha256({"b": 2, "a": 1})


def test_dataset_summary_records_split_and_class_counts() -> None:
    frame = pd.DataFrame(
        [
            {"class_id": "healthy", "split": "train", "leaf_group": "a"},
            {"class_id": "healthy", "split": "validation", "leaf_group": "b"},
            {"class_id": "blight", "split": "test", "leaf_group": "c"},
        ]
    )
    summary = dataset_summary(frame)
    assert summary["images"] == 3
    assert summary["leaf_groups"] == 3
    assert summary["by_split"] == {"train": 1, "validation": 1, "test": 1}
    assert summary["by_class_and_split"]["healthy"]["train"] == 1


def _write_valid_run(run_dir: Path, manifest: Path) -> None:
    run_dir.mkdir()
    for name in REQUIRED_ARTIFACTS:
        (run_dir / name).write_bytes(f"artifact:{name}".encode())
    hashes = artifact_hashes(run_dir, REQUIRED_ARTIFACTS)
    run = {
        "schema_version": 1,
        "run_name": "test-run",
        "manifest_sha256": sha256_file(manifest),
        "test_partition_used_for_training": False,
        "runtime": {
            "repository": {
                "commit": "a" * 40,
                "dirty": False,
            }
        },
        "dataset": {
            "images": 30,
            "leaf_groups": 30,
            "by_split": {"train": 20, "validation": 5, "test": 5},
        },
        "artifacts": hashes,
    }
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")


def test_validate_run_accepts_complete_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("relative_path,class_id\nleaf.jpg,healthy\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_valid_run(run_dir, manifest)
    assert validate_run(run_dir, manifest)["run_name"] == "test-run"


def test_validate_run_rejects_tampered_artifact(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("manifest", encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_valid_run(run_dir, manifest)
    (run_dir / "model.keras").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_run(run_dir, manifest)


def test_validate_run_rejects_dirty_source_by_default(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("manifest", encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_valid_run(run_dir, manifest)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["runtime"]["repository"]["dirty"] = True
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="clean source tree"):
        validate_run(run_dir, manifest)
    assert validate_run(run_dir, manifest, allow_dirty=True)["run_name"] == "test-run"


def test_validate_run_rejects_test_partition_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("manifest", encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_valid_run(run_dir, manifest)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["test_partition_used_for_training"] = True
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="test partition"):
        validate_run(run_dir, manifest)
