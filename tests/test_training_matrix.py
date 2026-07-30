from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_training_matrix import read_run, select_candidate


def make_run(
    root: Path, architecture: str, seed: int, loss: float, manifest: str = "a" * 64
) -> dict[str, object]:
    run_dir = root / f"{architecture}-seed{seed}"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "architecture": architecture,
                "seed": seed,
                "manifest_sha256": manifest,
                "config_sha256": "b" * 64,
                "test_partition_used_for_training": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps({"val_loss": [loss + 0.2, loss]}), encoding="utf-8"
    )
    return read_run(run_dir)


def test_selects_only_by_validation_loss_across_three_seeds(tmp_path: Path) -> None:
    records = [
        make_run(tmp_path, "mobilenetv3", 17, 0.5),
        make_run(tmp_path, "mobilenetv3", 42, 0.3),
        make_run(tmp_path, "mobilenetv3", 73, 0.4),
        make_run(tmp_path, "baseline", 42, 0.1),
    ]
    selected = select_candidate(records)
    assert selected["architecture"] == "mobilenetv3"
    assert selected["seed"] == 42


def test_rejects_mixed_manifests(tmp_path: Path) -> None:
    records = [
        make_run(tmp_path, "mobilenetv3", 17, 0.5),
        make_run(tmp_path, "mobilenetv3", 42, 0.3),
        make_run(tmp_path, "mobilenetv3", 73, 0.4, "c" * 64),
    ]
    with pytest.raises(ValueError, match="mixes dataset manifests"):
        select_candidate(records)


def test_rejects_run_without_locked_test_isolation(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "architecture": "mobilenetv3",
                "seed": 42,
                "manifest_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "test_partition_used_for_training": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps({"val_loss": [0.4]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="locked-test isolation"):
        read_run(run_dir)
