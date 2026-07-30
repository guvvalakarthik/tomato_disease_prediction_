from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (17, 42, 73)
ARCHITECTURES = ("baseline", "mobilenetv3")


def read_run(run_dir: Path) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    history_path = run_dir / "history.json"
    if not run_path.is_file() or not history_path.is_file():
        raise ValueError(f"Incomplete training run: {run_dir}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    validation_losses = history.get("val_loss", [])
    if not validation_losses:
        raise ValueError(f"Run has no validation loss: {run_dir}")
    if run.get("test_partition_used_for_training") is not False:
        raise ValueError(f"Run does not prove locked-test isolation: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "architecture": run["architecture"],
        "seed": int(run["seed"]),
        "best_validation_loss": float(min(validation_losses)),
        "manifest_sha256": run["manifest_sha256"],
        "config_sha256": run["config_sha256"],
    }


def select_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in records if r["architecture"] == "mobilenetv3"]
    if len(candidates) < 3:
        raise ValueError("Release selection requires at least three MobileNetV3 seeds")
    manifests = {r["manifest_sha256"] for r in records}
    if len(manifests) != 1:
        raise ValueError("Experiment matrix mixes dataset manifests")
    return min(
        candidates,
        key=lambda r: (r["best_validation_loss"], r["seed"], r["run_dir"]),
    )


def command_for(
    manifest: Path,
    dataset_root: Path,
    config: Path,
    output: Path,
    architecture: str,
    seed: int,
) -> list[str]:
    run_name = f"{architecture}-seed{seed}"
    return [
        sys.executable,
        "-m",
        "training.tomato_guard_ml.train",
        "--manifest",
        str(manifest),
        "--dataset-root",
        str(dataset_root),
        "--config",
        str(config),
        "--output",
        str(output / run_name),
        "--architecture",
        architecture,
        "--seed",
        str(seed),
        "--run-name",
        run_name,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a locked-manifest baseline/MobileNet multi-seed matrix"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("training/config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("model_version") != "1.0.0":
        raise ValueError("Release-1 matrix requires model_version 1.0.0")
    if len(set(args.seeds)) < 3:
        raise ValueError("At least three distinct seeds are required")
    if not args.manifest.is_file() or not args.dataset_root.is_dir():
        raise FileNotFoundError("Clean manifest and dataset root must exist")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty matrix: {args.output}")

    commands = [
        command_for(
            args.manifest,
            args.dataset_root,
            args.config,
            args.output,
            architecture,
            seed,
        )
        for architecture in ARCHITECTURES
        for seed in sorted(set(args.seeds))
    ]
    if args.dry_run:
        print(json.dumps({"commands": commands}, indent=2))
        return

    args.output.mkdir(parents=True)
    for command in commands:
        subprocess.run(command, check=True)
    records = [read_run(path) for path in sorted(args.output.iterdir()) if path.is_dir()]
    candidate = select_candidate(records)
    matrix = {
        "schema_version": 1,
        "selection_metric": "minimum_validation_loss",
        "test_or_field_used_for_selection": False,
        "runs": records,
        "selected_candidate": candidate,
    }
    (args.output / "experiment-matrix.json").write_text(
        json.dumps(matrix, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
