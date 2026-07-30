from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .provenance import sha256_file


REQUIRED_ARTIFACTS = ("best.keras", "model.keras", "history.json", "epochs.csv")


def validate_run(
    run_dir: Path,
    manifest: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise ValueError("run.json is missing")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("schema_version") != 1:
        raise ValueError("unsupported or missing run schema_version")
    if run.get("test_partition_used_for_training") is not False:
        raise ValueError("run does not attest that the test partition stayed locked")

    repository = run.get("runtime", {}).get("repository", {})
    if not repository.get("commit"):
        raise ValueError("source commit is missing")
    if repository.get("dirty") is True and not allow_dirty:
        raise ValueError("publishable runs must be produced from a clean source tree")

    recorded_hashes = run.get("artifacts", {})
    for name in REQUIRED_ARTIFACTS:
        path = run_dir / name
        if not path.is_file():
            raise ValueError(f"required artifact is missing: {name}")
        expected = recorded_hashes.get(name)
        if not expected:
            raise ValueError(f"artifact checksum is missing: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"artifact checksum mismatch: {name}")

    if manifest is not None:
        actual_manifest_hash = sha256_file(manifest)
        if actual_manifest_hash != run.get("manifest_sha256"):
            raise ValueError("manifest checksum does not match the training run")

    dataset = run.get("dataset", {})
    if not dataset.get("images") or not dataset.get("leaf_groups"):
        raise ValueError("dataset composition evidence is missing")
    if set(dataset.get("by_split", {})) != {"train", "validation", "test"}:
        raise ValueError("run must record train, validation, and locked test partitions")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a reproducible training run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    run = validate_run(args.run_dir, args.manifest, args.allow_dirty)
    print(
        json.dumps(
            {
                "status": "valid",
                "run_name": run["run_name"],
                "commit": run["runtime"]["repository"]["commit"],
            },
            indent=2,
        )
    )
