from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "pillow",
    "scikit-learn",
    "tensorflow",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def repository_state(repo_root: Path) -> dict[str, Any]:
    revision = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    return {
        "commit": revision,
        "dirty": None if status is None else bool(status),
        "remote": _git(repo_root, "remote", "get-url", "origin"),
    }


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def runtime_provenance(repo_root: Path, tensorflow: Any) -> dict[str, Any]:
    devices = [
        {"name": device.name, "type": device.device_type}
        for device in tensorflow.config.list_physical_devices()
    ]
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": dependency_versions(),
        "devices": devices,
        "repository": repository_state(repo_root),
    }


def dataset_summary(frame: Any) -> dict[str, Any]:
    by_split = {str(key): int(value) for key, value in frame["split"].value_counts().items()}
    by_class_split = (
        frame.groupby(["class_id", "split"]).size().unstack(fill_value=0).astype(int)
    )
    return {
        "images": int(len(frame)),
        "leaf_groups": int(frame["leaf_group"].nunique()),
        "by_split": by_split,
        "by_class_and_split": {
            str(class_id): {str(split): int(count) for split, count in row.items()}
            for class_id, row in by_class_split.iterrows()
        },
    }


def artifact_hashes(output: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256_file(output / name) for name in names if (output / name).is_file()}
