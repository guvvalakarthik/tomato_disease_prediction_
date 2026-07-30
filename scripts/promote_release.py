from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable


REQUIRED_FILES = (
    "model.onnx",
    "classifier_weights.npy",
    "metadata.json",
    "metrics.json",
    "field_metrics.json",
)
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate(candidate: Path) -> None:
    validator = Path(__file__).with_name("validate_release.py")
    subprocess.run([sys.executable, str(validator), str(candidate)], check=True)


def build_manifest(release_dir: Path, version: str, source_commit: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "version": version,
        "source_commit": source_commit,
        "promoted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "immutable": True,
        "files": {
            name: {"sha256": sha256_file(release_dir / name), "bytes": (release_dir / name).stat().st_size}
            for name in REQUIRED_FILES
        },
    }


def verify_manifest(release_dir: Path) -> dict[str, object]:
    manifest_path = release_dir / "release-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("release-manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("immutable") is not True:
        raise ValueError("release is not marked immutable")
    for name in REQUIRED_FILES:
        path = release_dir / name
        if not path.is_file():
            raise ValueError(f"release file is missing: {name}")
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if not expected or sha256_file(path) != expected:
            raise ValueError(f"release file checksum mismatch: {name}")
    return manifest


def promote_release(
    candidate: Path,
    releases_root: Path,
    version: str,
    source_commit: str,
    validator: Callable[[Path], None] = validate_candidate,
) -> Path:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("version must be semantic, for example 1.2.0 or 1.2.0-rc.1")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full 40-character Git SHA")
    missing = [name for name in REQUIRED_FILES if not (candidate / name).is_file()]
    if missing:
        raise ValueError(f"candidate is missing required files: {missing}")
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("version") != version:
        raise ValueError("candidate metadata version does not match requested version")

    validator(candidate)
    releases_root.mkdir(parents=True, exist_ok=True)
    target = releases_root / version
    if target.exists():
        raise FileExistsError(f"immutable release already exists: {target}")
    temporary = releases_root / f".{version}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        for name in REQUIRED_FILES:
            shutil.copy2(candidate / name, temporary / name)
        manifest = build_manifest(temporary, version, source_commit)
        (temporary / "release-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        verify_manifest(temporary)
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a validated immutable TomatoGuard release")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--releases-root", type=Path, default=Path("model/releases"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = verify_manifest(args.verify)
        print(f"Verified immutable release {manifest['version']}")
        return
    target = promote_release(
        args.candidate, args.releases_root, args.version, args.source_commit
    )
    print(f"Promoted immutable release: {target}")


if __name__ == "__main__":
    main()
