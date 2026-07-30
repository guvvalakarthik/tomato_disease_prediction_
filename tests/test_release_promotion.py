from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.promote_release import REQUIRED_FILES, promote_release, verify_manifest


VERSION = "1.2.0"
COMMIT = "a" * 40


def _candidate(tmp_path: Path, version: str = VERSION) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in REQUIRED_FILES:
        (candidate / name).write_bytes(f"candidate:{name}".encode())
    (candidate / "metadata.json").write_text(
        json.dumps({"version": version}), encoding="utf-8"
    )
    return candidate


def test_promotes_atomic_immutable_release(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    target = promote_release(
        candidate,
        tmp_path / "releases",
        VERSION,
        COMMIT,
        validator=lambda _: None,
    )
    manifest = verify_manifest(target)
    assert target.name == VERSION
    assert manifest["version"] == VERSION
    assert manifest["source_commit"] == COMMIT
    assert manifest["immutable"] is True
    assert set(manifest["files"]) == set(REQUIRED_FILES)


def test_refuses_to_overwrite_existing_release(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    releases = tmp_path / "releases"
    promote_release(candidate, releases, VERSION, COMMIT, validator=lambda _: None)
    with pytest.raises(FileExistsError, match="immutable release already exists"):
        promote_release(candidate, releases, VERSION, COMMIT, validator=lambda _: None)


def test_attestation_detects_tampered_artifact(tmp_path: Path) -> None:
    target = promote_release(
        _candidate(tmp_path),
        tmp_path / "releases",
        VERSION,
        COMMIT,
        validator=lambda _: None,
    )
    (target / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_manifest(target)


def test_rejects_metadata_version_mismatch(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, version="1.1.0")
    with pytest.raises(ValueError, match="metadata version"):
        promote_release(
            candidate,
            tmp_path / "releases",
            VERSION,
            COMMIT,
            validator=lambda _: None,
        )


@pytest.mark.parametrize(
    ("version", "commit", "message"),
    [
        ("latest", COMMIT, "semantic"),
        (VERSION, "abc123", "40-character"),
    ],
)
def test_rejects_untraceable_release_identity(
    tmp_path: Path, version: str, commit: str, message: str
) -> None:
    candidate = _candidate(tmp_path)
    with pytest.raises(ValueError, match=message):
        promote_release(
            candidate,
            tmp_path / "releases",
            version,
            commit,
            validator=lambda _: None,
        )


def test_validation_runs_before_copy(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)

    def reject(_: Path) -> None:
        raise RuntimeError("release gates failed")

    with pytest.raises(RuntimeError, match="release gates failed"):
        promote_release(candidate, tmp_path / "releases", VERSION, COMMIT, reject)
    assert not (tmp_path / "releases" / VERSION).exists()
