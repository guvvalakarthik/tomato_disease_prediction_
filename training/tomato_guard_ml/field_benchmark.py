from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "sample_id",
    "relative_path",
    "sha256",
    "class_id",
    "expert_reviewer",
    "expert_role",
    "reviewer_count",
    "adjudication_status",
    "review_status",
    "consent_recorded",
    "capture_date",
    "device_family",
    "lighting",
    "background",
    "location_bucket",
    "dataset_version",
}
ALLOWED_ADJUDICATION = {"agreed", "adjudicated"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Field image path must stay inside field root: {value}")
    return path


def validate_field_benchmark(
    manifest: Path,
    field_root: Path | None = None,
    class_ids: list[str] | None = None,
    minimum_samples: int = 300,
    minimum_per_class: int = 20,
    minimum_reviewers: int = 2,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(manifest, dtype=str).fillna("")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Field benchmark is missing columns: {sorted(missing)}")
    if len(frame) < minimum_samples:
        raise ValueError(
            f"Field benchmark has {len(frame)} samples; at least {minimum_samples} required"
        )

    empty_columns = [column for column in REQUIRED_COLUMNS if (frame[column].str.strip() == "").any()]
    if empty_columns:
        raise ValueError(f"Field benchmark has empty required values: {sorted(empty_columns)}")
    for column in ("sample_id", "relative_path", "sha256"):
        duplicated = frame.loc[frame[column].duplicated(keep=False), column].tolist()
        if duplicated:
            raise ValueError(f"Duplicate {column} values: {duplicated[:10]}")

    invalid_review = frame[
        (frame["review_status"].str.lower() != "confirmed")
        | (~frame["consent_recorded"].str.lower().isin({"true", "yes", "1"}))
    ]
    if not invalid_review.empty:
        raise ValueError(
            "Unconfirmed or unconsented field samples: "
            f"{invalid_review['sample_id'].head(10).tolist()}"
        )
    reviewers = pd.to_numeric(frame["reviewer_count"], errors="coerce")
    if reviewers.isna().any() or (reviewers < minimum_reviewers).any():
        raise ValueError(f"Every sample requires at least {minimum_reviewers} expert reviews")
    invalid_adjudication = ~frame["adjudication_status"].str.lower().isin(ALLOWED_ADJUDICATION)
    if invalid_adjudication.any():
        raise ValueError("adjudication_status must be agreed or adjudicated")

    unknown_classes = set(frame["class_id"])
    if class_ids is not None:
        unknown_classes -= set(class_ids)
        if unknown_classes:
            raise ValueError(f"Unknown field labels: {sorted(unknown_classes)}")
    counts = frame["class_id"].value_counts().sort_index()
    underrepresented = counts[counts < minimum_per_class]
    if not underrepresented.empty:
        raise ValueError(
            "Field classes below minimum representation: "
            f"{underrepresented.astype(int).to_dict()}"
        )

    versions = sorted(set(frame["dataset_version"]))
    if len(versions) != 1:
        raise ValueError("A locked field manifest must contain exactly one dataset_version")
    parsed_dates = pd.to_datetime(frame["capture_date"], errors="coerce", utc=True)
    if parsed_dates.isna().any():
        raise ValueError("capture_date contains invalid dates")

    for row in frame.itertuples(index=False):
        relative = _relative_path(row.relative_path)
        if not SHA256_PATTERN.fullmatch(row.sha256.lower()):
            raise ValueError(f"Invalid SHA-256 for sample {row.sample_id}")
        if field_root is not None:
            image_path = field_root / relative
            if not image_path.is_file():
                raise ValueError(f"Field image is missing: {row.relative_path}")
            if sha256_file(image_path) != row.sha256.lower():
                raise ValueError(f"Field image checksum mismatch: {row.sample_id}")

    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": versions[0],
        "locked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": sha256_file(manifest),
        "sample_count": int(len(frame)),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "reviewer_count": int(frame["expert_reviewer"].nunique()),
        "expert_roles": sorted(set(frame["expert_role"])),
        "device_families": sorted(set(frame["device_family"])),
        "lighting_conditions": sorted(set(frame["lighting"])),
        "backgrounds": sorted(set(frame["background"])),
        "location_buckets": sorted(set(frame["location_bucket"])),
        "minimum_independent_reviews": minimum_reviewers,
        "field_data_used_for_training": False,
        "calibration_refitted_on_field_data": False,
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and lock expert field evidence")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--field-root", type=Path)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int, default=300)
    parser.add_argument("--minimum-per-class", type=int, default=20)
    parser.add_argument("--minimum-reviewers", type=int, default=2)
    args = parser.parse_args()
    _, summary = validate_field_benchmark(
        args.manifest,
        args.field_root,
        minimum_samples=args.minimum_samples,
        minimum_per_class=args.minimum_per_class,
        minimum_reviewers=args.minimum_reviewers,
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Locked field benchmark evidence: {args.output_summary}")


if __name__ == "__main__":
    main()
