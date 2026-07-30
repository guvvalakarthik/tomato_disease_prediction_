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
    "consent_receipt_sha256",
    "lighting",
    "background",
    "location_bucket",
    "dataset_version",
}
ALLOWED_ADJUDICATION = {"agreed", "adjudicated"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

REVIEW_REQUIRED_COLUMNS = {
    "sample_id",
    "reviewer_id",
    "expert_role",
    "proposed_class_id",
    "review_outcome",
    "reviewed_at",
}
ALLOWED_REVIEW_OUTCOMES = {"independent", "adjudicator_decision"}


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
    review_ledger: Path | None = None,
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

    invalid_consent_receipts = ~frame["consent_receipt_sha256"].str.lower().map(
        lambda value: bool(SHA256_PATTERN.fullmatch(value))
    )
    if invalid_consent_receipts.any():
        raise ValueError("Every sample requires a valid private consent receipt SHA-256")

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

    if review_ledger is None:
        raise ValueError(
            "A normalized expert review ledger is required; reviewer_count claims are not evidence"
        )
    review_frame = pd.read_csv(review_ledger, dtype=str).fillna("")
    missing_review_columns = REVIEW_REQUIRED_COLUMNS - set(review_frame.columns)
    if missing_review_columns:
        raise ValueError(
            f"Expert review ledger is missing columns: {sorted(missing_review_columns)}"
        )
    empty_review_columns = [
        column
        for column in REVIEW_REQUIRED_COLUMNS
        if (review_frame[column].str.strip() == "").any()
    ]
    if empty_review_columns:
        raise ValueError(
            f"Expert review ledger has empty required values: {sorted(empty_review_columns)}"
        )
    if review_frame.duplicated(["sample_id", "reviewer_id"]).any():
        raise ValueError("A reviewer may submit only one final review per sample")
    unknown_samples = sorted(set(review_frame["sample_id"]) - set(frame["sample_id"]))
    if unknown_samples:
        raise ValueError(f"Review ledger contains unknown samples: {unknown_samples[:10]}")
    invalid_outcomes = sorted(
        set(review_frame["review_outcome"].str.lower()) - ALLOWED_REVIEW_OUTCOMES
    )
    if invalid_outcomes:
        raise ValueError(f"Unknown review outcomes: {invalid_outcomes}")
    if class_ids is not None:
        unknown_review_labels = sorted(
            set(review_frame["proposed_class_id"]) - set(class_ids)
        )
        if unknown_review_labels:
            raise ValueError(f"Unknown expert review labels: {unknown_review_labels}")
    reviewed_dates = pd.to_datetime(review_frame["reviewed_at"], errors="coerce", utc=True)
    if reviewed_dates.isna().any():
        raise ValueError("reviewed_at contains invalid timestamps")

    for manifest_row in frame.itertuples(index=False):
        sample_reviews = review_frame[review_frame["sample_id"] == manifest_row.sample_id]
        independent = sample_reviews[
            sample_reviews["review_outcome"].str.lower() == "independent"
        ]
        independent_count = int(independent["reviewer_id"].nunique())
        if independent_count < minimum_reviewers:
            raise ValueError(
                f"Sample {manifest_row.sample_id} has {independent_count} independent reviews; "
                f"at least {minimum_reviewers} required"
            )
        if int(manifest_row.reviewer_count) != independent_count:
            raise ValueError(
                f"Sample {manifest_row.sample_id} declares reviewer_count="
                f"{manifest_row.reviewer_count}, but ledger proves {independent_count}"
            )
        independent_labels = set(independent["proposed_class_id"])
        decisions = sample_reviews[
            sample_reviews["review_outcome"].str.lower() == "adjudicator_decision"
        ]
        if manifest_row.adjudication_status.lower() == "agreed":
            if independent_labels != {manifest_row.class_id}:
                raise ValueError(
                    f"Agreed sample {manifest_row.sample_id} does not have unanimous labels"
                )
            if not decisions.empty:
                raise ValueError(
                    f"Agreed sample {manifest_row.sample_id} must not include adjudication"
                )
        else:
            if len(independent_labels) < 2:
                raise ValueError(
                    f"Adjudicated sample {manifest_row.sample_id} lacks reviewer disagreement"
                )
            if (
                len(decisions) != 1
                or decisions.iloc[0]["proposed_class_id"] != manifest_row.class_id
            ):
                raise ValueError(
                    f"Adjudicated sample {manifest_row.sample_id} requires exactly one "
                    "matching adjudicator decision"
                )

    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset_version": versions[0],
        "locked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": sha256_file(manifest),
        "review_ledger_sha256": sha256_file(review_ledger),
        "review_record_count": int(len(review_frame)),
        "sample_count": int(len(frame)),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "reviewer_count": int(review_frame["reviewer_id"].nunique()),
        "expert_roles": sorted(set(review_frame["expert_role"])),
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
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--minimum-per-class", type=int, default=20)
    parser.add_argument("--minimum-reviewers", type=int, default=2)
    args = parser.parse_args()
    _, summary = validate_field_benchmark(
        args.manifest,
        args.field_root,
        minimum_samples=args.minimum_samples,
        minimum_per_class=args.minimum_per_class,
        minimum_reviewers=args.minimum_reviewers,
        review_ledger=args.reviews,
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Locked field benchmark evidence: {args.output_summary}")


if __name__ == "__main__":
    main()
