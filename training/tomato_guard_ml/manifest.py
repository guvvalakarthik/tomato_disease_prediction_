from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .constants import CLASS_FOLDER_TO_ID, IMAGE_EXTENSIONS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    bits = []
    for row in range(8):
        offset = row * 9
        bits.extend(
            pixels[offset + column] > pixels[offset + column + 1]
            for column in range(8)
        )
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def load_leaf_groups(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        required = {"relative_path", "leaf_group"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError(f"Leaf-group CSV must contain {sorted(required)}")
        return {
            row["relative_path"].replace("\\", "/"): row["leaf_group"]
            for row in rows
        }


def scan_dataset(root: Path, leaf_groups: dict[str, str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for folder_name, class_id in CLASS_FOLDER_TO_ID.items():
        class_dir = root / folder_name
        if not class_dir.is_dir():
            raise ValueError(f"Missing required class directory: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(root).as_posix()
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"Corrupt image: {relative}") from exc
            sha256 = sha256_file(path)
            perceptual_hash = difference_hash(path)
            records.append(
                {
                    "sample_id": sha256[:16],
                    "relative_path": relative,
                    "class_id": class_id,
                    "sha256": sha256,
                    "perceptual_hash": perceptual_hash,
                    "leaf_group": leaf_groups.get(relative, perceptual_hash),
                    "width": width,
                    "height": height,
                }
            )
    if not records:
        raise ValueError(f"No supported images found below {root}")
    return records


def validate_duplicates(records: list[dict[str, object]]) -> None:
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    for record in records:
        labels_by_hash[str(record["sha256"])].add(str(record["class_id"]))
    conflicts = {
        digest: labels for digest, labels in labels_by_hash.items() if len(labels) > 1
    }
    if conflicts:
        raise ValueError(f"Exact duplicates have conflicting labels: {conflicts}")


def assign_grouped_splits(
    records: list[dict[str, object]], seed: int
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    by_class_group: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        key = f"{record['class_id']}:{record['leaf_group']}"
        by_class_group[str(record["class_id"])][key].append(record)

    for class_id, groups in sorted(by_class_group.items()):
        items = list(groups.items())
        rng.shuffle(items)
        total = sum(len(group) for _, group in items)
        if len(items) < 3:
            raise ValueError(f"Class {class_id} needs at least three independent groups")
        train_target = total * 0.70
        validation_target = total * 0.15
        counts = Counter()
        for index, (_, group_records) in enumerate(items):
            groups_left = len(items) - index
            if counts["train"] < train_target and groups_left > 2:
                split = "train"
            elif counts["validation"] < validation_target and groups_left > 1:
                split = "validation"
            else:
                split = "test"
            for record in group_records:
                record["split"] = split
            counts[split] += len(group_records)

    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[str(record["leaf_group"])].add(str(record["split"]))
    leakage = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leakage:
        raise ValueError(f"Leaf groups cross dataset partitions: {leakage}")
    return records


def write_manifest(records: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "relative_path",
        "class_id",
        "split",
        "leaf_group",
        "sha256",
        "perceptual_hash",
        "width",
        "height",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_summary(
    records: list[dict[str, object]], manifest: Path, source: str, revision: str
) -> None:
    counts = Counter(
        (str(record["split"]), str(record["class_id"])) for record in records
    )
    summary = {
        "dataset": "PlantVillage tomato subset",
        "source": source,
        "source_revision": revision,
        "manifest_sha256": sha256_file(manifest),
        "sample_count": len(records),
        "counts": {
            f"{split}/{class_id}": count
            for (split, class_id), count in sorted(counts.items())
        },
        "warning": "PlantVillage images are controlled-background images, not field data.",
    }
    manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def validate_field_manifest(path: Path) -> None:
    required = {
        "sample_id",
        "relative_path",
        "class_id",
        "expert_reviewer",
        "review_status",
        "consent_recorded",
        "capture_date",
        "device_family",
        "lighting",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fields = set(rows[0].keys()) if rows else set()
    missing = required - fields
    if missing:
        raise ValueError(f"Field manifest is missing columns: {sorted(missing)}")
    invalid = [
        row["sample_id"]
        for row in rows
        if row["review_status"] != "confirmed"
        or row["consent_recorded"].lower() not in {"true", "yes", "1"}
    ]
    if invalid:
        raise ValueError(f"Unconfirmed or unconsented field samples: {invalid[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Build a clean-data manifest")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--leaf-groups", type=Path)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--revision", required=True)
    field = subparsers.add_parser("validate-field", help="Validate field evidence")
    field.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate-field":
        validate_field_manifest(args.manifest)
        print(f"Validated field manifest: {args.manifest}")
        return

    leaf_groups = load_leaf_groups(args.leaf_groups)
    records = scan_dataset(args.dataset_root, leaf_groups)
    validate_duplicates(records)
    assign_grouped_splits(records, args.seed)
    write_manifest(records, args.output)
    write_summary(records, args.output, args.source, args.revision)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
