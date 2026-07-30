from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from .modeling import (
    build_baseline_model,
    build_mobilenet_model,
    compile_model,
    unfreeze_backbone,
)
from .provenance import (
    artifact_hashes,
    dataset_summary,
    json_sha256,
    runtime_provenance,
    utc_now,
)


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(
    frame: pd.DataFrame,
    dataset_root: Path,
    class_ids: list[str],
    image_size: tuple[int, int],
    batch_size: int,
    training: bool,
    seed: int,
):
    import tensorflow as tf

    label_lookup = {class_id: index for index, class_id in enumerate(class_ids)}
    paths = [str(dataset_root / relative) for relative in frame["relative_path"]]
    labels = [label_lookup[class_id] for class_id in frame["class_id"]]

    def load(path, label):
        payload = tf.io.read_file(path)
        image = tf.io.decode_image(payload, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size, antialias=True)
        return tf.cast(image, tf.float32), label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return (
        dataset.with_options(options)
        .map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def merge_history(*histories) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reproducible TomatoGuard model")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("training/config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=["mobilenetv3", "baseline"], default="mobilenetv3"
    )
    parser.add_argument(
        "--run-name",
        help="Human-readable experiment name stored in run.json (defaults to output name)",
    )
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty run directory: {args.output}"
        )
    started_at = utc_now()
    started_clock = monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    set_determinism(seed)
    import tensorflow as tf

    frame = pd.read_csv(args.manifest)
    required = {"relative_path", "class_id", "split", "leaf_group", "sha256"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(frame.columns))}")
    group_counts = frame.groupby("leaf_group")["split"].nunique()
    if (group_counts > 1).any():
        raise ValueError("Manifest contains leaf-group leakage across splits")

    class_ids = [entry["id"] for entry in config["classes"]]
    image_size = tuple(config["image_size"])
    datasets = {
        split: build_dataset(
            frame[frame["split"] == split],
            args.dataset_root,
            class_ids,
            image_size,
            int(config["batch_size"]),
            split == "train",
            seed,
        )
        for split in ("train", "validation", "test")
    }

    if args.architecture == "mobilenetv3":
        model, backbone = build_mobilenet_model(image_size, len(class_ids), seed)
    else:
        model, backbone = build_baseline_model(image_size, len(class_ids), seed)
    compile_model(model, float(config["learning_rate_frozen"]))

    args.output.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output / "best.keras", monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(args.output / "epochs.csv"),
    ]
    frozen = model.fit(
        datasets["train"],
        validation_data=datasets["validation"],
        epochs=int(config["epochs_frozen"]),
        callbacks=callbacks,
    )

    finetuned = None
    if backbone is not None:
        unfreeze_backbone(backbone)
        compile_model(model, float(config["learning_rate_finetune"]))
        finetuned = model.fit(
            datasets["train"],
            validation_data=datasets["validation"],
            epochs=int(config["epochs_finetune"]),
            callbacks=callbacks,
        )

    model.save(args.output / "model.keras")
    history = merge_history(frozen, *([finetuned] if finetuned else []))
    (args.output / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    repo_root = Path(__file__).resolve().parents[2]
    run = {
        "schema_version": 1,
        "run_name": args.run_name or args.output.name,
        "architecture": args.architecture,
        "seed": seed,
        "manifest_sha256": manifest_hash(args.manifest),
        "config_sha256": json_sha256(config),
        "tensorflow_version": tf.__version__,
        "config": config,
        "dataset": dataset_summary(frame),
        "runtime": runtime_provenance(repo_root, tf),
        "arguments": {
            "manifest": str(args.manifest.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "config": str(args.config.resolve()),
            "output": str(args.output.resolve()),
        },
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_seconds": round(monotonic() - started_clock, 3),
        "test_partition_used_for_training": False,
        "artifacts": artifact_hashes(
            args.output, ("best.keras", "model.keras", "history.json", "epochs.csv")
        ),
    }
    (args.output / "run.json").write_text(
        json.dumps(run, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved training run to {args.output}")


if __name__ == "__main__":
    main()
