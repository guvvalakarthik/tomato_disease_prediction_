from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .calibration import softmax
from .evaluate import metrics_for, predict
from .field_benchmark import validate_field_benchmark
from .manifest import validate_field_manifest
from .train import build_dataset


def bootstrap_macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), len(labels))
        scores.append(
            f1_score(labels[indices], predictions[indices], average="macro", zero_division=0)
        )
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return {"lower_95": float(lower), "upper_95": float(upper)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the locked expert-reviewed field set")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--field-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("training/config.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validate_field_manifest(args.manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if not metadata.get("calibration", {}).get("fitted"):
        raise ValueError("Field evaluation requires frozen calibration metadata")

    import tensorflow as tf

    class_ids = [entry["id"] for entry in config["classes"]]
    frame, field_summary = validate_field_benchmark(
        args.manifest,
        args.field_root,
        class_ids,
        review_ledger=args.reviews,
    )
    represented = sorted(set(frame["class_id"]))
    missing = sorted(set(class_ids) - set(represented))

    dataset = build_dataset(
        frame,
        args.field_root,
        class_ids,
        tuple(config["image_size"]),
        int(config["batch_size"]),
        False,
        int(config["seed"]),
    )
    model = tf.keras.models.load_model(args.model)
    logits, labels = predict(model, dataset)
    probabilities = softmax(
        logits, float(metadata["calibration"]["temperature"])
    )
    metrics = metrics_for(probabilities, labels, class_ids)
    metrics["dataset"] = "expert-reviewed local field photographs"
    metrics["field_dataset"] = field_summary
    metrics["represented_classes"] = represented
    metrics["not_externally_validated_classes"] = missing
    metrics["macro_f1_bootstrap_95_ci"] = bootstrap_macro_f1(
        labels, probabilities.argmax(axis=1), seed=int(config["seed"])
    )
    threshold = float(metadata["rejection"]["confidence_threshold"])
    accepted = probabilities.max(axis=1) >= threshold
    metrics["acceptance_coverage"] = float(accepted.mean())
    metrics["uncertain_rate"] = float(1.0 - accepted.mean())
    metrics["accepted_accuracy"] = (
        float((probabilities.argmax(axis=1)[accepted] == labels[accepted]).mean())
        if accepted.any()
        else None
    )
    metrics["calibration_refitted_on_field_data"] = False
    metrics["field_data_used_for_training"] = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote field evaluation to {args.output}")


if __name__ == "__main__":
    main()
