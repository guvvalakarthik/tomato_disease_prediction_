from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .calibration import (
    brier_score,
    choose_rejection_threshold,
    expected_calibration_error,
    fit_temperature,
    softmax,
)
from .train import build_dataset


def predict(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in dataset:
        logits.append(np.asarray(model(images, training=False)))
        labels.append(np.asarray(batch_labels))
    if not logits:
        raise ValueError("Evaluation partition is empty")
    return np.concatenate(logits), np.concatenate(labels).astype(int)


def predict_unlabelled(model, paths: list[str], image_size: tuple[int, int], batch_size: int):
    import tensorflow as tf

    def load(path):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        return tf.cast(tf.image.resize(image, image_size, antialias=True), tf.float32)

    dataset = (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
    )
    return np.concatenate(
        [np.asarray(model(images, training=False)) for images in dataset]
    )


def metrics_for(
    probabilities: np.ndarray, labels: np.ndarray, class_ids: list[str]
) -> dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    return {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "nll": float(log_loss(labels, probabilities, labels=range(len(class_ids)))),
        "ece_15_bin": expected_calibration_error(probabilities, labels),
        "brier_score": brier_score(probabilities, labels),
        "per_class": classification_report(
            labels,
            predictions,
            labels=list(range(len(class_ids))),
            target_names=class_ids,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(class_ids)))
        ).tolist(),
    }


def plot_confusion(
    matrix: np.ndarray, class_ids: list[str], output: Path, normalize: bool
) -> None:
    values = matrix.astype(float)
    if normalize:
        denominator = values.sum(axis=1, keepdims=True)
        values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator != 0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        values,
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap="Greens",
        xticklabels=class_ids,
        yticklabels=class_ids,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_reliability(probabilities: np.ndarray, labels: np.ndarray, output: Path) -> None:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, 11)
    centers, accuracies = [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            centers.append(float(confidence[selected].mean()))
            accuracies.append(float(correct[selected].mean()))
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(centers, accuracies, marker="o", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Observed accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()



def plot_risk_coverage(probabilities: np.ndarray, labels: np.ndarray, output: Path) -> None:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    order = np.argsort(-confidence)
    ordered_correct = correct[order]
    coverage = np.arange(1, len(labels) + 1) / len(labels)
    risk = 1.0 - np.cumsum(ordered_correct) / np.arange(1, len(labels) + 1)
    plt.figure(figsize=(6, 5))
    plt.plot(coverage, risk)
    plt.xlabel("Accepted coverage")
    plt.ylabel("Error risk among accepted samples")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate TomatoGuard")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("training/config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ood-manifest", type=Path)
    parser.add_argument("--ood-root", type=Path)
    parser.add_argument("--ood-test-manifest", type=Path)
    parser.add_argument("--ood-test-root", type=Path)
    args = parser.parse_args()

    import tensorflow as tf

    config = json.loads(args.config.read_text(encoding="utf-8"))
    class_ids = [entry["id"] for entry in config["classes"]]
    image_size = tuple(config["image_size"])
    batch_size = int(config["batch_size"])
    frame = pd.read_csv(args.manifest)
    model = tf.keras.models.load_model(args.model)

    validation = build_dataset(
        frame[frame["split"] == "validation"],
        args.dataset_root,
        class_ids,
        image_size,
        batch_size,
        False,
        int(config["seed"]),
    )
    test = build_dataset(
        frame[frame["split"] == "test"],
        args.dataset_root,
        class_ids,
        image_size,
        batch_size,
        False,
        int(config["seed"]),
    )
    validation_logits, validation_labels = predict(model, validation)
    temperature = fit_temperature(validation_logits, validation_labels)
    validation_probabilities = softmax(validation_logits, temperature)

    ood_probabilities = None
    if args.ood_manifest:
        if not args.ood_root:
            raise ValueError("--ood-root is required with --ood-manifest")
        ood_frame = pd.read_csv(args.ood_manifest)
        ood_paths = [str(args.ood_root / path) for path in ood_frame["relative_path"]]
        ood_logits = predict_unlabelled(model, ood_paths, image_size, batch_size)
        ood_probabilities = softmax(ood_logits, temperature)

    rejection = choose_rejection_threshold(
        validation_probabilities, validation_labels, ood_probabilities
    )
    if ood_probabilities is not None:
        id_scores = validation_probabilities.max(axis=1)
        ood_scores = ood_probabilities.max(axis=1)
        rejection["ood_auroc"] = float(
            roc_auc_score(
                np.concatenate((np.ones(len(id_scores)), np.zeros(len(ood_scores)))),
                np.concatenate((id_scores, ood_scores)),
            )
        )

    test_logits, test_labels = predict(model, test)
    test_probabilities = softmax(test_logits, temperature)
    metrics = metrics_for(test_probabilities, test_labels, class_ids)
    metrics["rejection_validation"] = rejection
    test_accepted = test_probabilities.max(axis=1) >= float(
        rejection["confidence_threshold"]
    )
    metrics["selective"] = {
        "acceptance_coverage": float(test_accepted.mean()),
        "accepted_accuracy": (
            float(
                (
                    test_probabilities.argmax(axis=1)[test_accepted]
                    == test_labels[test_accepted]
                ).mean()
            )
            if test_accepted.any()
            else None
        ),
    }
    if args.ood_test_manifest:
        if not args.ood_test_root:
            raise ValueError("--ood-test-root is required with --ood-test-manifest")
        ood_test_frame = pd.read_csv(args.ood_test_manifest)
        ood_test_paths = [
            str(args.ood_test_root / path) for path in ood_test_frame["relative_path"]
        ]
        ood_test_logits = predict_unlabelled(model, ood_test_paths, image_size, batch_size)
        ood_test_probabilities = softmax(ood_test_logits, temperature)
        metrics["ood_test"] = {
            "sample_count": len(ood_test_paths),
            "false_acceptance_rate": float((ood_test_probabilities.max(axis=1) >= float(rejection["confidence_threshold"])).mean()),
            "auroc": float(roc_auc_score(np.concatenate((np.ones(len(test_probabilities)), np.zeros(len(ood_test_probabilities)))), np.concatenate((test_probabilities.max(axis=1), ood_test_probabilities.max(axis=1))))),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "version": config["model_version"],
        "artifact_format": "onnx",
        "input_size": list(image_size),
        "input_channels": 3,
        "classes": config["classes"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_commit": git_commit(),
        "dataset_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "frameworks": {"tensorflow": tf.__version__},
        "calibration": {"fitted": True, "temperature": temperature},
        "rejection": {"method": "maximum_softmax_probability", **rejection},
        "disclaimer": "Educational screening only. A prediction is not a definitive diagnosis; consult a qualified local agricultural expert before treatment.",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    matrix = np.asarray(metrics["confusion_matrix"])
    plot_confusion(matrix, class_ids, args.output / "confusion_matrix.png", False)
    plot_confusion(
        matrix, class_ids, args.output / "confusion_matrix_normalized.png", True
    )
    plot_reliability(test_probabilities, test_labels, args.output / "reliability.png")
    plot_risk_coverage(
        test_probabilities, test_labels, args.output / "risk_coverage.png"
    )
    print(f"Wrote locked-test evaluation to {args.output}")


if __name__ == "__main__":
    main()
