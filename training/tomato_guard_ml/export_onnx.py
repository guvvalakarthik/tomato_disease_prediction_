from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TomatoGuard to ONNX")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--field-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(args.model)
    features = model.get_layer("features").output
    export_model = tf.keras.Model(
        model.input, [model.output, features], name="tomatoguard_export"
    )
    input_shape = tuple(model.input_shape)
    signature = (
        tf.TensorSpec(input_shape, tf.float32, name="image"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "model.onnx"
    tf2onnx.convert.from_keras(
        export_model,
        input_signature=signature,
        opset=17,
        output_path=str(output_path),
    )
    classifier = model.get_layer("classifier")
    weights, _ = classifier.get_weights()
    np.save(args.output / "classifier_weights.npy", weights)
    metrics_path = args.output / "metrics.json"
    shutil.copyfile(args.metrics, metrics_path)
    field_metrics_path = args.output / "field_metrics.json"
    shutil.copyfile(args.field_metrics, field_metrics_path)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metadata["model_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata["classifier_weights_sha256"] = hashlib.sha256(
        (args.output / "classifier_weights.npy").read_bytes()
    ).hexdigest()
    metadata["metrics_sha256"] = hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    metadata["field_metrics_sha256"] = hashlib.sha256(
        field_metrics_path.read_bytes()
    ).hexdigest()
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()
