from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def validate_cam_contract(model, class_count: int) -> tuple[object, np.ndarray]:
    try:
        features_layer = model.get_layer("features")
        classifier = model.get_layer("classifier")
    except (ValueError, KeyError) as exc:
        raise ValueError("Model must expose named features and classifier layers") from exc
    feature_shape = tuple(features_layer.output.shape)
    if len(feature_shape) != 4:
        raise ValueError(f"features output must be rank 4 for CAM, got {feature_shape}")
    classifier_values = classifier.get_weights()
    if len(classifier_values) != 2:
        raise ValueError("classifier must contain a kernel and bias")
    weights = np.asarray(classifier_values[0])
    if weights.ndim != 2 or weights.shape[1] != class_count:
        raise ValueError(
            f"classifier kernel must be [channels, {class_count}], got {weights.shape}"
        )
    if feature_shape[-1] is not None and int(feature_shape[-1]) != weights.shape[0]:
        raise ValueError("features channels do not match classifier kernel")
    if int(model.output_shape[-1]) != class_count:
        raise ValueError("model output class count does not match metadata")
    return features_layer.output, weights


def compare_export_outputs(
    expected: list[np.ndarray], actual: list[np.ndarray], tolerance: float
) -> dict[str, object]:
    if len(actual) != 2:
        raise ValueError(f"CAM-compatible ONNX must expose exactly 2 outputs, got {len(actual)}")
    names = ("logits", "features")
    errors: dict[str, float] = {}
    for name, expected_value, actual_value in zip(names, expected, actual):
        expected_array = np.asarray(expected_value)
        actual_array = np.asarray(actual_value)
        if expected_array.shape != actual_array.shape:
            raise ValueError(
                f"{name} shape mismatch: {expected_array.shape} != {actual_array.shape}"
            )
        errors[name] = float(np.max(np.abs(expected_array - actual_array)))
    passed = all(value <= tolerance for value in errors.values())
    return {
        "passed": passed,
        "tolerance": tolerance,
        "maximum_absolute_error": errors,
        "output_contract": ["logits", "features"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TomatoGuard to ONNX")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--field-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    import tensorflow as tf
    import tf2onnx
    import onnxruntime as ort

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    class_count = len(metadata["classes"])
    model = tf.keras.models.load_model(args.model)
    features, weights = validate_cam_contract(model, class_count)
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
    batch_shape = (2, *(int(value) for value in input_shape[1:]))
    batch = np.random.default_rng(42).uniform(0.0, 255.0, batch_shape).astype(np.float32)
    expected_outputs = [
        np.asarray(value) for value in export_model(batch, training=False)
    ]
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    actual_outputs = session.run(
        None, {session.get_inputs()[0].name: batch}
    )
    validation = compare_export_outputs(
        expected_outputs, actual_outputs, args.tolerance
    )
    validation["output_names"] = [item.name for item in session.get_outputs()]
    validation["sample_count"] = len(batch)
    if not validation["passed"]:
        raise ValueError(f"ONNX runtime parity failed: {validation}")
    validation_path = args.output / "export_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    np.save(args.output / "classifier_weights.npy", weights)
    metrics_path = args.output / "metrics.json"
    shutil.copyfile(args.metrics, metrics_path)
    field_metrics_path = args.output / "field_metrics.json"
    shutil.copyfile(args.field_metrics, field_metrics_path)

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
    metadata["export_validation_sha256"] = hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    metadata["export_validation"] = validation
    metadata["cam_compatible"] = True
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()
