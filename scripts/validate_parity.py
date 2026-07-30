from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def load_images(paths: list[Path], size: tuple[int, int]) -> np.ndarray:
    images = []
    for path in paths:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            resized = ImageOps.fit(rgb, size, method=Image.Resampling.BILINEAR)
            images.append(np.asarray(resized, dtype=np.float32))
    return np.asarray(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TensorFlow and ONNX outputs")
    parser.add_argument("--keras-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    import tensorflow as tf

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    size = tuple(metadata["input_size"])
    paths = sorted(
        path
        for path in args.images.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.samples]
    if not paths:
        raise SystemExit("No parity images found")
    batch = load_images(paths, size)

    keras_model = tf.keras.models.load_model(args.keras_model)
    keras_logits = np.asarray(keras_model(batch, training=False))
    session = ort.InferenceSession(
        str(args.onnx_model), providers=["CPUExecutionProvider"]
    )
    onnx_logits = session.run(None, {session.get_inputs()[0].name: batch})[0]
    keras_probabilities = softmax(keras_logits)
    onnx_probabilities = softmax(onnx_logits)
    absolute = np.abs(keras_probabilities - onnx_probabilities)
    max_error = float(absolute.max())
    prediction_agreement = float(
        (keras_probabilities.argmax(axis=1) == onnx_probabilities.argmax(axis=1)).mean()
    )
    report = {
        "sample_count": len(paths),
        "maximum_probability_error": max_error,
        "mean_probability_error": float(absolute.mean()),
        "top1_agreement": prediction_agreement,
        "tolerance": args.tolerance,
        "passed": max_error <= args.tolerance and prediction_agreement == 1.0,
        "files": [path.as_posix() for path in paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(f"Runtime parity failed: {report}")
    print(f"Runtime parity passed with max error {max_error:.8f}")


if __name__ == "__main__":
    main()
