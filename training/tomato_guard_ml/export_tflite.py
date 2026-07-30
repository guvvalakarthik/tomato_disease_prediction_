from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def representative_images(
    manifest: Path, dataset_root: Path, image_size: tuple[int, int], samples: int
):
    frame = pd.read_csv(manifest)
    train = frame[frame["split"] == "train"].sort_values("sha256").head(samples)
    if train.empty:
        raise ValueError("representative manifest has no training samples")
    for relative_path in train["relative_path"]:
        with Image.open(dataset_root / relative_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
            yield [np.asarray(image, dtype=np.float32)[None, ...]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a validated TomatoGuard Keras model to TFLite")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantization", choices=["dynamic", "float16", "int8"], default="float16")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--representative-samples", type=int, default=256)
    args = parser.parse_args()

    import tensorflow as tf

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if not metadata.get("calibration", {}).get("fitted"):
        raise ValueError("edge export requires fitted calibration metadata")
    model = tf.keras.models.load_model(args.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]
    elif args.quantization == "int8":
        if args.manifest is None or args.dataset_root is None:
            raise ValueError("int8 export requires --manifest and --dataset-root")
        image_size = tuple(metadata["input_size"])
        converter.representative_dataset = lambda: representative_images(
            args.manifest,
            args.dataset_root,
            image_size,
            args.representative_samples,
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.int8

    payload = converter.convert()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    interpreter = tf.lite.Interpreter(model_path=str(args.output))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if list(input_detail["shape"]) != [1, *metadata["input_size"], 3]:
        raise ValueError(f"unexpected TFLite input shape: {input_detail['shape']}")
    if int(output_detail["shape"][-1]) != len(metadata["classes"]):
        raise ValueError(f"unexpected TFLite class output: {output_detail['shape']}")

    edge_metadata = {
        "schema_version": 1,
        "model_version": metadata["version"],
        "source_model_sha256": metadata.get("model_sha256"),
        "tflite_sha256": sha256_file(args.output),
        "quantization": args.quantization,
        "input_size": metadata["input_size"],
        "classes": metadata["classes"],
        "temperature": metadata["calibration"]["temperature"],
        "confidence_threshold": metadata["rejection"]["confidence_threshold"],
        "input_dtype": str(input_detail["dtype"]),
        "input_quantization": list(input_detail["quantization"]),
        "output_dtype": str(output_detail["dtype"]),
        "output_quantization": list(output_detail["quantization"]),
        "field_validation_inherited_from_release": True,
        "device_benchmark_completed": False,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(edge_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {args.output} ({args.quantization})")


if __name__ == "__main__":
    main()
