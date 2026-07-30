from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_interpreter(model: Path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    interpreter = Interpreter(model_path=str(model), num_threads=4)
    interpreter.allocate_tensors()
    return interpreter


def quantize_input(values: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError("quantized input has invalid scale")
    limits = np.iinfo(dtype)
    return np.clip(np.rint(values / scale + zero_point), limits.min, limits.max).astype(dtype)


def dequantize_output(values: np.ndarray, detail: dict) -> np.ndarray:
    if np.issubdtype(detail["dtype"], np.floating):
        return values.astype(np.float32)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError("quantized output has invalid scale")
    return (values.astype(np.float32) - zero_point) * scale


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max()
    values = np.exp(scaled)
    return values / values.sum()


def prepare_image(path: Path, input_size: list[int]) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize((input_size[1], input_size[0]), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)[None, ...]


def invoke(interpreter, batch: np.ndarray) -> tuple[np.ndarray, float]:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], quantize_input(batch, input_detail))
    started = time.perf_counter()
    interpreter.invoke()
    latency_ms = (time.perf_counter() - started) * 1000
    output = interpreter.get_tensor(output_detail["index"])[0]
    return dequantize_output(output, output_detail), latency_ms


def latency_summary(timings: list[float]) -> dict[str, float]:
    if not timings:
        raise ValueError("at least one timing is required")
    values = np.asarray(timings, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "maximum_ms": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and benchmark TomatoGuard TFLite on an edge device")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--benchmark-output", type=Path)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    interpreter = load_interpreter(args.model)
    batch = prepare_image(args.image, metadata["input_size"])
    for _ in range(args.warmup):
        invoke(interpreter, batch)
    timings = []
    logits = None
    for _ in range(args.runs):
        logits, latency = invoke(interpreter, batch)
        timings.append(latency)
    probabilities = softmax(logits, float(metadata["temperature"]))
    ranked = probabilities.argsort()[::-1][:3]
    top = [
        {
            "class_id": metadata["classes"][int(index)]["id"],
            "label": metadata["classes"][int(index)]["label"],
            "probability": float(probabilities[index]),
        }
        for index in ranked
    ]
    accepted = top[0]["probability"] >= float(metadata["confidence_threshold"])
    result = {
        "status": "predicted" if accepted else "uncertain",
        "prediction": top[0] if accepted else None,
        "top_predictions": top,
        "model_version": metadata["model_version"],
        "quantization": metadata["quantization"],
        "latency": latency_summary(timings),
    }
    print(json.dumps(result, indent=2))

    if args.benchmark_output:
        benchmark = {
            "schema_version": 1,
            "model_version": metadata["model_version"],
            "tflite_sha256": metadata["tflite_sha256"],
            "quantization": metadata["quantization"],
            "runs": args.runs,
            "warmup": args.warmup,
            "latency": result["latency"],
            "model_size_bytes": args.model.stat().st_size,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device_measurement": True,
            "includes_decode_resize": False,
        }
        args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_output.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
