from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values), value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a TomatoGuard ONNX release")
    parser.add_argument("release", type=Path)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metadata = json.loads(
        (args.release / "metadata.json").read_text(encoding="utf-8")
    )
    model_path = args.release / "model.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    height, width = metadata["input_size"]
    batch = np.zeros((1, height, width, 3), dtype=np.float32)

    for _ in range(10):
        session.run(None, {input_name: batch})
    timings = []
    for _ in range(args.runs):
        started = time.perf_counter()
        session.run(None, {input_name: batch})
        timings.append((time.perf_counter() - started) * 1000)

    report = {
        "model_version": metadata["version"],
        "runs": args.runs,
        "warm_latency_ms": {
            "p50": percentile(timings, 50),
            "p95": percentile(timings, 95),
            "maximum": max(timings),
        },
        "model_size_bytes": model_path.stat().st_size,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "onnxruntime": ort.__version__,
        "note": "Inference-only benchmark; API upload/decode and CAM latency are measured separately in deployment smoke tests.",
    }
    output = args.output or args.release / "benchmark.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
