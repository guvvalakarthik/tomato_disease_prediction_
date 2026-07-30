from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a TomatoGuard release bundle")
    parser.add_argument("release", type=Path)
    args = parser.parse_args()

    model_path = args.release / "model.onnx"
    metadata_path = args.release / "metadata.json"
    weights_path = args.release / "classifier_weights.npy"
    metrics_path = args.release / "metrics.json"
    field_metrics_path = args.release / "field_metrics.json"
    required = (model_path, metadata_path, weights_path, metrics_path, field_metrics_path)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing release files: {missing}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_sha256") != sha256(model_path):
        raise SystemExit("model.onnx checksum does not match metadata.json")
    if metadata.get("classifier_weights_sha256") != sha256(weights_path):
        raise SystemExit("classifier_weights.npy checksum does not match metadata.json")
    if metadata.get("metrics_sha256") != sha256(metrics_path):
        raise SystemExit("metrics.json checksum does not match metadata.json")
    if metadata.get("field_metrics_sha256") != sha256(field_metrics_path):
        raise SystemExit("field_metrics.json checksum does not match metadata.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    field_metrics = json.loads(field_metrics_path.read_text(encoding="utf-8"))
    if not metadata.get("calibration", {}).get("fitted"):
        raise SystemExit("Release calibration has not been fitted")
    if not metadata.get("rejection", {}).get("validated"):
        raise SystemExit("Release rejection threshold has not passed OOD validation")

    classes = metadata.get("classes", [])
    input_size = metadata.get("input_size")
    if len(classes) != 10 or not input_size or len(input_size) != 2:
        raise SystemExit("Release class or input contract is invalid")
    if float(metrics.get("macro_f1", 0.0)) < 0.90:
        raise SystemExit("Clean macro F1 release gate failed")
    for class_entry in classes:
        class_report = metrics.get("per_class", {}).get(class_entry["id"], {})
        if float(class_report.get("recall", 0.0)) < 0.80:
            raise SystemExit(f"Clean recall gate failed for {class_entry['id']}")
    if float(metrics.get("ece_15_bin", 1.0)) > 0.05:
        raise SystemExit("Clean calibration gate failed")
    if float(metrics.get("selective", {}).get("acceptance_coverage", 0.0)) < 0.80:
        raise SystemExit("Clean accepted-coverage gate failed")
    ood_test = metrics.get("ood_test")
    if not ood_test:
        raise SystemExit("Locked OOD test evidence is missing")
    if float(ood_test.get("false_acceptance_rate", 1.0)) > 0.10:
        raise SystemExit("Locked OOD false-acceptance gate failed")

    if int(field_metrics.get("sample_count", 0)) < 300:
        raise SystemExit("Field sample-count gate failed")
    if float(field_metrics.get("macro_f1", 0.0)) < 0.70:
        raise SystemExit("Field macro F1 gate failed")
    if float(field_metrics.get("ece_15_bin", 1.0)) > 0.10:
        raise SystemExit("Field calibration gate failed")
    if field_metrics.get("calibration_refitted_on_field_data") is not False:
        raise SystemExit("Field data must not refit calibration")
    if field_metrics.get("field_data_used_for_training") is not False:
        raise SystemExit("Field data must not be used for training")


    weights = np.load(weights_path)
    if weights.ndim != 2 or weights.shape[1] != len(classes):
        raise SystemExit("Classifier weights do not match the class contract")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if len(session.get_outputs()) < 2:
        raise SystemExit("ONNX model must return logits and a feature map")
    input_name = session.get_inputs()[0].name
    batch = np.zeros((1, int(input_size[0]), int(input_size[1]), 3), dtype=np.float32)
    logits, features, *_ = session.run(None, {input_name: batch})
    if logits.shape != (1, len(classes)) or not np.isfinite(logits).all():
        raise SystemExit(f"Unexpected or invalid logits: {logits.shape}")
    if features.ndim != 4 or features.shape[-1] != weights.shape[0]:
        raise SystemExit("Feature-map channels do not match classifier weights")
    print(f"Validated TomatoGuard release {metadata['version']} ({sha256(model_path)})")


if __name__ == "__main__":
    main()
