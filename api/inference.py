from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1.0)
    return _softmax(np.log(clipped) / temperature)


@dataclass(frozen=True)
class ModelMetadata:
    version: str
    classes: tuple[dict[str, str], ...]
    input_size: tuple[int, int]
    temperature: float
    confidence_threshold: float
    calibrated: bool
    disclaimer: str

    @classmethod
    def load(cls, path: Path) -> "ModelMetadata":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            version=raw["version"],
            classes=tuple(raw["classes"]),
            input_size=tuple(raw["input_size"]),
            temperature=float(raw.get("calibration", {}).get("temperature", 1.0)),
            confidence_threshold=float(
                raw.get("rejection", {}).get("confidence_threshold", 0.6)
            ),
            calibrated=bool(raw.get("calibration", {}).get("fitted", False)),
            disclaimer=raw.get(
                "disclaimer",
                "Educational screening only; consult a qualified agricultural expert.",
            ),
        )


class ModelRuntime:
    def __init__(self, model_path: Path, metadata_path: Path):
        self.model_path = model_path
        self.metadata = ModelMetadata.load(metadata_path)
        self.sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self._runtime: Any = None
        self._classifier_weights: np.ndarray | None = None
        self._kind = model_path.suffix.lower()

    def load(self) -> None:
        if self._kind == ".onnx":
            import onnxruntime as ort

            self._runtime = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            weights_path = self.model_path.with_name("classifier_weights.npy")
            if weights_path.exists():
                self._classifier_weights = np.load(weights_path)
        elif self._kind in {".h5", ".keras"}:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise RuntimeError(
                    "Legacy Keras model selected but TensorFlow is not installed. "
                    "Install the 'legacy' dependency group or export an ONNX model."
                ) from exc
            self._runtime = tf.keras.models.load_model(self.model_path)
        else:
            raise RuntimeError(f"Unsupported model artifact: {self.model_path}")

    def predict(
        self, batch: np.ndarray, include_explanation: bool = False
    ) -> tuple[np.ndarray, str | None]:
        if self._runtime is None:
            raise RuntimeError("Model runtime has not been loaded.")

        feature_map: np.ndarray | None = None
        if self._kind == ".onnx":
            input_name = self._runtime.get_inputs()[0].name
            outputs = self._runtime.run(None, {input_name: batch})
            raw = np.asarray(outputs[0][0], dtype=np.float64)
            if len(outputs) > 1:
                feature_map = np.asarray(outputs[1][0])
        else:
            raw = np.asarray(self._runtime.predict(batch, verbose=0)[0], dtype=np.float64)

        if np.all(raw >= 0.0) and np.isclose(np.sum(raw), 1.0, atol=1e-3):
            probabilities = raw
        else:
            probabilities = _softmax(raw)
        probabilities = _temperature_scale(probabilities, self.metadata.temperature)

        explanation = None
        if (
            include_explanation
            and feature_map is not None
            and self._classifier_weights is not None
        ):
            explanation = self._class_activation_map(
                feature_map, int(np.argmax(probabilities))
            )
        return probabilities, explanation

    def _class_activation_map(self, feature_map: np.ndarray, class_index: int) -> str:
        weights = self._classifier_weights[:, class_index]
        heatmap = np.maximum(np.tensordot(feature_map, weights, axes=([-1], [0])), 0)
        maximum = float(np.max(heatmap))
        if maximum > 0:
            heatmap /= maximum
        png = Image.fromarray(np.uint8(heatmap * 255))
        buffer = io.BytesIO()
        png.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
