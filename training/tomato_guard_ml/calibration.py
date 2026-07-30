from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def negative_log_likelihood(
    logits: np.ndarray, labels: np.ndarray, temperature: float
) -> float:
    probabilities = np.clip(softmax(logits, temperature), 1e-9, 1.0)
    return float(-np.log(probabilities[np.arange(len(labels)), labels]).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    result = minimize_scalar(
        lambda value: negative_log_likelihood(logits, labels, value),
        bounds=(0.05, 10.0),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(result.x)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            gap = abs(correctness[selected].mean() - confidence[selected].mean())
            ece += float(selected.mean() * gap)
    return ece


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    truth = np.eye(probabilities.shape[1])[labels]
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def choose_rejection_threshold(
    validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    ood_probabilities: np.ndarray | None = None,
    minimum_precision: float = 0.90,
    maximum_ood_acceptance: float = 0.10,
) -> dict[str, float | bool]:
    confidence = validation_probabilities.max(axis=1)
    correct = validation_probabilities.argmax(axis=1) == validation_labels
    ood_confidence = (
        ood_probabilities.max(axis=1) if ood_probabilities is not None else None
    )
    best: dict[str, float | bool] | None = None
    candidates = np.unique(np.concatenate(([0.0, 1.0], confidence)))
    for threshold in candidates:
        accepted = confidence >= threshold
        if not accepted.any():
            continue
        precision = float(correct[accepted].mean())
        coverage = float(accepted.mean())
        ood_acceptance = (
            float((ood_confidence >= threshold).mean())
            if ood_confidence is not None
            else 0.0
        )
        valid = precision >= minimum_precision and (
            ood_confidence is None or ood_acceptance <= maximum_ood_acceptance
        )
        candidate = {
            "confidence_threshold": float(threshold),
            "validation_precision": precision,
            "validation_coverage": coverage,
            "ood_acceptance": ood_acceptance,
            "validated": bool(valid and ood_confidence is not None),
        }
        if valid and (best is None or coverage > float(best["validation_coverage"])):
            best = candidate
    return best or {
        "confidence_threshold": 1.0,
        "validation_precision": 0.0,
        "validation_coverage": 0.0,
        "ood_acceptance": 0.0,
        "validated": False,
    }
