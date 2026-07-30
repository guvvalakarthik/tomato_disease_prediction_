from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _normalise(distribution: dict[str, float], keys: list[str]) -> list[float]:
    values = [max(0.0, float(distribution.get(key, 0.0))) for key in keys]
    total = sum(values)
    if total <= 0:
        raise ValueError("class distribution must contain positive mass")
    return [value / total for value in values]


def _kl(left: list[float], right: list[float]) -> float:
    return sum(a * math.log2(a / b) for a, b in zip(left, right) if a > 0 and b > 0)


def jensen_shannon(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    p = _normalise(left, keys)
    q = _normalise(right, keys)
    midpoint = [(a + b) / 2 for a, b in zip(p, q)]
    return (_kl(p, midpoint) + _kl(q, midpoint)) / 2


def drift_report(
    baseline: dict[str, object],
    current: dict[str, object],
    minimum_samples: int = 100,
    js_threshold: float = 0.10,
    uncertainty_delta_threshold: float = 0.15,
) -> dict[str, object]:
    sample_count = int(current.get("sample_count", 0))
    if sample_count < minimum_samples:
        return {
            "status": "insufficient_data",
            "sample_count": sample_count,
            "minimum_samples": minimum_samples,
            "alert": False,
        }
    divergence = jensen_shannon(
        baseline.get("class_distribution", {}), current.get("class_distribution", {})
    )
    uncertainty_delta = float(current.get("uncertain_rate", 0.0)) - float(
        baseline.get("uncertain_rate", 0.0)
    )
    reasons = []
    if divergence > js_threshold:
        reasons.append("class_distribution_shift")
    if abs(uncertainty_delta) > uncertainty_delta_threshold:
        reasons.append("uncertainty_rate_shift")
    return {
        "status": "alert" if reasons else "ok",
        "alert": bool(reasons),
        "sample_count": sample_count,
        "jensen_shannon_divergence": divergence,
        "uncertainty_rate_delta": uncertainty_delta,
        "reasons": reasons,
        "automatic_retraining_triggered": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check aggregate TomatoGuard prediction drift")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int, default=100)
    parser.add_argument("--js-threshold", type=float, default=0.10)
    parser.add_argument("--uncertainty-delta-threshold", type=float, default=0.15)
    args = parser.parse_args()
    report = drift_report(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.current.read_text(encoding="utf-8")),
        args.minimum_samples,
        args.js_threshold,
        args.uncertainty_delta_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["alert"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
