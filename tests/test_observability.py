from __future__ import annotations

import pytest

from api.metrics import MetricsRegistry
from scripts.check_drift import drift_report, jensen_shannon


def test_prometheus_metrics_are_bounded_and_privacy_safe() -> None:
    registry = MetricsRegistry()
    registry.observe_request("POST", "/v1/predict", 200, 120.0)
    registry.observe_prediction("uncertain", "tomato_early_blight")
    rendered = registry.render_prometheus()
    assert 'route="/v1/predict"' in rendered
    assert 'status="uncertain"' in rendered
    assert 'class_id="tomato_early_blight"' in rendered
    assert "filename" not in rendered
    assert "request_id" not in rendered


def test_histogram_buckets_are_cumulative() -> None:
    registry = MetricsRegistry()
    registry.observe_request("GET", "/health/ready", 200, 75.0)
    snapshot = registry.snapshot()
    buckets = snapshot["latency_buckets"]
    assert buckets.get(("/health/ready", 50), 0) == 0
    assert buckets[("/health/ready", 100)] == 1
    assert buckets[("/health/ready", 5000)] == 1
    assert buckets[("/health/ready", "+Inf")] == 1


def test_identical_distribution_has_zero_drift() -> None:
    distribution = {"healthy": 70, "blight": 30}
    assert jensen_shannon(distribution, distribution) == pytest.approx(0.0)


def test_drift_waits_for_minimum_sample_volume() -> None:
    report = drift_report(
        {"class_distribution": {"healthy": 1}, "uncertain_rate": 0.1},
        {
            "sample_count": 20,
            "class_distribution": {"blight": 1},
            "uncertain_rate": 0.9,
        },
    )
    assert report["status"] == "insufficient_data"
    assert report["alert"] is False


def test_drift_alert_never_triggers_automatic_retraining() -> None:
    report = drift_report(
        {
            "class_distribution": {"healthy": 90, "blight": 10},
            "uncertain_rate": 0.1,
        },
        {
            "sample_count": 500,
            "class_distribution": {"healthy": 10, "blight": 90},
            "uncertain_rate": 0.4,
        },
    )
    assert report["status"] == "alert"
    assert set(report["reasons"]) == {
        "class_distribution_shift",
        "uncertainty_rate_shift",
    }
    assert report["automatic_retraining_triggered"] is False
