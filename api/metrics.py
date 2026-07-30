from __future__ import annotations

import threading
from collections import Counter


LATENCY_BUCKETS_MS = (50, 100, 250, 500, 1000, 2000, 5000)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class MetricsRegistry:
    """Bounded, aggregate metrics that never retain uploads or user identifiers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[tuple[str, str, int]] = Counter()
        self._latency_buckets: Counter[tuple[str, int | str]] = Counter()
        self._latency_sum_ms: Counter[str] = Counter()
        self._latency_count: Counter[str] = Counter()
        self._predictions: Counter[tuple[str, str]] = Counter()

    def observe_request(self, method: str, route: str, status: int, latency_ms: float) -> None:
        route = route if route.startswith("/") else "unmatched"
        with self._lock:
            self._requests[(method, route, status)] += 1
            self._latency_sum_ms[route] += latency_ms
            self._latency_count[route] += 1
            for boundary in LATENCY_BUCKETS_MS:
                if latency_ms <= boundary:
                    self._latency_buckets[(route, boundary)] += 1
            self._latency_buckets[(route, "+Inf")] += 1

    def observe_prediction(self, status: str, class_id: str | None) -> None:
        safe_class = class_id or "none"
        with self._lock:
            self._predictions[(status, safe_class)] += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "requests": dict(self._requests),
                "latency_buckets": dict(self._latency_buckets),
                "latency_sum_ms": dict(self._latency_sum_ms),
                "latency_count": dict(self._latency_count),
                "predictions": dict(self._predictions),
            }

    def render_prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP tomatoguard_http_requests_total Completed HTTP requests.",
            "# TYPE tomatoguard_http_requests_total counter",
        ]
        for (method, route, status), count in sorted(snapshot["requests"].items()):
            lines.append(
                'tomatoguard_http_requests_total{method="%s",route="%s",status="%s"} %s'
                % (_escape(method), _escape(route), status, count)
            )
        lines.extend(
            [
                "# HELP tomatoguard_http_latency_ms Request latency in milliseconds.",
                "# TYPE tomatoguard_http_latency_ms histogram",
            ]
        )
        for (route, boundary), count in sorted(
            snapshot["latency_buckets"].items(), key=lambda item: (item[0][0], str(item[0][1]))
        ):
            lines.append(
                'tomatoguard_http_latency_ms_bucket{route="%s",le="%s"} %s'
                % (_escape(route), boundary, count)
            )
        for route, total in sorted(snapshot["latency_sum_ms"].items()):
            lines.append(f'tomatoguard_http_latency_ms_sum{{route="{_escape(route)}"}} {total:.6f}')
            lines.append(
                f'tomatoguard_http_latency_ms_count{{route="{_escape(route)}"}} '
                f'{snapshot["latency_count"][route]}'
            )
        lines.extend(
            [
                "# HELP tomatoguard_predictions_total Aggregate prediction outcomes.",
                "# TYPE tomatoguard_predictions_total counter",
            ]
        )
        for (status, class_id), count in sorted(snapshot["predictions"].items()):
            lines.append(
                'tomatoguard_predictions_total{status="%s",class_id="%s"} %s'
                % (_escape(status), _escape(class_id), count)
            )
        return "\n".join(lines) + "\n"
