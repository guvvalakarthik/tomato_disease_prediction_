from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SmokeResult:
    model_version: str
    frontend_bytes: int
    cors_origin: str


def _read(request: urllib.request.Request, timeout: float, opener: Callable):
    with opener(request, timeout=timeout) as response:
        return response.status, dict(response.headers), response.read()


def check_deployment(
    api_url: str,
    frontend_url: str,
    expected_model_version: str | None = None,
    timeout: float = 10.0,
    opener: Callable = urllib.request.urlopen,
) -> SmokeResult:
    api_url = api_url.rstrip("/")
    frontend_url = frontend_url.rstrip("/")
    _, _, live_body = _read(
        urllib.request.Request(f"{api_url}/health/live"), timeout, opener
    )
    live = json.loads(live_body)
    if live.get("status") != "ok":
        raise RuntimeError("API liveness did not report ok")

    _, _, ready_body = _read(
        urllib.request.Request(f"{api_url}/health/ready"), timeout, opener
    )
    ready = json.loads(ready_body)
    if ready.get("status") != "ok" or not ready.get("model_version"):
        raise RuntimeError("API readiness did not report a loaded model")
    if expected_model_version and ready["model_version"] != expected_model_version:
        raise RuntimeError(
            f"deployed model is {ready['model_version']}, expected {expected_model_version}"
        )

    _, _, frontend_body = _read(
        urllib.request.Request(frontend_url), timeout, opener
    )
    if b"TomatoGuard" not in frontend_body:
        raise RuntimeError("frontend response does not contain the TomatoGuard identity")

    preflight = urllib.request.Request(
        f"{api_url}/v1/predict",
        method="OPTIONS",
        headers={
            "Origin": frontend_url,
            "Access-Control-Request-Method": "POST",
        },
    )
    _, cors_headers, _ = _read(preflight, timeout, opener)
    allow_origin = cors_headers.get("Access-Control-Allow-Origin")
    if allow_origin != frontend_url:
        raise RuntimeError(
            f"CORS returned {allow_origin!r}; expected deployed frontend {frontend_url!r}"
        )
    return SmokeResult(
        model_version=ready["model_version"],
        frontend_bytes=len(frontend_body),
        cors_origin=allow_origin,
    )


def wait_for_deployment(
    api_url: str,
    frontend_url: str,
    expected_model_version: str | None,
    attempts: int,
    interval_seconds: float,
) -> SmokeResult:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return check_deployment(api_url, frontend_url, expected_model_version)
        except (RuntimeError, ValueError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(interval_seconds)
    raise RuntimeError(f"deployment did not become healthy: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a deployed TomatoGuard release")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--frontend-url", required=True)
    parser.add_argument("--expected-model-version")
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    result = wait_for_deployment(
        args.api_url,
        args.frontend_url,
        args.expected_model_version,
        args.attempts,
        args.interval_seconds,
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
