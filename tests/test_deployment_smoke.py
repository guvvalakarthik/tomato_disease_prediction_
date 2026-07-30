from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from scripts.smoke_deployment import check_deployment


@dataclass
class FakeResponse:
    body: bytes
    headers: dict[str, str]
    status: int = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self.body


def _opener(cors_origin: str = "https://app.example"):
    def open_request(request, timeout):
        assert timeout > 0
        if request.full_url.endswith("/health/live"):
            return FakeResponse(json.dumps({"status": "ok"}).encode(), {})
        if request.full_url.endswith("/health/ready"):
            return FakeResponse(
                json.dumps({"status": "ok", "model_version": "1.2.0"}).encode(),
                {},
            )
        if request.get_method() == "OPTIONS":
            return FakeResponse(b"", {"Access-Control-Allow-Origin": cors_origin})
        return FakeResponse(b"<title>TomatoGuard</title>", {})

    return open_request


def test_deployment_smoke_contract_passes() -> None:
    result = check_deployment(
        "https://api.example/",
        "https://app.example/",
        "1.2.0",
        opener=_opener(),
    )
    assert result.model_version == "1.2.0"
    assert result.cors_origin == "https://app.example"
    assert result.frontend_bytes > 0


def test_rejects_wrong_model_version() -> None:
    with pytest.raises(RuntimeError, match="expected 2.0.0"):
        check_deployment(
            "https://api.example",
            "https://app.example",
            "2.0.0",
            opener=_opener(),
        )


def test_rejects_permissive_or_wrong_cors() -> None:
    with pytest.raises(RuntimeError, match="CORS returned"):
        check_deployment(
            "https://api.example",
            "https://app.example",
            "1.2.0",
            opener=_opener("*"),
        )


def test_rejects_wrong_frontend_content() -> None:
    def opener(request, timeout):
        if request.full_url.endswith("/health/live"):
            return FakeResponse(b'{"status":"ok"}', {})
        if request.full_url.endswith("/health/ready"):
            return FakeResponse(b'{"status":"ok","model_version":"1.2.0"}', {})
        return FakeResponse(b"unrelated site", {})

    with pytest.raises(RuntimeError, match="TomatoGuard identity"):
        check_deployment(
            "https://api.example",
            "https://app.example",
            "1.2.0",
            opener=opener,
        )
