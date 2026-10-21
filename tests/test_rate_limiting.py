"""Rate-limit coverage for expensive analysis endpoints (issue #460)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import main
import rate_limit


def _make_request(
    headers: dict[str, str] | None = None,
    client: tuple[str, int] = ("127.0.0.1", 12345),
) -> Request:
    raw_headers = [(name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in (headers or {}).items()]
    scope: dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": raw_headers,
        "client": client,
    }
    return Request(scope)


@pytest.fixture(autouse=True)
def _reset_limiter() -> Iterator[None]:
    rate_limit.limiter.reset()
    yield
    rate_limit.limiter.reset()


def test_api_key_takes_precedence_over_forwarded_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit, "TRUST_PROXY_HEADERS", True)
    request = _make_request(
        {
            "X-API-Key": "service-key",
            "X-Forwarded-For": "198.51.100.1, 203.0.113.7",
        }
    )
    assert rate_limit.client_key(request) == "key:service-key"


def test_forwarded_header_is_ignored_when_proxy_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rate_limit, "TRUST_PROXY_HEADERS", False)
    request = _make_request({"X-Forwarded-For": "198.51.100.1, 203.0.113.7"})
    assert rate_limit.client_key(request) == "127.0.0.1"


def test_trusted_forwarded_header_uses_rightmost_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit, "TRUST_PROXY_HEADERS", True)
    request = _make_request({"X-Forwarded-For": "198.51.100.1, 203.0.113.7"})
    # The trusted proxy appends the peer address; the leftmost value is
    # client-controlled and must not mint a fresh bucket.
    assert rate_limit.client_key(request) == "203.0.113.7"


def test_direct_peer_is_used_without_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit, "TRUST_PROXY_HEADERS", True)
    request = _make_request(client=("192.0.2.9", 4321))
    assert rate_limit.client_key(request) == "192.0.2.9"


def test_adhkar_recommend_returns_clean_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit, "ANALYSIS_RATE_LIMIT_MAX", 1)
    client = TestClient(main.app)

    first = client.post("/adhkar/recommend", json={"query": "mercy"})
    assert first.status_code == 200

    second = client.post("/adhkar/recommend", json={"query": "mercy"})
    assert second.status_code == 429
    body = second.json()
    assert "detail" in body
    assert "hint" in body
    assert second.headers["Retry-After"] == "60"


def test_audio_hadith_verify_is_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit, "ANALYSIS_RATE_LIMIT_MAX", 1)
    client = TestClient(main.app)

    payload = {"transcript": "The best of you are those who learn the Quran and teach it.", "language": "en"}
    first = client.post("/audio-hadith/verify", json=payload)
    assert first.status_code == 200

    second = client.post("/audio-hadith/verify", json=payload)
    assert second.status_code == 429
    assert second.json()["detail"].startswith("Rate limit exceeded")
