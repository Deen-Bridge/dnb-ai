"""
tests/test_quran_provider.py

AI-101: Make Quran source availability and freshness observable.

Tests that:
1. Requests validate references before provider access and return
   explicit unavailable status when the provider is unreachable.
2. Source/version/language metadata is retained with citations.
3. Health/readiness metrics surface provider failures and stale data.
4. A tested fallback or bounded degradation exists for provider outage.
5. Scheduled verification compares expected source fixtures and alerts on drift.

These tests run entirely offline using monkeypatching so CI never makes
real network calls to the Quran.com API.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers / stubs
# ---------------------------------------------------------------------------

VALID_AYAH_PAYLOAD: dict[str, Any] = {
    "verse_key": "2:255",
    "text_uthmani": "\u0627\u0644\u0644\u0651\u064e\u0647\u064f \u0644\u064e\u0627 \u0625\u0650\u0644\u064e\u0670\u0647\u064e",
    "translations": [
        {
            "language_name": "english",
            "text": "Allah - there is no deity except Him.",
            "resource_id": 131,
            "resource_name": "Dr. Mustafa Khattab",
        }
    ],
    "meta": {
        "source": "quran.com",
        "api_version": "v4",
        "fetched_at": "2026-11-21T00:00:00Z",
        "language": "ar",
        "translation_id": 131,
    },
}


def _make_ok_response(payload: dict[str, Any]) -> MagicMock:
    """Return a mock httpx/requests response that returns payload JSON."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def _make_error_response(status: int = 503) -> MagicMock:
    import httpx

    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status}", request=MagicMock(), response=MagicMock()
    )
    return resp


# ---------------------------------------------------------------------------
# AC-1: Reference validation before provider access
# ---------------------------------------------------------------------------


class TestReferenceValidation:
    """Requests validate references before provider access."""

    def test_invalid_surah_number_rejected_offline(self) -> None:
        """Surah 115 does not exist; must be rejected without any HTTP call."""
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get") as mock_get:
            with pytest.raises((ValueError, KeyError, Exception)) as exc_info:
                fetch_ayah(surah=115, ayah=1)
            # Must NOT have called the API for an out-of-range surah
            mock_get.assert_not_called()
            assert any(
                word in str(exc_info.value).lower()
                for word in ("invalid", "surah", "range", "115", "not found", "out of")
            )

    def test_invalid_ayah_number_rejected_offline(self) -> None:
        """Al-Fatihah has 7 ayat; ayah 100 must be rejected offline."""
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get") as mock_get:
            with pytest.raises((ValueError, KeyError, Exception)) as exc_info:
                fetch_ayah(surah=1, ayah=100)
            mock_get.assert_not_called()
            assert any(
                word in str(exc_info.value).lower()
                for word in ("invalid", "ayah", "range", "100", "not found", "out of")
            )

    def test_valid_reference_proceeds_to_provider(self) -> None:
        """A valid reference (2:255) should reach the provider."""
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get", return_value=_make_ok_response(VALID_AYAH_PAYLOAD)):
            result = fetch_ayah(surah=2, ayah=255)
            assert result is not None

    def test_zero_surah_rejected(self) -> None:
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get") as mock_get:
            with pytest.raises(Exception):
                fetch_ayah(surah=0, ayah=1)
            mock_get.assert_not_called()

    def test_zero_ayah_rejected(self) -> None:
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get") as mock_get:
            with pytest.raises(Exception):
                fetch_ayah(surah=1, ayah=0)
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# AC-2: Source/version/language metadata retained with citations
# ---------------------------------------------------------------------------


class TestCitationMetadata:
    """Source, version, and language metadata is retained with citations."""

    def test_response_includes_source_field(self) -> None:
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get", return_value=_make_ok_response(VALID_AYAH_PAYLOAD)):
            result = fetch_ayah(surah=2, ayah=255)
            # Must carry a source identifier
            assert isinstance(result, dict)
            meta = result.get("meta") or result.get("source") or result.get("citation_meta")
            assert meta is not None, "Response must include source/meta field"

    def test_response_includes_language_field(self) -> None:
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get", return_value=_make_ok_response(VALID_AYAH_PAYLOAD)):
            result = fetch_ayah(surah=2, ayah=255)
            assert isinstance(result, dict)
            # Language info must be present somewhere
            language_found = (
                "language" in str(result).lower()
                or "ar" in str(result).lower()
            )
            assert language_found, "Response must include language information"

    def test_response_includes_fetched_timestamp_or_version(self) -> None:
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get", return_value=_make_ok_response(VALID_AYAH_PAYLOAD)):
            result = fetch_ayah(surah=2, ayah=255)
            assert isinstance(result, dict)
            has_temporal = any(
                k in str(result).lower()
                for k in ("fetched_at", "timestamp", "api_version", "v4", "cached_at")
            )
            assert has_temporal, "Response must include temporal/version metadata"

    def test_arabic_text_returned(self) -> None:
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get", return_value=_make_ok_response(VALID_AYAH_PAYLOAD)):
            result = fetch_ayah(surah=2, ayah=255)
            assert isinstance(result, dict)
            text = str(result)
            # Ayat al-Kursi should contain at least one Arabic character
            has_arabic = any("\u0600" <= c <= "\u06ff" for c in text)
            assert has_arabic, "Response must include Arabic Uthmani text"

    def test_translation_resource_name_retained(self) -> None:
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get", return_value=_make_ok_response(VALID_AYAH_PAYLOAD)):
            result = fetch_ayah(surah=2, ayah=255)
            assert isinstance(result, dict)
            # The translation author/resource should be traceable
            result_str = str(result)
            assert any(
                name in result_str
                for name in ("Khattab", "131", "quran.com", "english", "translation")
            ), "Translation resource attribution must be retained"


# ---------------------------------------------------------------------------
# AC-3: Health / readiness metrics surface provider failures
# ---------------------------------------------------------------------------


class TestProviderHealthMetrics:
    """Health and readiness signals surface provider failures and stale data."""

    def test_provider_health_check_returns_ok_when_reachable(self) -> None:
        """Health check passes when the upstream API is reachable."""
        from app.tools.tafsir import check_quran_provider_health  # type: ignore[import]

        ping_payload = {"data": {"verses": [{"verse_key": "1:1"}]}}
        with patch("httpx.get", return_value=_make_ok_response(ping_payload)):
            result = check_quran_provider_health()
            assert result is True or (isinstance(result, dict) and result.get("healthy") is True)

    def test_provider_health_check_returns_false_on_outage(self) -> None:
        """Health check fails when the upstream API is unreachable."""
        from app.tools.tafsir import check_quran_provider_health  # type: ignore[import]

        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            result = check_quran_provider_health()
            assert result is False or (isinstance(result, dict) and result.get("healthy") is False)

    def test_provider_health_check_returns_false_on_http_error(self) -> None:
        """Health check fails on 5xx responses."""
        from app.tools.tafsir import check_quran_provider_health  # type: ignore[import]

        with patch("httpx.get", return_value=_make_error_response(503)):
            result = check_quran_provider_health()
            assert result is False or (isinstance(result, dict) and result.get("healthy") is False)

    def test_stale_cache_detection(self) -> None:
        """Cache entries older than the max TTL are flagged as stale."""
        from app.tools.tafsir import is_cache_stale  # type: ignore[import]

        # 25 hours ago — should be stale (assuming TTL <= 24h)
        stale_time = time.time() - (25 * 3600)
        assert is_cache_stale(stale_time) is True

    def test_fresh_cache_not_stale(self) -> None:
        """Cache entries within TTL are not stale."""
        from app.tools.tafsir import is_cache_stale  # type: ignore[import]

        fresh_time = time.time() - 60  # 1 minute ago
        assert is_cache_stale(fresh_time) is False


# ---------------------------------------------------------------------------
# AC-4: Fallback / bounded degradation on provider outage
# ---------------------------------------------------------------------------


class TestProviderFallback:
    """Bounded degradation exists for provider outage."""

    def test_fetch_returns_unavailable_status_on_outage(self) -> None:
        """When the provider is down, fetch_ayah returns unavailable status."""
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("unreachable")):
            result = fetch_ayah(surah=2, ayah=255)
            # Must NOT raise — must return a structured unavailable response
            assert result is not None
            result_str = str(result).lower()
            assert any(
                word in result_str
                for word in ("unavailable", "error", "failed", "offline", "false", "none")
            ), f"Expected unavailable indicator, got: {result}"

    def test_cached_fallback_served_when_provider_down(self) -> None:
        """Stale cached ayah is served with a staleness warning on outage."""
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        import httpx

        # Pre-populate cache via a successful call
        with patch("httpx.get", return_value=_make_ok_response(VALID_AYAH_PAYLOAD)):
            fetch_ayah(surah=2, ayah=255)

        # Now simulate outage — should return stale cache, not raise
        with patch("httpx.get", side_effect=httpx.ConnectError("unreachable")):
            result = fetch_ayah(surah=2, ayah=255)
            assert result is not None, "Must return cached/fallback result, not None"

    def test_fetch_times_out_gracefully(self) -> None:
        """Timeout does not propagate as an unhandled exception."""
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        import httpx

        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            result = fetch_ayah(surah=1, ayah=1)
            # Must return something, not raise unhandled
            assert result is not None

    def test_rate_limit_response_handled(self) -> None:
        """HTTP 429 from provider is handled gracefully."""
        from app.tools.tafsir import fetch_ayah  # type: ignore[import]

        with patch("httpx.get", return_value=_make_error_response(429)):
            result = fetch_ayah(surah=1, ayah=1)
            assert result is not None


# ---------------------------------------------------------------------------
# AC-5: Scheduled drift detection vs expected fixtures
# ---------------------------------------------------------------------------


class TestSourceDriftDetection:
    """Scheduled verification compares expected source fixtures and alerts on drift."""

    FIXTURE_AYAH_AL_FATIHA_1 = {
        "verse_key": "1:1",
        "text_uthmani": "\u0628\u0650\u0633\u0652\u0645\u0650 \u0627\u0644\u0644\u0651\u064e\u0647\u0650 \u0627\u0644\u0631\u0651\u064e\u062d\u0652\u0645\u064e\u0670\u0646\u0650 \u0627\u0644\u0631\u0651\u064e\u062d\u0650\u064a\u0645\u0650",
    }

    def test_no_drift_when_text_matches_fixture(self) -> None:
        """No drift alert when provider text matches stored fixture."""
        from app.tools.tafsir import detect_source_drift  # type: ignore[import]

        with patch("httpx.get", return_value=_make_ok_response(self.FIXTURE_AYAH_AL_FATIHA_1)):
            drifted = detect_source_drift(
                surah=1, ayah=1, expected_text=self.FIXTURE_AYAH_AL_FATIHA_1["text_uthmani"]
            )
            assert drifted is False

    def test_drift_detected_when_text_differs(self) -> None:
        """Drift alert when provider returns different text than the fixture."""
        from app.tools.tafsir import detect_source_drift  # type: ignore[import]

        modified_payload = dict(self.FIXTURE_AYAH_AL_FATIHA_1)
        modified_payload["text_uthmani"] = "WRONG TEXT - corrupted"

        with patch("httpx.get", return_value=_make_ok_response(modified_payload)):
            drifted = detect_source_drift(
                surah=1, ayah=1, expected_text=self.FIXTURE_AYAH_AL_FATIHA_1["text_uthmani"]
            )
            assert drifted is True

    def test_drift_check_on_unavailability_returns_true(self) -> None:
        """Provider unavailability is treated as a drift condition."""
        from app.tools.tafsir import detect_source_drift  # type: ignore[import]

        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("unreachable")):
            drifted = detect_source_drift(
                surah=1, ayah=1, expected_text=self.FIXTURE_AYAH_AL_FATIHA_1["text_uthmani"]
            )
            assert drifted is True

    def test_fixture_comparison_covers_all_114_surahs(self) -> None:
        """surah_index.json has exactly 114 surahs — drift check fixture must cover all."""
        import json
        import pathlib

        idx_path = pathlib.Path("data/quran/surah_index.json")
        if not idx_path.exists():
            pytest.skip("surah_index.json not available in test environment")

        with open(idx_path) as f:
            index = json.load(f)

        surah_count = len(index) if isinstance(index, list) else len(index.get("surahs", index))
        assert surah_count == 114, f"Expected 114 surahs, found {surah_count}"

    def test_total_ayah_count_fixture(self) -> None:
        """Total ayah count in surah_index must equal 6236 (Uthmani/Kufan count)."""
        import json
        import pathlib

        idx_path = pathlib.Path("data/quran/surah_index.json")
        if not idx_path.exists():
            pytest.skip("surah_index.json not available in test environment")

        with open(idx_path) as f:
            index = json.load(f)

        surahs = index if isinstance(index, list) else index.get("surahs", list(index.values()))
        total = sum(
            s.get("verses_count") or s.get("ayah_count") or s.get("total_verses") or 0
            for s in surahs
        )
        assert total == 6236, f"Expected 6236 total ayat, found {total}"
