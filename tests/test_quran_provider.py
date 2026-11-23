"""Offline contract tests for the Quran.com adapter used by ``tafsir.py``.

The application exposes a root-level ``tafsir`` module, not an ``app`` package.
These tests exercise that implementation directly and never call the live API.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from semantic_cache import get_keyed_cache
from tafsir import (
    FakeTafsirSource,
    InvalidReference,
    QuranComTafsirSource,
    TafsirRequest,
    VerseText,
    build_tafsir_response,
    parse_tafsir_payload,
    validate_reference,
)

VALID_VERSE_PAYLOAD = {
    "verse": {
        "verse_key": "2:255",
        "text_uthmani": "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ",
        "translations": [{"text": "Allah - there is no deity except Him."}],
    }
}

IBN_KATHIR_PAYLOAD = {
    "resource_name": "Ibn Kathir (Abridged)",
    "translated_name": {"name": "Ibn Kathir (Abridged)"},
    "text": "<p>Allah is the Creator and Sustainer.</p>",
    "verses": {"2:255": {"id": 999}},
}


def run(coro):
    """Run one async provider operation in an isolated event loop."""
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def clear_tafsir_cache():
    cache = get_keyed_cache("tafsir")
    cache.clear()
    yield
    cache.clear()


def _response(status_code: int, payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


class TestReferenceValidation:
    def test_invalid_surah_is_rejected_offline(self) -> None:
        with pytest.raises(InvalidReference, match="Surah 115"):
            validate_reference(115, 1)

    def test_invalid_ayah_is_rejected_offline(self) -> None:
        with pytest.raises(InvalidReference, match="286 ayat"):
            validate_reference(2, 300)

    def test_valid_reference_is_normalized(self) -> None:
        assert validate_reference(2, 255).key == "2:255"


class TestQuranComAdapter:
    def test_verse_text_and_translation_are_returned_with_language(self) -> None:
        client = MagicMock()
        client.get = AsyncMock(return_value=_response(200, VALID_VERSE_PAYLOAD))
        source = QuranComTafsirSource(client=client)

        verse = run(source.fetch_verse("2:255", "en"))

        assert verse.arabic == VALID_VERSE_PAYLOAD["verse"]["text_uthmani"]
        assert verse.translation == "Allah - there is no deity except Him."
        assert verse.translation_language == "en"
        client.get.assert_awaited_once()

    def test_provider_connection_error_degrades_to_empty_verse(self) -> None:
        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.ConnectError("offline"))
        source = QuranComTafsirSource(client=client)

        verse = run(source.fetch_verse("2:255", "en"))

        assert verse == VerseText()

    def test_provider_http_error_degrades_to_empty_verse(self) -> None:
        client = MagicMock()
        client.get = AsyncMock(return_value=_response(503, {}))
        source = QuranComTafsirSource(client=client)

        verse = run(source.fetch_verse("2:255", "en"))

        assert verse == VerseText()

    def test_missing_translation_is_not_mislabeled(self) -> None:
        client = MagicMock()
        client.get = AsyncMock(return_value=_response(200, {"verse": {"text_uthmani": "Arabic"}}))
        source = QuranComTafsirSource(client=client)

        verse = run(source.fetch_verse("2:255", "en"))

        assert verse.arabic == "Arabic"
        assert verse.translation is None
        assert verse.translation_language is None


class TestTafsirAttribution:
    def test_provider_resource_metadata_is_retained(self) -> None:
        from tafsir import TAFSIR_REGISTRY

        attributed = parse_tafsir_payload(TAFSIR_REGISTRY["ibn-kathir"], "en", IBN_KATHIR_PAYLOAD)

        assert attributed is not None
        assert attributed.name == "Ibn Kathir (Abridged)"
        assert attributed.author == TAFSIR_REGISTRY["ibn-kathir"].author
        assert attributed.language == "english"
        assert attributed.verse_range == "2:255"

    def test_response_keeps_arabic_translation_and_tafsir_attribution(self) -> None:
        source = FakeTafsirSource(
            tafsirs={("en-tafisr-ibn-kathir", "2:255"): IBN_KATHIR_PAYLOAD},
            verses={
                "2:255": VerseText(
                    arabic=VALID_VERSE_PAYLOAD["verse"]["text_uthmani"],
                    translation="Allah - there is no deity except Him.",
                    translation_language="en",
                )
            },
        )

        response = run(build_tafsir_response(TafsirRequest(reference="2:255", tafsirs=["ibn-kathir"]), source))

        ayah = response.ayat[0]
        assert ayah.arabic == VALID_VERSE_PAYLOAD["verse"]["text_uthmani"]
        assert ayah.translation == "Allah - there is no deity except Him."
        assert ayah.translation_language == "en"
        assert ayah.tafsirs[0].name == "Ibn Kathir (Abridged)"
        assert ayah.tafsirs[0].author

    def test_missing_tafsir_is_reported_as_unavailable(self) -> None:
        response = run(
            build_tafsir_response(
                TafsirRequest(reference="2:255", tafsirs=["ibn-kathir"]),
                FakeTafsirSource(),
            )
        )

        assert response.ayat[0].tafsirs == []
        assert response.ayat[0].unavailable
