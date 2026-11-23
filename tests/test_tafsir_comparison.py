"""Offline tests for comparative tafsir source-boundary citations.

Every test runs against ``FakeTafsirSource``: no network, no model, no API key.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from semantic_cache import get_keyed_cache
from tafsir import (
    DEFAULT_COMPARISON_TAFSIR_KEYS,
    MIN_COMPARISON_TAFSIRS,
    TAFSIR_REGISTRY,
    FakeTafsirSource,
    InvalidReference,
    QuranComTafsirSource,
    TafsirComparisonRequest,
    VerseText,
    build_tafsir_comparison,
    resolve_comparison_tafsirs,
    router,
    set_source,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def clear_tafsir_cache():
    get_keyed_cache("tafsir").clear()
    yield
    get_keyed_cache("tafsir").clear()


_BASE_TEXT = (
    "<p>Human beings are in loss unless they believe, perform righteous deeds, "
    "and encourage one another to truth and patience.</p>"
)
_SECOND_TEXT = "<p>The passage emphasizes the limited nature of worldly time and accountability.</p>"
_THIRD_TEXT = "<p>The final quality is patient perseverance in obedience and communal counsel.</p>"


def _payload(work, text: str, *, wide: bool = False, verses: dict | None = None) -> dict:
    if verses is None:
        verses = {"103:1": {"id": 1}, "103:2": {"id": 2}, "103:3": {"id": 3}} if wide else {"103:2": {"id": 2}}
    return {
        "verses": verses,
        "resource_name": work.name,
        "translated_name": {"name": work.name},
        "text": text,
    }


def make_comparison_source(*, no_boundary_for: str | None = None) -> FakeTafsirSource:
    """Serve one comparison passage for every default work."""
    tafsirs: dict[tuple[str, str], dict] = {}
    for index, key in enumerate(DEFAULT_COMPARISON_TAFSIR_KEYS):
        work = TAFSIR_REGISTRY[key]
        if index < 10:
            text = _BASE_TEXT
        elif index == 10:
            text = _SECOND_TEXT
        else:
            text = _THIRD_TEXT
        verses: dict | None = None
        if key == no_boundary_for:
            verses = {}
        for slug in work.slugs.values():
            tafsirs[(slug, "103:2")] = _payload(
                work,
                text,
                wide=(key == DEFAULT_COMPARISON_TAFSIR_KEYS[0]),
                verses=verses,
            )
    return FakeTafsirSource(
        tafsirs=tafsirs,
        verses={
            "103:2": VerseText(
                arabic="إن الإنسان لفي خسر",
                translation="Indeed, mankind is in loss,",
                translation_language="en",
            )
        },
    )


def _build(**overrides):
    request = TafsirComparisonRequest(reference="103:2", **overrides)
    return run(build_tafsir_comparison(request, source=make_comparison_source()))


class TestComparisonSources:
    def test_default_set_is_within_budget(self):
        assert MIN_COMPARISON_TAFSIRS <= len(DEFAULT_COMPARISON_TAFSIR_KEYS) <= 12
        assert len(resolve_comparison_tafsirs(None)) == len(DEFAULT_COMPARISON_TAFSIR_KEYS)

    def test_every_source_carries_a_verse_boundary(self):
        response = _build()
        assert len(response.sources) == len(DEFAULT_COMPARISON_TAFSIR_KEYS)
        assert all(source.verse_range for source in response.sources)
        assert set(response.boundaries) == {source.key for source in response.sources}

    def test_wider_passage_boundary_is_reported_verbatim(self):
        response = _build()
        first_key = DEFAULT_COMPARISON_TAFSIR_KEYS[0]
        first = next(source for source in response.sources if source.key == first_key)
        assert first.verse_range == "103:1-3"
        assert response.boundaries[first_key] == "103:1-3"

    def test_missing_boundary_falls_back_to_the_compared_ayah(self):
        key = "saadi"
        request = TafsirComparisonRequest(reference="103:2")
        response = run(build_tafsir_comparison(request, source=make_comparison_source(no_boundary_for=key)))
        source = next(item for item in response.sources if item.key == key)
        assert source.verse_range is None
        assert response.boundaries[key] == "103:2"

    def test_unavailable_work_is_excluded_and_reported(self):
        source = make_comparison_source()
        missing_key = DEFAULT_COMPARISON_TAFSIR_KEYS[-1]
        missing_work = TAFSIR_REGISTRY[missing_key]
        source.tafsirs.pop((missing_work.slugs[missing_work.languages[0]], "103:2"))
        response = run(build_tafsir_comparison(TafsirComparisonRequest(reference="103:2"), source=source))
        assert missing_key not in response.boundaries
        assert missing_key not in {item.key for item in response.sources}
        assert any(item.key == missing_key for item in response.unavailable)


class TestComparisonCitations:
    def test_consensus_cites_source_boundaries(self):
        response = _build()
        assert response.consensus
        boundaries = {source.key: source.verse_range for source in response.sources}
        for consensus in response.consensus:
            assert set(consensus.source_boundaries) == set(consensus.source_keys)
            for key in consensus.source_keys:
                assert consensus.source_boundaries[key] == boundaries[key]

    def test_divergence_positions_cite_source_boundaries(self):
        response = _build()
        assert response.divergences
        boundaries = {source.key: source.verse_range for source in response.sources}
        for divergence in response.divergences:
            for position in divergence.positions:
                assert set(position.source_boundaries) == set(position.source_keys)
                for key in position.source_keys:
                    assert position.source_boundaries[key] == boundaries[key]

    def test_unique_insights_cite_source_boundaries(self):
        response = _build()
        assert response.unique_insights
        for insight in response.unique_insights:
            assert set(insight.source_boundaries) == set(insight.source_keys)

    def test_synthesis_cites_boundaries_in_text_and_data(self):
        response = _build()
        assert response.synthesis.source_boundaries
        assert "Passage boundaries" in response.synthesis.balance_note
        for key, boundary in response.synthesis.source_boundaries.items():
            assert key in response.boundaries
            assert boundary == response.boundaries[key]

    def test_provenance_documents_boundary_policy(self):
        response = _build()
        assert response.provenance.ruleset_version
        assert response.provenance.algorithm
        assert "boundary" in response.provenance.boundary_policy.lower()
        assert response.provenance.similarity_threshold == pytest.approx(0.35)
        assert response.uncertain is False


class TestComparisonFailureStates:
    def test_range_reference_is_rejected(self):
        with pytest.raises(InvalidReference):
            run(
                build_tafsir_comparison(
                    TafsirComparisonRequest(reference="103:1-2"),
                    source=make_comparison_source(),
                )
            )

    def test_too_few_works_are_rejected(self):
        request = TafsirComparisonRequest.model_construct(reference="103:2", tafsirs=["ibn-kathir"])
        with pytest.raises(InvalidReference):
            run(build_tafsir_comparison(request, source=make_comparison_source()))

    def test_unknown_works_are_rejected(self):
        request = TafsirComparisonRequest.model_construct(reference="103:2", tafsirs=["not-a-tafsir"])
        with pytest.raises(InvalidReference):
            run(build_tafsir_comparison(request, source=make_comparison_source()))


class TestComparisonEndpoint:
    def _client(self) -> TestClient:
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_endpoint_returns_boundaries(self):
        set_source(make_comparison_source())
        try:
            response = self._client().post("/tafsir/compare", json={"reference": "103:2"})
        finally:
            set_source(QuranComTafsirSource())
        assert response.status_code == 200
        body = response.json()
        assert body["boundaries"]
        assert body["provenance"]["ruleset_version"]
        assert all(item["source_boundaries"] for item in body["consensus"])

    def test_endpoint_rejects_range(self):
        response = self._client().post("/tafsir/compare", json={"reference": "103:1-2"})
        assert response.status_code == 400
        assert "exactly one ayah" in response.json()["detail"]

    def test_endpoint_rejects_too_few_works(self):
        response = self._client().post(
            "/tafsir/compare",
            json={"reference": "103:2", "tafsirs": ["ibn-kathir", "tabari"]},
        )
        assert response.status_code == 422
