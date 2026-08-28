"""Tests for the historical context injection module (issue #227)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from history import (
    ASBAB_AL_NUZUL,
    FIQH_TIMELINE,
    SCHOLARS,
    build_historical_context,
    router,
)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Pure builder: relevance detection
# ---------------------------------------------------------------------------


def test_asbab_detected_from_verse_reference() -> None:
    ctx = build_historical_context("What is the meaning of 5:90 about khamr?")
    refs = {e.reference for e in ctx.asbab}
    assert "5:90" in refs
    # Attribution is carried, never invented.
    entry = next(e for e in ctx.asbab if e.reference == "5:90")
    assert entry.attribution
    assert ctx.has_context


def test_unknown_verse_yields_no_asbab() -> None:
    ctx = build_historical_context("Explain 114:1 please")
    assert ctx.asbab == []


def test_fiqh_timeline_detected_and_scope_marked() -> None:
    ctx = build_historical_context("Why is alcohol forbidden in Islam?")
    keys = {t.key for t in ctx.timelines}
    assert "intoxicants" in keys
    timeline = next(t for t in ctx.timelines if t.key == "intoxicants")
    # Distinguishes time-bound vs universal rulings.
    assert timeline.scope == "universal"
    assert len(timeline.stages) >= 2


def test_time_bound_ruling_scope() -> None:
    ctx = build_historical_context("What is the ruling on smoking tobacco?")
    timeline = next(t for t in ctx.timelines if t.key == "tobacco")
    assert timeline.scope == "time-bound-to-knowledge"


def test_hadith_circumstance_detected() -> None:
    ctx = build_historical_context("Tell me about the hadith on intention (niyyah).")
    slugs = {h.slug for h in ctx.hadith_contexts}
    assert "actions-by-intentions" in slugs


def test_scholar_biography_detected_with_period() -> None:
    ctx = build_historical_context("What did al-Shafi'i and Ibn Taymiyya say?")
    keys = {s.key for s in ctx.scholars}
    assert "al-shafii" in keys
    assert "ibn-taymiyya" in keys
    shafii = next(s for s in ctx.scholars if s.key == "al-shafii")
    assert shafii.century_ce == 9


def test_context_block_is_injectable_and_attributed() -> None:
    ctx = build_historical_context("Meaning of 2:219 on wine, per al-Ghazali?")
    assert ctx.has_context
    block = ctx.context_block
    assert "asbab al-nuzul" in block.lower()
    assert "2:219" in block
    assert "al-Ghazali" in block


def test_no_match_yields_empty_context() -> None:
    ctx = build_historical_context("What time is the football match today?")
    assert not ctx.has_context
    assert ctx.context_block == ""
    assert ctx.asbab == []
    assert ctx.timelines == []


def test_deduplicates_repeated_reference() -> None:
    ctx = build_historical_context("5:90 and again 5:90")
    assert len([e for e in ctx.asbab if e.reference == "5:90"]) == 1


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


def test_post_context_route() -> None:
    resp = _client().post("/history/context", json={"text": "explain 9:5 and riba"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_context"] is True
    assert any(a["reference"] == "9:5" for a in body["asbab"])
    assert any(t["key"] == "riba" for t in body["timelines"])


def test_post_context_rejects_empty() -> None:
    resp = _client().post("/history/context", json={"text": ""})
    assert resp.status_code == 422


def test_get_asbab_route() -> None:
    resp = _client().get("/history/asbab/5/90")
    assert resp.status_code == 200
    assert resp.json()["reference"] == "5:90"


def test_get_asbab_not_found() -> None:
    resp = _client().get("/history/asbab/114/1")
    assert resp.status_code == 404


def test_get_scholar_route() -> None:
    resp = _client().get("/history/scholar/al-tabari")
    assert resp.status_code == 200
    body = resp.json()
    assert body["key"] == "al-tabari"
    assert body["century_ce"] == 10


def test_get_scholar_not_found() -> None:
    resp = _client().get("/history/scholar/nobody-here")
    assert resp.status_code == 404


def test_get_timeline_by_keyword() -> None:
    resp = _client().get("/history/timeline/usury")
    assert resp.status_code == 200
    assert resp.json()["key"] == "riba"


def test_get_timeline_not_found() -> None:
    resp = _client().get("/history/timeline/unknown-topic")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


def test_every_scholar_record_has_period_fields() -> None:
    for record in SCHOLARS.values():
        assert record["hijri"]
        assert record["gregorian"]
        assert isinstance(record["century_ce"], int)


def test_every_asbab_has_attribution() -> None:
    for record in ASBAB_AL_NUZUL.values():
        assert record["summary"]
        assert record["attribution"]


def test_every_timeline_has_stages_and_scope() -> None:
    for record in FIQH_TIMELINE.values():
        assert record["scope"]
        assert len(record["stages"]) >= 2
