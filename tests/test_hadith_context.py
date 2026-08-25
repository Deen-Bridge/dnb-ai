"""Tests for the contextual hadith interpretation module — no live API calls."""

import pytest

from hadith_context import (
    AskRequest,
    HadithContext,
    InterpretationResponse,
    answer_question,
    get_hadith_context,
    list_references,
    normalize_madhab,
    normalize_reference,
    router,
    synthesize_interpretation,
)

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sahih Bukhari 1", "bukhari:1"),
        ("sahih al-bukhari 1", "bukhari:1"),
        ("Bukhari:1", "bukhari:1"),
        ("bukhari 13", "bukhari:13"),
        ("Sahih Muslim 8", "muslim:8"),
        ("Arbain Nawawi 2", "nawawi:2"),
    ],
)
def test_normalize_reference(raw: str, expected: str) -> None:
    assert normalize_reference(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Hanafi", "hanafi"),
        ("shafi'i", "shafii"),
        ("Shafie", "shafii"),
        ("hanbalee", "hanbali"),
        (None, None),
        ("   ", None),
    ],
)
def test_normalize_madhab(raw: str | None, expected: str | None) -> None:
    assert normalize_madhab(raw) == expected


# ---------------------------------------------------------------------------
# Knowledge base retrieval
# ---------------------------------------------------------------------------


def test_list_references_nonempty() -> None:
    refs = list_references()
    assert "bukhari:1" in refs
    assert refs == sorted(refs)


def test_get_context_by_alias() -> None:
    # al-Nawawi hadith 2 aliases Sahih Muslim 8 (Hadith of Gabriel).
    ctx = get_hadith_context("Arbain Nawawi 2")
    assert ctx is not None
    assert ctx.reference == "muslim:8"


def test_get_context_unknown() -> None:
    assert get_hadith_context("bukhari:99999") is None


def test_record_completeness() -> None:
    # Every acceptance-criteria facet must be populated for each record.
    for ref in list_references():
        ctx = get_hadith_context(ref)
        assert isinstance(ctx, HadithContext)
        assert ctx.historical_context
        assert ctx.key_points
        assert ctx.applications
        assert ctx.commentaries
        assert ctx.contradictions
        assert ctx.contemporary_perspectives
        # At least one commentary attributes a named madhab reading.
        assert any(c.madhab for c in ctx.commentaries)
        # Contemporary perspective present alongside classical commentary.
        assert any(c.era == "classical" for c in ctx.commentaries)


# ---------------------------------------------------------------------------
# Interpretation synthesis
# ---------------------------------------------------------------------------


def test_synthesize_basic() -> None:
    ctx = get_hadith_context("bukhari:1")
    assert ctx is not None
    interp = synthesize_interpretation(ctx)
    assert isinstance(interp, InterpretationResponse)
    assert interp.reference == "bukhari:1"
    assert interp.madhab is None
    assert interp.summary
    assert interp.related  # complementary hadiths surfaced


def test_synthesize_madhab_reorders_without_dropping() -> None:
    ctx = get_hadith_context("bukhari:13")
    assert ctx is not None
    base = synthesize_interpretation(ctx)
    hanbali = synthesize_interpretation(ctx, madhab="Hanbali")
    assert hanbali.madhab == "hanbali"
    # Nothing dropped — same set of commentaries, reordered.
    assert {c.work for c in hanbali.commentaries} == {c.work for c in base.commentaries}
    # A hanbali or cross-madhab reading now leads.
    assert hanbali.commentaries[0].madhab in ("hanbali", None)
    assert "hanbali" in hanbali.summary


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------


def test_ask_matches_by_topic() -> None:
    resp = answer_question("What is the meaning of intention in deeds?")
    assert resp.matched is True
    assert resp.reference == "bukhari:1"
    assert resp.sources


def test_ask_history_facet() -> None:
    resp = answer_question("What is the historical occasion of this hadith?", reference="muslim:8")
    assert resp.matched is True
    assert "Gabriel" in resp.answer


def test_ask_application_facet() -> None:
    resp = answer_question("How do I practically apply this?", reference="bukhari:1")
    assert resp.answer.startswith("Practical guidance:")


def test_ask_contradiction_facet() -> None:
    resp = answer_question("Does this conflict with other texts?", reference="bukhari:13")
    assert "Reconciliation:" in resp.answer


def test_ask_no_match() -> None:
    resp = answer_question("xyzzy quux frobnicate")
    assert resp.matched is False
    assert resp.key_points == []


def test_ask_request_model_validates() -> None:
    req = AskRequest(question="What does ihsan mean?")
    assert req.reference is None


# ---------------------------------------------------------------------------
# Router wiring
# ---------------------------------------------------------------------------


def test_router_prefix_and_routes() -> None:
    assert router.prefix == "/hadith-context"
    paths = {route.path for route in router.routes}  # type: ignore[attr-defined]
    assert "/hadith-context/interpret" in paths
    assert "/hadith-context/ask" in paths
    assert "/hadith-context/list" in paths
