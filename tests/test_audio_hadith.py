"""Tests for the audio Hadith verification module — no network, no live ASR."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio_hadith import (
    PassthroughTranscriber,
    Transcriber,
    extract_isnad,
    find_matches,
    match_score,
    normalize_text,
    router,
    tokenize,
    verify_transcript,
)

# ---------------------------------------------------------------------------
# Normalization and tokenization
# ---------------------------------------------------------------------------


def test_normalize_strips_diacritics_and_punctuation():
    assert normalize_text("Actions, are but by INTENTIONS!") == "actions are but by intentions"


def test_normalize_folds_arabic_diacritics():
    # Same base letters with and without tashkeel normalize identically.
    assert normalize_text("إِنَّمَا") == normalize_text("انما")


def test_tokenize_drops_stopwords_and_folds_synonyms():
    tokens = tokenize("The Messenger of Allah reported the deeds")
    assert "prophet" in tokens  # messenger -> prophet
    assert "allah" in tokens
    assert "narrate" in tokens  # reported -> narrate
    assert "the" not in tokens and "of" not in tokens


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_match_score_identical_is_one():
    toks = tokenize("Actions are but by intentions")
    assert match_score(toks, toks) == 1.0


def test_match_score_empty_is_zero():
    assert match_score([], ["prophet"]) == 0.0


def test_match_score_partial_between_zero_and_one():
    a = tokenize("Actions are but by intentions and reward")
    b = tokenize("Actions are but by intentions")
    score = match_score(a, b)
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Corpus matching
# ---------------------------------------------------------------------------


def test_find_matches_ranks_correct_hadith_first():
    matches = find_matches("Actions are but by intentions and every person gets what they intended")
    assert matches[0].hadith_id == "intentions"
    assert matches[0].collection == "Sahih al-Bukhari"
    assert matches[0].score >= 0.75


def test_find_matches_respects_limit():
    assert len(find_matches("a good word is charity", limit=2)) == 2


# ---------------------------------------------------------------------------
# Isnad extraction
# ---------------------------------------------------------------------------


def test_extract_isnad_finds_named_narrator():
    isnad = extract_isnad("Narrated by Umar ibn al-Khattab, the Prophet said...")
    assert any("Umar" in n.name for n in isnad)
    assert isnad[0].bio


def test_extract_isnad_handles_alias():
    isnad = extract_isnad("On the authority of Abu Huraira that the Prophet said...")
    assert any("Abu Hurayrah" in n.name for n in isnad)


def test_extract_isnad_empty_when_no_marker():
    assert extract_isnad("The lawful is clear and the unlawful is clear.") == []


# ---------------------------------------------------------------------------
# End-to-end verification
# ---------------------------------------------------------------------------


def test_verify_authentic_narration_is_verified():
    result = verify_transcript(
        "Narrated by Umar ibn al-Khattab: Actions are but by intentions, "
        "and every person will have only what they intended."
    )
    assert result.verified is True
    assert result.grade == "sahih"
    assert result.best_match is not None
    assert result.best_match.reference == "Bukhari 1"
    assert any("Umar" in n.name for n in result.isnad)


def test_verify_unknown_text_not_verified():
    result = verify_transcript("The quarterly revenue projections exceeded market expectations.")
    assert result.verified is False
    assert result.grade is None
    assert "No authenticated narration" in result.note


def test_verify_partial_flags_misquotation():
    # Right hadith, mangled wording -> matched but flagged.
    result = verify_transcript("Deeds are only judged by the intention behind every single one")
    assert result.flagged_misquotation is True


# ---------------------------------------------------------------------------
# Transcriber interface
# ---------------------------------------------------------------------------


def test_passthrough_transcriber_returns_input():
    assert PassthroughTranscriber().transcribe("hello") == "hello"


def test_transcriber_base_is_abstract():
    class Dummy(Transcriber):
        pass

    import pytest

    with pytest.raises(NotImplementedError):
        Dummy().transcribe("x")


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_endpoint_verifies_known_hadith():
    resp = _client().post("/audio-hadith/verify", json={"transcript": "A good word is charity"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["verified"] is True
    assert body["grade"] == "sahih"
    assert body["best_match"]["reference"] == "Bukhari 2989"


def test_endpoint_rejects_empty_transcript():
    resp = _client().post("/audio-hadith/verify", json={"transcript": "   "})
    assert resp.status_code == 422
