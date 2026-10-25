"""Extended tests for Arabic speech-to-text optimization (#244).

Covers dialect handling (including Maghrebi), Arabic/English code-switching, diacritic
handling, the expanded lexicon, streaming, batch and error-rate evaluation. Everything is
deterministic and offline: there is no acoustic model, network call or clock dependency.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import arabic_stt_optimizer as stt
from arabic_stt_optimizer import (
    ArabicSTTOptimizationRequest,
    STTStreamRequest,
    edit_distance,
    error_rates,
    find_latin_segments,
    has_arabic,
    normalize_latin_formulas,
    optimize_arabic_stt,
    optimize_stream,
    optimize_text,
    resolve_dialect,
    split_stream,
    strip_diacritics,
    to_msa,
)

SHADDA_SALAH = "\u0627\u0644\u0635\u0651\u064e\u0644\u0627\u0647"  # الصَّلاه (diacritized, ta marbuta as ha)
PLAIN_SALAH_WRONG = "\u0627\u0644\u0635\u0644\u0627\u0647"  # الصلاه
KNOWN_DIALECTS = {"msa", "egyptian", "gulf", "levantine", "maghrebi"}


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(stt.router)
    return TestClient(app)


def _request(transcript: str, **overrides: Any) -> ArabicSTTOptimizationRequest:
    payload: dict[str, Any] = {"candidates": [{"transcript": transcript}]}
    payload.update(overrides)
    return ArabicSTTOptimizationRequest.model_validate(payload)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def test_strip_diacritics_removes_harakat_and_tatweel() -> None:
    diacritized = "\u0627\u0644\u0635\u0651\u064e\u0644\u0627\u0629\u064f"
    assert strip_diacritics(diacritized) == "\u0627\u0644\u0635\u0644\u0627\u0629"
    assert strip_diacritics("\u0643\u0640\u0640\u062a\u0627\u0628") == "\u0643\u062a\u0627\u0628"


def test_strip_diacritics_leaves_plain_and_latin_text_alone() -> None:
    assert strip_diacritics("hello world") == "hello world"
    assert strip_diacritics("") == ""


def test_has_arabic() -> None:
    assert has_arabic("مرحبا") is True
    assert has_arabic("hello قال") is True
    assert has_arabic("hello") is False
    assert has_arabic("") is False


def test_find_latin_segments_returns_unique_words_in_order() -> None:
    assert find_latin_segments("قال hello ثم hello world") == ["hello", "world"]
    assert find_latin_segments("لا يوجد") == []


# ---------------------------------------------------------------------------
# Orthographic corrections (expanded lexicon)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wrong", "right", "category"),
    [
        ("ان شاء الله", "إن شاء الله", "hamza_restoration"),
        ("الاسلام", "الإسلام", "hamza_restoration"),
        ("الرحمان", "الرحمن", "divine_name_spelling"),
        ("الجنه", "الجنة", "ta_marbuta_restoration"),
        ("الاخره", "الآخرة", "madda_and_ta_marbuta"),
        ("عائشه", "عائشة", "name_spelling"),
        ("ابو هريره", "أبو هريرة", "name_spelling"),
        ("الله اكبر", "الله أكبر", "hamza_restoration"),
        ("لا اله الا الله", "لا إله إلا الله", "hamza_restoration"),
        ("الحمدلله", "الحمد لله", "formula_spacing"),
        ("سبحانه وتعالي", "سبحانه وتعالى", "alif_maqsura_restoration"),
        ("القران", "القرآن", "madda_restoration"),
    ],
)
def test_lexicon_entries_are_corrected(wrong: str, right: str, category: str) -> None:
    text, corrections = optimize_text(wrong, preserve_diacritics=True, dialect=None, normalize_latin=False)

    assert text == right
    assert [(c.original_span, c.corrected_span, c.category) for c in corrections] == [(wrong, right, category)]


def test_correct_text_is_left_unchanged() -> None:
    text, corrections = optimize_text("ما حكم الصلاة", preserve_diacritics=True, dialect=None, normalize_latin=False)

    assert text == "ما حكم الصلاة"
    assert corrections == []


def test_whitespace_is_collapsed() -> None:
    text, _ = optimize_text("  ما   حكم  ", preserve_diacritics=True, dialect=None, normalize_latin=False)

    assert text == "ما حكم"


def test_correction_applies_after_a_prefix_letter() -> None:
    text, _ = optimize_text("والزكاه", preserve_diacritics=True, dialect=None, normalize_latin=False)

    assert text == "والزكاة"


def test_correction_does_not_fire_inside_a_longer_word() -> None:
    text, corrections = optimize_text("مكهرباء", preserve_diacritics=True, dialect=None, normalize_latin=False)

    assert text == "مكهرباء"
    assert corrections == []


# ---------------------------------------------------------------------------
# Diacritics
# ---------------------------------------------------------------------------


def test_diacritized_words_are_never_damaged_when_preserved() -> None:
    text, corrections = optimize_text(SHADDA_SALAH, preserve_diacritics=True, dialect=None, normalize_latin=False)

    assert text == SHADDA_SALAH
    assert corrections == []


def test_stripping_diacritics_lets_the_correction_apply() -> None:
    text, corrections = optimize_text(SHADDA_SALAH, preserve_diacritics=False, dialect=None, normalize_latin=False)

    assert text == "الصلاة"
    assert [c.original_span for c in corrections] == ["الصلاه"]


def test_preserve_diacritics_flag_is_honoured_end_to_end() -> None:
    kept = optimize_arabic_stt(_request(SHADDA_SALAH, preserve_diacritics=True))
    stripped = optimize_arabic_stt(_request(SHADDA_SALAH, preserve_diacritics=False))

    assert kept.optimized_transcript == SHADDA_SALAH
    assert stripped.optimized_transcript == "الصلاة"
    assert PLAIN_SALAH_WRONG not in stripped.optimized_transcript


def test_islamic_terms_are_detected_in_diacritized_text() -> None:
    result = optimize_arabic_stt(_request("الصَّلاةُ والزَّكاةُ"))

    assert "الصلاة" in result.detected_islamic_terms
    assert "الزكاة" in result.detected_islamic_terms


# ---------------------------------------------------------------------------
# Dialects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hint", ["egyptian", "Egyptian", "  GULF ", "levantine", "msa", "maghrebi"])
def test_a_recognised_hint_wins_over_detection(hint: str) -> None:
    dialect, confidence, source = resolve_dialect("كلام عادي", hint)

    assert dialect == hint.strip().lower()
    assert confidence is None
    assert source == "hint"


def test_an_unrecognised_hint_is_ignored_and_the_dialect_detected() -> None:
    dialect, confidence, source = resolve_dialect("كلام عادي", "klingon")

    assert dialect in KNOWN_DIALECTS
    assert confidence is not None
    assert source == "detected"


def test_text_without_arabic_has_no_detected_dialect() -> None:
    assert resolve_dialect("hello world", None) == (None, None, None)
    assert resolve_dialect("", None) == (None, None, None)


def test_a_hint_is_used_even_for_text_without_arabic() -> None:
    assert resolve_dialect("hello", "gulf") == ("gulf", None, "hint")


def test_plain_msa_is_reported_as_msa() -> None:
    dialect, confidence, source = resolve_dialect("ما حكم الصلاة في السفر", None)

    assert (dialect, source) == ("msa", "detected")
    assert confidence == 0.95


def test_egyptian_is_detected_from_distinctive_words() -> None:
    dialect, _, source = resolve_dialect("انا عايز اروح دلوقتي", None)

    assert (dialect, source) == ("egyptian", "detected")


@pytest.mark.parametrize(
    ("text", "expected_confidence"),
    [("كلام بزاف", 0.75), ("علاش بزاف", 0.9)],
)
def test_maghrebi_is_detected_from_distinctive_words(text: str, expected_confidence: float) -> None:
    dialect, confidence, source = resolve_dialect(text, None)

    assert (dialect, source) == ("maghrebi", "detected")
    assert confidence == expected_confidence


def test_maghrebi_words_are_not_matched_inside_longer_words() -> None:
    assert stt._count_maghrebi_markers("كلام بزافة") == 0
    assert stt._count_maghrebi_markers("بزاف") == 1


def test_maghrebi_text_is_rewritten_to_msa() -> None:
    assert to_msa("علاش بزاف", "maghrebi") == "لماذا كثيرا"
    assert to_msa("ديال", "maghrebi") == "ديال"


def test_msa_and_non_arabic_text_are_not_rewritten() -> None:
    assert to_msa("عايز دلوقتي", "msa") == "عايز دلوقتي"
    assert to_msa("hello", "egyptian") == "hello"


def test_egyptian_text_is_rewritten_to_msa_with_the_shared_classifier() -> None:
    msa = to_msa("انا عايز اروح دلوقتي", "egyptian")

    assert "الآن" in msa
    assert "دلوقتي" not in msa


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ("egyptian", "ايه"),  # "what" in Egyptian: must not become "verse"
        ("levantine", "ايه"),  # "yes" in Levantine
        ("msa", "آية"),
        ("gulf", "آية"),
    ],
)
def test_a_word_that_means_something_else_in_a_dialect_is_not_rewritten(hint: str, expected: str) -> None:
    result = optimize_arabic_stt(_request("ايه", dialect_hint=hint))

    assert result.optimized_transcript == expected
    assert result.detected_dialect == hint
    assert result.dialect_source == "hint"


# ---------------------------------------------------------------------------
# Response fields: dialect, MSA, code-switching, Latin formulas
# ---------------------------------------------------------------------------


def test_response_reports_the_detected_dialect() -> None:
    result = optimize_arabic_stt(_request("انا عايز اروح دلوقتي"))

    assert result.detected_dialect == "egyptian"
    assert result.dialect_source == "detected"
    assert result.dialect_confidence is not None


def test_msa_transcript_is_only_returned_when_requested() -> None:
    assert optimize_arabic_stt(_request("انا عايز اروح دلوقتي")).msa_transcript is None

    result = optimize_arabic_stt(_request("انا عايز اروح دلوقتي", normalize_to_msa=True))

    assert result.msa_transcript is not None
    assert "الآن" in result.msa_transcript


def test_code_switching_is_flagged_for_mixed_arabic_and_english() -> None:
    result = optimize_arabic_stt(_request("ما حكم الصلاه in Islam"))

    assert result.code_switching is True
    assert result.latin_segments == ["in", "Islam"]
    assert "الصلاة" in result.optimized_transcript  # the Arabic part is still corrected
    assert result.optimized_transcript.endswith("in Islam")  # the English part is untouched


def test_pure_arabic_or_pure_english_is_not_code_switching() -> None:
    assert optimize_arabic_stt(_request("ما حكم الصلاة")).code_switching is False
    english = optimize_arabic_stt(_request("what is the ruling on prayer"))
    assert english.code_switching is False
    assert english.detected_dialect is None


@pytest.mark.parametrize(
    ("latin", "arabic"),
    [
        ("inshallah", "إن شاء الله"),
        ("Insha Allah", "إن شاء الله"),
        ("alhamdulillah", "الحمد لله"),
        ("SubhanAllah", "سبحان الله"),
        ("masha'allah", "ما شاء الله"),
        ("Bismillah", "بسم الله"),
        ("astaghfirullah", "أستغفر الله"),
        ("Allahu Akbar", "الله أكبر"),
        ("assalamu alaikum", "السلام عليكم"),
    ],
)
def test_latin_formulas_are_written_in_arabic_when_requested(latin: str, arabic: str) -> None:
    text, corrections = normalize_latin_formulas(f"قال {latin} غدا")

    assert text == f"قال {arabic} غدا"
    assert [c.category for c in corrections] == ["latin_formula"]
    assert corrections[0].original_span.lower() == latin.lower()


def test_latin_formulas_are_left_alone_by_default() -> None:
    result = optimize_arabic_stt(_request("قال inshallah غدا"))

    assert "inshallah" in result.optimized_transcript


def test_latin_formulas_option_applies_end_to_end() -> None:
    result = optimize_arabic_stt(_request("قال inshallah غدا", normalize_latin_terms=True))

    assert result.optimized_transcript == "قال إن شاء الله غدا"
    assert any(c.category == "latin_formula" for c in result.corrections)


def test_latin_formula_is_not_matched_inside_a_longer_word() -> None:
    text, corrections = normalize_latin_formulas("xinshallahx")

    assert text == "xinshallahx"
    assert corrections == []


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def test_the_candidate_with_more_islamic_vocabulary_wins_at_equal_confidence() -> None:
    request = ArabicSTTOptimizationRequest.model_validate(
        {
            "candidates": [
                {"transcript": "كلام عام غير مرتبط", "acoustic_confidence": 0.8},
                {"transcript": "ما حكم الصلاة والزكاة والصيام", "acoustic_confidence": 0.8},
            ]
        }
    )

    assert optimize_arabic_stt(request).best_raw_transcript == "ما حكم الصلاة والزكاة والصيام"


def test_vocabulary_counts_for_diacritized_candidates_too() -> None:
    request = ArabicSTTOptimizationRequest.model_validate(
        {
            "candidates": [
                {"transcript": "كلام عام غير مرتبط", "acoustic_confidence": 0.8},
                {"transcript": "الصَّلاةُ والزَّكاةُ والصِّيامُ والحَجُّ", "acoustic_confidence": 0.8},
            ]
        }
    )

    best = optimize_arabic_stt(request).best_raw_transcript

    assert strip_diacritics(best) == "الصلاة والزكاة والصيام والحج"


def test_confidence_is_boosted_per_correction() -> None:
    result = optimize_arabic_stt(_request("الصلاه والزكاه"))

    assert result.confidence == 0.88  # 0.8 + 2 corrections * 0.04
    assert optimize_arabic_stt(_request("الصلاة")).confidence == 0.8


def test_confidence_never_exceeds_the_cap() -> None:
    request = ArabicSTTOptimizationRequest.model_validate(
        {"candidates": [{"transcript": "الصلاه والزكاه والجنه", "acoustic_confidence": 0.98}]}
    )

    assert optimize_arabic_stt(request).confidence == 0.99


# ---------------------------------------------------------------------------
# Real-time streaming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "is_final", "expected"),
    [
        ("ما حكم الصلا", False, ("ما حكم", "الصلا")),  # the last word may be incomplete
        ("ما حكم الصلا ", False, ("ما حكم الصلا", "")),  # trailing space: the word is complete
        ("ما حكم الصلا", True, ("ما حكم الصلا", "")),  # the utterance is over
        ("ما", False, ("", "ما")),
        ("", False, ("", "")),
        ("   ", False, ("", "")),
    ],
)
def test_split_stream(text: str, is_final: bool, expected: tuple[str, str]) -> None:
    assert split_stream(text, is_final) == expected


def test_stream_corrects_complete_words_and_holds_back_the_unfinished_one() -> None:
    response = optimize_stream(STTStreamRequest(partial_transcript="ما حكم الصلاه والزك"))

    assert response.stable_text == "ما حكم الصلاة"
    assert response.pending_text == "والزك"
    assert response.full_text == "ما حكم الصلاة والزك"
    assert response.is_final is False
    assert [c.corrected_span for c in response.corrections] == ["الصلاة"]


def test_stream_final_result_corrects_every_word() -> None:
    response = optimize_stream(STTStreamRequest(partial_transcript="ما حكم الصلاه والزكاه", is_final=True))

    assert response.stable_text == "ما حكم الصلاة والزكاة"
    assert response.pending_text == ""
    assert response.is_final is True


def test_stream_with_a_single_unfinished_word_has_nothing_stable_yet() -> None:
    response = optimize_stream(STTStreamRequest(partial_transcript="الصلا"))

    assert response.stable_text == ""
    assert response.pending_text == "الصلا"
    assert response.corrections == []


def test_stream_honours_the_dialect_hint() -> None:
    response = optimize_stream(STTStreamRequest(partial_transcript="ايه ", dialect_hint="egyptian"))

    assert response.stable_text == "ايه"


# ---------------------------------------------------------------------------
# Error-rate measurement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [
        ([], [], 0),
        (["a"], [], 1),
        ([], ["a", "b"], 2),
        (["a", "b", "c"], ["a", "b", "c"], 0),
        (["a", "b", "c"], ["a", "x", "c"], 1),
        (["a", "b", "c"], ["a", "c"], 1),
        (list("kitten"), list("sitting"), 3),
    ],
)
def test_edit_distance(a: list[str], b: list[str], distance: int) -> None:
    assert edit_distance(a, b) == distance
    assert edit_distance(b, a) == distance  # symmetric


def test_identical_text_has_zero_error_rates() -> None:
    assert error_rates("ما حكم الصلاة", "ما حكم الصلاة") == (0.0, 0.0)


def test_one_wrong_word_out_of_four_is_a_25_percent_word_error_rate() -> None:
    wer, cer = error_rates("ما حكم الصلاة اليوم", "ما حكم الصلاه اليوم")

    assert wer == 0.25
    assert 0.0 < cer < 0.1


def test_error_rates_ignore_punctuation_and_diacritics_by_default() -> None:
    assert error_rates("ما حكم الصلاة؟", "ما حكم الصَّلاةُ") == (0.0, 0.0)


def test_diacritics_count_as_errors_when_asked() -> None:
    wer, _ = error_rates("الصَّلاةُ", "الصلاة", compare_diacritics=True)

    assert wer == 1.0


def test_an_empty_hypothesis_is_a_total_error() -> None:
    assert error_rates("ما حكم", "") == (1.0, 1.0)


def test_an_empty_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="no comparable text"):
        error_rates("؟؟", "ما حكم")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_optimize_endpoint_returns_the_new_fields(client: TestClient) -> None:
    response = client.post(
        "/stt-optimizer/optimize",
        json={"candidates": [{"transcript": "ما حكم الصلاه in Islam"}], "normalize_to_msa": True},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["optimized_transcript"] == "ما حكم الصلاة in Islam"
    assert body["code_switching"] is True
    assert body["latin_segments"] == ["in", "Islam"]
    assert body["msa_transcript"] is not None


def test_stream_endpoint(client: TestClient) -> None:
    response = client.post("/stt-optimizer/stream", json={"partial_transcript": "ما حكم الصلاه والزك"})

    assert response.status_code == 200
    assert response.json()["stable_text"] == "ما حكم الصلاة"


def test_batch_endpoint_processes_items_independently(client: TestClient) -> None:
    response = client.post(
        "/stt-optimizer/batch",
        json={
            "items": [
                {"candidates": [{"transcript": "الصلاه"}]},
                {"candidates": [{"transcript": "ايه"}], "dialect_hint": "egyptian"},
            ]
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["count"] == 2
    assert [r["optimized_transcript"] for r in body["results"]] == ["الصلاة", "ايه"]


def test_batch_endpoint_limits_the_number_of_items(client: TestClient) -> None:
    item = {"candidates": [{"transcript": "الصلاه"}]}

    assert client.post("/stt-optimizer/batch", json={"items": []}).status_code == 422
    assert client.post("/stt-optimizer/batch", json={"items": [item] * 51}).status_code == 422
    assert client.post("/stt-optimizer/batch", json={"items": [item] * 50}).status_code == 200


def test_evaluate_endpoint_shows_the_improvement(client: TestClient) -> None:
    response = client.post(
        "/stt-optimizer/evaluate",
        json={"candidates": [{"transcript": "ما حكم الصلاه والزكاه"}], "reference": "ما حكم الصلاة والزكاة"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["raw_wer"] == 0.5
    assert body["optimized_wer"] == 0.0
    assert body["wer_improvement"] == 0.5
    assert body["raw_accuracy"] == 0.5
    assert body["optimized_accuracy"] == 1.0
    assert body["optimized_transcript"] == "ما حكم الصلاة والزكاة"
    assert "corpus" in body["note"]


def test_evaluate_endpoint_rejects_oversized_input_and_empty_reference(client: TestClient) -> None:
    too_long = "ا" * (stt.MAX_EVAL_CHARS + 1)
    oversized = {"candidates": [{"transcript": too_long}], "reference": "ما"}
    blank_reference = {"candidates": [{"transcript": "ما حكم"}], "reference": "؟؟"}

    assert client.post("/stt-optimizer/evaluate", json=oversized).status_code == 422
    assert client.post("/stt-optimizer/evaluate", json=blank_reference).status_code == 422


def test_existing_endpoint_contract_is_unchanged(client: TestClient) -> None:
    transcript = "ما حكم الصلاه والزكاه في سوره البقرة ان شاء الله"
    response = client.post(
        "/stt-optimizer/optimize",
        json={"candidates": [{"transcript": transcript, "acoustic_confidence": 0.85}]},
    )

    body = response.json()
    assert body["optimized_transcript"] == "ما حكم الصلاة والزكاة في سورة البقرة إن شاء الله"
    assert len(body["corrections"]) >= 4
