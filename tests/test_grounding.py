"""Tests for source-grounded generation enforcement (``grounding.py``).

Everything here runs offline: ``grounding`` is pure Python with no LLM or
network calls, so no mocks are needed to keep the tests deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import grounding
from grounding import (
    AgentResponse,
    GroundingAction,
    GroundingReport,
    SourceIndex,
    apply_grounding_rules,
    check_entailment,
    compute_grounding,
    detect_hallucinations,
    index_from_strings,
    index_from_tafsir_context,
    normalize_terminology,
    sentence_fidelity,
    split_sentences,
    synthesize_responses,
    token_overlap_ratio,
    validate_synthesis,
)

ZAKAT_SOURCE = "Zakat is due on wealth held for a full lunar year. The nisab for gold is twenty mithqal."


def _zakat_index() -> SourceIndex:
    return index_from_strings([ZAKAT_SOURCE], source_id="fiqh")


# ---------------------------------------------------------------------------
# Sentence splitting and tokenisation helpers
# ---------------------------------------------------------------------------


def test_split_sentences_returns_empty_list_for_blank_text() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n\t ") == []


def test_split_sentences_splits_on_terminal_punctuation() -> None:
    sentences = split_sentences("First claim. Second claim! Third claim; Fourth claim:")
    assert sentences == ["First claim.", "Second claim!", "Third claim;", "Fourth claim:"]


def test_token_overlap_ratio_grounded_sentence_is_high() -> None:
    assert token_overlap_ratio("Zakat is due on wealth held for a year", ZAKAT_SOURCE) == 1.0


def test_token_overlap_ratio_unrelated_sentence_is_zero() -> None:
    assert token_overlap_ratio("Interest rates rose sharply in London", ZAKAT_SOURCE) == 0.0


def test_token_overlap_ratio_handles_stopword_only_input() -> None:
    assert token_overlap_ratio("the and of", ZAKAT_SOURCE) == 0.0


# ---------------------------------------------------------------------------
# Source index
# ---------------------------------------------------------------------------


def test_source_index_ranks_the_matching_passage_first() -> None:
    index = index_from_strings(
        [
            "Gold and silver are the two metals subject to nisab.",
            "The prayer times begin at dawn.",
        ]
    )
    results = index.find_relevant("nisab on gold and silver")
    assert results
    assert results[0][0] == 0
    assert results[0][1] == 1.0
    best = index.best_matching_passage("nisab on gold and silver")
    assert best is not None
    assert "nisab" in best.text


def test_source_index_empty_query_has_no_matches() -> None:
    index = _zakat_index()
    assert index.find_relevant("") == []
    assert index.best_overlap("the and of") == 0.0
    assert index.best_matching_passage("") is None


def test_source_index_from_texts_sets_source_ids() -> None:
    index = SourceIndex.from_texts(["alpha", "beta"], source_id="tafsir")
    assert [p.source_id for p in index.passages] == ["tafsir:0", "tafsir:1"]


def test_index_from_strings_is_empty_for_no_text() -> None:
    index = index_from_strings([])
    assert index.passages == []
    assert index.best_overlap("zakat") == 0.0


# ---------------------------------------------------------------------------
# Sentence fidelity: grounded vs ungrounded
# ---------------------------------------------------------------------------


def test_grounded_sentence_scores_high() -> None:
    index = _zakat_index()
    assert sentence_fidelity("Zakat is due on wealth held for a full lunar year.", index) >= 0.9


def test_ungrounded_sentence_scores_low() -> None:
    index = _zakat_index()
    assert sentence_fidelity("The Prophet rode a bicycle to Madinah in 1998.", index) < 0.15


def test_blank_sentence_is_vacuously_grounded() -> None:
    assert sentence_fidelity("   ", _zakat_index()) == 1.0


def test_detect_hallucinations_marks_grounded_and_ungrounded_sentences() -> None:
    index = _zakat_index()
    verdicts = detect_hallucinations(
        "The nisab for gold is twenty mithqal. Camels were first domesticated in Australia.",
        index,
    )
    assert len(verdicts) == 2
    assert verdicts[0].supported is True
    assert verdicts[1].supported is False
    assert "nisab" in verdicts[0].closest_source


def test_detect_hallucinations_threshold_override_flips_verdict() -> None:
    index = _zakat_index()
    text = "The Prophet rode a bicycle to Madinah in 1998."
    assert detect_hallucinations(text, index)[0].supported is False
    assert detect_hallucinations(text, index, overlap_threshold=0.0)[0].supported is True


# ---------------------------------------------------------------------------
# Aggregate grounding report
# ---------------------------------------------------------------------------


def test_compute_grounding_passes_for_grounded_answer() -> None:
    report = compute_grounding(
        "Zakat is due on wealth held for a full lunar year. The nisab for gold is twenty mithqal.",
        _zakat_index(),
    )
    assert report.sentence_count == 2
    assert report.supported_count == 2
    assert report.unsupported_count == 0
    assert report.hallucination_ratio == 0.0
    assert report.passed is True


def test_compute_grounding_fails_for_fabricated_answer() -> None:
    report = compute_grounding(
        "Camels were first domesticated in Australia. Bitcoin mining consumes electricity in Iceland.",
        _zakat_index(),
    )
    assert report.hallucination_ratio == 1.0
    assert report.passed is False
    assert all(not verdict.supported for verdict in report.verdicts)


def test_compute_grounding_ungrounded_with_empty_context() -> None:
    """No retrieved sources at all means nothing can be grounded."""
    report = compute_grounding("Zakat is due on wealth held for a full lunar year.", SourceIndex())
    assert report.sentence_count == 1
    assert report.supported_count == 0
    assert report.hallucination_ratio == 1.0
    assert report.passed is False


def test_compute_grounding_empty_answer_passes_trivially() -> None:
    report = compute_grounding("", _zakat_index())
    assert report.sentence_count == 0
    assert report.fidelity_score == 1.0
    assert report.passed is True


def test_compute_grounding_respects_explicit_thresholds() -> None:
    ungrounded = "Camels were first domesticated in Australia."
    strict = compute_grounding(ungrounded, _zakat_index())
    lenient = compute_grounding(ungrounded, _zakat_index(), fidelity_floor=0.0, hallucination_limit=1.0)
    assert strict.passed is False
    assert lenient.passed is True


def test_grounding_report_to_dict_is_json_safe() -> None:
    report = compute_grounding("Zakat is due on wealth.", _zakat_index())
    payload = report.to_dict()
    assert payload["sentence_count"] == report.sentence_count
    assert payload["passed"] is report.passed
    assert isinstance(payload["fidelity_score"], float)


# ---------------------------------------------------------------------------
# Blocking rules
# ---------------------------------------------------------------------------


def test_rules_pass_a_grounded_report() -> None:
    decision = apply_grounding_rules(compute_grounding("Zakat is due on wealth.", _zakat_index()))
    assert decision.action is GroundingAction.PASS
    assert decision.reason == ""


def test_rules_block_a_mostly_unsupported_report() -> None:
    decision = apply_grounding_rules(
        compute_grounding(
            "Camels were first domesticated in Australia. Bitcoin mining consumes electricity in Iceland.",
            _zakat_index(),
        )
    )
    assert decision.action is GroundingAction.BLOCK
    assert "Hallucination ratio" in decision.reason
    assert decision.to_dict()["action"] == "block"


def test_rules_warn_when_fidelity_is_low_but_hallucination_is_within_limit() -> None:
    report = GroundingReport(
        fidelity_score=0.10,
        hallucination_ratio=0.20,
        sentence_count=5,
        supported_count=4,
        unsupported_count=1,
        passed=False,
    )
    decision = apply_grounding_rules(report)
    assert decision.action is GroundingAction.WARN
    assert "below floor" in decision.reason


# ---------------------------------------------------------------------------
# Entailment heuristic
# ---------------------------------------------------------------------------


def test_check_entailment_supports_paraphrase_with_shared_content() -> None:
    result = check_entailment(ZAKAT_SOURCE, "Zakat is due on wealth held for a full lunar year.")
    assert result.supported is True
    assert result.score == 1.0


def test_check_entailment_empty_claim_is_supported() -> None:
    result = check_entailment(ZAKAT_SOURCE, "")
    assert result.supported is True
    assert result.reason == "empty claim"


def test_check_entailment_rejects_empty_source() -> None:
    result = check_entailment("", "Zakat is due on wealth")
    assert result.supported is False
    assert result.reason == "empty source"


def test_check_entailment_flags_novel_entities() -> None:
    result = check_entailment(
        "Zakat is due on wealth",
        "Imam Bukhari narrated that Zakat is due on camels",
    )
    assert result.supported is False
    assert "novel entities" in result.reason


# ---------------------------------------------------------------------------
# Index builders and fail-open behavior
# ---------------------------------------------------------------------------


def test_index_from_tafsir_context_handles_none() -> None:
    index = index_from_tafsir_context(None)
    assert index.passages == []


def test_index_from_tafsir_context_reads_passages() -> None:
    @dataclass
    class FakeTafsirContext:
        passages: list[str]

    index = index_from_tafsir_context(FakeTafsirContext(passages=["  Ayah about patience. ", ""]))
    assert len(index.passages) == 1
    assert index.passages[0].text == "Ayah about patience."
    assert index.passages[0].source_id == "tafsir"


def test_index_from_tafsir_context_ignores_unexpected_shape() -> None:
    index = index_from_tafsir_context(object())
    assert index.passages == []


# ---------------------------------------------------------------------------
# Multi-agent synthesis
# ---------------------------------------------------------------------------


def test_synthesize_responses_merges_duplicates_with_attribution() -> None:
    report = synthesize_responses(
        [
            AgentResponse(
                agent_id="fiqh", text="Zakat is due on wealth held for one lunar year.", citations=["Quran 9:60"]
            ),
            AgentResponse(
                agent_id="hadith", text="Zakat is due on wealth held for one lunar year.", citations=["Bukhari 1454"]
            ),
        ],
        source_index=_zakat_index(),
    )
    assert len(report.sentences) == 1
    assert report.redundancy_removed == 1
    assert set(report.sentences[0].agent_ids) == {"fiqh", "hadith"}
    assert set(report.sentences[0].citations) == {"Quran 9:60", "Bukhari 1454"}
    assert validate_synthesis(report) is True


def test_synthesize_responses_empty_input_is_not_deliverable() -> None:
    report = synthesize_responses([])
    assert report.text == ""
    assert report.sentences == []
    assert validate_synthesis(report) is False


def test_synthesize_responses_normalizes_terminology() -> None:
    report = synthesize_responses(
        [AgentResponse(agent_id="fiqh", text="zakaah is due on wealth.")],
        terminology_map={"zakaah": "Zakat"},
    )
    assert report.text == "Zakat is due on wealth."


def test_normalize_terminology_is_case_insensitive_and_word_bounded() -> None:
    assert normalize_terminology("Zakaah and zakaah", {"zakaah": "zakat"}) == "zakat and zakat"
    # Substrings inside other words must not be rewritten.
    assert normalize_terminology("zakaahs", {"zakaah": "zakat"}) == "zakaahs"


def test_grounding_block_message_is_exposed() -> None:
    assert "authentic sources" in grounding.GROUNDING_BLOCK_MESSAGE
