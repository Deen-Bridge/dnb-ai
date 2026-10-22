"""Tests for the synthesis module."""

import pytest

from synthesis import (
    AttributionTracker,
    ContradictionDetector,
    NarrativeGenerator,
    SynthesisEngine,
    SynthesisError,
    TermNormalizer,
)


def test_term_normalizer_strips_stopwords_and_applies_synonyms() -> None:
    normalized = TermNormalizer.normalize("The AI implementation is an application!")

    assert normalized == "ai implement app"
    assert TermNormalizer.normalize("") == ""


def test_contradiction_detector_finds_antonym_pairs() -> None:
    contradictions = ContradictionDetector.detect_contradictions(
        [
            "The rate will increase.",
            "The rate will decrease.",
        ]
    )

    assert len(contradictions) == 1
    assert contradictions[0]["reason"] == "Antonym pair"
    assert contradictions[0]["confidence"] == 0.7


def test_contradiction_detector_finds_negation_mismatch() -> None:
    contradictions = ContradictionDetector.detect_contradictions(
        [
            "Prayer is required.",
            "Prayer is never required.",
        ]
    )

    assert len(contradictions) == 1
    assert contradictions[0]["reason"] == "Negation mismatch"


def test_contradiction_detector_ignores_unrelated_statements() -> None:
    assert ContradictionDetector.detect_contradictions(["Zakat is due.", "Pilgrimage is costly."]) == []


def test_attribution_tracker_consolidates_duplicate_claims() -> None:
    tracker = AttributionTracker()
    tracker.add_attribution("Zakat is due.", "agent_a", 0.8)
    tracker.add_attribution("zakat is due!", "agent_b", 0.6)

    consolidated = tracker.consolidate()

    assert len(consolidated) == 1
    assert consolidated[0]["claim"] == "Zakat is due."
    assert set(consolidated[0]["agents"]) == {"agent_a", "agent_b"}
    assert consolidated[0]["confidence"] == pytest.approx(0.7)


def test_narrative_generator_returns_empty_for_no_segments() -> None:
    assert NarrativeGenerator.generate([], ["overview"]) == ""


def test_synthesis_engine_rejects_empty_input() -> None:
    with pytest.raises(SynthesisError, match="No agent outputs"):
        SynthesisEngine().synthesize([])


def test_synthesis_engine_builds_narrative_and_attribution() -> None:
    result = SynthesisEngine().synthesize(
        [
            {"agent": "quran_agent", "content": "Overview: Zakat is due. Details: It purifies wealth."},
            {"agent": "fiqh_agent", "content": "Conclusion: Pay it annually."},
        ]
    )

    assert "## Overview" in result["synthesized_text"]
    assert "## Details" in result["synthesized_text"]
    assert "## Conclusion" in result["synthesized_text"]
    assert "## References" in result["synthesized_text"]
    assert result["quality"]["attribution_accuracy"] == 1.0
    assert result["contradictions"] == []


def test_synthesis_engine_reports_contradictions() -> None:
    result = SynthesisEngine().synthesize(
        [
            {"agent": "agent_a", "content": "The rate will increase."},
            {"agent": "agent_b", "content": "The rate will decrease."},
        ]
    )

    assert len(result["contradictions"]) == 1
    assert "## Contradictions Detected" in result["synthesized_text"]
    assert "Antonym pair" in result["synthesized_text"]
