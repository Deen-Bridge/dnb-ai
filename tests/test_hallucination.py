"""Tests for the hallucination detection module."""

from hallucination import (
    BenchmarkExample,
    HallucinationSeverity,
    HallucinationType,
    detect_fabricated_citations,
    detect_scholar_position_errors,
    detect_temporal_errors,
    detect_unsupported_claims,
    load_benchmark_dataset,
    run_benchmark,
    scan_for_hallucinations,
)


def test_severity_weights_are_ordered() -> None:
    assert HallucinationSeverity.MINOR.weight == 0.25
    assert HallucinationSeverity.MODERATE.weight == 0.50
    assert HallucinationSeverity.MAJOR.weight == 0.75
    assert HallucinationSeverity.CRITICAL.weight == 1.0


def test_detects_misquoted_quran_citation() -> None:
    flags = detect_fabricated_citations('Quran 1:1 "This is a fabricated translation string."')

    assert len(flags) == 1
    assert flags[0].hallucination_type is HallucinationType.FABRICATED_CITATION
    assert flags[0].severity is HallucinationSeverity.MAJOR


def test_detects_impossible_surah_reference() -> None:
    flags = detect_fabricated_citations("As Surah 115 says, be patient.")

    assert len(flags) == 1
    assert flags[0].hallucination_type is HallucinationType.FABRICATED_VERSE
    assert flags[0].severity is HallucinationSeverity.CRITICAL
    assert "115" in flags[0].description


def test_detects_known_fabricated_hadith() -> None:
    result = scan_for_hallucinations("The Prophet said: Seek knowledge even if in China.")

    assert result.hallucination_detected is True
    assert result.max_severity is HallucinationSeverity.MAJOR
    assert any(flag.hallucination_type is HallucinationType.FABRICATED_HADITH for flag in result.flags)


def test_detects_absolute_unsupported_claim() -> None:
    flags = detect_unsupported_claims("All scholars unanimously agree that this is recommended.")

    assert len(flags) == 1
    assert flags[0].hallucination_type is HallucinationType.UNSUPPORTED_CLAIM
    assert flags[0].severity is HallucinationSeverity.MODERATE


def test_detects_temporal_error_for_badr() -> None:
    flags = detect_temporal_errors("The Battle of Badr took place in 600 CE.")

    assert len(flags) == 1
    assert flags[0].hallucination_type is HallucinationType.TEMPORAL_CONFUSION
    assert flags[0].expected_fact is not None
    assert "624" in flags[0].expected_fact


def test_detects_cross_school_position_error() -> None:
    flags = detect_scholar_position_errors("The Hanafi school holds that qunut in fajr is recommended.")

    assert len(flags) == 1
    assert flags[0].hallucination_type is HallucinationType.SCHOLAR_POSITION_ERROR
    assert flags[0].severity is HallucinationSeverity.MAJOR


def test_clean_text_produces_no_flags() -> None:
    result = scan_for_hallucinations("Patience is encouraged in the Quran.")

    assert result.hallucination_detected is False
    assert result.hallucination_count == 0
    assert result.max_severity is None
    assert result.severity_score == 0.0
    assert result.category_breakdown == {}


def test_scan_deduplicates_and_aggregates_flags() -> None:
    result = scan_for_hallucinations(
        "The Prophet said: Seek knowledge even if in China. The Hanafi school holds that qunut in fajr is recommended."
    )

    descriptions = [flag.description for flag in result.flags]
    assert len(descriptions) == len(set(descriptions))
    assert result.hallucination_count == len(result.flags)
    assert result.category_breakdown["fabricated_hadith"] == 1
    assert result.category_breakdown["scholar_position_error"] == 1


def test_benchmark_runner_handles_empty_input() -> None:
    result = run_benchmark([])

    assert result.total_examples == 0
    assert result.detection_rate == 0.0
    assert result.average_severity_score == 0.0


def test_benchmark_runner_computes_detection_rate() -> None:
    examples = [
        BenchmarkExample(
            id="fv_1",
            category=HallucinationType.FABRICATED_VERSE,
            severity=HallucinationSeverity.CRITICAL,
            prompt="What does Surah 115 say?",
            response="In Surah 115:15, Allah says something.",
            expected_hallucinations=["Reference to non-existent Surah 115"],
        )
    ]

    result = run_benchmark(examples)

    assert result.total_examples == 1
    assert result.detection_rate == 1.0
    assert result.critical_detection_rate == 1.0
    assert result.fabricated_verse_detection_rate == 1.0
    assert result.severity_distribution["critical"] == 1
    assert result.pass_criteria["fabricated_verse_detection_100%"] is True


def test_load_benchmark_dataset_reads_bundled_examples() -> None:
    examples = load_benchmark_dataset()

    assert len(examples) > 0
    assert all(example.id for example in examples)
    assert any(example.category is HallucinationType.FABRICATED_VERSE for example in examples)
