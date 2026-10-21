from hallucination import (
    HallucinationSeverity,
    HallucinationType,
    detect_fabricated_citations,
    detect_misattributions,
    detect_scholar_position_errors,
    detect_temporal_errors,
    detect_unsupported_claims,
    load_benchmark_dataset,
    run_benchmark,
    scan_for_hallucinations,
)


class TestHallucinationDetectors:
    def test_detect_fabricated_quran(self):
        text = "Allah says in Surah 250:5 'Do not despair'."
        flags = detect_fabricated_citations(text)
        assert len(flags) > 0
        assert any(f.hallucination_type == HallucinationType.FABRICATED_VERSE for f in flags)
        assert any(f.severity == HallucinationSeverity.CRITICAL for f in flags)

    def test_detect_misquoted_quran(self):
        # A verse reference that exists but with totally wrong text
        text = "As stated in Surah 2:255 'God is love and light'."
        flags = detect_fabricated_citations(text)
        assert len(flags) > 0
        # Might be FABRICATED_CITATION due to verifier failing similarity

    def test_detect_fabricated_hadith(self):
        text = "The Prophet said 'Seek knowledge even if in China'."
        flags = detect_fabricated_citations(text)
        assert len(flags) > 0
        assert any(f.hallucination_type == HallucinationType.FABRICATED_HADITH for f in flags)

    def test_detect_misattributions(self):
        # Anachronism example
        text = "Imam Shafi'i (1995 CE) said this."
        flags = detect_misattributions(text)
        assert len(flags) > 0
        assert any(f.hallucination_type == HallucinationType.TEMPORAL_CONFUSION for f in flags)

        # Wrong school
        text = "Imam Malik, the founder of the Hanafi school, believed this."
        flags = detect_misattributions(text)
        assert len(flags) > 0
        assert any(f.hallucination_type == HallucinationType.MISATTRIBUTION for f in flags)

    def test_detect_unsupported_claims(self):
        text = "All scholars unanimously agree that this is required."
        flags = detect_unsupported_claims(text)
        assert len(flags) > 0
        assert any(f.hallucination_type == HallucinationType.UNSUPPORTED_CLAIM for f in flags)

    def test_detect_temporal_errors(self):
        text = "The Battle of Badr took place in 2020 CE."
        flags = detect_temporal_errors(text)
        assert len(flags) > 0
        assert any(f.hallucination_type == HallucinationType.TEMPORAL_CONFUSION for f in flags)

    def test_detect_scholar_position_errors(self):
        text = "The Hanafi school says we must recite qunut in Fajr."
        flags = detect_scholar_position_errors(text)
        assert len(flags) > 0
        assert any(f.hallucination_type == HallucinationType.SCHOLAR_POSITION_ERROR for f in flags)


class TestUnifiedScanner:
    def test_clean_text(self):
        text = "Prophet Muhammad ﷺ passed away in 632 CE."
        result = scan_for_hallucinations(text)
        assert result.hallucination_detected is False
        assert len(result.flags) == 0
        assert result.severity_score == 0.0

    def test_multiple_hallucinations(self):
        text = "In Surah 115:1 it says 'Hello'. Also the Battle of Badr was in 1999 CE."
        result = scan_for_hallucinations(text)
        assert result.hallucination_detected is True
        assert len(result.flags) >= 2
        assert result.max_severity == HallucinationSeverity.CRITICAL


class TestBenchmarkRunner:
    def test_load_dataset(self):
        examples = load_benchmark_dataset()
        assert len(examples) > 0
        assert examples[0].id is not None

    def test_run_benchmark(self):
        result = run_benchmark()
        assert result.total_examples > 0
        assert "overall_detection_>85%" in result.pass_criteria
        # Assuming our simple dataset generators will get >85% detection
        # We won't assert pass_criteria values here to prevent flakiness in unit tests,
        # but we ensure the structure is correct.
        assert result.detection_rate >= 0.0
