"""Tests for the Quran accuracy benchmark (#122)."""

from quran_benchmark import (
    _VERSE_DATA,
    BenchmarkResult,
    CaseResult,
    exact_match,
    print_report,
    reference_valid,
    run_benchmark,
    sequence_similarity,
    token_overlap,
)

# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_identical(self):
        assert exact_match("Hello", "Hello") is True

    def test_case_insensitive(self):
        assert exact_match("In the name of Allah", "in the name of allah") is True

    def test_whitespace_stripped(self):
        assert exact_match("  Hello  ", "Hello") is True

    def test_different(self):
        assert exact_match("Hello", "World") is False

    def test_empty(self):
        assert exact_match("", "") is True


class TestTokenOverlap:
    def test_identical(self):
        assert token_overlap("the cat sat", "the cat sat") == 1.0

    def test_no_overlap(self):
        assert token_overlap("apple orange", "grape banana") == 0.0

    def test_partial(self):
        score = token_overlap("the big cat sat", "the cat")
        assert 0.3 < score < 0.8

    def test_empty_expected(self):
        assert token_overlap("", "") == 1.0

    def test_empty_predicted(self):
        assert token_overlap("hello", "") == 0.0


class TestSequenceSimilarity:
    def test_identical(self):
        assert sequence_similarity("Hello world", "Hello world") == 1.0

    def test_similar(self):
        score = sequence_similarity("the cat sat", "the cat sat down")
        assert 0.6 < score < 1.0

    def test_different(self):
        score = sequence_similarity("apple", "banana")
        assert score < 0.5


class TestReferenceValid:
    def test_valid_first_surah(self):
        assert reference_valid(1, 1) is True
        assert reference_valid(1, 7) is True

    def test_invalid_ayah(self):
        assert reference_valid(1, 8) is False
        assert reference_valid(1, 0) is False

    def test_invalid_surah(self):
        assert reference_valid(0, 1) is False
        assert reference_valid(115, 1) is False

    def test_surah_2_has_286_ayahs(self):
        assert reference_valid(2, 286) is True
        assert reference_valid(2, 287) is False


# ---------------------------------------------------------------------------
# Ground truth data integrity
# ---------------------------------------------------------------------------


class TestGroundTruthData:
    def test_has_minimum_cases(self):
        assert len(_VERSE_DATA) >= 30

    def test_all_required_fields(self):
        required = {
            "id",
            "category",
            "surah",
            "ayah",
            "english",
            "tags",
            "related_verses",
            "partial_quote",
            "wrong_options",
        }
        for case in _VERSE_DATA:
            missing = required - set(case.keys())
            assert not missing, f"{case['id']} missing: {missing}"

    def test_all_surah_numbers_in_range(self):
        for case in _VERSE_DATA:
            assert 1 <= case["surah"] <= 114, f"{case['id']}: surah {case['surah']}"

    def test_categories_are_valid(self):
        valid_cats = {"exact_lookup", "translation_fidelity", "partial_match", "cross_reference", "edge_case"}
        for case in _VERSE_DATA:
            assert case["category"] in valid_cats, f"{case['id']}: {case['category']}"

    def test_all_references_valid(self):
        for case in _VERSE_DATA:
            assert reference_valid(case["surah"], case["ayah"]), (
                f"{case['id']}: invalid ref {case['surah']}:{case['ayah']}"
            )

    def test_no_duplicate_ids(self):
        ids = [c["id"] for c in _VERSE_DATA]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"

    def test_categories_covered(self):
        cats = {c["category"] for c in _VERSE_DATA}
        assert cats == {"exact_lookup", "translation_fidelity", "partial_match", "cross_reference", "edge_case"}


# ---------------------------------------------------------------------------
# Benchmark runner (self-test mode — uses ground truth as prediction)
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    def test_self_test_passes(self):
        """When prediction == ground truth, pass rate should be 100%."""
        result = run_benchmark()
        assert result.total_cases == len(_VERSE_DATA)
        assert result.pass_rate == 1.0

    def test_self_test_all_references_valid(self):
        result = run_benchmark()
        for cr in result.results:
            assert cr.reference_valid, f"{cr.case_id}: reference invalid"

    def test_category_counts(self):
        result = run_benchmark()
        assert result.category_counts["exact_lookup"] >= 10
        assert result.category_counts["translation_fidelity"] >= 5
        assert result.category_counts["partial_match"] >= 5
        assert result.category_counts["cross_reference"] >= 5
        assert result.category_counts["edge_case"] >= 5

    def test_latencies_recorded(self):
        result = run_benchmark()
        assert len(result.latencies_ms) == result.total_cases
        assert all(lat >= 0 for lat in result.latencies_ms)

    def test_category_rates(self):
        result = run_benchmark()
        for cat in result.category_counts:
            rate = result.category_rate(cat)
            assert 0.0 <= rate <= 1.0


class TestBenchmarkWithBadPredictions:
    def test_wrong_translation_fails(self):
        """A completely wrong translation should produce a low pass rate."""
        bad_data = [
            {
                "id": "bad-001",
                "category": "exact_lookup",
                "surah": 1,
                "ayah": 1,
                "arabic": "",
                "english": "In the name of Allah, the Entirely Merciful, the Especially Merciful.",
                "tags": [],
                "related_verses": [],
                "partial_quote": "",
                "wrong_options": [],
            }
        ]

        def bad_lookup(surah: int, ayah: int) -> str:
            return "Completely wrong translation with no similarity."

        result = run_benchmark(test_data=bad_data, translation_lookup_fn=bad_lookup)
        assert result.pass_rate < 1.0

    def test_custom_test_data(self):
        data = [
            {
                "id": "custom-001",
                "category": "translation_fidelity",
                "surah": 2,
                "ayah": 286,
                "arabic": "",
                "english": "Allah does not charge a soul except within its capacity.",
                "tags": [],
                "related_verses": [],
                "partial_quote": "",
                "wrong_options": [],
            }
        ]

        def good_lookup(surah: int, ayah: int) -> str:
            return "Allah does not charge a soul except within its capacity."

        result = run_benchmark(test_data=data, translation_lookup_fn=good_lookup)
        assert result.pass_rate == 1.0


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


class TestPrintReport:
    def test_pass_returns_zero(self):
        result = run_benchmark()
        code = print_report(result)
        assert code == 0

    def test_fail_returns_one(self):
        result = BenchmarkResult(
            total_cases=1,
            results=[
                CaseResult(
                    case_id="x",
                    category="exact_lookup",
                    passed=False,
                    exact_match_score=False,
                    token_overlap_score=0.0,
                    sequence_similarity=0.0,
                    reference_valid=True,
                ),
            ],
            category_counts={"exact_lookup": 1},
            category_passed={},
        )
        code = print_report(result)
        assert code == 1
