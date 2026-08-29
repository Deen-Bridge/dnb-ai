"""Offline tests for the Arabic language benchmark."""

from pathlib import Path

import pytest

from scripts.arabic_benchmark import (
    ArabicBenchmarkEvaluator,
    BenchmarkCase,
    HumanRating,
    compare_models,
    corpus_bleu,
    diacritical_accuracy,
    generate_cases,
    load_dataset,
    script_handling_score,
    validate_cases,
    write_dataset,
)


def test_generated_dataset_has_required_coverage() -> None:
    cases = generate_cases()
    assert len(cases) == 420
    assert validate_cases(cases) == []
    assert {case.variety for case in cases} >= {"msa", "classical", "dialect"}
    assert {case.dialect for case in cases} >= {"msa", "classical", "egyptian", "levantine", "gulf", "maghrebi"}
    assert {case.task for case in cases} == {
        "comprehension",
        "generation",
        "diacritics",
        "grammar",
        "terminology",
        "script_handling",
        "cultural_context",
    }
    assert {case.difficulty for case in cases} == {"easy", "medium", "hard"}


def test_generate_rejects_too_small_benchmark() -> None:
    with pytest.raises(ValueError, match="at least 400"):
        generate_cases(399)


def test_dataset_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "arabic.jsonl"
    written = write_dataset(path)
    loaded = load_dataset(path)
    assert loaded == written
    assert path.read_text(encoding="utf-8").startswith('{"id": "arabic-0001"')


def test_bleu_and_diacritical_accuracy() -> None:
    text = "إِنَّ مَعَ الْعُسْرِ يُسْرًا"
    assert corpus_bleu(text, text) == pytest.approx(1.0)
    assert diacritical_accuracy(text, text) == pytest.approx(1.0)
    assert diacritical_accuracy(text, "إن مع العسر يسرا") < 0.5


def test_script_handling_detects_arabic_and_corruption() -> None:
    assert script_handling_score("العربية لغة جميلة") > 0.9
    assert script_handling_score("????") == 0.0


def test_evaluator_scores_all_metrics_and_success_criteria() -> None:
    cases = generate_cases(420)
    responses = {case.id: case.reference_answer for case in cases}
    ratings = [HumanRating(cases[0].id, "reviewer-01", 4.8, 4.7, 4.8, 4.9, 4.8)]
    report = ArabicBenchmarkEvaluator(grammar_checker=lambda _text: 0, perplexity_scorer=lambda _text: 2.0).evaluate(
        cases,
        responses,
        ratings,
        english_baseline_accuracy=0.95,
    )
    assert report["total_cases"] == 420
    assert report["overall"]["bleu"] == pytest.approx(1.0)
    assert report["overall"]["diacritical_accuracy"] == pytest.approx(1.0)
    assert report["overall"]["grammar_error_rate"] == pytest.approx(0.0)
    assert report["overall"]["perplexity"] == pytest.approx(2.0)
    assert report["human_evaluation"]["fluency"] == pytest.approx(4.8)
    assert all(report["success_criteria"].values())


def test_evaluator_requires_complete_responses() -> None:
    case = generate_cases(400)[0]
    with pytest.raises(ValueError, match="Missing responses"):
        ArabicBenchmarkEvaluator().evaluate([case], {})


def test_terminology_and_forbidden_terms() -> None:
    case = BenchmarkCase(
        id="custom-001",
        variety="classical",
        dialect="classical",
        formality="formal",
        difficulty="hard",
        task="terminology",
        prompt="عرّف الإسناد.",
        reference_answer="الإسناد سلسلة رواة الحديث.",
        required_terms=("الإسناد", "الرواة"),
        forbidden_terms=("غير مهم",),
    )
    good = ArabicBenchmarkEvaluator(grammar_checker=lambda _text: 0).score(case, "الإسناد هو سلسلة الرواة.")
    bad = ArabicBenchmarkEvaluator(grammar_checker=lambda _text: 0).score(case, "الإسناد غير مهم.")
    assert good["terminology_accuracy"] == 1.0
    assert good["cultural_appropriateness"] == 1.0
    assert bad["terminology_accuracy"] == 0.5
    assert bad["cultural_appropriateness"] == 0.0


def test_human_rating_range_is_validated() -> None:
    with pytest.raises(ValueError, match="fluency"):
        HumanRating("case", "reviewer", 6.0, 4.0, 4.0, 4.0, 4.0)


def test_compare_models_ranks_stronger_report_first() -> None:
    strong = {
        "overall": {
            "comprehension_accuracy": 0.95,
            "bleu": 0.9,
            "diacritical_accuracy": 0.99,
            "terminology_accuracy": 0.96,
            "script_handling": 1.0,
            "grammar_error_rate": 0.01,
        }
    }
    weak = {
        "overall": {
            "comprehension_accuracy": 0.5,
            "bleu": 0.4,
            "diacritical_accuracy": 0.6,
            "terminology_accuracy": 0.5,
            "script_handling": 0.8,
            "grammar_error_rate": 0.2,
        }
    }
    leaderboard = compare_models({"deen-bridge": strong, "specialized-baseline": weak})
    assert leaderboard[0]["model"] == "deen-bridge"
