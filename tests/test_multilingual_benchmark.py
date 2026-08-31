from __future__ import annotations

import pytest

from scripts.eval_multilingual import (
    BenchmarkItem,
    BenchmarkValidationError,
    HumanEvaluation,
    Prediction,
    Translation,
    bleu_score,
    evaluate,
    script_is_valid,
    terminology_accuracy,
    validate_parallel_dataset,
)

LANGUAGES = ("en", "ar", "ur", "tr", "ms", "fr")


def make_items(count: int = 2) -> list[BenchmarkItem]:
    questions = {
        "en": "What is Tawhid?",
        "ar": "ما هو التوحيد؟",
        "ur": "توحید کیا ہے؟",
        "tr": "Tevhid nedir?",
        "ms": "Apakah tauhid?",
        "fr": "Qu'est-ce que le tawhid ?",
    }
    answers = {
        "en": "Tawhid means the oneness of Allah.",
        "ar": "التوحيد يعني إفراد الله بالعبادة.",
        "ur": "توحید سے مراد اللہ کی وحدانیت ہے۔",
        "tr": "Tevhid, Allah'ın birliği demektir.",
        "ms": "Tauhid bermaksud keesaan Allah.",
        "fr": "Le tawhid désigne l'unicité d'Allah.",
    }
    terms = {
        "en": ("Tawhid",),
        "ar": ("التوحيد",),
        "ur": ("توحید",),
        "tr": ("Tevhid",),
        "ms": ("Tauhid",),
        "fr": ("tawhid",),
    }
    return [
        BenchmarkItem(
            item_id=f"item-{index}",
            translations={
                language: Translation(questions[language], answers[language]) for language in LANGUAGES
            },
            terms=terms,
        )
        for index in range(count)
    ]


def make_predictions(items: list[BenchmarkItem]) -> list[Prediction]:
    return [
        Prediction(
            item_id=item.item_id,
            language=language,
            answer=item.translations[language].reference_answer,
            accurate=True,
            comet=0.95,
        )
        for item in items
        for language in LANGUAGES
    ]


def make_human_evaluations(items: list[BenchmarkItem]) -> list[HumanEvaluation]:
    return [
        HumanEvaluation(
            item_id=item.item_id,
            language=language,
            evaluator_id=f"reviewer-{language}",
            cultural_appropriateness=5,
            terminology=5,
            tone=5,
            source_appropriateness=5,
        )
        for item in items
        for language in LANGUAGES
    ]


def test_parallel_dataset_requires_complete_language_coverage() -> None:
    items = make_items()
    assert validate_parallel_dataset(items, minimum_questions=2) == LANGUAGES

    incomplete = list(items)
    first = incomplete[0]
    incomplete[0] = BenchmarkItem(
        item_id=first.item_id,
        translations={key: value for key, value in first.translations.items() if key != "fr"},
        terms=first.terms,
    )
    with pytest.raises(BenchmarkValidationError, match="languages on every question"):
        validate_parallel_dataset(incomplete, minimum_questions=2)


def test_parallel_dataset_enforces_question_count_and_unique_ids() -> None:
    items = make_items()
    with pytest.raises(BenchmarkValidationError, match="at least 200"):
        validate_parallel_dataset(items)

    duplicate = [items[0], items[0]]
    with pytest.raises(BenchmarkValidationError, match="duplicate question ids"):
        validate_parallel_dataset(duplicate, minimum_questions=2)


def test_bleu_and_terminology_metrics() -> None:
    answer = "Tawhid means the oneness of Allah."
    assert bleu_score(answer, answer) == pytest.approx(1.0)
    assert bleu_score(answer, "") == 0.0
    assert terminology_accuracy(answer, ["Tawhid", "Allah"]) == 1.0
    assert terminology_accuracy(answer, ["Tawhid", "Zakat"]) == 0.5
    assert terminology_accuracy(answer, []) is None


def test_script_validation_supports_arabic_urdu_and_latin_scripts() -> None:
    assert script_is_valid("التوحيد يعني إفراد الله بالعبادة.", "ar")
    assert script_is_valid("توحید سے مراد اللہ کی وحدانیت ہے۔", "ur")
    assert script_is_valid("Tevhid, Allah'ın birliği demektir.", "tr")
    assert script_is_valid("L'unicité d'Allah.", "fr")
    assert not script_is_valid("This is not Arabic", "ar")
    assert not script_is_valid("invalid replacement \ufffd", "en")


def test_complete_evaluation_reports_success_and_language_parity() -> None:
    items = make_items()
    predictions = make_predictions(items)
    human_evaluations = make_human_evaluations(items)

    report = evaluate(
        items,
        predictions,
        human_evaluations,
        semantic_scorer=lambda first, second: 1.0 if first and second else 0.0,
        minimum_questions=2,
    )

    assert report["benchmark"]["language_count"] == 6
    assert report["aggregate"]["cross_lingual_semantic_equivalence"] == 1.0
    assert report["aggregate"]["islamic_term_accuracy"] == 1.0
    assert report["aggregate"]["script_correctness"] == 1.0
    assert report["performance_gap_from_english"]["fr"] == 0.0
    assert all(report["success_criteria"].values())
    assert report["passed"] is True


def test_evaluation_rejects_missing_predictions() -> None:
    items = make_items()
    predictions = make_predictions(items)
    with pytest.raises(BenchmarkValidationError, match="missing 1 predictions"):
        evaluate(items, predictions[:-1], minimum_questions=2)


def test_accuracy_gap_failure_is_reported() -> None:
    items = make_items()
    predictions = make_predictions(items)
    predictions = [
        Prediction(
            item_id=prediction.item_id,
            language=prediction.language,
            answer=prediction.answer,
            accurate=False if prediction.language == "fr" else prediction.accurate,
            comet=prediction.comet,
        )
        for prediction in predictions
    ]
    report = evaluate(
        items,
        predictions,
        make_human_evaluations(items),
        semantic_scorer=lambda _first, _second: 0.95,
        minimum_questions=2,
    )

    assert report["performance_gap_from_english"]["fr"] == 1.0
    assert report["success_criteria"]["per_language_accuracy_within_5_percent_of_english"] is False
    assert report["success_criteria"]["no_language_degradation_above_10_percent"] is False
    assert report["passed"] is False
