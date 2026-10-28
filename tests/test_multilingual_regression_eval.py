"""Tests for deterministic multilingual Islamic and Stellar USDC regression evaluations (#529)."""

from __future__ import annotations

from multilingual_regression_eval import REVIEWED_EVAL_CASES, evaluate_multilingual_response


def test_reviewed_cases_cover_arabic_swahili_hausa_and_nigerian_english() -> None:
    langs = {c.language for c in REVIEWED_EVAL_CASES}
    assert {"ar", "sw", "ha", "en-NG"}.issubset(langs)


def test_evaluator_detects_missing_claim_and_disallowed_failure_modes() -> None:
    case = REVIEWED_EVAL_CASES[0]
    good = evaluate_multilingual_response(case, "آية الكرسي هي الآية 2:255 من سورة البقرة.")
    assert good.passed is True

    bad = evaluate_multilingual_response(case, "unknown reference with fabricated_citation")
    assert bad.passed is False
    assert len(bad.failures) == 2
