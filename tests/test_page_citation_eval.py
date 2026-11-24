"""Offline evaluation gate for page-analysis citation extraction (#854).

Runs the labeled synthetic fixture through the analyzer and fails on any drop
in precision, recall, invalid-reference rejection, or provenance. No network,
no credentials, and the FastAPI app is never imported.
"""

from __future__ import annotations

from scripts.eval_page_citations import THRESHOLDS, benchmark, evaluate, load_cases


def test_fixture_covers_every_required_category() -> None:
    categories = {case["category"] for case in load_cases()}
    assert {"arabic", "english", "mixed", "ocr_noise", "invalid", "no_reference"} <= categories


def test_evaluation_meets_thresholds() -> None:
    report = evaluate(load_cases())
    assert report["failures"] == []
    for name, floor in THRESHOLDS.items():
        assert report[name] >= floor, name
    assert report["no_reference_false_positives"] == 0


def test_unverified_matches_are_not_counted_as_validated() -> None:
    report = evaluate(load_cases())
    # Bare number pairs and number-less collection names stay unverified.
    assert report["accepted_by_status"]["unverified"] >= 3


def test_benchmark_reports_zero_provider_cost() -> None:
    result = benchmark(runs=2)
    assert result["provider_calls"] == 0
    assert result["provider_cost_usd"] == 0.0
    assert [shape["blocks"] for shape in result["shapes"]] == [40, 400, 2000]
    assert all(shape["p95_ms"] >= shape["p50_ms"] >= 0 for shape in result["shapes"])
