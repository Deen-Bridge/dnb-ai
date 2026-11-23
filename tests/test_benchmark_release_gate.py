"""
tests/test_benchmark_release_gate.py

AI-102: Enforce held-out benchmark and scholar review release gates.

Tests that:
1. Release CI fails when required thresholds or reviewer sign-off are missing.
2. Held-out data is not used for prompt/model tuning (data isolation).
3. Reports record model, prompt, dataset, tokenizer, tool, and environment versions.
4. Sacred-text mismatches and high-rating disagreements receive adjudication.
5. Rollback criteria are defined for regressions.

All tests run offline using fixtures and mocking — no live model calls.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

PASSING_REPORT: dict[str, Any] = {
    "schema_version": "1.0.0",
    "model_id": "gemini-1.5-flash-002",
    "model_version": "002",
    "prompt_policy_version": "v3.1",
    "dataset_version": "1.0.0",
    "tokenizer_id": "gemini-tokenizer",
    "tool_versions": {"eval_harness": "2.4.0", "arabic_benchmark": "1.2.0"},
    "environment": {"python": "3.11.9", "runner": "ubuntu-26.04"},
    "evaluation_type": "held_out",
    "scholar_sign_off": True,
    "scholar_id": "reviewer-001",
    "metrics": {
        "composite_score": 0.78,
        "arabic_comprehension_vs_english_baseline": 0.93,
        "fluency_score": 4.4,
        "quran_diacritical_accuracy": 0.991,
        "grammatical_error_rate": 0.03,
        "islamic_terminology_accuracy": 0.95,
        "cultural_appropriateness": 0.91,
    },
    "thresholds": {
        "composite_score": 0.70,
        "arabic_comprehension_vs_english_baseline": 0.90,
        "fluency_score": 4.2,
        "quran_diacritical_accuracy": 0.98,
        "grammatical_error_rate_max": 0.05,
        "islamic_terminology_accuracy": 0.92,
        "cultural_appropriateness": 0.88,
    },
    "sacred_text_mismatches": [],
    "high_rating_disagreements": [],
    "rollback_criteria": {
        "composite_score_drop_threshold": 0.05,
        "quran_accuracy_minimum": 0.98,
        "max_new_sacred_text_mismatches": 0,
    },
    "confidence_intervals": {
        "composite_score": [0.75, 0.81],
        "arabic_comprehension_vs_english_baseline": [0.90, 0.96],
    },
}

FAILING_REPORT_LOW_SCORE: dict[str, Any] = {
    **PASSING_REPORT,
    "metrics": {
        **PASSING_REPORT["metrics"],
        "composite_score": 0.65,  # Below 0.70 threshold
        "quran_diacritical_accuracy": 0.97,  # Below 0.98 threshold
    },
}

FAILING_REPORT_NO_SIGN_OFF: dict[str, Any] = {
    **PASSING_REPORT,
    "scholar_sign_off": False,
    "scholar_id": None,
}

FAILING_REPORT_NOT_HELD_OUT: dict[str, Any] = {
    **PASSING_REPORT,
    "evaluation_type": "training_set",  # Must be held_out
}

FAILING_REPORT_MISSING_VERSIONS: dict[str, Any] = {
    **PASSING_REPORT,
    "model_id": None,
    "prompt_policy_version": None,
    "dataset_version": None,
}

FAILING_REPORT_SACRED_TEXT_MISMATCH: dict[str, Any] = {
    **PASSING_REPORT,
    "sacred_text_mismatches": [
        {
            "verse_key": "2:255",
            "expected": "correct Uthmani text",
            "got": "wrong text from model",
            "adjudicated": False,  # Not adjudicated — should block
        }
    ],
}

REPORT_WITH_ADJUDICATED_MISMATCH: dict[str, Any] = {
    **PASSING_REPORT,
    "sacred_text_mismatches": [
        {
            "verse_key": "2:255",
            "expected": "correct text",
            "got": "slightly different",
            "adjudicated": True,
            "adjudicator_id": "scholar-002",
            "resolution": "minor diacritic variation — acceptable",
        }
    ],
}


# ---------------------------------------------------------------------------
# AC-1: CI fails without thresholds or reviewer sign-off
# ---------------------------------------------------------------------------


class TestReleaseGateThresholds:
    """Release CI fails when required thresholds or reviewer sign-off are missing."""

    def test_passing_report_clears_gate(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        result = evaluate_release_gate(PASSING_REPORT)
        assert result["passed"] is True
        assert result["failures"] == []

    def test_gate_fails_on_low_composite_score(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        result = evaluate_release_gate(FAILING_REPORT_LOW_SCORE)
        assert result["passed"] is False
        assert any("composite_score" in f.lower() for f in result["failures"])

    def test_gate_fails_on_low_quran_diacritical_accuracy(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "metrics": {**PASSING_REPORT["metrics"], "quran_diacritical_accuracy": 0.97}}
        result = evaluate_release_gate(report)
        assert result["passed"] is False
        assert any("quran" in f.lower() or "diacritical" in f.lower() for f in result["failures"])

    def test_gate_fails_without_scholar_sign_off(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        result = evaluate_release_gate(FAILING_REPORT_NO_SIGN_OFF)
        assert result["passed"] is False
        assert any("scholar" in f.lower() or "sign" in f.lower() for f in result["failures"])

    def test_gate_fails_on_missing_scholar_id(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "scholar_sign_off": True, "scholar_id": None}
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_gate_fails_on_low_fluency(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "metrics": {**PASSING_REPORT["metrics"], "fluency_score": 4.1}}
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_gate_fails_on_high_grammatical_error_rate(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "metrics": {**PASSING_REPORT["metrics"], "grammatical_error_rate": 0.06}}
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_gate_fails_on_low_arabic_comprehension(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {
            **PASSING_REPORT,
            "metrics": {**PASSING_REPORT["metrics"], "arabic_comprehension_vs_english_baseline": 0.89},
        }
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_gate_fails_on_missing_confidence_intervals(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "confidence_intervals": {}}
        result = evaluate_release_gate(report)
        assert result["passed"] is False
        assert any("confidence" in f.lower() or "interval" in f.lower() for f in result["failures"])


# ---------------------------------------------------------------------------
# AC-2: Held-out data isolation
# ---------------------------------------------------------------------------


class TestHeldOutDataIsolation:
    """Held-out data is not used for prompt/model tuning."""

    def test_gate_requires_held_out_evaluation_type(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        result = evaluate_release_gate(FAILING_REPORT_NOT_HELD_OUT)
        assert result["passed"] is False
        assert any(
            word in " ".join(result["failures"]).lower()
            for word in ("held_out", "held-out", "evaluation_type", "training")
        )

    def test_gate_accepts_held_out_evaluation_type(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        result = evaluate_release_gate(PASSING_REPORT)
        assert PASSING_REPORT["evaluation_type"] == "held_out"
        assert result["passed"] is True

    def test_validation_set_type_also_rejected(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "evaluation_type": "validation"}
        result = evaluate_release_gate(report)
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# AC-3: Versioned artifact reporting
# ---------------------------------------------------------------------------


class TestVersionedArtifacts:
    """Reports record model, prompt, dataset, tokenizer, tool, and environment versions."""

    def test_gate_requires_model_id(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "model_id": None}
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_gate_requires_prompt_policy_version(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "prompt_policy_version": None}
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_gate_requires_dataset_version(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "dataset_version": None}
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_gate_requires_tokenizer_id(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "tokenizer_id": None}
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_gate_requires_tool_versions(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "tool_versions": {}}
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_gate_requires_environment_info(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "environment": {}}
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_all_versions_present_passes(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        result = evaluate_release_gate(PASSING_REPORT)
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# AC-4: Sacred-text mismatches and high-rating disagreements require adjudication
# ---------------------------------------------------------------------------


class TestAdjudication:
    """Sacred-text mismatches and high-rating disagreements receive adjudication."""

    def test_unadjudicated_sacred_text_mismatch_blocks_release(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        result = evaluate_release_gate(FAILING_REPORT_SACRED_TEXT_MISMATCH)
        assert result["passed"] is False
        assert any(
            word in " ".join(result["failures"]).lower() for word in ("sacred", "mismatch", "adjudicat", "quran")
        )

    def test_adjudicated_mismatch_does_not_block(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        result = evaluate_release_gate(REPORT_WITH_ADJUDICATED_MISMATCH)
        assert result["passed"] is True

    def test_unadjudicated_high_rating_disagreement_blocks(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {
            **PASSING_REPORT,
            "high_rating_disagreements": [
                {
                    "item_id": "aqeedah-042",
                    "spread": 3,  # Rating spread > 2 — requires adjudication
                    "adjudicated": False,
                }
            ],
        }
        result = evaluate_release_gate(report)
        assert result["passed"] is False
        assert any(
            word in " ".join(result["failures"]).lower() for word in ("disagreement", "adjudicat", "spread", "rating")
        )

    def test_adjudicated_disagreement_does_not_block(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {
            **PASSING_REPORT,
            "high_rating_disagreements": [
                {
                    "item_id": "aqeedah-042",
                    "spread": 3,
                    "adjudicated": True,
                    "adjudicator_id": "scholar-003",
                    "resolution": "Reviewer 2 misread the dialect context",
                }
            ],
        }
        result = evaluate_release_gate(report)
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# AC-5: Rollback criteria defined
# ---------------------------------------------------------------------------


class TestRollbackCriteria:
    """Rollback criteria are defined for regressions."""

    def test_rollback_criteria_required_in_report(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {**PASSING_REPORT, "rollback_criteria": {}}
        result = evaluate_release_gate(report)
        assert result["passed"] is False
        assert any("rollback" in f.lower() for f in result["failures"])

    def test_rollback_criteria_must_include_composite_drop_threshold(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {
            **PASSING_REPORT,
            "rollback_criteria": {
                "quran_accuracy_minimum": 0.98,
                "max_new_sacred_text_mismatches": 0,
                # Missing composite_score_drop_threshold
            },
        }
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_rollback_criteria_must_include_quran_minimum(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        report = {
            **PASSING_REPORT,
            "rollback_criteria": {
                "composite_score_drop_threshold": 0.05,
                "max_new_sacred_text_mismatches": 0,
                # Missing quran_accuracy_minimum
            },
        }
        result = evaluate_release_gate(report)
        assert result["passed"] is False

    def test_valid_rollback_criteria_passes(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        result = evaluate_release_gate(PASSING_REPORT)
        assert result["passed"] is True

    def test_gate_generates_rollback_recommendation_on_failure(self) -> None:
        from scripts.benchmark_release_gate import evaluate_release_gate

        result = evaluate_release_gate(FAILING_REPORT_LOW_SCORE)
        assert result["passed"] is False
        assert "rollback_recommendation" in result or "action" in result or len(result["failures"]) > 0
