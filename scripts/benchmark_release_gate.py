"""
scripts/benchmark_release_gate.py

AI-102: Enforce held-out benchmark and scholar review release gates.

Evaluates a benchmark report dict (or JSON file) against all required
release criteria. Returns a structured result with passed=True/False
and a list of failure reasons.

Usage (CLI):
    python scripts/benchmark_release_gate.py report.json
    echo $?   # 0 = passed, 1 = failed
"""

from __future__ import annotations

import json
import sys
from typing import Any

REQUIRED_THRESHOLDS: dict[str, dict[str, Any]] = {
    "composite_score": {"key": "metrics.composite_score", "min": 0.70, "type": "min"},
    "arabic_comprehension": {"key": "metrics.arabic_comprehension_vs_english_baseline", "min": 0.90, "type": "min"},
    "fluency_score": {"key": "metrics.fluency_score", "min": 4.2, "type": "min"},
    "quran_diacritical_accuracy": {"key": "metrics.quran_diacritical_accuracy", "min": 0.98, "type": "min"},
    "grammatical_error_rate": {"key": "metrics.grammatical_error_rate", "max": 0.05, "type": "max"},
    "islamic_terminology_accuracy": {"key": "metrics.islamic_terminology_accuracy", "min": 0.92, "type": "min"},
    "cultural_appropriateness": {"key": "metrics.cultural_appropriateness", "min": 0.88, "type": "min"},
}

REQUIRED_VERSIONED_FIELDS: list[str] = [
    "model_id",
    "model_version",
    "prompt_policy_version",
    "dataset_version",
    "tokenizer_id",
]

REQUIRED_ROLLBACK_KEYS: list[str] = [
    "composite_score_drop_threshold",
    "quran_accuracy_minimum",
    "max_new_sacred_text_mismatches",
]


def _get_nested(d: dict[str, Any], dotted_key: str) -> Any:
    keys = dotted_key.split(".")
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def evaluate_release_gate(report: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate a benchmark report against all release gate criteria.

    Returns:
        {
            "passed": bool,
            "failures": list[str],
            "rollback_recommendation": str | None,
        }
    """
    failures: list[str] = []

    # ---- 1. Held-out evaluation type ----
    eval_type = report.get("evaluation_type")
    if eval_type != "held_out":
        failures.append(
            f"evaluation_type must be 'held_out', got '{eval_type}'. "
            "Do not evaluate against training or validation splits."
        )

    # ---- 2. Scholar sign-off ----
    if not report.get("scholar_sign_off"):
        failures.append("scholar_sign_off is False or missing. A qualified scholar must sign off before release.")
    if not report.get("scholar_id"):
        failures.append("scholar_id is missing. The sign-off reviewer must be identified by ID.")

    # ---- 3. Metric thresholds ----
    for name, spec in REQUIRED_THRESHOLDS.items():
        value = _get_nested(report, spec["key"])
        if value is None:
            failures.append(f"Missing required metric: {spec['key']}")
            continue
        if spec["type"] == "min" and value < spec["min"]:
            failures.append(f"{name}: {value:.4f} is below required minimum {spec['min']}.")
        elif spec["type"] == "max" and value > spec["max"]:
            failures.append(f"{name}: {value:.4f} exceeds maximum allowed {spec['max']}.")

    # ---- 4. Confidence intervals ----
    ci = report.get("confidence_intervals") or {}
    if not ci:
        failures.append(
            "confidence_intervals is missing or empty. Report must include confidence intervals for primary metrics."
        )

    # ---- 5. Versioned artifact fields ----
    for field in REQUIRED_VERSIONED_FIELDS:
        if not report.get(field):
            failures.append(f"Required versioned field '{field}' is missing or null.")

    tool_versions = report.get("tool_versions") or {}
    if not tool_versions:
        failures.append("tool_versions is missing or empty. Record eval harness and tool versions.")

    environment = report.get("environment") or {}
    if not environment:
        failures.append("environment is missing or empty. Record Python version, runner, etc.")

    # ---- 6. Sacred-text mismatches ----
    mismatches = report.get("sacred_text_mismatches") or []
    for m in mismatches:
        if not m.get("adjudicated"):
            failures.append(
                f"Unadjudicated sacred-text mismatch for {m.get('verse_key', '?')}: "
                "every Quran/Hadith mismatch must be reviewed by a qualified scholar."
            )

    # ---- 7. High-rating disagreements ----
    disagreements = report.get("high_rating_disagreements") or []
    for d in disagreements:
        if not d.get("adjudicated"):
            failures.append(
                f"Unadjudicated high-rating disagreement for item '{d.get('item_id', '?')}' "
                f"(spread={d.get('spread')}). Items with spread > 2 require expert adjudication."
            )

    # ---- 8. Rollback criteria ----
    rollback = report.get("rollback_criteria") or {}
    if not rollback:
        failures.append("rollback_criteria is missing or empty. Define rollback thresholds before releasing.")
    else:
        for key in REQUIRED_ROLLBACK_KEYS:
            if key not in rollback:
                failures.append(f"rollback_criteria is missing required key: '{key}'.")

    # ---- Build result ----
    passed = len(failures) == 0
    rollback_rec = None
    if not passed:
        rollback_rec = (
            "Do not deploy. Address all failures above, obtain new scholar sign-off, "
            "and re-run the full held-out evaluation before retrying the release gate."
        )

    return {
        "passed": passed,
        "failures": failures,
        "rollback_recommendation": rollback_rec,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/benchmark_release_gate.py <report.json>")
        sys.exit(2)

    report_path = sys.argv[1]
    with open(report_path) as f:
        report = json.load(f)

    result = evaluate_release_gate(report)

    print(json.dumps(result, indent=2))

    if result["passed"]:
        print("\n[benchmark-release-gate] PASSED - release criteria met.")
        sys.exit(0)
    else:
        print(f"\n[benchmark-release-gate] FAILED - {len(result['failures'])} issue(s) found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
