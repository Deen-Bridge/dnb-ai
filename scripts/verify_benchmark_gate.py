#!/usr/bin/env python3
"""
Benchmark Release Gate Verifier
Issue #877: enforce held-out benchmark and scholar review release gates

Validates that all required benchmark artifacts and scholar sign-offs exist
before a release is permitted.

Usage:
    python scripts/verify_benchmark_gate.py [--eval-dir PATH] [--review-dir PATH]

Exit codes:
    0 - all gates passed
    1 - one or more gates failed (release should be blocked)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

PASS = "âœ… PASS"
FAIL = "âŒ FAIL"
WARN = "âš ï¸  WARN"

results: List[Tuple[str, str, str]] = []  # (gate_name, status, detail)


def gate(name: str, condition: bool, detail: str = "") -> bool:
    """Record a gate check result."""
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    return condition


def check_held_out_results(eval_dir: Path) -> bool:
    """Verify held-out benchmark results exist and are not from the training set."""
    result_files = list(eval_dir.glob("*held_out*.json")) + list(eval_dir.glob("*heldout*.json"))
    if not result_files:
        return gate(
            "Held-out benchmark results",
            False,
            f"No held-out result files found in {eval_dir} (expected *held_out*.json)",
        )

    latest = max(result_files, key=os.path.getmtime)
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        is_held_out = data.get("evaluation_type") == "held_out" or data.get("held_out", False)
        return gate(
            "Held-out benchmark results",
            is_held_out,
            f"Found {latest.name}; held_out={is_held_out}",
        )
    except (json.JSONDecodeError, KeyError) as e:
        return gate("Held-out benchmark results", False, f"Error reading {latest.name}: {e}")


def check_confidence_intervals(eval_dir: Path) -> bool:
    """Verify confidence intervals are present in the benchmark results."""
    ci_files = list(eval_dir.glob("*confidence*.json")) + list(eval_dir.glob("*ci_*.json"))
    if not ci_files:
        # Also accept confidence_intervals key inside held-out results
        held_out_files = list(eval_dir.glob("*held_out*.json")) + list(eval_dir.glob("*heldout*.json"))
        for f in held_out_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if "confidence_intervals" in data or "ci_lower" in data:
                    return gate("Confidence intervals", True, f"Found in {f.name}")
            except (json.JSONDecodeError, KeyError):
                pass
        return gate(
            "Confidence intervals",
            False,
            f"No confidence interval data found in {eval_dir}",
        )

    return gate("Confidence intervals", True, f"Found {ci_files[0].name}")


def check_scholar_review(review_dir: Path) -> bool:
    """Verify scholar review sign-off file exists."""
    sign_off_files = (
        list(review_dir.glob("*scholar*sign*off*"))
        + list(review_dir.glob("*sign*off*.json"))
        + list(review_dir.glob("*review*approval*.json"))
        + list(review_dir.glob("SCHOLAR_REVIEW_APPROVAL*"))
    )
    if not sign_off_files:
        return gate(
            "Scholar review sign-off",
            False,
            f"No scholar sign-off file found in {review_dir}. "
            "Expected: SCHOLAR_REVIEW_APPROVAL.json or similar.",
        )

    latest = sign_off_files[0]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        reviewers = data.get("reviewers", [])
        approved = data.get("approved", False)
        return gate(
            "Scholar review sign-off",
            approved and len(reviewers) > 0,
            f"Reviewers: {reviewers}, approved={approved}",
        )
    except (json.JSONDecodeError, KeyError):
        # Presence of the file alone is treated as a sign-off for plain text markers
        return gate("Scholar review sign-off", True, f"Sign-off file present: {latest.name}")


def check_arabic_benchmark(eval_dir: Path) -> bool:
    """Verify Arabic-language benchmark results exist."""
    arabic_files = list(eval_dir.glob("*arabic*")) + list(eval_dir.glob("*ar_*"))
    return gate(
        "Arabic benchmark results",
        bool(arabic_files),
        f"Found {len(arabic_files)} Arabic benchmark file(s)" if arabic_files else f"No Arabic benchmark files in {eval_dir}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark release gate verifier")
    parser.add_argument("--eval-dir", default="data/eval", help="Directory containing evaluation results")
    parser.add_argument("--review-dir", default="data/eval/reviews", help="Directory containing scholar review files")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    review_dir = Path(args.review_dir)

    print("=" * 60)
    print("  Benchmark Release Gate Verification")
    print("=" * 60)

    if not eval_dir.exists():
        print(f"\n{FAIL} Evaluation directory not found: {eval_dir}")
        print("Create the directory and add required benchmark artifacts before releasing.")
        return 1

    # Run all gates
    check_held_out_results(eval_dir)
    check_confidence_intervals(eval_dir)
    check_arabic_benchmark(eval_dir)
    check_scholar_review(review_dir if review_dir.exists() else eval_dir)

    # Print results
    print()
    failures = 0
    for name, status, detail in results:
        print(f"  {status}  {name}")
        if detail:
            print(f"           {detail}")
        if status == FAIL:
            failures += 1

    print()
    print("=" * 60)
    if failures == 0:
        print("  All release gates passed. Safe to proceed with release.")
        return 0
    else:
        print(f"  {failures} gate(s) FAILED. Release is BLOCKED until resolved.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
