#!/usr/bin/env python3
"""Export down-rated feedback records as evaluation-dataset candidates.

Targets the evaluation-harness dataset format (issue #16). Each emitted entry
carries ``needs_review: true`` — a human curator MUST supply an
``expected_answer`` before any record enters the golden set.

This script intentionally NEVER generates expected answers for religious
content. That decision belongs to qualified scholars and the maintainers of
the evaluation harness. ``answer_draft`` is included only so a reviewer can
assess the failure; it is never treated as ground truth.

Backend: the export reads from whichever feedback store the service is
configured to use — Redis when ``REDIS_URL`` is set, SQLite otherwise — so a
Redis-backed deployment exports from the live store, not a stale local
``feedback.db``. Pass ``--db`` to force a specific SQLite file.

Usage
-----
    python scripts/export_eval_candidates.py [options]

    --output PATH      Write JSONL here (default: stdout)
    --db PATH          Force this SQLite DB, ignoring REDIS_URL
    --min-categories N Only include records with at least N categories tagged
    --limit N          Max records to read (default: 2000)

Output (one JSON object per line):
    {"question", "category", "categories", "needs_review", "source",
     "feedback_id", "model_name", "answer_draft", "comment"}

Near-duplicate prompts (same first 120 chars after normalization) are
deduplicated: the first occurrence wins.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feedback import (  # noqa: E402
    FeedbackRecord,
    FeedbackStore,
    SQLiteFeedbackStore,
    build_store,
)

_WS = re.compile(r"\s+")

_TAXONOMY_TO_HARNESS_CATEGORY: dict[str, str] = {
    "incorrect_information": "factual_accuracy",
    "wrong_or_missing_citation": "citation_quality",
    "one_sided_fiqh_answer": "fiqh_balance",
    "too_vague": "answer_completeness",
    "too_long": "answer_conciseness",
    "wrong_language": "language",
    "poor_adab": "adab",
    "refused_unnecessarily": "refusal",
    "other": "other",
}


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace for near-duplicate detection."""
    return _WS.sub(" ", text.lower().strip())


def _primary_category(categories: list[str]) -> str:
    """First recognized category, mapped to the harness label, else 'other'."""
    for cat in categories:
        if cat in _TAXONOMY_TO_HARNESS_CATEGORY:
            return _TAXONOMY_TO_HARNESS_CATEGORY[cat]
    return "other"


def to_candidate(record: FeedbackRecord) -> dict[str, Any] | None:
    """Convert a FeedbackRecord to an eval-harness candidate, or None.

    Returns None when there is no prompt snapshot — without the question there
    is no useful candidate to review.
    """
    if not record.prompt:
        return None
    return {
        "question": record.prompt,
        "category": _primary_category(record.categories),
        "categories": record.categories,
        "needs_review": True,
        "source": "user_feedback",
        "feedback_id": record.feedback_id,
        "model_name": record.model_name or "unknown",
        "answer_draft": record.answer or "",
        "comment": record.comment or "",
    }


def build_candidates(records: list[FeedbackRecord], min_categories: int = 0) -> list[dict[str, Any]]:
    """Deduplicated candidates from *records*, first occurrence winning."""
    seen_prompts: set = set()
    candidates: list[dict[str, Any]] = []
    for record in records:
        if min_categories and len(record.categories) < min_categories:
            continue
        candidate = to_candidate(record)
        if candidate is None:
            continue
        norm = _normalise(record.prompt or "")[:120]
        if norm in seen_prompts:
            continue
        seen_prompts.add(norm)
        candidates.append(candidate)
    return candidates


def _select_store(db_path: str | None) -> FeedbackStore:
    """The SQLite file when --db is given, otherwise the configured backend."""
    if db_path:
        return SQLiteFeedbackStore(db_path=db_path)
    return build_store()


def export(
    output_path: str | None,
    min_categories: int = 0,
    limit: int = 2000,
    db_path: str | None = None,
) -> int:
    """Run the export; return the number of candidates written."""
    store = _select_store(db_path)
    records = store.list_records(rating="down", limit=limit)
    candidates = build_candidates(records, min_categories=min_categories)

    out = open(output_path, "w", encoding="utf-8") if output_path else sys.stdout
    try:
        for candidate in candidates:
            out.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    finally:
        if output_path:
            out.close()
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export down-rated feedback as evaluation-dataset candidates.")
    parser.add_argument("--output", metavar="PATH", help="Output JSONL file (default: stdout)")
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="Force this SQLite DB path, ignoring REDIS_URL (default: configured backend)",
    )
    parser.add_argument(
        "--min-categories",
        metavar="N",
        type=int,
        default=0,
        help="Only include records with at least N failure categories (default: 0)",
    )
    parser.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=2000,
        help="Max feedback records to read (default: 2000)",
    )
    args = parser.parse_args()

    # A negative --limit reaches SQLite as "no limit" and loads the whole table
    # before dedup; a negative --min-categories is meaningless. Reject both.
    if args.limit < 0:
        parser.error("--limit must be a non-negative integer")
    if args.min_categories < 0:
        parser.error("--min-categories must be a non-negative integer")

    count = export(
        output_path=args.output,
        min_categories=args.min_categories,
        limit=args.limit,
        db_path=args.db,
    )
    print(f"Exported {count} candidates.", file=sys.stderr)


if __name__ == "__main__":
    main()
