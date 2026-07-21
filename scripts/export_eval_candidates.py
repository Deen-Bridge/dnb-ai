#!/usr/bin/env python3
"""Export down-rated feedback records as evaluation-dataset candidates.

Targets the dataset format defined by issue #16 (evaluation harness).
Each emitted entry carries ``needs_review: true`` — a human curator MUST
supply ``expected_answer`` before any record enters the golden set.

This script intentionally NEVER generates expected answers for religious
content.  That decision belongs to qualified scholars and the maintainers
of the evaluation harness.

Usage
-----
    python scripts/export_eval_candidates.py [options]

    --output PATH      Write JSONL to this file  (default: stdout)
    --db PATH          SQLite DB path             (default: feedback.db)
    --min-categories N Only include records with at least N categories tagged
                       (default: 0 — include all down-rated)
    --limit N          Max records to read        (default: 2000)

Output format (one JSON object per line)
-----------------------------------------
{
  "question":      "<user prompt>",
  "category":      "<primary feedback taxonomy label | 'other'>",
  "categories":    ["<all taxonomy labels tagged by user>"],
  "needs_review":  true,
  "source":        "user_feedback",
  "feedback_id":   "<uuid>",
  "model_name":    "<model that produced the answer>",
  "answer_draft":  "<the model answer that was flagged>",
  "comment":       "<optional user comment>"
}

``answer_draft`` is included so a human reviewer can quickly assess the
failure — it is NOT treated as a ground-truth expected answer.

Near-duplicate prompts (same first 120 chars after normalisation) are
deduplicated: the first occurrence wins.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feedback import SQLiteFeedbackStore, FeedbackRecord  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")

_TAXONOMY_TO_HARNESS_CATEGORY: Dict[str, str] = {
    "incorrect_information":      "factual_accuracy",
    "wrong_or_missing_citation":  "citation_quality",
    "one_sided_fiqh_answer":      "fiqh_balance",
    "too_vague":                  "answer_completeness",
    "too_long":                   "answer_conciseness",
    "wrong_language":             "language",
    "poor_adab":                  "adab",
    "refused_unnecessarily":      "refusal",
    "other":                      "other",
}


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace for near-duplicate detection."""
    return _WS.sub(" ", text.lower().strip())


def _primary_category(categories: List[str]) -> str:
    """Return the first recognised category, falling back to 'other'."""
    for cat in categories:
        if cat in _TAXONOMY_TO_HARNESS_CATEGORY:
            return _TAXONOMY_TO_HARNESS_CATEGORY[cat]
    return "other"


def _to_candidate(record: FeedbackRecord) -> Optional[Dict[str, Any]]:
    """Convert a FeedbackRecord to an eval-harness candidate dict."""
    if not record.prompt:
        return None  # No prompt snapshot — cannot create a useful candidate
    return {
        "question":    record.prompt,
        "category":    _primary_category(record.categories),
        "categories":  record.categories,
        "needs_review": True,
        "source":      "user_feedback",
        "feedback_id": record.feedback_id,
        "model_name":  record.model_name or "unknown",
        "answer_draft": record.answer or "",
        "comment":     record.comment or "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export(
    db_path: str,
    output_path: Optional[str],
    min_categories: int,
    limit: int,
) -> int:
    """Run the export; return the number of candidates written."""
    store = SQLiteFeedbackStore(db_path=db_path)
    records = store.list_records(rating="down", limit=limit)

    seen_prompts: Dict[str, str] = {}  # normalised prefix → feedback_id
    candidates: List[Dict[str, Any]] = []

    for record in records:
        if min_categories and len(record.categories) < min_categories:
            continue

        candidate = _to_candidate(record)
        if candidate is None:
            continue

        norm = _normalise(record.prompt or "")[:120]
        if norm in seen_prompts:
            # Near-duplicate — skip; first occurrence already queued
            continue
        seen_prompts[norm] = record.feedback_id
        candidates.append(candidate)

    out = open(output_path, "w", encoding="utf-8") if output_path else sys.stdout
    try:
        for c in candidates:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
    finally:
        if output_path:
            out.close()

    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export down-rated feedback as evaluation-dataset candidates."
    )
    parser.add_argument("--output", metavar="PATH", help="Output JSONL file (default: stdout)")
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=os.getenv("FEEDBACK_DB_PATH", "feedback.db"),
        help="SQLite DB path (default: feedback.db or $FEEDBACK_DB_PATH)",
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

    count = export(
        db_path=args.db,
        output_path=args.output,
        min_categories=args.min_categories,
        limit=args.limit,
    )
    print(f"Exported {count} candidates.", file=sys.stderr)


if __name__ == "__main__":
    main()
