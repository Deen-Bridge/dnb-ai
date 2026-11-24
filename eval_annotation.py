"""Annotation workflow for failed evaluations.

Every evaluation case that fails must carry a human annotation saying *why* it
failed and *what* will be done about it. Annotations live in a versioned JSONL
file next to the eval data and reference cases by ``eval_id`` only — they never
copy the question, the model answer or any user text.

The CI gate scores a small sanitized fixture of cases through the offline
scorer (``RegressionRunner`` + ``evaluate_single_response``), then checks the
annotation file against a versioned baseline:

* every failing case has exactly one valid annotation (coverage threshold);
* no annotation points at a case that now passes (stale) or does not exist;
* triage rate and open-critical limits from the baseline hold;
* a case marked ``resolved`` must actually pass.

No network, API keys or provider responses are needed.

Usage::

    python eval_annotation.py check
    python eval_annotation.py queue --results run.json [--annotations a.jsonl] [--output stubs.jsonl]

``check`` exits 1 with one ``FAIL:`` line per problem. ``queue`` turns the
failed items of an eval run (``RegressionRunner.run_suite`` or
``scripts/eval_islamic_qa.py`` output) into ``open`` annotation stubs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ANNOTATION_DIR = ROOT / "data" / "eval" / "annotations"
DEFAULT_CASES = ANNOTATION_DIR / "cases.jsonl"
DEFAULT_ANNOTATIONS = ANNOTATION_DIR / "annotations.jsonl"
DEFAULT_BASELINE = ANNOTATION_DIR / "baseline.json"

# What went wrong. The first block mirrors the offline scorer's signals; the
# rest lines up with the user-feedback taxonomy in feedback.py so annotated
# failures and down-rated chats can be compared.
FAILURE_CATEGORIES = frozenset(
    {
        "missing_key_point",
        "prohibited_content",
        "missed_abstention",
        "unnecessary_abstention",
        "incorrect_information",
        "wrong_or_missing_citation",
        "one_sided_fiqh_answer",
        "wrong_language",
        "poor_adab",
        "benchmark_defect",
        "scorer_defect",
        "other",
    }
)

# Deciding religious ground truth is a scholar's call, never an engineer's
# (same rule as review.py and scripts/export_eval_candidates.py).
SCHOLAR_REVIEW_CATEGORIES = frozenset({"benchmark_defect", "incorrect_information", "one_sided_fiqh_answer"})

SEVERITIES = ("critical", "major", "minor")
STATUSES = ("open", "triaged", "resolved", "wont_fix")
ACTIONS = frozenset(
    {
        "prompt_fix",
        "retrieval_fix",
        "safety_policy_fix",
        "model_change",
        "dataset_fix",
        "scorer_fix",
        "accept_risk",
    }
)

REQUIRED_FIELDS = frozenset({"eval_id", "failure_category", "severity", "status", "annotator", "annotated_at"})
OPTIONAL_FIELDS = frozenset({"rationale", "action", "requires_scholar_review"})
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS

RATIONALE_MAX_CHARS = 500
# A role handle such as "eval-maintainer" — not a person's name or email.
_ANNOTATOR_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,39}$")
_EVAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9]{8,}|AIza[0-9A-Za-z_-]{20,}|api[_-]?key|bearer\s+\S+|password)", re.I)


class AnnotationInputError(ValueError):
    """A fixture, annotation or baseline file is missing or malformed."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise AnnotationInputError(f"{path.name}: file not found")
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationInputError(f"{path.name}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(record, dict):
                raise AnnotationInputError(f"{path.name}:{lineno}: expected a JSON object")
            records.append(record)
    return records


def _read_json(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise AnnotationInputError(f"{path.name}: file not found")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise AnnotationInputError(f"{path.name}: invalid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise AnnotationInputError(f"{path.name}: expected a JSON object")
    return data


def load_cases(path: Path | str = DEFAULT_CASES) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def load_annotations(path: Path | str = DEFAULT_ANNOTATIONS) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def load_baseline(path: Path | str = DEFAULT_BASELINE) -> dict[str, Any]:
    baseline = _read_json(path)
    missing = [k for k in ("version", "owner", "rollback") if not baseline.get(k)]
    if missing:
        raise AnnotationInputError(f"baseline missing {', '.join(missing)}")
    return baseline


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _contains_sensitive_text(text: str) -> bool:
    return bool(_EMAIL_RE.search(text) or _SECRET_RE.search(text))


def validate_annotation(record: dict[str, Any]) -> list[str]:
    """Return every contract violation in *record* (empty when valid)."""
    errors: list[str] = []

    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        errors.append(f"missing field(s): {', '.join(missing)}")
    unknown = sorted(record.keys() - ALLOWED_FIELDS)
    if unknown:
        # Rejecting unknown keys keeps answers/prompts from being pasted in.
        errors.append(f"unknown field(s): {', '.join(unknown)}")

    eval_id = record.get("eval_id")
    if "eval_id" in record and not (isinstance(eval_id, str) and _EVAL_ID_RE.match(eval_id)):
        errors.append("eval_id must be a short identifier")

    category = record.get("failure_category")
    if "failure_category" in record and category not in FAILURE_CATEGORIES:
        errors.append(f"failure_category {category!r} is not in the taxonomy")

    severity = record.get("severity")
    if "severity" in record and severity not in SEVERITIES:
        errors.append(f"severity must be one of {', '.join(SEVERITIES)}")

    status = record.get("status")
    if "status" in record and status not in STATUSES:
        errors.append(f"status must be one of {', '.join(STATUSES)}")

    annotator = record.get("annotator")
    if "annotator" in record and not (isinstance(annotator, str) and _ANNOTATOR_RE.match(annotator)):
        errors.append("annotator must be a lowercase role handle (no names or emails)")

    annotated_at = record.get("annotated_at")
    if "annotated_at" in record:
        try:
            date.fromisoformat(str(annotated_at))
        except ValueError:
            errors.append("annotated_at must be an ISO date (YYYY-MM-DD)")

    rationale = record.get("rationale", "")
    if not isinstance(rationale, str):
        errors.append("rationale must be a string")
        rationale = ""
    elif len(rationale) > RATIONALE_MAX_CHARS:
        errors.append(f"rationale exceeds {RATIONALE_MAX_CHARS} characters")
    elif _contains_sensitive_text(rationale):
        errors.append("rationale looks like it contains an email address or secret")

    action = record.get("action")
    if action is not None and action not in ACTIONS:
        errors.append(f"action {action!r} is not a known remediation")

    scholar_review = record.get("requires_scholar_review", False)
    if not isinstance(scholar_review, bool):
        errors.append("requires_scholar_review must be true or false")

    # Cross-field rules only apply to a triaged annotation.
    if status in STATUSES and status != "open":
        if not rationale.strip():
            errors.append(f"status {status!r} requires a rationale")
        if action is None:
            errors.append(f"status {status!r} requires an action")
    if status == "wont_fix" and action not in (None, "accept_risk"):
        errors.append("status 'wont_fix' requires action 'accept_risk'")
    if action == "accept_risk" and severity == "critical":
        errors.append("a critical failure cannot be closed as accept_risk")
    if category in SCHOLAR_REVIEW_CATEGORIES and scholar_review is not True:
        errors.append(f"failure_category {category!r} requires requires_scholar_review: true")
    if category == "benchmark_defect" and action not in (None, "dataset_fix"):
        errors.append("failure_category 'benchmark_defect' requires action 'dataset_fix'")

    return errors


# ---------------------------------------------------------------------------
# Runs and the annotation queue
# ---------------------------------------------------------------------------


def run_items(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-item rows from a RegressionRunner summary or an eval_islamic_qa report."""
    items = results.get("details", results.get("results"))
    if not isinstance(items, list):
        raise AnnotationInputError("run results have no 'details' or 'results' list")
    return [item for item in items if isinstance(item, dict) and "id" in item]


def failed_items(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in run_items(results) if not item.get("passed", False)]


def suggest_category(item: dict[str, Any]) -> str:
    """Best first guess from the scorer's diagnostics; a human confirms it."""
    if item.get("abstention_correct") is False:
        return "missed_abstention"
    if item.get("must_not_include_pass") is False:
        return "prohibited_content"
    recall = item.get("key_point_recall")
    precision = item.get("must_include_precision")
    if (isinstance(recall, int | float) and recall < 1) or (isinstance(precision, int | float) and precision < 1):
        return "missing_key_point"
    return "other"


def build_queue(
    results: dict[str, Any],
    existing: list[dict[str, Any]] | None = None,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Open annotation stubs for failed items that are not annotated yet.

    Severity starts at ``major`` so nothing untriaged looks harmless.
    """
    annotated = {a.get("eval_id") for a in existing or []}
    stamp = (today or date.today()).isoformat()
    stubs = []
    for item in sorted(failed_items(results), key=lambda i: str(i["id"])):
        if item["id"] in annotated:
            continue
        category = suggest_category(item)
        stubs.append(
            {
                "eval_id": item["id"],
                "failure_category": category,
                "severity": "major",
                "status": "open",
                "annotator": "unassigned",
                "annotated_at": stamp,
                "rationale": "",
                "requires_scholar_review": category in SCHOLAR_REVIEW_CATEGORIES,
            }
        )
    return stubs


def score_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Score fixture cases with their canned ``candidate_answer`` (offline)."""
    from regression import RegressionRunner

    answers: dict[str, str] = {}
    for case in cases:
        question = case.get("question")
        if not isinstance(question, str) or not isinstance(case.get("candidate_answer"), str):
            raise AnnotationInputError(f"case {case.get('id')!r} needs a question and candidate_answer")
        if question in answers:
            raise AnnotationInputError(f"case {case.get('id')!r} repeats another case's question")
        answers[question] = case["candidate_answer"]

    class _FixtureRunner(RegressionRunner):
        def load_dataset(self) -> list[dict[str, Any]]:
            return cases

    return _FixtureRunner().run_suite(lambda question: answers[question])


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def evaluate_annotations(
    results: dict[str, Any], annotations: list[dict[str, Any]], baseline: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Check *annotations* against a scored run. Returns (summary, messages)."""
    messages: list[str] = []
    items = {str(item["id"]): item for item in run_items(results)}
    failed_ids = {i for i, item in items.items() if not item.get("passed", False)}

    by_id: dict[str, dict[str, Any]] = {}
    for index, annotation in enumerate(annotations, start=1):
        label = str(annotation.get("eval_id") or f"annotation #{index}")
        for error in validate_annotation(annotation):
            messages.append(f"{label}: {error}")
        if label in by_id:
            messages.append(f"{label}: duplicate annotation (keep one per eval case)")
        by_id.setdefault(label, annotation)

    for eval_id, annotation in sorted(by_id.items()):
        if eval_id not in items:
            if not eval_id.startswith("annotation #"):
                messages.append(f"{eval_id}: annotation refers to an unknown eval case")
        elif eval_id not in failed_ids:
            if annotation.get("status") == "resolved":
                continue  # fixed and kept for the record
            messages.append(f"{eval_id}: case now passes; mark the annotation 'resolved' or remove it")
        elif annotation.get("status") == "resolved":
            messages.append(f"{eval_id}: marked resolved but still failing; reopen it or land the fix")

    unannotated = sorted(failed_ids - by_id.keys())
    for eval_id in unannotated:
        item = items[eval_id]
        messages.append(
            f"{eval_id}: failed (score {item.get('score', item.get('composite_score'))}) with no annotation; "
            f"run `python eval_annotation.py queue` and triage it"
        )

    covering = [by_id[i] for i in sorted(failed_ids) if i in by_id]
    failed_total = len(failed_ids)
    coverage = len(covering) / failed_total if failed_total else 1.0
    triaged = [a for a in covering if a.get("status") != "open"]
    triaged_rate = len(triaged) / failed_total if failed_total else 1.0
    open_critical = sorted(
        str(a["eval_id"]) for a in covering if a.get("status") == "open" and a.get("severity") == "critical"
    )

    min_failed = int(baseline.get("min_failed_cases", 0))
    if failed_total < min_failed:
        messages.append(
            f"fixture produced {failed_total} failed cases, baseline requires {min_failed}; "
            "the scorer or fixture changed — review before updating the baseline"
        )
    min_coverage = float(baseline.get("min_annotation_coverage", 1.0))
    if coverage < min_coverage:
        messages.append(f"annotation coverage {coverage:.4f} below threshold {min_coverage}")
    min_triaged = float(baseline.get("min_triaged_rate", 0.0))
    if triaged_rate < min_triaged:
        messages.append(f"triaged rate {triaged_rate:.4f} below threshold {min_triaged}")
    max_open_critical = int(baseline.get("max_open_critical", 0))
    if len(open_critical) > max_open_critical:
        messages.append(
            f"{len(open_critical)} critical failure(s) still open ({', '.join(open_critical)}); "
            f"limit is {max_open_critical}"
        )

    summary = {
        "baseline_version": baseline.get("version"),
        "evaluated": len(items),
        "failed": failed_total,
        "annotated": len(covering),
        "coverage": round(coverage, 4),
        "triaged_rate": round(triaged_rate, 4),
        "open_critical": len(open_critical),
        "by_category": dict(sorted(Counter(str(a.get("failure_category")) for a in covering).items())),
        "by_status": dict(sorted(Counter(str(a.get("status")) for a in covering).items())),
    }
    return summary, messages


def check(
    cases: Path | str = DEFAULT_CASES,
    annotations: Path | str = DEFAULT_ANNOTATIONS,
    baseline: Path | str = DEFAULT_BASELINE,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Return (ok, summary, actionable_messages). Bad input is reported, not raised."""
    try:
        base = load_baseline(baseline)
        results = score_cases(load_cases(cases))
        summary, messages = evaluate_annotations(results, load_annotations(annotations), base)
    except AnnotationInputError as exc:
        return False, {}, [str(exc)]
    return (not messages), summary, messages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    ok, summary, messages = check(args.cases, args.annotations, args.baseline)
    print(json.dumps(summary, sort_keys=True))
    for message in messages:
        print(f"FAIL: {message}", file=sys.stderr)
    return 0 if ok else 1


def _cmd_queue(args: argparse.Namespace) -> int:
    try:
        results = _read_json(args.results)
        existing = load_annotations(args.annotations) if args.annotations else []
        stubs = build_queue(results, existing)
    except AnnotationInputError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    lines = "".join(json.dumps(stub, ensure_ascii=False) + "\n" for stub in stubs)
    if args.output:
        Path(args.output).write_text(lines, encoding="utf-8")
    else:
        sys.stdout.write(lines)
    print(f"{len(stubs)} failed case(s) need annotation.", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Annotation workflow for failed evaluations.")
    sub = parser.add_subparsers(dest="command")

    check_parser = sub.add_parser("check", help="CI gate: every failed fixture case is annotated")
    check_parser.add_argument("--cases", default=str(DEFAULT_CASES))
    check_parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    check_parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    check_parser.set_defaults(func=_cmd_check)

    queue_parser = sub.add_parser("queue", help="emit open annotation stubs for a run's failures")
    queue_parser.add_argument("--results", required=True, help="run summary JSON")
    queue_parser.add_argument("--annotations", help="existing annotations to skip")
    queue_parser.add_argument("--output", help="write JSONL here (default: stdout)")
    queue_parser.set_defaults(func=_cmd_queue)

    argv = sys.argv[1:] if argv is None else argv
    # A bare `python eval_annotation.py` runs the gate, like incident_replay.py.
    args = parser.parse_args(argv or ["check"])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
