from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import eval_annotation as ea
from regression import RegressionRunner


def _annotation(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "eval_id": "case-1",
        "failure_category": "missing_key_point",
        "severity": "major",
        "status": "triaged",
        "annotator": "eval-maintainer",
        "annotated_at": "2026-11-24",
        "rationale": "Omits the nisab condition.",
        "action": "prompt_fix",
        "requires_scholar_review": False,
    }
    record.update(overrides)
    return record


def _run(*items: tuple[str, bool]) -> dict[str, Any]:
    return {"details": [{"id": i, "domain": "d", "score": 1.0 if ok else 0.0, "passed": ok} for i, ok in items]}


LENIENT_BASELINE = {"version": 1, "owner": "o", "rollback": "r", "min_annotation_coverage": 1.0}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Versioned fixture and baseline
# ---------------------------------------------------------------------------


class TestFixtureAndBaseline:
    def test_cases_are_sanitized_and_unique(self):
        cases = ea.load_cases()
        assert len({c["id"] for c in cases}) == len(cases)
        assert len({c["question"] for c in cases}) == len(cases)
        for case in cases:
            text = json.dumps(case).lower()
            assert "@" not in text and "api_key" not in text and "sk-" not in text

    def test_annotations_are_valid(self):
        for annotation in ea.load_annotations():
            assert ea.validate_annotation(annotation) == [], annotation["eval_id"]

    def test_baseline_is_owned_versioned_and_rollbackable(self):
        base = ea.load_baseline()
        assert base["owner"] and base["rollback"] and base["version"] >= 1
        assert 0 < base["min_annotation_coverage"] <= 1
        assert 0 <= base["min_triaged_rate"] <= 1
        assert base["min_failed_cases"] >= 1

    def test_baseline_without_owner_is_rejected(self, tmp_path):
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"version": 1, "rollback": "r"}), encoding="utf-8")
        with pytest.raises(ea.AnnotationInputError, match="owner"):
            ea.load_baseline(path)


# ---------------------------------------------------------------------------
# Annotation contract
# ---------------------------------------------------------------------------


class TestValidateAnnotation:
    def test_valid_annotation(self):
        assert ea.validate_annotation(_annotation()) == []

    def test_open_stub_needs_no_rationale_or_action(self):
        stub = _annotation(status="open", rationale="", annotator="unassigned")
        del stub["action"]
        assert ea.validate_annotation(stub) == []

    def test_missing_and_unknown_fields(self):
        record = _annotation(answer="model text pasted here")
        del record["severity"]
        errors = ea.validate_annotation(record)
        assert any("missing field(s): severity" in e for e in errors)
        assert any("unknown field(s): answer" in e for e in errors)

    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"failure_category": "vibes"}, "taxonomy"),
            ({"severity": "huge"}, "severity"),
            ({"status": "done"}, "status"),
            ({"action": "rewrite_everything"}, "remediation"),
            ({"annotator": "Jane Doe"}, "role handle"),
            ({"annotator": "someone@example.com"}, "role handle"),
            ({"annotated_at": "yesterday"}, "ISO date"),
            ({"eval_id": ""}, "eval_id"),
            ({"requires_scholar_review": "yes"}, "true or false"),
            ({"rationale": "x" * (ea.RATIONALE_MAX_CHARS + 1)}, "exceeds"),
        ],
    )
    def test_field_rules(self, overrides, fragment):
        assert any(fragment in e for e in ea.validate_annotation(_annotation(**overrides)))

    @pytest.mark.parametrize(
        "rationale",
        ["contact reviewer at someone@example.com", "used key sk-abcdefghijklmnop", "set API_KEY first"],
    )
    def test_rationale_rejects_personal_data_and_secrets(self, rationale):
        errors = ea.validate_annotation(_annotation(rationale=rationale))
        assert any("email address or secret" in e for e in errors)

    def test_triaged_requires_rationale_and_action(self):
        record = _annotation(rationale="  ")
        del record["action"]
        errors = ea.validate_annotation(record)
        assert "status 'triaged' requires a rationale" in errors
        assert "status 'triaged' requires an action" in errors

    def test_wont_fix_must_accept_risk(self):
        errors = ea.validate_annotation(_annotation(status="wont_fix", action="prompt_fix"))
        assert "status 'wont_fix' requires action 'accept_risk'" in errors

    def test_critical_cannot_accept_risk(self):
        errors = ea.validate_annotation(_annotation(severity="critical", status="wont_fix", action="accept_risk"))
        assert "a critical failure cannot be closed as accept_risk" in errors

    @pytest.mark.parametrize("category", sorted(ea.SCHOLAR_REVIEW_CATEGORIES))
    def test_religious_ground_truth_needs_scholar_review(self, category):
        action = "dataset_fix" if category == "benchmark_defect" else "prompt_fix"
        errors = ea.validate_annotation(_annotation(failure_category=category, action=action))
        assert any("requires requires_scholar_review: true" in e for e in errors)
        ok = _annotation(failure_category=category, action=action, requires_scholar_review=True)
        assert ea.validate_annotation(ok) == []

    def test_benchmark_defect_is_fixed_in_the_dataset(self):
        record = _annotation(failure_category="benchmark_defect", requires_scholar_review=True, action="prompt_fix")
        assert "failure_category 'benchmark_defect' requires action 'dataset_fix'" in ea.validate_annotation(record)


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class TestQueue:
    @pytest.mark.parametrize(
        ("diagnostics", "expected"),
        [
            ({"abstention_correct": False}, "missed_abstention"),
            ({"must_not_include_pass": False, "key_point_recall": 0.0}, "prohibited_content"),
            ({"key_point_recall": 0.5, "must_include_precision": 1.0}, "missing_key_point"),
            ({"must_include_precision": 0.0}, "missing_key_point"),
            ({}, "other"),
        ],
    )
    def test_suggest_category(self, diagnostics, expected):
        assert ea.suggest_category({"id": "x", **diagnostics}) == expected

    def test_build_queue_skips_passing_and_annotated(self):
        run = _run(("b", False), ("a", False), ("c", True), ("d", False))
        stubs = ea.build_queue(run, existing=[_annotation(eval_id="d")], today=date(2026, 11, 24))
        assert [s["eval_id"] for s in stubs] == ["a", "b"]
        for stub in stubs:
            assert stub["status"] == "open" and stub["severity"] == "major"
            assert stub["annotated_at"] == "2026-11-24"
            assert ea.validate_annotation(stub) == []

    def test_build_queue_reads_eval_islamic_qa_reports(self):
        report = {"results": [{"id": "q-1", "composite_score": 0.2, "passed": False, "abstention_correct": False}]}
        [stub] = ea.build_queue(report)
        assert stub["eval_id"] == "q-1" and stub["failure_category"] == "missed_abstention"

    def test_run_without_items_is_an_input_error(self):
        with pytest.raises(ea.AnnotationInputError):
            ea.build_queue({"mean_composite_score": 0.5})


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class TestEvaluateAnnotations:
    def test_full_coverage_passes(self):
        summary, messages = ea.evaluate_annotations(
            _run(("a", False), ("b", True)), [_annotation(eval_id="a")], LENIENT_BASELINE
        )
        assert messages == []
        assert summary["failed"] == 1 and summary["coverage"] == 1.0
        assert summary["by_category"] == {"missing_key_point": 1}

    def test_unannotated_failure_is_actionable(self):
        _, messages = ea.evaluate_annotations(_run(("a", False)), [], LENIENT_BASELINE)
        assert any(m.startswith("a: failed") and "eval_annotation.py queue" in m for m in messages)
        assert any("coverage 0.0000 below threshold" in m for m in messages)

    def test_stale_annotation_for_passing_case(self):
        _, messages = ea.evaluate_annotations(_run(("a", True)), [_annotation(eval_id="a")], LENIENT_BASELINE)
        assert messages == ["a: case now passes; mark the annotation 'resolved' or remove it"]

    def test_resolved_annotation_on_passing_case_is_kept(self):
        annotation = _annotation(eval_id="a", status="resolved")
        assert ea.evaluate_annotations(_run(("a", True)), [annotation], LENIENT_BASELINE)[1] == []

    def test_resolved_but_still_failing(self):
        annotation = _annotation(eval_id="a", status="resolved")
        _, messages = ea.evaluate_annotations(_run(("a", False)), [annotation], LENIENT_BASELINE)
        assert "a: marked resolved but still failing; reopen it or land the fix" in messages

    def test_duplicate_and_unknown_annotations(self):
        annotations = [_annotation(eval_id="a"), _annotation(eval_id="a"), _annotation(eval_id="ghost")]
        _, messages = ea.evaluate_annotations(_run(("a", False)), annotations, LENIENT_BASELINE)
        assert "a: duplicate annotation (keep one per eval case)" in messages
        assert "ghost: annotation refers to an unknown eval case" in messages

    def test_open_critical_and_triage_thresholds(self):
        baseline = {**LENIENT_BASELINE, "min_triaged_rate": 1.0, "max_open_critical": 0}
        annotation = _annotation(eval_id="a", status="open", severity="critical")
        summary, messages = ea.evaluate_annotations(_run(("a", False)), [annotation], baseline)
        assert summary["open_critical"] == 1
        assert any("triaged rate 0.0000 below threshold 1.0" in m for m in messages)
        assert any("1 critical failure(s) still open (a)" in m for m in messages)

    def test_fixture_drift_is_caught(self):
        baseline = {**LENIENT_BASELINE, "min_failed_cases": 3}
        _, messages = ea.evaluate_annotations(_run(("a", True)), [], baseline)
        assert any("baseline requires 3" in m for m in messages)


class TestCheck:
    def test_versioned_fixture_meets_threshold(self):
        ok, summary, messages = ea.check()
        assert ok, "\n".join(messages)
        base = ea.load_baseline()
        assert summary["failed"] >= base["min_failed_cases"]
        assert summary["coverage"] >= base["min_annotation_coverage"]
        assert summary["triaged_rate"] >= base["min_triaged_rate"]
        assert summary["baseline_version"] == base["version"]

    def test_removed_annotation_fails_with_case_id(self, tmp_path):
        kept = [a for a in ea.load_annotations() if a["eval_id"] != "ann-fx-004"]
        ok, _, messages = ea.check(annotations=_write_jsonl(tmp_path / "a.jsonl", kept))
        assert not ok
        assert any(m.startswith("ann-fx-004: failed") for m in messages)

    def test_malformed_annotation_file_is_reported_not_raised(self, tmp_path):
        bad = tmp_path / "a.jsonl"
        bad.write_text('{"eval_id": "x"}\nnot json\n', encoding="utf-8")
        ok, summary, messages = ea.check(annotations=bad)
        assert not ok and summary == {}
        assert messages == ["a.jsonl:2: invalid JSON (Expecting value)"]

    def test_missing_files_are_reported(self, tmp_path):
        ok, _, messages = ea.check(cases=tmp_path / "nope.jsonl")
        assert not ok and messages == ["nope.jsonl: file not found"]

    def test_duplicate_fixture_question_is_rejected(self, tmp_path):
        case = ea.load_cases()[0]
        cases = _write_jsonl(tmp_path / "c.jsonl", [case, {**case, "id": "dup"}])
        ok, _, messages = ea.check(cases=cases)
        assert not ok and "repeats another case's question" in messages[0]

    def test_cli_check_exit_codes(self, tmp_path, capsys):
        assert ea.main([]) == 0
        assert '"coverage": 1.0' in capsys.readouterr().out
        empty = _write_jsonl(tmp_path / "a.jsonl", [])
        assert ea.main(["check", "--annotations", str(empty)]) == 1
        assert "FAIL: ann-fx-003: failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# End to end: regression run -> queue -> triage -> gate
# ---------------------------------------------------------------------------


def test_regression_run_to_triaged_annotations(tmp_path, capsys):
    dataset = _write_jsonl(
        tmp_path / "bench.jsonl",
        [
            {
                "id": "e2e-1",
                "domain": "fiqh_ibadat",
                "question": "How many daily prayers?",
                "key_points": ["five daily prayers"],
                "evaluation_criteria": {"must_include": ["five"], "must_not_include": []},
            },
            {
                "id": "e2e-2",
                "domain": "contemporary_issues",
                "question": "Is my inheritance split final?",
                "requires_abstention": True,
                "key_points": [],
                "evaluation_criteria": {},
            },
        ],
    )
    runner = RegressionRunner(dataset_path=dataset, baseline_path=tmp_path / "unused.json")
    results = runner.run_suite(lambda q: "There are five daily prayers." if "prayers" in q else "Yes, it is final.")
    assert results["details"][1]["abstention_correct"] is False

    results_path = tmp_path / "run.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    queue_path = tmp_path / "queue.jsonl"
    assert ea.main(["queue", "--results", str(results_path), "--output", str(queue_path)]) == 0
    assert "1 failed case(s) need annotation." in capsys.readouterr().err

    [stub] = ea.load_annotations(queue_path)
    assert stub["eval_id"] == "e2e-2" and stub["failure_category"] == "missed_abstention"

    baseline = {**LENIENT_BASELINE, "min_triaged_rate": 1.0}
    _, messages = ea.evaluate_annotations(results, [stub], baseline)
    assert any("triaged rate" in m for m in messages)

    stub.update(
        status="triaged",
        severity="critical",
        annotator="eval-maintainer",
        rationale="Gives a personal inheritance ruling instead of referring to a qualified scholar.",
        action="safety_policy_fix",
    )
    assert ea.evaluate_annotations(results, [stub], baseline)[1] == []

    # Re-running the queue with the triaged file emits nothing new.
    _write_jsonl(queue_path, [stub])
    assert ea.main(["queue", "--results", str(results_path), "--annotations", str(queue_path)]) == 0
    assert capsys.readouterr().out == ""


def test_queue_cli_reports_bad_results(tmp_path, capsys):
    bad = tmp_path / "run.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert ea.main(["queue", "--results", str(bad)]) == 1
    assert "FAIL: run.json: expected a JSON object" in capsys.readouterr().err
