"""Prompt version lock, evaluation compatibility and the offline CI gate (#829)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from prompts.registry import DEFAULT_SYSTEM_PROMPT, PromptRegistry, PromptTemplate, parse_version
from prompts.versioning import DEFAULT_LOCK_PATH, build_lock, check_lock, is_eval_compatible, load_lock, write_lock
from regression import RegressionRunner
from scripts import check_prompt_eval
from scripts.check_prompt_eval import DEFAULT_BASELINE, DEFAULT_FIXTURE, GateReport, run_gate, write_baseline

ROLLBACK = "git revert the baseline commit"


def _variant(**changes: Any) -> PromptTemplate:
    return dataclasses.replace(DEFAULT_SYSTEM_PROMPT, **changes)


def _without_context() -> PromptTemplate:
    """A plausible-looking edit that silently drops retrieved context."""
    text = DEFAULT_SYSTEM_PROMPT.template.replace("Retrieved context:\n{retrieved_context}\n\n", "")
    return _variant(
        version="1.1.0",
        template=text,
        variables=frozenset(DEFAULT_SYSTEM_PROMPT.variables - {"retrieved_context"}),
    )


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    """Copies of the committed fixture and baseline so tests never touch the repo."""
    fixture = tmp_path / "fixture.jsonl"
    baseline = tmp_path / "baseline.json"
    fixture.write_bytes(DEFAULT_FIXTURE.read_bytes())
    baseline.write_bytes(DEFAULT_BASELINE.read_bytes())
    return {"fixture": fixture, "baseline": baseline, "lock": tmp_path / "lock.json"}


def _gate(template: PromptTemplate, paths: dict[str, Path]) -> GateReport:
    registry = PromptRegistry([template])
    write_lock(registry, paths["lock"])
    return run_gate(
        template.name,
        registry=registry,
        fixture_path=paths["fixture"],
        baseline_path=paths["baseline"],
        lock_path=paths["lock"],
    )


# ---------------------------------------------------------------------------
# Template validation and fingerprints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["1", "1.0", "v1.0.0", "1.0.0-beta", "01.0.0", ""])
def test_parse_version_rejects_non_semver(version: str) -> None:
    with pytest.raises(ValueError, match="Invalid prompt version"):
        parse_version(version)


def test_parse_version_orders_numerically() -> None:
    assert parse_version("1.10.0") > parse_version("1.9.3")


def test_template_rejects_invalid_version() -> None:
    with pytest.raises(ValueError, match="Invalid prompt version"):
        _variant(version="latest")


def test_template_rejects_undeclared_placeholder() -> None:
    with pytest.raises(ValueError, match="do not match declared variables"):
        _variant(template=DEFAULT_SYSTEM_PROMPT.template + "\nTone: {tone}")


def test_template_rejects_declared_but_unused_variable() -> None:
    text = DEFAULT_SYSTEM_PROMPT.template.replace("Question intent: {intent}\n", "")
    with pytest.raises(ValueError, match="do not match declared variables"):
        _variant(template=text)


def test_fingerprint_tracks_content_not_metadata() -> None:
    fingerprint = DEFAULT_SYSTEM_PROMPT.fingerprint
    assert fingerprint.startswith("sha256:") and len(fingerprint) == 71
    assert _variant(version="9.9.9", changelog="reworded notes").fingerprint == fingerprint
    assert _variant(template=DEFAULT_SYSTEM_PROMPT.template + " ").fingerprint != fingerprint


def test_eval_compatibility_follows_major_version() -> None:
    assert is_eval_compatible("1.0.0", "1.4.2")
    assert is_eval_compatible("2.3.0", "2.0.0")
    assert not is_eval_compatible("1.9.9", "2.0.0")


# ---------------------------------------------------------------------------
# Version lock
# ---------------------------------------------------------------------------


def test_committed_lock_matches_registry() -> None:
    from prompts.registry import registry

    assert check_lock(registry, load_lock(DEFAULT_LOCK_PATH)) == []


def test_lock_round_trip_is_deterministic(tmp_path: Path) -> None:
    registry = PromptRegistry([DEFAULT_SYSTEM_PROMPT])
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    write_lock(registry, first)
    write_lock(registry, second)
    assert first.read_bytes() == second.read_bytes()
    assert check_lock(registry, load_lock(first)) == []


def test_lock_detects_text_change_without_version_bump() -> None:
    lock = build_lock(PromptRegistry([DEFAULT_SYSTEM_PROMPT]))
    edited = PromptRegistry([_variant(template=DEFAULT_SYSTEM_PROMPT.template + "\nCite sources.")])
    errors = check_lock(edited, lock)
    assert len(errors) == 1
    assert "text changed but its version is still 1.0.0" in errors[0]
    assert "--write-lock" in errors[0]


def test_lock_detects_version_bump_without_regenerated_lock() -> None:
    lock = build_lock(PromptRegistry([DEFAULT_SYSTEM_PROMPT]))
    errors = check_lock(PromptRegistry([_variant(version="1.0.1")]), lock)
    assert errors == [
        "prompt 'islamic_context' is version 1.0.1 but the lock records 1.0.0; "
        "run `python -m scripts.check_prompt_eval --write-lock` and commit the lock with the change"
    ]


def test_lock_detects_unlocked_and_stale_templates() -> None:
    lock = build_lock(PromptRegistry([DEFAULT_SYSTEM_PROMPT]))
    renamed = PromptRegistry([_variant(name="islamic_context_v2")])
    errors = check_lock(renamed, lock)
    assert any("'islamic_context_v2' is not in the lock" in error for error in errors)
    assert any("lock entry 'islamic_context' has no registered template" in error for error in errors)


@pytest.mark.parametrize(
    ("lock", "message"),
    [
        (None, "lock file is missing"),
        ({"schema_version": 99, "owner": "x", "templates": {}}, "schema_version 99"),
        ({"schema_version": 1, "owner": "", "templates": {}}, "no owner"),
    ],
)
def test_lock_rejects_missing_or_malformed_documents(lock: dict | None, message: str) -> None:
    errors = check_lock(PromptRegistry([DEFAULT_SYSTEM_PROMPT]), lock)
    assert len(errors) == 1 and message in errors[0]


def test_load_lock_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_lock(path)
    assert load_lock(tmp_path / "missing.json") is None


# ---------------------------------------------------------------------------
# Baseline compatibility in RegressionRunner
# ---------------------------------------------------------------------------


def _runner_with_prompt_baseline(tmp_path: Path, prompt_version: str | None) -> RegressionRunner:
    runner = RegressionRunner(baseline_path=tmp_path / "b.json")
    runner.save_baseline(
        {"mean_composite_score": 0.9, "pass_rate": 1.0, "domain_scores": {}},
        owner="team",
        version="1",
        rollback_plan=ROLLBACK,
        prompt_name="islamic_context",
        prompt_version=prompt_version,
    )
    return runner


def test_compare_accepts_same_major_prompt(tmp_path: Path) -> None:
    runner = _runner_with_prompt_baseline(tmp_path, "1.0.0")
    current = {"mean_composite_score": 0.9, "pass_rate": 1.0, "domain_scores": {}}
    comparison = runner.compare_with_baseline(current, prompt_name="islamic_context", prompt_version="1.3.0")
    assert comparison["compatible"] is True
    assert comparison["alert"] == "OK"


def test_compare_flags_major_prompt_change(tmp_path: Path) -> None:
    runner = _runner_with_prompt_baseline(tmp_path, "1.0.0")
    current = {"mean_composite_score": 0.95, "pass_rate": 1.0, "domain_scores": {}}
    comparison = runner.compare_with_baseline(current, prompt_name="islamic_context", prompt_version="2.0.0")
    assert comparison["compatible"] is False
    assert comparison["alert"].startswith("INCOMPATIBLE BASELINE")
    assert "regenerate and review the baseline" in comparison["alert"]


def test_compare_flags_other_prompt_or_unversioned_baseline(tmp_path: Path) -> None:
    runner = _runner_with_prompt_baseline(tmp_path, None)
    current = {"mean_composite_score": 0.9, "pass_rate": 1.0, "domain_scores": {}}
    comparison = runner.compare_with_baseline(current, prompt_name="other", prompt_version="1.0.0")
    assert comparison["incompatibilities"] == [
        "baseline is for prompt 'islamic_context', not 'other'",
        "baseline does not record metadata.prompt_version",
    ]


def test_compare_reports_malformed_baseline_prompt_version(tmp_path: Path) -> None:
    runner = _runner_with_prompt_baseline(tmp_path, "latest")
    comparison = runner.compare_with_baseline({"mean_composite_score": 0.9}, prompt_version="1.0.0")
    assert comparison["compatible"] is False
    assert "Invalid prompt version 'latest'" in comparison["incompatibilities"][0]


def test_compare_without_prompt_version_keeps_legacy_behaviour(tmp_path: Path) -> None:
    runner = _runner_with_prompt_baseline(tmp_path, None)
    comparison = runner.compare_with_baseline({"mean_composite_score": 0.9})
    assert comparison["compatible"] is True
    assert comparison["alert"] == "OK"


# ---------------------------------------------------------------------------
# Offline gate (evaluation tests on the deterministic fixture)
# ---------------------------------------------------------------------------


def test_committed_gate_passes_with_threshold() -> None:
    report = run_gate()
    assert report.failures == []
    assert report.results is not None
    assert report.results["total_evaluated"] >= 6
    assert report.results["mean_composite_score"] >= check_prompt_eval.MIN_MEAN_SCORE


def test_committed_baseline_is_versioned_with_owner_and_rollback() -> None:
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    meta = baseline["metadata"]
    assert meta["owner"] and meta["version"] and meta["rollback_plan"]
    assert meta["prompt_name"] == DEFAULT_SYSTEM_PROMPT.name
    assert is_eval_compatible(meta["prompt_version"], DEFAULT_SYSTEM_PROMPT.version)
    assert meta["model_id"] == check_prompt_eval.REPLAY_MODEL_ID


def test_committed_baseline_has_no_prompt_or_answer_text() -> None:
    raw = DEFAULT_BASELINE.read_text(encoding="utf-8")
    for line in DEFAULT_FIXTURE.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert record["question"] not in raw
        assert record["reference_answer"] not in raw


def test_gate_fails_when_prompt_drops_retrieved_context(workspace: dict[str, Path]) -> None:
    report = _gate(_without_context(), workspace)
    assert not report.passed
    assert any("below the 0.90 threshold" in failure for failure in report.failures)
    assert any(failure.startswith("REGRESSION DETECTED") for failure in report.failures)
    assert any("item pc-001" in line for line in report.lines)


def test_gate_fails_for_unlocked_text_change(workspace: dict[str, Path]) -> None:
    registry = PromptRegistry([DEFAULT_SYSTEM_PROMPT])
    write_lock(registry, workspace["lock"])
    edited = PromptRegistry([_variant(template=DEFAULT_SYSTEM_PROMPT.template + "\nBe brief.")])
    report = run_gate(
        registry=edited,
        fixture_path=workspace["fixture"],
        baseline_path=workspace["baseline"],
        lock_path=workspace["lock"],
    )
    assert report.failures and "version is still 1.0.0" in report.failures[0]


def test_gate_requires_rebaseline_on_major_bump(workspace: dict[str, Path]) -> None:
    report = _gate(_variant(version="2.0.0"), workspace)
    assert report.failures == [
        "INCOMPATIBLE BASELINE: baseline was recorded with prompt 1.0.0; current prompt 2.0.0 "
        "is a new major version, so regenerate and review the baseline"
    ]

    bumped = _variant(version="2.0.0")
    errors = write_baseline(
        bumped.name,
        owner="team",
        baseline_version="2.0.0",
        rollback_plan=ROLLBACK,
        registry=PromptRegistry([bumped]),
        fixture_path=workspace["fixture"],
        baseline_path=workspace["baseline"],
        lock_path=workspace["lock"],
    )
    assert errors == []
    assert _gate(bumped, workspace).passed

    runner = RegressionRunner(baseline_path=workspace["baseline"])
    assert runner.rollback_baseline() is True
    restored = runner.load_baseline()
    assert restored is not None
    assert restored["metadata"]["prompt_version"] == "1.0.0"


def test_gate_rejects_variables_outside_contract(workspace: dict[str, Path]) -> None:
    extended = _variant(
        version="1.1.0",
        template=DEFAULT_SYSTEM_PROMPT.template + "\nTone: {tone}",
        variables=frozenset(DEFAULT_SYSTEM_PROMPT.variables | {"tone"}),
    )
    report = _gate(extended, workspace)
    assert len(report.failures) == 1
    assert "['tone']" in report.failures[0] and "VARIABLE_CONTRACT" in report.failures[0]


def test_gate_flags_fixture_size_change_and_missing_baseline(workspace: dict[str, Path]) -> None:
    lines = workspace["fixture"].read_text(encoding="utf-8").splitlines()
    workspace["fixture"].write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    report = _gate(DEFAULT_SYSTEM_PROMPT, workspace)
    assert any("fixture has 5 item(s) but the baseline covers 6" in failure for failure in report.failures)

    workspace["baseline"].unlink()
    report = _gate(DEFAULT_SYSTEM_PROMPT, workspace)
    assert report.failures == ["baseline baseline.json is missing; create it with --write-baseline"]


def test_gate_validates_fixture_records(workspace: dict[str, Path]) -> None:
    record = {"id": "pc-001", "domain": "aqeedah", "question": "q"}
    workspace["fixture"].write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    report = _gate(DEFAULT_SYSTEM_PROMPT, workspace)
    assert "fixture record pc-001 is missing retrieved_context, reference_answer" in report.failures
    assert "fixture id pc-001 is duplicated" in report.failures

    workspace["fixture"].write_text("", encoding="utf-8")
    assert _gate(DEFAULT_SYSTEM_PROMPT, workspace).failures == ["fixture has no records"]


def test_gate_reports_unknown_template(workspace: dict[str, Path]) -> None:
    registry = PromptRegistry([DEFAULT_SYSTEM_PROMPT])
    write_lock(registry, workspace["lock"])
    report = run_gate("missing", registry=registry, lock_path=workspace["lock"])
    assert report.failures == ["prompt 'missing' is not registered; choose one of ['islamic_context']"]


def test_write_baseline_refuses_unlocked_prompt(workspace: dict[str, Path]) -> None:
    errors = write_baseline(
        DEFAULT_SYSTEM_PROMPT.name,
        owner="team",
        baseline_version="1",
        rollback_plan=ROLLBACK,
        fixture_path=workspace["fixture"],
        baseline_path=workspace["baseline"],
        lock_path=workspace["lock"],
    )
    assert errors[0] == "refusing to write a baseline for an unlocked prompt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_passes_and_output_is_sanitized(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_prompt_eval.main([]) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out
    assert "islamic_context@1.0.0" in out
    for line in DEFAULT_FIXTURE.read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["question"] not in out


def test_cli_json_reports_failures(workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--json", "--fixture", str(workspace["fixture"]), "--lock", str(workspace["lock"])]
    assert check_prompt_eval.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert "lock file is missing" in report["failures"][0]


@pytest.mark.parametrize("argv", [["--write-baseline", "--owner", "team"], ["--min-score", "1.5"]])
def test_cli_rejects_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        check_prompt_eval.main(argv)
    assert excinfo.value.code == 2
