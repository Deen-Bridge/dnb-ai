"""Offline gate for prompt versioning and evaluation compatibility.

Run from the repository root (no API keys or network access needed)::

    python -m scripts.check_prompt_eval              # CI check
    python -m scripts.check_prompt_eval --write-lock # after a reviewed prompt change
    python -m scripts.check_prompt_eval --write-baseline \\
        --owner "AI quality maintainers" --baseline-version 1.1.0 \\
        --rollback-plan "git revert the baseline commit"

The check fails (exit code 1) when:

* a registered prompt's text changed without a version bump, or the version
  lock is stale (see ``prompts/versioning.py``);
* the evaluation harness can no longer render the prompt, i.e. the template
  needs a variable outside the published ``VARIABLE_CONTRACT``;
* the deterministic fixture scores below ``--min-score``, drops more than the
  tolerances below the committed baseline, or changes size;
* the baseline is missing owner/version/rollback metadata, or was recorded
  for a different prompt or an incompatible (different major) prompt version.

The "model" is a replay of the fixture's reference answers that only answers
when the rendered prompt still carries the user's question and the retrieved
context. A template that drops either input therefore loses score, which is
the compatibility signal this gate measures without calling a provider. A
live model can be evaluated with the same runner by passing a real callable
to ``RegressionRunner.run_suite``.

Output contains item ids, domains and scores only; prompt text and answers
are never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompts.registry import (
    DEFAULT_SYSTEM_PROMPT,
    VARIABLE_CONTRACT,
    PromptRegistry,
    PromptTemplate,
    registry as default_registry,
)
from prompts.versioning import DEFAULT_LOCK_PATH, check_lock, load_lock, write_lock
from regression import EVAL_DIR, RegressionRunner

FIXTURE_DIR = EVAL_DIR / "prompt_compat"
DEFAULT_FIXTURE = FIXTURE_DIR / "fixture.jsonl"
DEFAULT_BASELINE = FIXTURE_DIR / "baseline.json"

# Thresholds. The replay fixture is deterministic, so any drop is a real
# change in how the prompt carries evaluation inputs, not sampling noise.
MIN_MEAN_SCORE = 0.90
SCORE_TOLERANCE = 0.02
DOMAIN_TOLERANCE = 0.05
PASS_RATE_TOLERANCE = 0.0

REPLAY_MODEL_ID = "replay-fixture"
REQUIRED_FIXTURE_FIELDS = ("id", "domain", "question", "retrieved_context", "reference_answer")


@dataclass
class GateReport:
    template_name: str
    template_version: str = ""
    failures: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    results: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "template": self.template_name,
            "template_version": self.template_version,
            "passed": self.passed,
            "failures": self.failures,
            "summary": self.lines,
        }


def build_render_variables(record: dict[str, Any]) -> dict[str, Any]:
    """Map a fixture record onto every variable in the published contract."""
    return {
        "madhhab": record.get("madhhab") or "not specified",
        "language": record.get("language") or "en",
        "knowledge_level": record.get("knowledge_level") or "intermediate",
        "intent": record.get("intent") or record["domain"],
        "retrieved_context": record.get("retrieved_context", ""),
        "user_question": record["question"],
    }


class ReplayModel:
    """Deterministic stand-in for the LLM.

    Returns a record's reference answer only when the rendered prompt
    contains both that record's question and its retrieved context.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def __call__(self, prompt: str) -> str:
        for record in self._records:
            if record["question"] in prompt and record["retrieved_context"] in prompt:
                return str(record["reference_answer"])
        return ""


def validate_fixture(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return ["fixture has no records"]
    errors = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        label = record.get("id") or f"line {index}"
        missing = [name for name in REQUIRED_FIXTURE_FIELDS if not record.get(name)]
        if missing:
            errors.append(f"fixture record {label} is missing {', '.join(missing)}")
        if record.get("id") in seen:
            errors.append(f"fixture id {label} is duplicated")
        seen.add(str(record.get("id")))
    return errors


def make_prompt_model(template: PromptTemplate, records: list[dict[str, Any]]) -> Callable[[str], str]:
    """Wrap the replay model so every question is routed through ``template``."""
    replay = ReplayModel(records)
    by_question = {record["question"]: record for record in records}

    def model(question: str) -> str:
        variables = build_render_variables(by_question[question])
        prompt = template.render(**{name: variables[name] for name in template.variables})
        return replay(prompt)

    return model


def run_suite(template: PromptTemplate, runner: RegressionRunner) -> tuple[dict[str, Any] | None, list[str]]:
    """Run the fixture through ``template``; return (results, failures)."""
    unsupported = sorted(set(template.variables) - set(VARIABLE_CONTRACT))
    if unsupported:
        return None, [
            f"prompt '{template.name}' needs variable(s) {unsupported} that the evaluation harness does not "
            "supply; add them to VARIABLE_CONTRACT and build_render_variables, or keep the old variables"
        ]

    records = runner.load_dataset()
    fixture_errors = validate_fixture(records)
    if fixture_errors:
        return None, fixture_errors
    return runner.run_suite(make_prompt_model(template, records)), []


def run_gate(
    template_name: str = DEFAULT_SYSTEM_PROMPT.name,
    *,
    registry: PromptRegistry = default_registry,
    fixture_path: Path = DEFAULT_FIXTURE,
    baseline_path: Path = DEFAULT_BASELINE,
    lock_path: Path = DEFAULT_LOCK_PATH,
    min_score: float = MIN_MEAN_SCORE,
) -> GateReport:
    report = GateReport(template_name=template_name)

    lock_errors = check_lock(registry, load_lock(lock_path))
    report.failures.extend(lock_errors)
    report.lines.append("prompt lock: " + ("OK" if not lock_errors else f"{len(lock_errors)} problem(s)"))

    try:
        template = registry.get(template_name)
    except KeyError:
        report.failures.append(f"prompt '{template_name}' is not registered; choose one of {registry.list_names()}")
        return report
    report.template_version = template.version
    report.lines.append(f"prompt: {template.name}@{template.version} ({template.fingerprint[:19]})")

    runner = RegressionRunner(dataset_path=fixture_path, baseline_path=baseline_path)
    results, suite_errors = run_suite(template, runner)
    report.failures.extend(suite_errors)
    if results is None:
        return report
    report.results = results

    mean = results["mean_composite_score"]
    report.lines.append(
        f"fixture: {results['total_evaluated']} item(s), mean {mean:.4f} (min {min_score:.2f}), "
        f"pass_rate {results['pass_rate']:.4f}"
    )
    if mean < min_score:
        report.failures.append(f"fixture mean score {mean:.4f} is below the {min_score:.2f} threshold")
    for item in results["details"]:
        if not item["passed"]:
            report.lines.append(f"  item {item['id']} ({item['domain']}) failed with score {item['score']}")

    report.failures.extend(_check_baseline(runner, template, results))
    return report


def _check_baseline(runner: RegressionRunner, template: PromptTemplate, results: dict[str, Any]) -> list[str]:
    baseline = runner.load_baseline()
    if baseline is None:
        return [f"baseline {runner.baseline_path.name} is missing; create it with --write-baseline"]

    failures = [f"baseline {error}" for error in runner.baseline_metadata_errors()]
    if baseline.get("total_evaluated") != results["total_evaluated"]:
        failures.append(
            f"fixture has {results['total_evaluated']} item(s) but the baseline covers "
            f"{baseline.get('total_evaluated')}; regenerate the baseline with --write-baseline"
        )

    comparison = runner.compare_with_baseline(
        results,
        tolerance=SCORE_TOLERANCE,
        domain_tolerance=DOMAIN_TOLERANCE,
        pass_rate_tolerance=PASS_RATE_TOLERANCE,
        prompt_name=template.name,
        prompt_version=template.version,
    )
    if not comparison["compatible"] or comparison["degraded"]:
        failures.append(comparison["alert"])
    return failures


def write_baseline(
    template_name: str,
    *,
    owner: str,
    baseline_version: str,
    rollback_plan: str,
    registry: PromptRegistry = default_registry,
    fixture_path: Path = DEFAULT_FIXTURE,
    baseline_path: Path = DEFAULT_BASELINE,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> list[str]:
    """Record a new baseline for ``template_name``; returns blocking errors."""
    lock_errors = check_lock(registry, load_lock(lock_path))
    if lock_errors:
        return ["refusing to write a baseline for an unlocked prompt", *lock_errors]
    template = registry.get(template_name)
    runner = RegressionRunner(dataset_path=fixture_path, baseline_path=baseline_path)
    results, errors = run_suite(template, runner)
    if results is None:
        return errors
    runner.save_baseline(
        results,
        owner=owner,
        version=baseline_version,
        rollback_plan=rollback_plan,
        model_id=REPLAY_MODEL_ID,
        prompt_name=template.name,
        prompt_version=template.version,
        prompt_fingerprint=template.fingerprint,
    )
    return []


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--template", default=DEFAULT_SYSTEM_PROMPT.name, help="prompt template to evaluate")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--min-score", type=float, default=MIN_MEAN_SCORE)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--write-lock", action="store_true", help="regenerate the prompt version lock")
    parser.add_argument("--write-baseline", action="store_true", help="record a new evaluation baseline")
    parser.add_argument("--owner", help="baseline owner (required with --write-baseline)")
    parser.add_argument("--baseline-version", help="baseline version (required with --write-baseline)")
    parser.add_argument("--rollback-plan", help="baseline rollback plan (required with --write-baseline)")
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be between 0 and 1")
    if args.write_baseline and not (args.owner and args.baseline_version and args.rollback_plan):
        parser.error("--write-baseline requires --owner, --baseline-version and --rollback-plan")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.write_lock:
        lock = write_lock(default_registry, args.lock)
        print(f"Wrote {args.lock} with {len(lock['templates'])} template(s).")
        return 0

    if args.write_baseline:
        errors = write_baseline(
            args.template,
            owner=args.owner,
            baseline_version=args.baseline_version,
            rollback_plan=args.rollback_plan,
            fixture_path=args.fixture,
            baseline_path=args.baseline,
            lock_path=args.lock,
        )
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        if not errors:
            print(f"Wrote baseline {args.baseline}.")
        return 1 if errors else 0

    report = run_gate(
        args.template,
        fixture_path=args.fixture,
        baseline_path=args.baseline,
        lock_path=args.lock,
        min_score=args.min_score,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        for line in report.lines:
            print(line)
        for failure in report.failures:
            print(f"FAIL: {failure}")
        print("RESULT: " + ("PASS" if report.passed else "FAIL"))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
