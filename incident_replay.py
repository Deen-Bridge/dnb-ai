"""Deterministic production-incident replay.

Replays sanitized past incidents (data/eval/incident_replay/incidents.jsonl)
through the offline hallucination scanner and compares the outcome against a
versioned baseline. No network, API keys or provider responses are needed.

Usage: python incident_replay.py  (exit code 1 on failure)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPLAY_DIR = ROOT / "data" / "eval" / "incident_replay"
DEFAULT_FIXTURE = REPLAY_DIR / "incidents.jsonl"
DEFAULT_BASELINE = REPLAY_DIR / "baseline.json"


def load_incidents(path: Path | str = DEFAULT_FIXTURE) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_baseline(path: Path | str = DEFAULT_BASELINE) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def replay_incidents(incidents: list[dict[str, Any]]) -> dict[str, Any]:
    from hallucination import scan_for_hallucinations

    failures: list[str] = []
    for inc in incidents:
        result = scan_for_hallucinations(inc["response"])
        types = {f.hallucination_type.value for f in result.flags}
        expected = bool(inc["expect_detected"])
        if result.hallucination_detected != expected:
            failures.append(
                f"{inc['id']}: expected detected={expected}, got {result.hallucination_detected} ({inc['description']})"
            )
        elif expected and inc.get("expect_type") and inc["expect_type"] not in types:
            failures.append(f"{inc['id']}: expected type {inc['expect_type']}, got {sorted(types)}")
    total = len(incidents)
    passed = total - len(failures)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "failures": failures,
    }


def check(
    fixture: Path | str = DEFAULT_FIXTURE, baseline: Path | str = DEFAULT_BASELINE
) -> tuple[bool, dict[str, Any], list[str]]:
    """Return (ok, results, actionable_messages)."""
    base = load_baseline(baseline)
    results = replay_incidents(load_incidents(fixture))
    messages = list(results["failures"])
    if results["total"] < base["min_incidents"]:
        messages.append(f"fixture has {results['total']} incidents, baseline requires {base['min_incidents']}")
    if results["pass_rate"] < base["min_pass_rate"]:
        messages.append(f"pass_rate {results['pass_rate']} below threshold {base['min_pass_rate']}")
    return (not messages), results, messages


def main() -> int:
    ok, results, messages = check()
    print(json.dumps({k: v for k, v in results.items() if k != "failures"}))
    for m in messages:
        print(f"FAIL: {m}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
