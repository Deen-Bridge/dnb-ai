# Production Incident Replay

Deterministic, offline replay of sanitized past incidents through the hallucination scanner.

- **Fixture:** `data/eval/incident_replay/incidents.jsonl` (id, description, response, expectations).
- **Baseline/threshold:** `data/eval/incident_replay/baseline.json` (version, owner, `min_pass_rate`, `min_incidents`, rollback).
- **Runner:** `python incident_replay.py` (exit 1 with per-incident `FAIL:` messages) or `pytest tests/test_incident_replay.py` (runs in the existing CI pytest job).

## Behavior and failure modes
Each incident must be flagged (or not flagged, for false-positive guards) with the expected type. Any mismatch fails with the incident id and reason; a pass rate under the threshold or fewer incidents than `min_incidents` also fails.

## Data handling
Fixtures must contain no secrets, personal data, user prompts or provider responses; use synthetic or redacted text only. The test checks for obvious secret markers. Needs no API keys or network.

## Observability
Output is a one-line JSON summary on stdout plus failures on stderr.

## Adding an incident
Append a sanitized line to `incidents.jsonl`, raise `min_incidents` in `baseline.json`, and bump `version`.

## Rollback
Revert the commit that changed the fixture or baseline. The check is test-only and has no runtime or deployment impact.
