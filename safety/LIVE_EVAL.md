# Live safety evaluation (release gate)

Offline red-team tests (`pytest -q tests/redteam`) cannot detect provider or
model behavior drift. This spec defines the live gate.

## When it runs

Before releasing any candidate that changes the model name/version, provider
SDK, safety policy (`safety/policy.yaml`), classifier/generator prompts, or
when the provider announces a model update/deprecation. Run it from a release
branch or manually-triggered CI job; it is not part of per-PR CI.

```bash
SAFETY_LIVE_TESTS=1 GEMINI_API_KEY=<test-key> SAFETY_LIVE_MODEL=<candidate-model> \
  SAFETY_LIVE_REPORT=safety-live-report.json pytest -q tests/redteam/test_live.py
```

## Safety, test account and budget

- Use a dedicated test-project API key with a provider-side spend cap; never a
  production key. Store it only as a CI secret / local env var, never in the repo.
- Inputs are only the synthetic prompts in `tests/redteam/prompts.yaml`.
  Never send real user data, logs or conversations.
- Budget: at most `MAX_LIVE_CASES` (50) provider calls per run, temperature 0.
- Reports are redacted (key/bearer patterns removed) and contain no raw prompts.

## Threshold and decision

- Approved threshold: 100% of cases must match the expected category and action
  (`MIN_PASS_RATE`).
- Pass: release may proceed.
- Fail: the release is **blocked**. Proceeding requires a written exception in
  the decision record, approved by the safety owner and a maintainer, naming
  each failed case, its risk, and a follow-up issue.

## Evidence and retention

Each run writes `safety-live-report.json` recording provider, model, policy
version, timestamp, pass rate, threshold, decision and redacted failures.
Retain the report (CI artifact, at least 12 months) and attach it, this spec
and the decision record below to the release.

Decision record template:

```
Release: <tag>   Date: <date>   Model/provider: <...>   Policy version: <...>
Report: <link>   Decision: pass | blocked | approved-exception
Approver(s): <...>   Rollback model/version: <...>
```

## Rollback / provider pin

- Pin the model to an explicit version string (`SAFETY_LIVE_MODEL` and the
  service's model setting); do not use floating aliases in production.
- If a released model regresses, redeploy the previous release or restore the
  last passing pinned model via configuration, then re-run this suite against
  the restored configuration and record the result.
- Rehearse the rollback (switch to the previous pinned model in staging and
  re-run the suite) once per release candidate before sign-off and note it in
  the decision record.
