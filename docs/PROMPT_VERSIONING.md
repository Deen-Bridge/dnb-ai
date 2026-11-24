# Prompt versioning and evaluation compatibility

Owner: AI quality maintainers (`owner` in `prompts/prompt_versions.lock.json` and
`metadata.owner` in the baseline). Related: [MODEL_PROMPT_ROLLBACK.md](MODEL_PROMPT_ROLLBACK.md).

## Contract

| Piece | Location | Purpose |
| ----- | -------- | ------- |
| Prompt templates | `prompts/registry.py` | Each `PromptTemplate` has a `MAJOR.MINOR.PATCH` version, a changelog, and declared variables that must match its `{placeholders}` exactly. Invalid templates fail at import. |
| Fingerprint | `PromptTemplate.fingerprint` | SHA-256 of name + variables + text. Version and changelog are excluded, so a text edit without a version bump is detectable. |
| Version lock | `prompts/prompt_versions.lock.json` | Committed record of each template's version and fingerprint. |
| Fixture | `data/eval/prompt_compat/fixture.jsonl` | Six synthetic Q&A items across five domains (one requires abstention). No user data. |
| Baseline | `data/eval/prompt_compat/baseline.json` | Aggregate and per-item scores (ids only) with `owner`, `version`, `rollback_plan`, `prompt_name`, `prompt_version`, `prompt_fingerprint`. |
| Gate | `python -m scripts.check_prompt_eval` | Offline CI check. Exits 1 with one `FAIL:` line per problem. |

### Version semantics

- **PATCH**: wording or typo fixes that do not change the expected behaviour.
- **MINOR**: behaviour changes that are still comparable to the existing baseline.
- **MAJOR**: scores are no longer comparable (e.g. new output format). The gate reports
  `INCOMPATIBLE BASELINE` until a new baseline is written in the same change.

### Measured thresholds

The fixture is scored with the offline rubric from `scripts/eval_islamic_qa.py`
(`evaluate_single_response`). The "model" replays each item's reference answer only
when the rendered prompt still contains that item's question **and** its retrieved
context, so a template that stops passing either input to the model loses score.

| Check | Threshold |
| ----- | --------- |
| Mean composite score | at least `0.90` (`--min-score`) |
| Mean drop vs baseline | at most `0.02` |
| Per-domain drop vs baseline | at most `0.05` |
| Pass-rate drop vs baseline | `0.0` (no item may start failing) |
| Fixture size | must equal the baseline's `total_evaluated` |

## Failure modes and what the gate reports

| Situation | Gate output (abridged) | Fix |
| --------- | ---------------------- | --- |
| Text edited, version unchanged | `text changed but its version is still X` | Bump the version, add a changelog entry, run `--write-lock`. |
| Version bumped, lock not regenerated | `is version X but the lock records Y` | Run `--write-lock` and commit the lock. |
| Template added/removed/renamed | `not in the lock` / `has no registered template` | Run `--write-lock`. |
| Template needs a variable the eval harness cannot supply | `needs variable(s) [...]` | Add it to `VARIABLE_CONTRACT` and `build_render_variables`. |
| Prompt drops question/context, or quality falls | `below the 0.90 threshold`, `REGRESSION DETECTED`, failing item ids | Fix the template, or justify and re-baseline. |
| Major version bump | `INCOMPATIBLE BASELINE` | Run `--write-baseline` and get the new baseline reviewed. |
| Baseline missing or lacks owner/version/rollback | `baseline ... missing` | Run `--write-baseline` with all metadata flags. |

## Changing a prompt

```bash
# 1. Edit the template in prompts/registry.py, bump `version`, update `changelog`.
# 2. Regenerate the lock:
python -m scripts.check_prompt_eval --write-lock
# 3. Major bump or intentional fixture change only: write a new baseline.
python -m scripts.check_prompt_eval --write-baseline \
  --owner "AI quality maintainers" --baseline-version 2.0.0 \
  --rollback-plan "git revert this commit, redeploy, re-run the gate"
# 4. Confirm, then commit the template, lock and baseline together:
python -m scripts.check_prompt_eval
pytest -q tests/test_prompt_versioning.py
```

`--write-baseline` refuses to run while the lock is stale. Use `--json` to get
machine-readable output.

## Configuration

No environment variables or secrets. The gate never calls a model provider and
can run on any checkout. Thresholds are constants in
`scripts/check_prompt_eval.py`. CI uses the defaults. `--min-score` overrides
the score floor for local experiments only. To evaluate a live model, pass a real
callable to `RegressionRunner.run_suite` with a separate baseline. Do not
overwrite the replay baseline.

## Data handling

The fixture and baseline are synthetic and contain no personal data. The baseline
and the gate output contain only item ids, domains and scores. Prompt text, questions
and answers are never printed or stored in the baseline. A test enforces this.

## Observability

Each run prints the template name, version and a fingerprint prefix, the lock
status, the fixture mean and pass rate, and the id/domain/score of each failing item.
`RegressionRunner.compare_with_baseline(..., prompt_name=..., prompt_version=...)`
returns `compatible` and `incompatibilities` alongside the existing `degraded`/`reasons`.

## Rollback

1. `git revert` the commit that changed the prompt. The template, lock and baseline
   ship together, so one revert restores all three. Then redeploy.
2. If only a local baseline write needs undoing, call
   `RegressionRunner(baseline_path="data/eval/prompt_compat/baseline.json").rollback_baseline()`.
   This restores `baseline.json.prev`, which is git-ignored.
3. Re-run `python -m scripts.check_prompt_eval` and confirm `RESULT: PASS`.

The change is additive at runtime. `PromptTemplate` now rejects invalid versions
and placeholder/variable mismatches when defined. The existing template already
complied, and `main.py` does not import the registry, so serving behaviour is
unchanged.
