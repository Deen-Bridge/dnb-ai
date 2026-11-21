# Safe model and prompt rollback

Owner: AI quality maintainers (record the individual owner in baseline `metadata.owner`).

## Contract
- `RegressionRunner` (`regression.py`) compares an evaluation run to a versioned baseline
  (`data/eval/baseline_results.json`). Comparison fails on a mean-score drop beyond `tolerance`
  (default 0.02), and optionally on per-domain (`domain_tolerance`) or `pass_rate_tolerance` drops.
  Alerts list each failing reason.
- Baselines should be saved with `owner`, `version`, `rollback_plan` (plus optional `model_id`,
  `prompt_version`). `baseline_metadata_errors()` reports missing fields.
- `save_baseline` keeps the previous file as `baseline_results.json.prev`;
  `rollback_baseline()` restores it. A missing baseline reports no regression (never blocks).

## Rollback procedure
1. Revert the model/prompt change (config or commit revert) and redeploy.
2. If the baseline was updated for that change, call `RegressionRunner().rollback_baseline()`
   or `git revert` the baseline commit.
3. Re-run the suite with a deterministic stub callable (no secrets needed) and confirm `OK`.

## Data handling
Baselines contain only aggregate scores and item ids — no prompts, user data, keys or provider responses.

## CI
`pytest tests/test_regression.py` runs offline with a deterministic fixture and stub model.
