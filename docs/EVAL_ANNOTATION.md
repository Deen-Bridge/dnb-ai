# Annotation Workflow for Failed Evaluations

Every failing evaluation case gets a short, structured human annotation that says why it failed and what happens next. Annotations are versioned in the repo and checked in CI, so a failure can't go unexplained and a "fixed" case can't quietly start failing again.

- **Module:** `eval_annotation.py` (no network, no API keys)
- **Fixture:** `data/eval/annotations/cases.jsonl` holds sanitized cases, each with a canned `candidate_answer`.
- **Annotations:** `data/eval/annotations/annotations.jsonl` holds one line per failing case.
- **Baseline and thresholds:** `data/eval/annotations/baseline.json` (`version`, `owner`, `rollback`, plus the thresholds below)
- **Tests:** `tests/test_eval_annotation.py`
- **CI:** the `Lint and Test` job runs the tests and then `python eval_annotation.py check`.

## Workflow

1. Run an evaluation. Either `RegressionRunner().run_suite(...)` or `scripts/eval_islamic_qa.py` works. Save its JSON summary.
2. Generate stubs for the failures that aren't annotated yet:

   ```bash
   python eval_annotation.py queue --results run.json \
       --annotations data/eval/annotations/annotations.jsonl --output stubs.jsonl
   ```

   Each stub starts with `status: open` and `severity: major`. Its `failure_category` is a first guess taken from the scorer's diagnostics: a missed abstention, prohibited content or a missing key point.
3. Triage each stub. Confirm or correct the category and set the severity. Write a short `rationale`, then choose an `action`. Set `status` to `triaged` or `wont_fix`.
4. Append the triaged lines to the annotation file and open a PR. When the fix lands and the case passes, set `status: resolved`.

## Annotation contract

| Field | Rule |
| --- | --- |
| `eval_id` | Required. The id of the eval case. Allow one annotation per case. |
| `failure_category` | Required. One of `missing_key_point`, `prohibited_content`, `missed_abstention`, `unnecessary_abstention`, `incorrect_information`, `wrong_or_missing_citation`, `one_sided_fiqh_answer`, `wrong_language`, `poor_adab`, `benchmark_defect`, `scorer_defect`, `other`. |
| `severity` | Required. `critical`, `major` or `minor`. |
| `status` | Required. `open`, `triaged`, `resolved` or `wont_fix`. |
| `annotator` | Required. A lowercase role handle such as `eval-maintainer`. Never use a person's name or email. |
| `annotated_at` | Required. An ISO date. |
| `rationale` | Required unless `open`. At most 500 characters. Rejected if it contains an email address or a secret-like token. |
| `action` | Required unless `open`. One of `prompt_fix`, `retrieval_fix`, `safety_policy_fix`, `model_change`, `dataset_fix`, `scorer_fix`, `accept_risk`. |
| `requires_scholar_review` | Must be `true` for `incorrect_information`, `one_sided_fiqh_answer` and `benchmark_defect`. |

Cross-field rules:

- `wont_fix` must use the `accept_risk` action.
- A `critical` failure can never be closed as `accept_risk`.
- A `benchmark_defect` is fixed through `dataset_fix`. Engineers never decide religious ground truth; scholar review comes first, the same rule as `review.py` and `scripts/export_eval_candidates.py`.

Unknown fields are rejected. This keeps model answers, user prompts and provider responses out of the file.

## Behavior and failure modes

`check` scores the fixture with the offline scorer and compares the annotations against that run. It exits with status 1 and prints one `FAIL:` line per problem:

| Message | Meaning / fix |
| --- | --- |
| `<id>: failed (score …) with no annotation` | Run `queue` and triage the case. |
| `<id>: case now passes; mark the annotation 'resolved' or remove it` | The annotation is stale. |
| `<id>: marked resolved but still failing` | The fix didn't hold. Reopen the annotation. |
| `<id>: duplicate annotation` / `refers to an unknown eval case` | Clean up the annotation file. |
| `<id>: <contract violation>` | Fix the field named in the message. |
| `annotation coverage … below threshold` | Coverage is below `min_annotation_coverage`. |
| `triaged rate … below threshold` | Too many annotations are still `open` (`min_triaged_rate`). |
| `N critical failure(s) still open` | More than `max_open_critical` critical failures are untriaged. |
| `fixture produced N failed cases, baseline requires M` | The scorer or fixture changed. Review before you touch the baseline. |
| `<file>:<line>: invalid JSON` / `file not found` | The input is malformed. The check reports it and doesn't crash. |

On success, stdout carries a one-line JSON summary for observability. It includes the evaluated, failed and annotated counts, the coverage, the triaged rate and per-category and per-status counts.

## Data handling

The fixture and the annotations use synthetic or redacted text only. Never commit user prompts, personal data, API keys or provider responses. The tests check for obvious secret and email markers. Reviewers can reproduce everything with `python eval_annotation.py check` and `pytest -q tests/test_eval_annotation.py`, and no secrets are needed.

## Changing the baseline

Raise `min_failed_cases` when you add failing cases, and bump `version`. The owner listed in `baseline.json` must approve any threshold change.

## Rollback

Revert the commit that changed `cases.jsonl`, `annotations.jsonl` or `baseline.json`. The workflow is tooling for the eval data and CI only: it doesn't touch the runtime service, its configuration or the deployment. Rolling back has no effect on production traffic.
