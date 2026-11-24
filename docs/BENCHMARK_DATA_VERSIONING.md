# Benchmark data versioning and provenance

Owner: AI quality & evaluation maintainers (recorded as `owner` in the manifest).

Every committed evaluation dataset is pinned in `data/eval/benchmark_manifest.json` and verified
offline by `benchmark_manifest.py`. The check needs no API keys, network or provider responses, so
any reviewer can reproduce it from a clean clone.

## Contract

Each dataset entry records:

| Field | Meaning |
| --- | --- |
| `path`, `format` | Repo-relative file (`json` or `jsonl`). Paths outside the repo are rejected. |
| `records_key` | For `json` files: the list that holds the records (e.g. `results`, `examples`). |
| `version` | Semantic version of the data (`MAJOR.MINOR.PATCH`). |
| `sha256` | Hash of the file with CRLF folded to LF, so Windows and Linux checkouts agree. |
| `records`, `min_records` | Pinned record count and the floor it may never drop below. |
| `source`, `license` | Where the data came from (builder script, issue, curation) and its license. |
| `contains_personal_data` | Must be `false`. |
| `history` | Append-only `{version, sha256}` list; the last entry must match the current pin. |

The manifest also carries `owner`, `rollback`, `scan_dirs` (directories where every `.json`/`.jsonl`
file must be versioned) and `ignore` (glob patterns for local outputs such as
`data/eval/live_results*.json`).

## Failure modes

`python benchmark_manifest.py` prints a one-line JSON summary on stdout and exits 1 with one
`FAIL:` line per problem on stderr:

- **Data changed without a version bump.** The hash differs from the pin. Bump `version`, then run
  `python benchmark_manifest.py --update`, or revert the data change.
- **Below threshold.** The record count fell under `min_records`.
- **Missing or malformed file.** The message names the dataset, file and (for JSONL) line.
- **Missing provenance.** An owner, rollback note, source, license or history entry is missing.
- **Secret-like content.** An email address, API key, GitHub token, Stellar secret seed or bearer
  token pattern is present. Only the kind is reported; the matched text is never echoed.
- **Unversioned file.** A new data file under `data/eval/` has no manifest entry. This includes a
  `baseline_results.json` saved by `RegressionRunner`: a committed baseline has to be versioned.

## Changing a dataset

1. Edit or regenerate the data (for example with `scripts/build_islamic_qa_benchmark.py`).
2. Bump the dataset's `version`: patch for corrections, minor for added records, major for removed
   records or a schema change. Raise `min_records` when records are added on purpose.
3. Run `python benchmark_manifest.py --update`. It re-pins `sha256` and `records` and appends to
   `history`. It refuses to run if the version was not bumped.
4. Commit the data and the manifest together.

`generate_benchmark.py` assigns `uuid4` ids, so regenerating `data/hallucination_benchmark.json`
always produces new bytes and always needs a version bump.

## Data handling

Benchmark files must be synthetic or redacted. They must not contain user prompts, personal data,
API keys, tokens or provider responses. The manifest stores only hashes, counts and provenance text.

## Observability

CI runs `python benchmark_manifest.py` and `pytest -q tests/test_benchmark_manifest.py`. The
summary line lists every dataset's version, so the CI log shows which data a run evaluated against.

## Rollback

Revert the commit that changed the dataset and its manifest entry (`git revert <sha>`). The previous
version and hash remain visible in `history`. The check is CI-only: it does not ship in the request
path and has no runtime or deployment impact.
