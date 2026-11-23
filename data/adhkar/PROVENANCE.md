# Adhkar Corpus — Data Provenance

## Summary

`data/adhkar.json` is a curated, hand-verified collection of Islamic
supplications (adhkar / du'a) used by the `/adhkar/recommend` endpoint.

This document is the authoritative provenance record for the corpus. It
defines the sources, grading policy, schema, versioning rules, verification
process, and operational implications (configuration, observability, and
rollback) required for production readiness under audit item AI-066.

## Ownership and contract

- **Data owner:** the DeenBridge AI knowledge team. Changes require review by
  at least one maintainer with Islamic-studies review responsibility in
  addition to the standard engineering review.
- **Consumer contract:** `/adhkar/recommend` returns only entries that
  conform to the schema below. Every returned entry carries its `source`,
  `reference`, and `grading`, so responses are traceable to versioned data.
- **Uncertainty:** where a translation or transliteration is contested or
  where scholarly grading differs across traditions, the entry must not be
  added until the disagreement is resolved or explicitly noted in a
  `notes` field (a MINOR schema addition).

## Sources

| Source | Language | Rights |
|---|---|---|
| *Hisnul Muslim* (Fortress of the Muslim) by Sa'id ibn ʿAli ibn Wahf al-Qahtani | Arabic + English translation | Public domain — widely republished with no copyright claim by the author |
| *Sahih al-Bukhari* (hadith collection) | Arabic | Classical text, public domain |

Translations are taken from widely-used, freely-available English renderings
that have been used in Islamic educational contexts without restriction.
The specific translation edition and its publisher must be recorded in the
change commit message whenever translation text is added or replaced.

## Grading policy

Only supplications classified as **authentic** (ṣaḥīḥ) or **good** (ḥasan)
by mainstream Sunni hadith scholarship are included. No weak (ḍaʿīf) or
fabricated (mawḍūʿ) supplications appear in this corpus. The `grading` field
on each entry records the applied classification.

Entries whose grading is disputed among scholars are excluded until the
dispute is documented in the entry's `notes` field and reviewed.

## Schema

Each entry carries the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Stable unique identifier (e.g., `morning_1`). Never reused after deletion. |
| `title` | string | yes | Short English title. |
| `arabic` | string | yes | Arabic text with full tashkeel (diacritics). |
| `transliteration` | string | yes | Latin-script transliteration. |
| `translation` | string | yes | English meaning. |
| `category` | string | yes | Functional category (e.g., `morning`, `evening`, `before_sleep`, `distress`). |
| `source` | string | yes | Collection name (e.g., `Hisnul Muslim`, `Sahih al-Bukhari`). |
| `reference` | string | yes | Specific reference within the source (book number, hadith number, or entry number). |
| `grading` | string | yes | Authenticity grade: `authentic` (ṣaḥīḥ) or `good` (ḥasan). |
| `notes` | string | no | Optional scholarly or translation caveat (added in MINOR schema change). |

## Input validation and failure modes

The loader must treat the corpus as untrusted-by-default structure:

- Missing required fields, duplicate `id` values, or unknown `grading`
  values cause the file to be rejected at load time; the service must fail
  closed and report the failure through its health/status endpoint rather than
  serving partial data.
- Empty `category` filters return an empty result set with an explicit
  "no matches" status, not a fallback to unrelated entries.
- Arabic text must be stored in Unicode NFC form with tashkeel present;
  transliteration and translation must be non-empty strings.
- Language-specific edge cases (right-to-left rendering, mixed Arabic/Latin
  content in `reference` fields) are validated by the test suite rather than
  silently normalized.

## Versioning

The file carries a top-level `version` field (semantic versioning,
`MAJOR.MINOR.PATCH`) and a `schema_version` integer.

- **PATCH** bump: wording corrections, transliteration fixes, or reference corrections that do not add or remove entries.
- **MINOR** bump: new entries added, or non-breaking schema additions (such as `notes`).
- **MAJOR** bump: entries removed, schema fields removed or renamed.

Bump `last_updated` to the ISO-8601 date of the change whenever the file is edited.

## Verification and release evidence

Each corpus change must attach, in the pull request:

- The implementing commit hash.
- CI links for schema validation and unit tests covering the loader.
- A diff summary showing added, removed, and modified entry `id`s.
- A changed-file malware/supply-chain scan result (see below).
- A redacted verification note confirming the entries were checked against
  the cited reference. Secrets, tokens, and private provider responses must
  never be attached.

## Observability

- Every API response includes `loaded_at` (time the in-memory copy was loaded)
  and the corpus `version`, so operators can correlate responses with the
  deployed data.
- Load failures and schema validation failures are logged at error level with
  the offending entry `id` and field name only; entry text is not logged.

## Configuration and data handling

- The corpus path is bundled with the service and is not configurable per
  request.
- The corpus contains no personal data and no per-user state. User-scoped
  data, if any, is stored separately and is outside the scope of this file.

## Rollback

Because the file is bundled in the repository, rolling back is a git revert.
The `loaded_at` field in every API response shows when the in-memory copy was
loaded; a process restart is required after updating the file on disk.
Rollback must restore the previous `version` and `last_updated` values along
with the entries.

## Malware / supply-chain note

This file contains only JSON-encoded Arabic/Latin text and integers.
No executable code, URLs, or external dependencies are embedded.
Reviewers can diff any update against the previous commit to verify no
executable or binary content was introduced. Any update must pass the
changed-file malware scan before merge, and the scan result is recorded in
the pull request.
