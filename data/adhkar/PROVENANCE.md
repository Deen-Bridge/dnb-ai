# Adhkar Corpus — Data Provenance

## Summary

`data/adhkar.json` is a curated, hand-verified collection of Islamic
supplications (adhkar / du'a) used by the `/adhkar/recommend` endpoint.

## Sources

| Source | Language | Rights |
|---|---|---|
| *Hisnul Muslim* (Fortress of the Muslim) by Sa'id ibn ʿAli ibn Wahf al-Qahtani | Arabic + English translation | Public domain — widely republished with no copyright claim by the author |
| *Sahih al-Bukhari* (hadith collection) | Arabic | Classical text, public domain |

Translations are taken from widely-used, freely-available English renderings
that have been used in Islamic educational contexts without restriction.

## Grading policy

Only supplications classified as **authentic** (ṣaḥīḥ) or **good** (ḥasan)
by mainstream Sunni hadith scholarship are included. No weak (ḍaʿīf) or
fabricated (mawḍūʿ) supplications appear in this corpus. The `grading` field
on each entry records the applied classification.

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

## Versioning

The file carries a top-level `version` field (semantic versioning,
`MAJOR.MINOR.PATCH`) and a `schema_version` integer.

- **PATCH** bump: wording corrections, transliteration fixes, or reference corrections that do not add or remove entries.
- **MINOR** bump: new entries added, or non-breaking schema additions.
- **MAJOR** bump: entries removed, schema fields removed or renamed.

Bump `last_updated` to the ISO-8601 date of the change whenever the file is edited.

## Rollback

Because the file is bundled in the repository, rolling back is a git revert.
The `loaded_at` field in every API response shows when the in-memory copy was
loaded; a process restart is required after updating the file on disk.

## Malware / supply-chain note

This file contains only JSON-encoded Arabic/Latin text and integers.
No executable code, URLs, or external dependencies are embedded.
Reviewers can diff any update against the previous commit to verify no
executable or binary content was introduced.
