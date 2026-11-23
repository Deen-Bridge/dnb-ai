# Quran concordance relevance and explainability

The Quranic concordance (`concordance.py`, mounted at `/concordance`) turns the
bundled theme taxonomy and `data/theme_verses.json` mappings into topic
discovery. This note records the relevance filtering, the per-verse explanation
contract, the dataset provenance, and the rollback plan.

## Behavior and failure modes

- **Topic search/browse** accepts English or Arabic queries; search is covered by
  the existing taxonomy normalizer and tests.
- **Browse relevance filters:** `min_relevance`, `context_type`
  (`primary`/`secondary`), and now `surah` narrow the verse list. Filters compose,
  and the returned `verses` are still ordered by relevance.
- **Surah filtering** was ported from `dev` (#492). Values outside 1–114 are
  rejected with 422; a surah with no mapped verses returns `verses: []` and a
  200, leaving the theme's `total_verses` unchanged.
- **Unknown topics** return 404 with a hint to list/search topics first.
- **Multi-topic queries** keep AND/OR semantics; an invalid operator is a 422.

## Versioned data and provenance

Every concordance response now carries `provenance`, read from the mapping
store's `dataset_info()`:

- `dataset_version`, `schema_version`, `source`, `curation`
- `content_sha256` (detects silent data edits)
- `mapping_count`, `skipped_invalid`
- `uncertainty_note` — states that mappings are a curated, non-exhaustive aid and
  that relevance scores are editorial estimates, not scholarly rulings.

This exposes the exact bundled revision that produced a result instead of
implying the mapping is exhaustive or authoritative.

## Explainability

Each `VerseHit` now reports:

- `theme_id` — the theme whose mapping matched the verse;
- `relevance_score` plus a coarse `relevance_label` (`high` ≥ 0.8,
  `moderate` ≥ 0.5, `low` otherwise);
- `context_type` — primary or secondary;
- `annotation` — the curated note, when present;
- `explanation` — a plain-language sentence naming the verse, theme, context,
  relevance band, and any annotation.

## Data handling

- Fully offline and deterministic: the taxonomy and curated mapping are bundled;
  no network, model, or API key is required.
- No user-scoped state or prompts are stored; the module is read-only over the
  bundled data, so there is nothing to isolate or reset per user.
- Optional surah names in frequency rows come from the bundled corpus index, so
  the response stays offline.

## Observability

Watch `provenance.content_sha256`/`dataset_version` (unexpected data drift),
`provenance.skipped_invalid` (malformed mappings at load), `uncertainty_note`
(curated, non-exhaustive coverage), and `relevance_label` distribution per
query. Empty `verses` with a non-zero `total_verses` indicates an active filter
(including `surah`).

## Reproduction

```bash
python -m pytest tests/test_concordance.py -q
```

All tests run offline against the bundled data and the ASGI app; no secrets or
network are needed.

## Malware scan

No executables or new dependencies are introduced. CI runs the repository's
`PR Malware Guard` on every changed file; its detector tests can be run locally
with:

```bash
node --test .github/scripts/pr-malware-guard.test.cjs
```

## Rollback

Revert the commit that changed `concordance.py`, the concordance tests, and this
note. Changes are additive response fields plus one optional query parameter, so
no data migration is required and existing clients keep working.
