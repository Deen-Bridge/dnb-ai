# Public Knowledge Catalog and Dataset Update Governance (#848)

- **Audit ID**: PROD-AUDIT AI-072
- **Scope**: the public, non-user-specific knowledge this service serves: the
  bundled datasets under `data/`, the runtime sources the knowledge modules
  read, and the `public_knowledge.py` discovery catalog. It also covers the
  boundary between that public data and user-scoped learner state.
- **Review locations**: `hadith*.py`, `tafsir.py`, `concordance.py`,
  `faraid.py`, `study.py`, `memorization_planner.py`, plus `public_knowledge.py`
  and `study_store.py`.
- **Owner**: to be confirmed by the `dnb-ai` maintainers in review. The roles in
  section 3 are written so that each can be assigned to a named team.

## 1. Inventory of public knowledge sources

| Source | Served by | Canonical upstream | License / rights | Version pin | Provenance record | Integrity check | How uncertainty is expressed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `data/hadith/<collection>.json` (10 collections) | `hadith.py`, `hadith_research.py`, `citations.py`, `page_analysis.py` | `fawazahmed0/hadith-api` | The Unlicense (public domain); grading metadata only, no translated text | `source.ref = "1"`, `source.fetched = 2026-08-30` in every file | `data/hadith/PROVENANCE.md` | `scripts/build_hadith_data.py`; `tests/test_hadith.py`; `tests/test_knowledge_data_provenance.py` | Weakest grade wins across graders; `UNKNOWN` where no grade exists; chain type kept separate from strength |
| `data/hadith/variants.json` | `hadith_research.py` | Curated | Bibliographic facts | `version` field | `provenance` field | `tests/test_knowledge_data_provenance.py` | Variants listed side by side, not merged |
| `data/rijal/narrators.json` | `hadith_research.py` | Taqrib al-Tahdhib, Siyar A'lam al-Nubala', Mizan al-I'tidal | Bibliographic facts | `version: 1` | `provenance` field | `tests/test_knowledge_data_provenance.py` | Declared advisory and non-exhaustive; the more cautious reading is the headline where critics disagree |
| `data/quran/surah_index.json` | `tafsir.py`, `citations.py`, `page_analysis.py`, `memorization_planner.py` (cross-checked) | Mushaf facts (Kufan count) | Facts, no copyright | Build script asserts 114 surahs and 6,236 ayat | `data/quran/PROVENANCE.md` | `scripts/build_surah_index.py`; `tests/test_knowledge_data_provenance.py` | Not applicable: closed reference table |
| `memorization_planner._SURAH_DATA` | `memorization_planner.py` (`/memorization/surahs`) | Embedded copy of the surah index plus difficulty and juz data | As above | Code | This document | `tests/test_knowledge_data_provenance.py` asserts it matches the canonical index | Difficulty tiers are heuristic estimates |
| `data/theme_verses.json` + `.meta.json` | `thematic_quran.py`, `concordance.py` | Curated from the tafsir/concordance tradition | Curation | `dataset_version: 1.0.0`, `schema_version: 1` | `theme_verses.meta.json` | `python scripts/build_theme_mappings.py --check` (CI) | Declared "Not exhaustive" |
| `data/adhkar.json` | `adhkar.py` | Hisnul Muslim; Sahih al-Bukhari | Public domain | `version: 1.0.0`, `last_updated: 2024-01-01` | `data/adhkar/PROVENANCE.md` | `tests/test_adhkar.py`; `tests/test_knowledge_data_provenance.py` | `grading` per entry; only sahih/hasan included |
| `data/quran_uthmani.json` | `corpus.py` | Uthmani text | Public domain text | **None**; holds surahs 1, 2 and 114 only | `citations.py` docstring | None | Callers must not use it for bounds checks (see `citations.py`) |
| Ayah text and translations (runtime) | `tafsir.py` | Quran.com API v4 (`QURAN_API_BASE`) | Per Quran.com terms | API version `v4` | `data/quran/PROVENANCE.md` | Request-time; failures return "unavailable" | Missing text is reported, never filled in |
| Tafsir text (runtime) | `tafsir.py` | `spa5k/tafsir_api` via jsDelivr (`TAFSIR_API_BASE`) | Per upstream | **`@main`, not pinned** | `tafsir.py` comments, `TAFSIR_REGISTRY` | Request-time | Each excerpt is attributed to a named work and author |
| Inheritance rules | `faraid.py` (router **not mounted** in `main.py`) | Qur'an 4:11, 4:12, 4:176; Sunni majority position | Scripture and fiqh | Code | `faraid.py` constants | `tests/test_faraid.py` (added here) | `DISCLAIMER` on every result; see the critical finding in section 10 |
| Study curriculum | `study.py` | Curated topics | Curation | `CURRICULUM_VERSION = "1.0.0"`, generator version | Response `provenance` metadata | `tests/test_study.py` | Provenance and seed returned with each generation |
| Discovery catalog | `public_knowledge.py` | Supplied by the caller per record | Per record (`license`) | Catalog `version` and per-record `dataset_version` | Per-record `source`, `reviewed_by`, `last_reviewed` | `tests/test_public_knowledge.py` | Per-record `caveat` |

## 2. The catalog contract (`public_knowledge.py`)

A record is searchable only when **all** of these hold. Anything else fails
closed.

| Rule | Failure |
| :--- | :--- |
| `published` is true | dropped at construction and not kept; only counted in `unpublished_count` |
| `kind`, `title`, `source`, `license`, `reviewed_by` are non-blank | listed in `rejected` with the reason |
| `dataset_version` is a label such as `1`, `1.0.0` or `2026-08-30` (no spaces or markup) | rejected |
| `last_reviewed` is an ISO date, not in the future | rejected |
| `url` is a site path (`/events/1`) or an `https://` URL | rejected; `javascript:`, `data:`, `http:` and `//host` are refused |
| `(kind, url)` was not already accepted | the later duplicate is rejected |

Rejected entries expose only `kind`, `url` and `reasons`. Their title and
summary are withheld, because a record that failed review must not leak its
content through the audit list.

Search keeps the original behavior. Every query term must appear in the title
or summary. An empty query returns nothing, so an empty search never lists the
catalog. Records come back in insertion order, and `kind` filters exactly.
Matching is now case-insensitive across scripts:

- Latin transliteration marks are ignored: `Ṣaḥīḥ` matches `sahih`, and
  `Qurʾān`, `Qur'an` and `Quran` match each other;
- Arabic tashkeel, tatweel and presentation forms are folded (`ﷲ` matches
  `الله`);
- alef, ya and ta-marbuta variants fold through
  `quran_recitation.normalize_arabic`, the normalizer the recitation module
  already uses.

Queries over 256 characters or 16 distinct terms raise `ValueError`.

Every result is the full record, so every output carries its `source`,
`license`, `dataset_version`, `reviewed_by`, `last_reviewed` and `caveat`.
`reviewed_by` names a role or team (for example `content-review`), never a
person's name or email.

## 3. Roles

| Role | Responsibility |
| :--- | :--- |
| **Source owner** | Keeps one dataset or runtime source current: pins the upstream, runs its build script, maintains its provenance record. |
| **Content reviewer** | Checks religious substance: grades, rulings, attributions, translations, caveats. Required for any change to hadith grades, rijal reliability, adhkar grading, faraid rules, or theme mappings. |
| **Maintainer** | Reviews the code and data diff, confirms the checks in section 4 passed, merges, and owns the deployment. A maintainer may not be the sole reviewer of their own data change. |

## 4. Update and publication workflow

1. **Propose.** Open an issue naming the dataset, the upstream ref or source,
   and why it is changing (correction, addition, upstream release).
2. **Build, never hand-edit generated data.** Regenerate from the pinned
   upstream with the dataset's build script (`scripts/build_*.py`). Curated
   files (`adhkar.json`, `narrators.json`, `variants.json`) are edited by hand
   in a dedicated PR that changes nothing else.
3. **Version.** Bump the dataset's version: patch for corrections, minor for
   additions, major for schema or meaning changes. Update the pinned
   `source.ref` and `fetched` date, or `last_updated`.
4. **Record provenance.** Update the dataset's `PROVENANCE.md` or `provenance`
   field: upstream, ref, license, what is and is not bundled, and why.
5. **Check.** All of these must pass in CI:
   `tests/test_knowledge_data_provenance.py`, the dataset's own tests, and any
   `--check` script, for example
   `python scripts/build_theme_mappings.py --check`.
6. **Review.** One maintainer, plus a content reviewer for the changes listed
   in section 3.
7. **Publish.** Merge to `main` and deploy. For catalog records, publish by
   setting `published=True` with complete provenance in the same change.
   Records missing provenance cannot be served, so a partial publish fails
   closed.

## 5. Freshness

| What | Cadence |
| :--- | :--- |
| Catalog records | Re-review at least every 180 days. `PublicKnowledgeCatalog.needs_review(as_of=..., max_age_days=180)` lists overdue records. They stay searchable: staleness feeds the review queue, it does not hide content. |
| Pinned upstream datasets (hadith, surah index) | Check the upstream for a new release each quarter; adopt it only through section 4. |
| Curated datasets (adhkar, rijal, theme verses) | Review once a year. `adhkar.json` reports `last_updated: 2024-01-01`, so it is due now. |
| Runtime sources (Quran.com, tafsir CDN) | Pin a version where the upstream allows it; check endpoint health in monitoring. |

## 6. Uncertainty and failure handling

- Unknown or missing grades are reported as `UNKNOWN` or unverified, never as
  authentic.
- A missing bundled file degrades to "not found" or "unverified" rather than an
  invented answer (`BundledGradingSource.get` returns `None`; `corpus.py` loads
  an empty corpus).
- Runtime source failures (timeouts, 4xx/5xx) are logged with the path only and
  reported as unavailable. Text is never substituted.
- An invalid catalog record is excluded and reported in `rejected`; the rest of
  the catalog keeps serving.
- Faraid results always carry the disclaimer and name the juristic basis.

## 7. Public data and user-scoped data

Public knowledge and learner state are separate by construction:

- `public_knowledge.py` and every dataset in section 1 hold no user
  identifiers and are identical for every caller. Personal data must never be
  added to them.
- Learner state lives only in `study_store.py`. It is opt-in
  (`opt_in_personalization` defaults to false), keyed by an opaque learner ID,
  and stored in Redis with a one-year TTL when `REDIS_URL` is set, or otherwise
  in a 5,000-entry in-memory LRU. `DELETE /study/users/{user_id}` erases it.
  That reset is idempotent and leaves other learners untouched; both properties
  are tested.
- **Fixed in this change:** whitespace-only learner IDs used to collapse onto
  one shared key, `study:user:`, so different anonymous callers could read and
  reset each other's progress. Blank IDs are now treated as anonymous (not
  tracked), the store refuses them, and the progress routes reject IDs that are
  blank or padded with whitespace with 422. The reset log line used to print
  the first eight characters of the raw ID while calling it a hash; it now logs
  a real SHA-256 prefix.
- `memorization_planner.py` stores no learner state: callers pass prior
  progress in each request.
- **Isolation depends on the caller.** The service does not authenticate
  learners. The calling backend must authenticate the learner and pass only
  that learner's ID. `main.verify_api_key` exists, but on `main` it is not
  attached to any route; see section 10.

## 8. Observability

Log counts and identifiers of operations, never content or personal data
(`docs/RETRIEVAL_LOGGING.md`). Useful signals: catalog size, rejected and
unpublished counts at load, and the size of the `needs_review` queue;
runtime-source error rates by host; study reset counts by hashed key.

## 9. Changes in this PR

- `public_knowledge.py`: the provenance contract, fail-closed validation, the
  `rejected` audit list, catalog `version`, Arabic- and transliteration-aware
  search, query bounds, and the `needs_review` freshness queue.
- `memorization_planner.py`: all 114 Arabic surah names were stored as mojibake
  (UTF-8 decoded as Latin-1; `الفاتحة` was stored as `Ø§Ù„ÙØ§ØªØ­Ø©`) and
  served that way by `/memorization/surahs`. They are replaced with the
  canonical names from `data/quran/surah_index.json`. Five milestone badges
  with corrupted emoji are repaired the same way. The repair was mechanical
  (cp1252 bytes re-decoded as UTF-8), and every repaired name equals the
  canonical index exactly.
- `study.py` / `study_store.py`: the user-ID isolation fixes from section 7.
- `faraid.py`: fails closed instead of answering when an heir type is listed
  more than once (two daughters used to produce one share of 9/8 of the
  estate), and refuses any result whose shares exceed the estate. No ruling
  was changed.
- Tests: `tests/test_public_knowledge.py` (rewritten for the contract),
  `tests/test_study_store.py`, `tests/test_knowledge_data_provenance.py`,
  `tests/test_faraid.py`, and new isolation cases in `tests/test_study.py`.

**Reproduce** without secrets or network:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q tests/test_public_knowledge.py tests/test_study_store.py tests/test_faraid.py \
          tests/test_knowledge_data_provenance.py tests/test_study.py tests/test_memorization_planner.py
```

**Deployment**: code-only. There is no migration, configuration or new
dependency. Client impact:

1. `PublicRecord` keeps its first five positional fields; the provenance
   fields are keyword arguments with empty defaults. A published record built
   without provenance is now excluded from search. Nothing on `main`
   constructs a catalog outside tests, so no runtime caller is affected.
2. `/memorization/surahs` now returns correct Arabic names and badges. Any
   client that compared against the corrupted strings must update.
3. `faraid.distribute` raises `ValueError` (the route maps it to 400) for a
   repeated heir type. The route is not mounted, so no client is affected.
4. `/study/generate` with a whitespace-only `user_id` no longer records
   progress. `GET` and `DELETE /study/users/{user_id}` return 422 for blank or
   whitespace-padded IDs, instead of operating on the shared blank key.

**Rollback**: revert the PR. Nothing is migrated. After a rollback the catalog
serves records without provenance again, the planner serves the corrupted
names again, and blank learner IDs share one key again. Before rolling back,
erase the shared blank-ID record if one exists (`DELETE` the raw key
`study:user:` in Redis); the routes cannot reach it once this change ships.

## 10. Findings left for follow-up

- **Critical: faraid rulings.** `faraid.py` applies fixed base shares no matter
  which other heirs are present. With an estate of 1,200 it gives a wife 1/4
  alongside a son (Qur'an 4:12: 1/8), a husband 1/2 alongside a daughter
  (4:12: 1/4), a son and daughter equal halves (4:11: two to one), and a
  mother 1/2 with only the father (4:11: 1/3). `tests/test_faraid.py` records
  these as strict expected failures. The router is not mounted in `main.py`;
  keep it unmounted until the rules are corrected and content-reviewed in a
  dedicated issue.
- **Unpinned tafsir source.** `TAFSIR_API_BASE` defaults to
  `spa5k/tafsir_api@main`, so upstream changes reach production unreviewed.
  Pin a tag or commit (overridable through the environment) after confirming
  which one is current.
- **Partial, unversioned corpus.** `data/quran_uthmani.json` has no version or
  source metadata and holds three surahs. Version it, or document it as a
  fixture.
- **No service authentication on routes.** `verify_api_key` is defined in
  `main.py` but not attached to any route, including the per-learner study
  routes. Confirm the intended deployment boundary and attach it where needed.
- **Embedded surah table.** `memorization_planner._SURAH_DATA` duplicates the
  canonical surah index. It is now tested for agreement, but deriving the
  shared fields from `data/quran/surah_index.json` would remove the duplicate.
- **Arabic-script names in `hadith.py`.** Collection aliases are Latin-only, so
  Arabic citations of collections are not recognized anywhere (see
  `docs/PAGE_ANALYSIS_CONTRACT.md`).
