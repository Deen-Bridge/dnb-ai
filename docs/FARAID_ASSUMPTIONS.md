# Faraid assumptions and edge cases

`faraid.py` is the offline Islamic inheritance (faraid) calculator exposed at
`POST /faraid`, with its rule-set assumptions available at
`GET /faraid/assumptions`. This note records the production contract, the
versioned provenance, the uncertainty handling, and the rollback plan.

## Behavior and failure modes

- **Exact arithmetic:** every share is a `fractions.Fraction`; monetary amounts
  are derived from the fraction with `Decimal` and rounded to whole units.
- **Input validation:** unknown heir labels, an empty heir list, more than one
  husband, a husband and wife recorded together, more than 100 heirs, a
  non-positive estate, and `count < 1` are rejected (400/422) with an actionable
  hint. Duplicate heir groups are aggregated, never silently overwritten.
- **Arabic/English labels:** heir types are accepted in English, common
  transliterations, legacy names (`grandson`, `paternal_grandfather`, ...), and
  Arabic script (e.g. `زوجة`, `ابن`, `بنت`). Labels are Unicode-normalized
  (diacritics, alef/ya/ta-marbuta variants) before lookup.
- **Empty/failure states:** the pure `distribute()` function raises `ValueError`
  for a non-positive estate or no heirs; the HTTP layer maps these to 4xx.

## Versioned provenance and uncertainty

- `FARAID_RULESET_VERSION` (semantic version), `FARAID_MADHAB`
  (`sunni_majority`), `FARAID_BASIS`, and `FARAID_SOURCES` pin the rule set.
- Every `POST /faraid` response includes `provenance` (`ruleset_version`,
  `madhab`, `basis`, `sources`) plus `uncertain` and `warnings`.
- `GET /faraid/assumptions` returns a `FaraidAssumptions` record with the
  version, supported heirs, the six Quranic fractions, the awl/radd/hajb rules,
  the known uncertainty rules, and `valid`/`errors` from invariant checks.
- The invariant checks recompute known cases (wife with descendants, awl,
  radd, hajb) and fail `valid` if any rule drifts. The validator never raises.

## Rules implemented (majority Sunni position)

- Quranic fixed shares for spouse, parents, daughters, grandmother, and sisters
  (Surah an-Nisa 4:11-12, 4:176), including the conditional spouse share
  (1/8 vs 1/4, 1/4 vs 1/2), the mother's 1/6 vs 1/3, and daughter/sister
  aggregation (1/2 vs shared 2/3).
- Residuary (`asaba`) distribution with the male:female 2:1 ratio, including
  daughters/granddaughters with sons, and full sisters with a full brother or
  with daughters (asaba ma'a al-ghayr).
- Blocking (`hajb`) of nearer relatives, and proportional reduction (`awl`) and
  return of surplus (`radd`, spouse excluded).
- Multiple wives share the collective spouse share; multiple grandmothers share
  the single 1/6.

## Known uncertainty (flagged in the response)

- **Grandfather with siblings (al-jadd wa al-ikhwa):** simplified residuary
  treatment; the schools differ, so the response is marked `uncertain`.
- **Unallocated residue:** when there is no eligible residuary or radd
  recipient (for example, only a spouse), the remainder is reported as
  unallocated (bayt al-mal / dhawu al-arham depending on jurisdiction).
- **Out of scope:** debts, bequests (wasiyyah), maternal half-siblings, and
  distant heirs (dhawu al-arham) are not modelled.

## Data handling

- Fully offline and deterministic; no network, secrets, or API keys.
- Stateless: the engine keeps no user-scoped or cached state, so there is
  nothing to isolate, migrate, or reset between users.

## Observability

Watch `provenance.ruleset_version` (unexpected drift), `uncertain` (result needs
review), `warnings` (why), `awl_applied`/`awl_denominator`, `radd_applied`, and
`hajb_applied`. `GET /faraid/assumptions.valid` is a cheap startup/audit probe.

## Reproduction

```bash
python -m pytest tests/test_faraid.py tests/test_faraid_assumptions.py -q
```

All tests are offline and use committed code only.

## Malware scan

No executables or new dependencies are introduced. CI runs the repository's
`PR Malware Guard` on every changed file; its detector tests can be run locally
with:

```bash
node --test .github/scripts/pr-malware-guard.test.cjs
```

## Rollback

Revert the commit that changed `faraid.py`, `main.py`, the two faraid test files,
and this note. The change only wires in a previously unmounted router and adds
response fields, so no data migration is required. Removing the
`faraid_router` include restores the previous (unmounted) behavior.
