# Tafsir comparison and source boundaries

`POST /tafsir/compare` compares one ayah across at least ten named tafsir works,
clusters the retrieved passages deterministically, and reports consensus,
divergence, and singleton observations. This note records the production
contract, how every citation is tied to its source boundary, and the rollback
plan.

## Behavior and failure modes

- **One ayah per comparison:** a range such as `103:1-2` is rejected with 400 —
  mixing passages that cover different ranges would break boundary attribution.
- **Work budget:** the request accepts 10–12 distinct works; the default set is
  every registered work except al-Jalalayn (which is available on request), so
  the default stays inside the comparison budget.
- **Graceful degradation:** a work with no entry for the ayah, or one that is not
  published in a usable language, is reported in `unavailable` and never treated
  as evidence.
- **Model optional:** the response is fully deterministic and offline. A model is
  used for the prose synthesis only when `GEMINI_API_KEY` is set; if it is absent
  or returns invalid citations, the deterministic fallback is returned.
- **Citation guardrails:** generated claims that cite unknown source keys, too few
  keys, or non-distinct divergence positions are rejected and replaced by the
  deterministic analysis.

## Source boundaries

Every retrieved passage carries `verse_range` when the source states it (for
example `103:1-3`). The response makes that boundary explicit in four places:

1. `sources[*].verse_range` — the range the retrieved passage covers.
2. `boundaries` — a map of every retrieved source key to its boundary. A source
   that states no boundary is cited against the compared ayah, so a claim is
   never left unbounded.
3. `consensus[*].source_boundaries`, `divergences[*].positions[*].source_boundaries`,
   and `unique_insights[*].source_boundaries` — each citation carries the boundary
   of every source key it names.
4. `synthesis.source_boundaries` plus a `Passage boundaries: key=range` line in
   `synthesis.balance_note`.

The synthesis prompt also labels each evidence passage with `covers=<range>` and
instructs the model never to attribute a passage that covers a different ayah
range without noting the boundary.

## Provenance and uncertainty

`provenance` reports the ruleset version, the deterministic clustering algorithm,
the similarity threshold in effect, and the boundary policy. `uncertain` is true
when no high-support consensus grouping was found, so a comparison is never
presented as more settled than its evidence.

## Data handling

- Retrieval is the existing tafsir pipeline; text is cached per `slug|ayah` (and
  verse text per `ayah|language`) through the keyed cache. The cache holds only
  public tafsir text — no user data, prompts, or secrets.
- No user-scoped state is persisted; the cache is resettable with the existing
  keyed-cache clear used by tests (`get_keyed_cache("tafsir").clear()`).
- The optional synthesis model receives public tafsir text and ayah metadata only.

## Observability

Watch `uncertain` (no consensus), `unavailable` (degraded coverage),
`provenance.similarity_threshold` (clustering drift), and `boundaries` (a source
whose passage covers a wider range than the compared ayah). Per-source
`verse_range` distinguishes a single-ayah reading from a multi-ayah passage.

## Reproduction

```bash
python -m pytest tests/test_tafsir.py tests/test_tafsir_comparison.py -q
```

All tests run offline against `FakeTafsirSource`; no network, no model, no API
key.

## Malware scan

No executables or new dependencies are introduced. CI runs the repository's
`PR Malware Guard` on every changed file; its detector tests can be run locally
with:

```bash
node --test .github/scripts/pr-malware-guard.test.cjs
```

## Rollback

Revert the commit that changed `tafsir.py`, added the comparison tests, and added
this note. The endpoint and models are additive, so removing them restores the
previous behavior with no data migration; cached entries remain valid.
