# Cross-reference graph consistency (AI-064, #840)

Owner: Knowledge and learning (`tafsir.py`).

The ayah cross-reference graph behind `GET /cross-references/{ayah_key}` and
`GET /cross-reference-network` is validated every time it is loaded. Edges that
break a rule are never served, and the result is published as a versioned
integrity report.

## Production behaviour

| Rule | Severity | Effect |
| --- | --- | --- |
| Record does not fit the `CrossReference` schema (missing field, unknown `reference_type`, not an object) | error | edge rejected |
| `source`/`target` is not a single `surah:ayah` key, names a surah outside 1–114 or an ayah past the surah's length (Kufan count, `data/quran/surah_index.json`) | error | edge rejected |
| Self-reference (`source == target` after normalisation) | error | edge rejected |
| Empty `context` | error | edge rejected |
| Duplicate edge: same unordered pair and type, including `B→A` after `A→B` | error | later copy rejected |
| Key needed normalising (leading zeros, spaces, Arabic-Indic `٢:٢٥٥` or Persian `۲:۲۵۵` digits) | warning | kept under the canonical ASCII key |
| Same pair stored with a different `reference_type` | warning | kept |
| Missing `source_work` (no named tafsir) or empty `commentary` | warning | kept, flagged as unattributed |

Lookups normalise the requested key the same way, so `/cross-references/٢:٢٨٢`
returns the same links as `/cross-references/2:282`.

## Failure modes

| Condition | `data_source` | Served graph |
| --- | --- | --- |
| `data/quran/cross_references.json` absent (current deployment) | `seed` | reviewed seed list in `tafsir.py` |
| File unreadable, invalid JSON, or top level not a list | `seed_fallback` | seed list |
| File valid, some edges rejected | `file` | the valid edges only. Before this change, one bad record discarded the whole file |
| File valid but every edge rejected | `seed_fallback` | seed list. An empty graph would read as "no links exist" |
| File is `[]` | `file` | empty graph (an explicit choice by the data owner) |

## Provenance and versioning

`GET /cross-reference-graph/report` returns:

- `schema_version`: the version of the validation rules (`cross-reference-graph/v1`).
- `data_source`: `file`, `seed` or `seed_fallback`.
- `data_version`: the first 16 hex characters of a SHA-256 over the accepted edges,
  canonicalised. The same data always gives the same version, and any edit changes it.
- `consistent`: `false` if any error-level issue was found.
- Edge counts and every issue, each with its code, severity and the position of the raw edge.

Each served edge keeps its `source_work`. Edges without one are listed as
`missing_provenance` in the report.

## Configuration and data handling

- No new environment variables. The data path is still `data/quran/cross_references.json`.
- The graph is built once per process, on first use (`get_cross_reference_graph`).
  To reload it after replacing the file, restart the process or call
  `get_cross_reference_graph.cache_clear()`.
- The graph is global reference data and holds no user data, so there is no
  user-scoped state to isolate or reset.

## Observability

When a load produces issues, one `WARNING` line is logged on the `tafsir`
logger. It gives the data version, the source, the accepted/total counts and a
count per issue code. Edge text (context, commentary) is never logged.

## Rollback

The change touches only `tafsir.py`. Reverting the commit restores the previous
loader. The routes and response shapes of `/cross-references/{ayah_key}` and
`/cross-reference-network` are unchanged. The only addition is the new report
endpoint. The seed edge `2:255 → 3:2` changes type from `repeated_story` to
`thematic_parallel`, because neither ayah narrates a story.

## Reproducing

```bash
pytest -q tests/test_cross_reference_graph.py
```

The tests are offline and need no secrets. They cover each rule, Arabic/English
key edge cases, the empty and fallback states, determinism of the data version,
the HTTP routes, and an evaluation of the graph that ships.
