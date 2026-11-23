# Narrator biography and isnad analysis provenance

The Hadith Research Agent (``hadith_research.py``) resolves narrator names
against the bundled rijāl (narrator biography) reference and analyzes a supplied
chain (isnad) for continuity and reliability. This note records the contract,
data provenance, uncertainty handling, and rollback for that capability.

## Behavior and failure modes

- **Narrator lookup** folds Arabic script and English transliterations to one
  key (Unicode normalization, Arabic tashkeel/accent stripping, alef/ya/
  ta-marbuta folding, ``bin``/``b.`` → ``ibn``, punctuation removal). Exact
  name/alias hits are high confidence.
- **Partial matches** are accepted only when a known form appears as a whole
  contiguous run of words inside a longer phrase. They are reported as
  ``match_method: "partial"`` with ``match_confidence: 0.6`` and add a warning;
  short aliases such as ``ali`` no longer match inside an unrelated name such as
  ``Khalid``.
- **Unknown narrators** are reported as ``matched: false`` and set ``flagged``
  and ``uncertain`` rather than being silently accepted.
- **Continuity** is checked on adjacent *known* narrators: a reversed generation
  order or a multi-generation jump is flagged. With fewer than two resolvable
  narrators, ``continuous`` is ``null`` and the result is marked ``uncertain``.
- **Empty/failure states:** an empty chain returns no narrators with
  ``continuous: null``; the ``POST /hadith-research/chain`` endpoint rejects an
  empty ``narrators`` list with HTTP 422. Blank/whitespace names never resolve.

## Versioned data and provenance

- Data file: ``data/rijal/narrators.json`` (``version``, ``owner``,
  ``last_updated``, ``provenance``, ``disclaimer``).
- Facts are drawn from classical rijāl references (Ibn Hajar's *Taqrib
  al-Tahdhib*, al-Dhahabi's *Siyar A'lam al-Nubala'* and *Mizan al-I'tidal*).
  The set is a curated, advisory starter subset, not an exhaustive dictionary.
- Every isnad response includes a ``provenance`` object with ``dataset``,
  ``version``, ``source``, ``narrator_count``, and ``disclaimer``. Each matched
  profile also carries ``source`` (``data/rijal/narrators.json v<version>``),
  ``match_method``, and ``match_confidence``.

## Uncertainty

``uncertain`` is ``true`` when any narrator is unknown, matched only partially,
or lacks a generation label, or when too few narrators resolve to judge
continuity. Callers must treat ``uncertain: true`` as "needs verification", not
as an authenticity verdict. All output remains advisory regardless of this flag.

## Data handling

- Fully offline and deterministic; no network, model, or external lookup. No
  API keys or secrets are used or required to reproduce behavior.
- The module is read-only and keeps no per-user state. There is therefore no
  user-scoped data to isolate, migrate, or reset; cache entries are derived
  purely from the committed JSON file.

## Observability

Response fields to alert on: ``flagged`` (reliability/continuity concern),
``uncertain`` (identity or generation uncertainty), per-entry
``match_method``/``match_confidence``, and ``provenance.version`` (detect an
unexpected dataset version in production).

## Reproduction

```bash
pytest -q tests/test_hadith_research.py
```

The tests are offline and use only the committed ``data/rijal/narrators.json``;
they cover Arabic and transliterated names, diacritics, ``bin``/``b.`` forms,
empty input, partial-match uncertainty, unknown narrators, empty chains, and the
versioned provenance fields.

## Malware scan

No executable payloads or new dependencies are introduced; the change is Python
and JSON only. CI runs the repository's ``PR Malware Guard`` on every changed
file. The guard's detector tests can be run locally with:

```bash
node --test .github/scripts/pr-malware-guard.test.cjs
```

## Rollback

Revert the commit that changed ``hadith_research.py``,
``data/rijal/narrators.json``, the tests, and this note. The response models only
gained fields and the data version bump is additive, so no data migration or
schema downgrade is required; older clients that ignore the new fields keep
working.
