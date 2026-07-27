# Hadith grading dataset — provenance

## Source

- Repository: [`fawazahmed0/hadith-api`](https://github.com/fawazahmed0/hadith-api)
- Pinned ref: tag `1`
- License: [The Unlicense](https://github.com/fawazahmed0/hadith-api/blob/1/LICENSE) — public domain
- Fetched: see the `source.fetched` field in each `data/hadith/<collection>.json` file
- Collections bundled: Sahih al-Bukhari, Sahih Muslim, Sunan Abu Dawud, Jami
  at-Tirmidhi, Sunan an-Nasai, Sunan Ibn Majah, Muwatta Malik (the two
  Sahihs, the four Sunan, and the Muwatta)

## What is bundled, and why

Only grading metadata is extracted: collection, book number, hadith number
(both the global sequential number and the book-local number), Arabic
number, and each grader's name and raw grade string. **The translated
hadith text itself is not bundled.** The upstream source aggregates
translations from several third-party sites of unclear individual copyright
status even though the API wrapper itself is public domain; grading
metadata (which grader assigned which grade to which numbered hadith) is
factual/bibliographic information, not a creative work, so it carries no
such ambiguity. This also keeps this service's scope on authenticity
grading, matching this issue — a full bundled hadith *text* corpus, if ever
wanted, is a separate concern (mirroring how bundled Quran text is handled
separately per issue #40).

## Why Bukhari and Muslim have no per-hadith grader

The source data has an empty `grades` array for every hadith in both
collections. This is not missing data — it reflects the classical scholarly
position that both collections were authenticated by consensus (ijma') of
the community of hadith scholars in their entirety, so individual hadith
within them are not separately graded the way hadith in the four Sunan are.
Both collections are recorded here with grade `SAHIH` and grader
`"Scholarly consensus (Bukhari and Muslim)"`.

## Normalization policy

Implemented in `hadith.py` (`parse_grade_string`, `aggregate_strength`,
`aggregate_chain_type`) and shared by the build script and the runtime
lookup, so there is exactly one normalizer.

Each grader's raw string (e.g. "Hasan Sahih", "Sahih Lighairihi", "Mauquf
Daif", "-") is decomposed into two independent axes:

- **Strength** — one of `SAHIH`, `HASAN`, `DAIF`, `MAWDU`, or `UNKNOWN`
  (blank/unrecognized). Composite strings are resolved to the *weaker* word
  ("Hasan Sahih" → `HASAN`). "Munkar" and "Shadh" are treated as `DAIF`
  (both are recognized weak-hadith subtypes); "Mawdu" and "Batil" are
  treated as `MAWDU` (fabricated/void).
- **Chain type** — `MARFU` (attributed to the Prophet, peace be upon him —
  the default) unless the raw string says `MAUQUF` (stopped at a Companion),
  `MAQTU` (stopped at a Successor), or `MURSAL` (a Successor narrating
  directly from the Prophet, omitting the Companion link). These describe
  *what* is being graded, not how strong the grading is — a hadith graded
  "Mauquf Sahih" has an authentically transmitted chain, but to a Companion's
  own statement, not to the Prophet, and is never presented as a Prophetic
  hadith regardless of that chain strength.

When a hadith has multiple graders and they disagree, the **overall
strength is the weakest strength any grader assigned**, and the **overall
chain type is the most cautious (non-marfu) type any grader flagged**. This
is a deliberately conservative, safety-first policy — every individual
grader's raw string is preserved in the `graders` array of each record, so
nothing is hidden, but the headline grade this service surfaces never
overstates a hadith's authenticity relative to what any of its named
graders said.

## File format

`data/hadith/<collection>.json`:

```json
{
  "collection": "abudawud",
  "name": "Sunan Abu Dawud",
  "source": {"repo": "fawazahmed0/hadith-api", "ref": "1", "fetched": "2026-07-23"},
  "hadiths": [
    {
      "n": 3,
      "book": 1,
      "bn": 3,
      "an": 3,
      "grade": "DAIF",
      "chain": "MARFU",
      "graders": [
        {"g": "Al-Albani", "raw": "Daif", "s": "DAIF", "c": "MARFU"}
      ]
    }
  ]
}
```

`n` is the global sequential hadith number (the number most citations use,
e.g. "Sunan Abu Dawud 3"); `book`/`bn` are the book number and book-local
hadith number, used when a citation explicitly names a book.

## Update process

Re-run the build script whenever the upstream data changes or a new tag is
published:

```bash
python scripts/build_hadith_data.py
```

This re-downloads the pinned edition files and regenerates
`data/hadith/*.json` from scratch. It requires network access and is not
run in CI — CI tests run entirely offline against the committed JSON files.
To track a newer upstream release, bump `SOURCE_TAG` in
`scripts/build_hadith_data.py` after checking its release notes for
breaking format changes.
