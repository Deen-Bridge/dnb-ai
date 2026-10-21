# Language Module Contributor Guide

This guide explains how the Deen Bridge AI service (`dnb-ai`) handles languages and how to
add or extend a language module without reverse-engineering the existing ones.

Existing language modules to use as reference:

| Module | Location | Router mounted in `main.py`? | Chat pipeline integration |
|---|---|---|---|
| Swahili (reference implementation) | `swahili/` | Yes (`/swahili`, `main.py:361`) | Yes — detection, prompt enhancement, response headers |
| Arabic dialect | `arabic_dialect/` | Yes (`/arabic-dialect`, `main.py:365`) | No |
| Arabic OCR (Arabic-script manuscripts) | `arabic_ocr.py` | Yes (`/arabic-ocr`, `main.py:357`) | No |
| Malay / Indonesian | `malay/` | No (no FastAPI router) | No (only `ms`/`id` in the language whitelist) |
| Urdu | `urdu_processor.py` | No (router defined at line 21, never imported by `main.py`) | No (only `ur` in the language whitelist) |
| Swahili legacy facade | `swahili_processor.py` | No | No — root-level re-export shim; prefer importing from `swahili/` |

---

## 1. How language selection works in `/chat`

There is **no generic `detect_language()`**. Language is resolved in three layers:

1. **Caller-supplied BCP-47 code.** `ChatRequest.language` (`main.py:437`) is validated by
   `normalize_language()` (`main.py:729`) against the `SUPPORTED_LANGUAGES` whitelist
   (`main.py:697`): `ar, en, ur, ms, fr, tr, id, bn, fa, ha, sw, tl`. Unrecognized codes log a
   warning and fall back to `None` (auto-detect) — they never return 422. Base-code fallback is
   applied too (`en-GB` → `en`).
2. **LLM auto-detect when omitted.** The streaming path injects `LANGUAGE_INSTRUCTIONS`
   (`main.py:712`) plus `response_language: <code>` or `auto` into the system context
   (`main.py:1640-1647`). The effective code is echoed back in `ChatResponse.language`
   (`main.py:471`) so the frontend can set `dir="rtl"` (see README, "Multilingual support").
3. **Offline keyword heuristic (Swahili only, non-streaming `/chat`).** `main.py:992-1014`
   checks a keyword list against the prompt; if `analyze_swahili()` detects at least two Islamic
   terms, `effective_language` is overridden to `"sw"`.

Per-request language extras (all optional, Swahili is the only complete template):

- **Detection/analysis block** in `/chat` — `main.py:992-1014`.
- **Prompt injection** in `generate()` — cultural/system context block at `main.py:1170-1175`
  (streaming: `LANGUAGE_INSTRUCTIONS` at `main.py:1640-1647`).
- **Response headers** — `X-Swahili-Dialect`, `X-Swahili-Terms-Detected`,
  `X-Swahili-Code-Switching` at `main.py:1361-1364`.
- **Tafsir language** — passed to `tafsir_retriever()` (`main.py:1020`), defaults to
  `DEFAULT_TAFSIR_LANGUAGE = "en"`.

---

## 2. The interface / contract a language module implements

A language module is a package (or single file) that provides five things: models, a
terminology catalog, dialect handling, a facade function, and an optional FastAPI router.
Swahili is the reference implementation; arabic_dialect is the lean variant.

### 2.1 Package layout

```text
<lang>/
  __init__.py        # re-export models, singletons, router, and the facade
  models.py          # enums + Pydantic request/response models
  terminology.py     # catalog engine + module-level singleton
  dialects.py        # dialect markers, classifier, normalization
  router.py          # APIRouter(prefix="/<lang>", tags=["<lang>"])
  # optional stage modules (swahili has all of them):
  loanwords.py       # morphology / borrowed-word analysis
  code_switching.py  # mixed-language segment detection
  cultural_context.py# regional institutions, rituals, fiqh notes
  generator.py       # system instructions + prompt enhancement + response validator
  optimizer.py       # query normalization/expansion for retrieval
```

### 2.2 Models (`models.py`)

- A `str`-Enum for dialects/variety (e.g. `SwahiliDialect` in `swahili/models.py:10`,
  `ArabicDialect` in `arabic_dialect/models.py:15`) with an `UNKNOWN` member.
- A category/domain enum for terminology (e.g. `IslamicDomain`,
  `swahili/models.py:32`).
- Pydantic request models with `text: str = Field(..., min_length=1)`.
- One aggregate result model carrying:
  - `original_text: str`
  - `normalized_text: str` (or `normalized_msa` in arabic_dialect)
  - per-stage sub-results (tokens, detected terms, dialect, code-switch, cultural context)
  - `processing_time_ms: float = 0.0` (the facade leaves `0.0`; HTTP routes fill real timing)

### 2.3 Terminology catalog (`terminology.py`)

A catalog class with this method set:

```python
class <Lang>Terminology:
    def __init__(self, data_file: Path = DATA_PATH) -> None: ...
    def get_term_by_id(self, term_id: str) -> <Term> | None: ...
    def lookup_term(self, text: str) -> <Term> | None: ...
    def search_terms(self, query=None, category=None, limit: int = 50) -> list[<Term>]: ...
    def extract_terms_from_text(self, text: str) -> list[<Term>]: ...

terminology_db = <Lang>Terminology()   # module-level singleton
```

Conventions:

- Load data **once at import time** into a module-level singleton.
- Missing data file must **not crash** the import — log a warning (swahili) or fall back to a
  small hardcoded list (malay).
- Resolve the data path relative to the module file, never the working directory:
  - `swahili/terminology.py:14` — `Path(__file__).resolve().parent.parent / "data" / "swahili_islamic_terms.json"`
  - `urdu_processor.py:23` — `Path(__file__).resolve().parent / "data" / "urdu_islamic_terms.json"`
  - **Do not** copy `malay/terminology.py:14`, which uses a CWD-relative
    `os.path.join("data", ...)` and breaks when the server runs from another directory.
- Match multi-word terms before single words (see `extract_terms_from_text`,
  `swahili/terminology.py:159`).

### 2.4 Dialect handling (`dialects.py`)

```python
DIALECT_MARKERS: dict[<DialectEnum>, dict] = {
    # "words": {dialectal: standard/MSA}, "patterns": [regex, ...], "is_coastal": bool
}

class <Lang>DialectClassifier:
    def classify_dialect(self, text: str) -> <Profile>: ...
    def normalize_to_standard(self, text: str) -> tuple[str, dict[str, str]]: ...

dialect_classifier = <Lang>DialectClassifier()   # module-level singleton
```

Scoring convention used by both swahili and arabic_dialect: word markers score `+1.5`,
regex patterns score `+2.0`; confidence is `min(0.98, 0.60 + score * 0.1)`. A zero-score input
falls back to the standard variety with high confidence (Sanifu `0.90` for swahili,
MSA `0.95` for arabic_dialect). Normalization returns both the normalized string and the map
of replacements applied.

### 2.5 Facade (`__init__.py`)

Every package exposes a single facade that runs all stages:

```python
def analyze_<lang>(text: str) -> <AggregateResult>: ...
```

Example: `analyze_swahili()` (`swahili/__init__.py:26`) chains tokenize → term extraction →
loanwords → dialect classification → code-switching → cultural context → dialect normalization
and returns the aggregate with `processing_time_ms=0.0`. The `__init__.py` also re-exports all
models, engine singletons, and `router` via `__all__`.

### 2.6 HTTP router (`router.py`)

```python
router = APIRouter(prefix="/<lang>", tags=["<lang>"])
```

Canonical endpoint quartet (see `swahili/router.py:32` and `arabic_dialect/router.py:25`):

| Endpoint | Purpose |
|---|---|
| `POST /<lang>/analyze` | Full facade pipeline with real timing + per-segment results |
| `POST /<lang>/normalize` | Dialect → standard/MSA normalization |
| `GET /<lang>/dialects` | List supported dialects/variety |
| `GET /<lang>/terms` | Catalog search (`query`, `category`/`dialect`, `limit` capped at 150) |
| `GET /<lang>/terms/{term_id}` | Single term; `HTTPException` 404 on miss |

Optional extras: `POST /<lang>/code-switch`, `POST /<lang>/enhance-prompt`,
`POST /<lang>/process` (urdu), `POST /<lang>/transliterate`.

### 2.7 Design rules

- **Offline and deterministic** — no network calls, no API keys; engines must work in tests
  without `GEMINI_API_KEY`.
- **Graceful degradation** — missing resources warn, they don't raise.
- **Pure library core** — the facade and engines must be importable without FastAPI app state
  so unit tests can exercise them directly.

---

## 3. Catalogs and resources

All language resources live in `data/` at the repository root:

| File | Loaded by |
|---|---|
| `data/swahili_islamic_terms.json` (~120 terms) | `swahili/terminology.py:14` |
| `data/urdu_islamic_terms.json` | `urdu_processor.py:23` |
| `data/malay_indonesian_islamic_terms.json` | `malay/terminology.py:14` (CWD-relative — see warning above) |
| *(none — lexicon embedded in Python as `_RAW_TERMS`)* | `arabic_dialect/terminology.py:18` |

Use an embedded Python lexicon only for small, fully-offline dialect lexicons; use a JSON file
once the catalog grows beyond a few dozen terms.

### JSON envelope

Every catalog file follows the same envelope:

```json
{
  "version": "1.0.0",
  "language": "sw",
  "total_terms": 120,
  "description": "…",
  "terms": [ { "…": "…" } ]
}
```

### Per-term fields

Common across catalogs: `id`, `<lang>_term`, `arabic_original` (Arabic script),
`arabic_transliteration` (or `transliteration`), `english_equivalent`, `category`,
`definition_<lang>`, `variants` (optional).

- **IDs** are namespaced `<lang>-<domain>-<slug>`: `"sw-ibada-swala"`, `"ur-ibadat-salah"`,
  `"ms-id-ibada-solat"`, `"eg-3ayez"` (arabic_dialect uses dialect prefix instead of domain).
- **`category`** must match a value of the module's domain enum (`IslamicDomain` for swahili).
- **Swahili extras:** `variants_sw`, `dialect_notes`, `common_misspellings`, `related_terms`.
- **Malay extras:** parallel `malay_term` / `indonesian_term` / `jawi_term` and
  `definition_ms` / `definition_id`.

Builders for these files live in `scripts/` (e.g. `build_hadith_data.py`,
`build_surah_index.py` follow the same pattern).

---

## 4. Steps to add a new language module

Work from a branch off `dev` (or `main` if `dev` is unavailable); PRs target `dev` per
`CONTRIBUTING.md`.

### Step 1 — Add the language code to the whitelist

Add your BCP-47 code to `SUPPORTED_LANGUAGES` (`main.py:697`) and, if prompt guidance is
needed, extend `LANGUAGE_INSTRUCTIONS` (`main.py:712`) with a policy paragraph for the language
(script rules, quote rules, honorifics — mirror the existing Swahili/Arabic entries).

### Step 2 — Create the package

```text
<lang>/
  __init__.py      # exports + analyze_<lang>() facade
  models.py        # Dialect/domain enums, request models, aggregate result model
  terminology.py   # catalog class + terminology_db singleton, Path(__file__)-anchored data path
  dialects.py      # DIALECT_MARKERS + classifier with classify_dialect/normalize_to_standard
  router.py        # APIRouter with the canonical endpoint quartet
```

Add optional stage modules (`loanwords.py`, `code_switching.py`, `cultural_context.py`,
`generator.py`, `optimizer.py`) only where the language actually needs them — start with
`cultural_context.py` and `generator.py` if the language has regional fiqh variance or a
distinct script.

### Step 3 — Add the catalog

Create `data/<lang>_islamic_terms.json` using the envelope and field conventions in §3.
Anchor the load path with `Path(__file__).resolve().parent[.parent] / "data" / …` and make a
missing file a warning, not a crash.

### Step 4 — Mount the router in `main.py`

```python
from <lang> import router as <lang>_router          # near main.py:41 (imports)
app.include_router(<lang>_router)                    # near main.py:356-367 (mounts)
```

Add a one-line comment referencing the tracking issue, as done for arabic_dialect
(`main.py:365`, issue #136).

### Step 5 (optional) — Wire into the `/chat` pipeline

Use the Swahili block as the template:

1. Detection heuristic in `/chat` (`main.py:992-1014`): keyword list → facade call →
   override `effective_language` when enough terms are detected.
2. Prompt enhancement inside `generate()` (`main.py:1170-1175`) or `LANGUAGE_INSTRUCTIONS`
   in the stream path (`main.py:1640-1647`).
3. `X-<Lang>-*` response headers (`main.py:1361-1364`).
4. Pass the effective language to `tafsir_retriever()` if tafsir applies.

If you skip this step, the language still works at generation level via the whitelist +
LLM auto-detect (this is the current state of malay and urdu).

### Step 6 — Add tests

Create `tests/test_<lang>.py`. Follow existing patterns:

- **Engine unit tests** import singletons directly from the package and assert on Pydantic
  fields with realistic sentences (`tests/test_swahili.py` sections 1–6,
  `tests/test_urdu_processor.py`).
- **Endpoint tests** need `main` imported. Set the env stub *before* importing:
  ```python
  import os
  os.environ.setdefault("GEMINI_API_KEY", "test-key")
  from main import app
  ```
  Then use either a sync `TestClient(app)` (`tests/test_swahili.py:24`) or async
  `httpx.ASGITransport` + `AsyncClient` (`tests/test_arabic_dialect.py:25`). The autouse
  `_fast_session_store` fixture in `tests/conftest.py` keeps `/chat` tests hermetic.
- **Router-only tests** can mount the router on a scratch app instead of importing `main`
  (`tests/test_urdu_processor.py:64`):
  ```python
  app = FastAPI()
  app.include_router(router)
  client = TestClient(app)
  ```
- Cover: catalog load + lookups, dialect classification + normalization, facade output shape,
  every HTTP endpoint (including a 404 case), and — if wired — the `/chat` headers/behavior.
- Multilingual parity checks (script validity, terminology accuracy, ≤10% quality degradation
  vs English) live in `tests/test_multilingual.py` and `scripts/eval_multilingual.py`.

### Step 7 — Add a named CI step

CI (`.github/workflows/ci.yml`) runs **explicit per-file pytest steps** — there is no
`pytest tests/` catch-all. Add a named step for your test file next to the existing ones
(`tests/test_swahili.py` at line 138, `tests/test_arabic_dialect.py` at line 141), or your
tests will never run in CI.

### Step 8 — Lint, type-check, commit

```bash
ruff check . && ruff format . && mypy .
pytest -q tests/test_<lang>.py
```

`main.py` is strictly typed (`disallow_untyped_defs`), so new functions wired into it must
have full annotations. Use Conventional Commits
(`feat(<lang>): …`).

---

## 5. RTL and script-specific considerations

Languages using Arabic script (Arabic, Urdu, Persian, Jawi Malay) or written right-to-left
need extra care. There is **no shared bidi utility module** — each module owns its script
handling today.

### 5.1 Shared building blocks to reuse

- **`arabic_ocr.py`** has the most complete Arabic-script toolkit:
  - `ARABIC_DIACRITICS` (line 354), `is_arabic_diacritic()` (385), `strip_diacritics()` (390),
    `count_diacritics()` (395)
  - `normalize_arabic()` (line 400) — removes tatweel, folds alef variants `آأإٱ→ا`,
    `ى→ي`, `ة→ه`, collapses whitespace
  - `HAMZA_FORMS` (361), `COMMON_OCR_ERRORS` (370, dot-confusion pairs)
- **`urdu_processor.py`** has the most sophisticated normalization:
  - `normalize_urdu()` (line 215) — NFKC + keyboard-variant folding (`ي→ی`, `ك→ک`, `ه→ہ`),
    strips tatweel/ZWJ/ZWNJ/BOM, optional diacritic strip, punctuation folding `?→؟`, `,→،`
  - `analyze_script()` (line 229) — dominant/mixed script profiling across
    Arabic, Latin, and Devanagari ranges
  - `transliterate_urdu()` (line 313) — curated term transliteration before a generic map
- **`page_analysis.py`** handles layout-level RTL: `PageInput.rtl` (line 108) and
  `determine_reading_order()` reverse column order for RTL pages.
- **Inline detection regex**: Arabic-script spans are matched with `[\u0600-\u06FF]`
  (`swahili/code_switching.py:91`, `swahili/generator.py:73`) and wider Urdu/Persian ranges
  `[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]` (`urdu_processor.py:49`).

### 5.2 Rules for new RTL/script modules

1. **Never transliterate or translate Quranic text.** Quran must stay in Arabic script with a
   translation alongside, cited as `surah:ayah` — this rule is encoded in
   `LANGUAGE_INSTRUCTIONS` (`main.py:712`) and `SWAHILI_SYSTEM_INSTRUCTIONS`
   (`swahili/generator.py:15`).
2. **Normalize before matching.** Fold keyboard/Unicode variants and strip diacritics (or
   match with an optional-diacritic pattern) before term lookup, or catalog lookups will miss.
3. **Preserve diacritics by default;** make stripping an opt-in flag
   (`preserve_diacritics: bool = True`, `urdu_processor.py:215`).
4. **Keep script detection separate from language detection.** A Latin-script Swahili or
   Urdu-English mix is still the same language — use `analyze_script()`-style profiling for
   script, the whitelist/heuristic for language.
5. **RTL direction (`dir="rtl"`) is a frontend contract.** Echo the effective code in
   `ChatResponse.language` and let the frontend set direction (README "Multilingual support").
6. **RTL layout for documents/images** belongs to `page_analysis.py` (`rtl` flag), not the
   chat pipeline.
7. **Quote punctuation** must be language-appropriate (`؟` vs `?`, `،` vs `,`) — fold it
   during normalization, don't let it leak into term matching.

---

## 6. Files to touch (checklist)

```text
New:
  <lang>/__init__.py
  <lang>/models.py
  <lang>/terminology.py
  <lang>/dialects.py
  <lang>/router.py               (+ optional stage modules)
  data/<lang>_islamic_terms.json
  tests/test_<lang>.py
  docs/languages.md              (update the module table in §1)

Modified:
  main.py                        # SUPPORTED_LANGUAGES (697), LANGUAGE_INSTRUCTIONS (712),
                                 # router import (~41), include_router (~356-367),
                                 # optional chat integration (992-1014, 1170-1175, 1361-1364,
                                 # 1640-1647)
  .github/workflows/ci.yml       # named pytest step for tests/test_<lang>.py
  README.md                      # multilingual support table (296-330), if user-facing
```

## 7. Acceptance criteria for a new language module

- [ ] Language code whitelisted in `SUPPORTED_LANGUAGES`; prompt policy in `LANGUAGE_INSTRUCTIONS`
- [ ] Package implements the contract in §2 (models, catalog singleton, dialect classifier, facade, router)
- [ ] Catalog follows the `data/` JSON envelope; path anchored with `Path(__file__)`; missing file warns
- [ ] Router mounted in `main.py` with an issue-referencing comment
- [ ] `tests/test_<lang>.py` covers engines, facade, and every endpoint (+ `/chat` if wired)
- [ ] Named pytest step added to `ci.yml`
- [ ] `ruff check . && ruff format . && mypy .` pass
- [ ] RTL/script rules in §5 applied if the language uses Arabic script
