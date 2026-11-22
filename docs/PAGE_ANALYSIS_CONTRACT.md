# Manuscript OCR and Page-Analysis Citation Contract (#854)

- **Audit ID**: PROD-AUDIT AI-078
- **Scope**: the boundary from a manuscript page image (`arabic_ocr.py`) to
  positioned OCR text blocks and the citations extracted from them
  (`page_analysis.py`), plus the shared checks both rely on
  (`media_upload_limits.py`). Audio, video, calligraphy and recitation routes
  were reviewed for the same input boundary and are unchanged; see
  [Follow-ups](#follow-ups).
- **Owner**: to be confirmed by the `dnb-ai` maintainers in review (proposed:
  whoever owns the multimodal routes).
- **Contract version**: `2`, returned as `contract_version` on every
  `/page-analysis/analyze` response.

## 1. Pipeline and data flow

```
page image ──► /arabic-ocr/process | /process-file      (consent, size, signature)
                    │  OCR engine (local stub today; no provider call)
                    ▼
             OCRResult.lines[*] {text, confidence, bbox}
                    │  caller maps each positioned line/block to a PageBlock
                    ▼
             /page-analysis/analyze  ── PageInput {width, height, blocks[], rtl, consent}
                    │  deterministic: geometry + text heuristics, no I/O
                    ▼
             PageAnalysisResponse {elements, reading_order, structure,
                                   references, rejected_references, warnings}
```

The page analyzer never sees pixels and never calls an external service. It
reads two bundled, versioned datasets: the 114-surah index
(`data/quran/surah_index.json`, through `tafsir.surah_by_number`,
`surah_by_name` and `is_canonical_surah_name`) and the hadith number index
(`data/hadith/*.json`, through `hadith.get_default_source`).

**OCR to page-analysis mapping.** For each positioned line or block the OCR
engine returns, send one `PageBlock` with `x`, `y`, `width`, `height` from its
bounding box, `text` from its text, and `confidence` from the engine's
confidence for that line or block. If the engine reports no confidence, omit
`confidence`; do not send a default. Lines without a bounding box cannot be
placed on the page and should not be sent.

## 2. Inputs and limits

| Input | Limit | Enforced by | Rejection |
| :--- | :--- | :--- | :--- |
| OCR image, base64 (`/process`, `/detect-style`, `/preprocess`) | encoded length checked **before** decoding; decoded at most 10 MB | `media_upload_limits.decode_base64_image`, `validate_image_upload` | 413 |
| OCR image, multipart (`/process-file`) | at most 10 MB; the route reads at most 10 MB + 1 byte | `validate_image_upload` | 413 |
| Base64 syntax | strict alphabet and padding; ASCII whitespace (line wrapping) allowed | `decode_base64_image` | 422 |
| Data-URL prefix | only `data:image/(png\|jpeg\|jpg\|tiff\|webp);base64,` | `decode_base64_image` | 422 |
| Empty image | rejected | `validate_image_upload` | 400 |
| `/detect-style` `text_sample` | 20,000 characters | Pydantic | 422 |
| Page width / height | `0 < value <= 20,000`, finite | Pydantic (`MAX_PAGE_DIMENSION`) | 422 |
| Blocks per page | 5,000 | Pydantic (`MAX_PAGE_BLOCKS`) | 422 |
| Block geometry, `font_size` | `0 <= value <= 20,000`, finite (no NaN / Infinity) | Pydantic | 422 |
| Block `confidence` | `0.0 <= value <= 1.0`, optional | Pydantic | 422 |
| Characters per block | 20,000 | Pydantic (`MAX_PAGE_BLOCK_TEXT_CHARS`) | 422 |
| Characters per page (sum of blocks) | 200,000 | `PageInput` validator (`MAX_PAGE_TEXT_CHARS`) | 422 |
| References returned per page | 1,000 accepted and 1,000 rejected | `analyze_page` (`MAX_REFERENCES_PER_PAGE`) | not an error: lists are cut and a warning says so |

Page limits are checked during request validation, before the analyzer runs.
The 413 block-count branch that used to sit in the route could never run,
because request validation rejects more than 5,000 blocks with 422 first; it
was replaced by the shared `validate_page_layout_input` call, which keeps the
same consent message. Request validation still parses the whole JSON body, so
the hosting layer's request-size limit remains the outer bound on body size.

### Supported image formats

The format is taken from the decoded bytes' signature. Filenames, data-URL
media types and declared `Content-Type` values are never trusted.

| Signature | OCR routes | Why |
| :--- | :--- | :--- |
| PNG, JPEG, TIFF, WebP | accepted | single-page raster images that common Arabic OCR engines read |
| PDF | **415** | a multi-page document container, not a page image; the shared sniffer still recognizes it for other callers |
| anything else | **415** | unknown or corrupted header |

Signature acceptance does not prove that a future engine can decode every file
(for example a multi-page TIFF or an unusual WebP variant). The current engine
is a stub that decodes nothing, so the decode path must be re-verified with
the real engine before it is enabled.

## 3. Consent

`consent_to_process` must be exactly `true` on `/page-analysis/analyze` and on
every image-taking OCR route (`/process`, `/process-file`, `/detect-style`,
`/preprocess`). An omitted field or `false` returns **400** with
`"Processing requires explicit user consent (consent_to_process=true)."`.
Consent is checked before the image is decoded and before any preprocessing,
style detection, OCR or analysis, so a request without it is never processed.
A request that also fails schema validation (for example a page wider than
20,000 units) gets that 422 first, because FastAPI validates the body before
the route runs. Consent means the user agrees to have the
submitted image or text processed for this one request. It is not consent to
store it, to send it to a third party, or to use it for training.

**This is a behavior change.** Before, an omitted field was treated as `true`.
Clients that omitted it now receive 400 and must send
`"consent_to_process": true` (JSON) or `consent_to_process=true` (multipart
form).

## 4. Retention and logging

- **Stored**: nothing. Images, OCR text and analysis results live only for the
  request; no route writes them to disk, a cache, a database or a queue.
- **Returned to the caller only**: `OCRResult.image_hash` (SHA-256 of the
  submitted image). It is stable for identical images, so it can link two
  requests for the same image; it is not logged.
- **Provider retention**: none, because no provider is called. Any future
  provider integration must document its own retention and region settings
  before it is enabled.
- **Logged**: one `page_analysis completed` INFO line per successful analysis
  with counts only: `blocks`, `text_chars`, `validated`, `unverified`,
  `rejected`, `requires_review`, `elapsed_ms`. Image bytes, base64 payloads,
  OCR text, matched citation text, filenames and `image_hash` are never logged.
  This follows `docs/RETRIEVAL_LOGGING.md`.

## 5. Output: what each reference field claims

`ExtractedReference` keeps every v1 field with its v1 meaning and adds the rest.

| Field | Meaning |
| :--- | :--- |
| `type` | `quran` or `hadith` (v1). |
| `reference` | Qur'an: `surah:ayah` of the first ayah (v1, unchanged for ranges). Hadith: canonical collection name (v1). |
| `text`, `block_index` | Exact matched substring and its source block (v1). |
| `surah`, `ayah`, `collection` | As v1. `collection` now uses the canonical names from `hadith.COLLECTION_NAMES`. |
| `start`, `end` | Character offsets: `blocks[block_index].text[start:end] == text`. |
| `ayah_end` | Last ayah of a validated range (`2:255-256` gives `256`); null for a single ayah. |
| `surah_name` | Canonical surah name from the surah index, never from the page text. |
| `collection_key`, `hadith_number`, `book` | Canonical collection key (`bukhari`), and the number(s) written right after the name, if any. |
| `status` | `validated` or `unverified`, defined below. |
| `status_reason` | Plain-language reason for the status. |
| `ocr_confidence` | The source block's `confidence` exactly as supplied, or null. |
| `ocr_confidence_band` | `high` (>= 0.75), `medium` (>= 0.40), `low` (< 0.40), from `evaluate_confidence_and_fallback`; `unavailable` when no confidence was supplied. |
| `requires_review` | `true` unless `status` is `validated` **and** the band is `high` or `medium`. |

`RejectedReference` (`type`, `text`, `block_index`, `start`, `end`, `reason`)
lists citation-shaped matches that cannot point at real data.
`PageAnalysisResponse.warnings` gives page-level counts of unavailable or low
OCR confidence, unverified candidates, rejections, truncation, and empty pages.
It never echoes page text.

### Extraction is not verification

Extraction (stage 1, `find_reference_candidates`) finds citation-shaped text
and records where it came from. Verification (stage 2,
`verify_reference_candidate`) checks each candidate against the bundled data.
The two never merge:

| Candidate | Result |
| :--- | :--- |
| Qur'an with a marker (`Surah`, `Sura`, `Surat`, `S`/`S.`, `Q`, `Quran`/`Qur'an`, `سورة`), in brackets (`[2:153]`), or as a bracketed canonical, Arabic, or `Surah`-prefixed surah name (`[البقرة: ١٥٣]`, `(Al-Baqarah: 255-256)`, `(Surah Light, 35)`), and the surah, ayah and range exist | `validated` |
| Bracketed surah name matched only through an alias that is also an ordinary word (`(Time: 3)`, `[cow: 5]`) | `unverified`: the bracket may not be a citation |
| Bare `N:M` with no marker that exists in the index (`10:30`, `John 3:16`) | `unverified`: it may be a time, a ratio or another text's verse |
| Surah outside 1-114, ayah 0 or past the surah's length, or a range that runs backwards or past the end | **rejected** with a reason |
| Hadith collection plus a number that exists in the bundled index | `validated` |
| Hadith collection plus a number not in the index | `unverified`: editions number differently |
| Hadith collection named with no number | `unverified` |
| Bare `Muslim` or `Malik` with no number | not extracted: these are ordinary words and names far more often than citations |
| Collections not in `hadith.py` (for example Musnad Ahmad) | not extracted |
| Clock times such as `10:30:15`, brackets naming no surah such as `(page: 12)` | not extracted |

`validated` only means the citation is well formed and points at something that
exists in the bundled data. It does **not** mean the OCR read the page
correctly, that quoted text matches the ayah, or that a hadith is authentic.
Hadith grading is never looked up or returned here; use the hadith endpoints
for that. Only `validated` references can make a block a `reference-citation`
element, so an unverified `10:30` cannot change the page layout.

Arabic-Indic (`٠-٩`) and Extended Arabic-Indic (`۰-۹`) digits are read as
numbers.

### Confidence and fallback

The analyzer does not compute a confidence of its own. A regex match is not
evidence of a correct reading, so no score is derived from it. Upstream OCR
confidence is carried from the source block unchanged. Without it, the band is
`unavailable` and the reference requires review. A low band also requires
review and adds a warning. The safe fallback is always the same: return
the candidate with its status, band and reason, and never promote it.

## 6. Failure modes

| Condition | Behavior |
| :--- | :--- |
| Missing or false consent | 400; nothing is processed |
| Malformed base64, disallowed data URL | 422 |
| Oversized image (encoded or decoded) | 413 |
| Empty image | 400 |
| Unsupported or spoofed format, PDF on OCR | 415 |
| Page or block over a limit, non-finite or negative geometry, confidence out of range | 422 |
| Empty page or only blank blocks | 200; no references; warning `No text was found on the page` |
| Low or missing OCR confidence | 200; `requires_review=true`; warning |
| Invalid citation | 200; listed in `rejected_references`; warning |
| Unknown hadith collection | 200; not extracted |
| More than 1,000 references or rejections | 200; lists cut to the first 1,000 in block order; warning |

## 7. Observability

Monitor, without content:

- request count and status code per route (400 consent, 413 size, 415 format,
  422 validation);
- from the `page_analysis completed` log line: block and character volume,
  `elapsed_ms`, and the `validated` / `unverified` / `rejected` /
  `requires_review` counts;
- OCR confidence band distribution and provider cost per page, once a real
  engine is enabled.

A rising share of `rejected` or `unverified` usually means worse OCR input or a
new citation style. A rising share of 400 responses after deploying this change
means a client is not sending consent.

## 8. Evaluation, cost and latency

Reproduce without secrets or network:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q tests/test_page_analysis.py tests/test_page_citation_eval.py \
          tests/test_arabic_ocr.py tests/test_manuscript_page_api.py tests/test_media_upload_limits.py
python scripts/eval_page_citations.py --bench --runs 50
```

**Evaluation fixture**: `tests/fixtures/page_citation_eval.json`, 29 synthetic
pages in seven categories: Arabic, English, mixed, OCR-like noise, invalid
references, no references, and bare numbers. Results:

| Metric | Result |
| :--- | :--- |
| Precision (type, reference, block, status) | 1.00 (24 / 24) |
| Recall | 1.00 (24 / 24) |
| Invalid-reference rejection | 1.00 (4 / 4) |
| Provenance (`text[start:end] == text`) | 1.00 |
| False positives on no-reference pages | 0 |
| Accepted by status | 19 validated, 5 unverified; 15 require review |

These are regression gates on a small curated set (enforced by
`tests/test_page_citation_eval.py`). They are **not** an estimate of accuracy
on real manuscripts, and no claim about general OCR accuracy is made.

**Latency baseline** for `analyze_page`, 50 runs per shape, RTL two-column
pages, each block holding three citations. Measured on Python 3.13, Linux
x86_64, Intel Xeon @ 2.30 GHz, 2 vCPU:

| Blocks | Characters | p50 | p95 |
| :--- | :--- | :--- | :--- |
| 40 | 4,800 | 3.8 ms | 4.9 ms |
| 400 | 48,000 | 37.0 ms | 48.9 ms |
| 2,000 | 200,000 (page maximum) | 147.8 ms | 179.4 ms |

Adversarial pages at the size limits (every block 20,000 characters of
brackets, colon-digit runs, repeated markers, or Arabic brackets) completed in
15-110 ms for ten blocks, with no regex backtracking blow-up.

**Cost**: zero. The page analyzer and the OCR stub make no provider calls,
which `scripts/eval_page_citations.py --bench` records as `provider_calls: 0`.
This baseline does **not** measure a real OCR engine. Before any provider is
enabled, measure its per-page latency and cost separately, using approved
non-personal test images, and add it to this section.

## 9. Configuration

None. The limits are constants in `media_upload_limits.py` and `page_analysis.py`.
No environment variables, secrets, or feature flags are introduced. No
external OCR provider is enabled by this change.

## 10. Deployment

Code-only. There is no migration and no configuration change. Compatibility for
clients:

1. **Breaking (deliberate):** an omitted `consent_to_process` now returns 400 on
   `/page-analysis/analyze`, `/arabic-ocr/process` and `/arabic-ocr/process-file`.
   `/arabic-ocr/detect-style` and `/arabic-ocr/preprocess` now require consent
   too. Every client of these routes must send `true` before this ships.
2. **Breaking (deliberate):** PDF uploads to OCR routes return 415. Base64 with
   non-alphabet characters, or with a data-URL prefix other than an image type,
   returns 422 instead of being decoded leniently.
3. **Stricter validation:** block text over 20,000 characters, page text over
   200,000, block coordinates over 20,000, non-finite numbers, and block
   `confidence` outside [0, 1] return 422.
4. **Changed meaning:** Qur'an references whose ayah exceeds the surah's
   length (for example `1:200`) were returned before and are now in
   `rejected_references`. A bare `Muslim` (no number) and every `Ahmad`
   mention are no longer extracted. Three collection names change spelling to
   match `hadith.py`: `Sunan Abi Dawud` becomes `Sunan Abu Dawud`,
   `Jami' at-Tirmidhi` becomes `Jami at-Tirmidhi`, and `Sunan an-Nasa'i`
   becomes `Sunan an-Nasai`. References are now ordered left to right within
   a block, regardless of type.
5. **Additive:** new response fields (section 5), the optional block
   `confidence`, `rejected_references`, `warnings`, `contract_version`.

## 11. Rollback

Revert the PR. Nothing is persisted, so there is no data to migrate or clean
up. After a rollback, omitted consent is treated as `true` again, PDFs are
accepted by OCR, invalid citations such as `1:200` are returned as references
again, and the v2 fields disappear from responses. A client that already reads
`status` or `rejected_references` must handle their absence, and should treat
an absent `status` as unverified. Clients that send `consent_to_process: true`
keep working on both versions, so the safe order is to update clients first,
then deploy.

## Known limitations

- Hadith collections are recognized only by the Latin-script aliases in
  `hadith.py`. Arabic-script names (`صحيح البخاري`, `رواه مسلم`) are not yet
  recognized; adding them belongs in `hadith.py`, so every caller benefits.
- Musnad Ahmad and other collections absent from `hadith.py` are not extracted.
- A bare `N:M` that exists in the surah index is reported as an unverified
  Qur'an candidate even when it is a time or another scripture's verse. That is
  why it is never validated.
- The OCR engine is a stub. Its confidence values are fixtures, not measurements.

## Follow-ups

- `audio_hadith.py`, `video_analysis.py`, `quran_recitation.py` and
  `calligraphy.py` still default `consent_to_process` to `true`. Applying the
  same affirmative-consent rule there is a separate, client-visible change.
- `docs/MULTIMODAL_UPLOAD_LIMITS.md` lists limits for those routes that are not
  all enforced in code (for example image pixel dimensions). This PR corrects
  the OCR and page-layout rows only.
- `OCREngineConfig.api_endpoint` is accepted from the request body. It is
  unused while the engine is a stub, but a real `custom` engine must take its
  endpoint from server configuration, never from the caller, to avoid
  server-side request forgery.
- The `dev` branch wraps page analysis with a rate limiter (`rate_limit.py`)
  and sanitizes extracted text (`multimodal_budget.py`, #859). Neither module
  exists on `main`, so neither was ported here.
