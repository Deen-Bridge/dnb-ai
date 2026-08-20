<div align="center">

# 🕌 Deen Bridge — AI Service

**The FastAPI service behind Deen Bridge's Islamic-knowledge AI assistant, powered by Google Gemini.**

[![CI](https://github.com/Deen-Bridge/dnb-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Deen-Bridge/dnb-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](CONTRIBUTING.md)
[![Python](https://img.shields.io/badge/Python-3.12-3776ab.svg)](https://www.python.org/)

[Live API](https://dnb-ai.onrender.com) · [Web App](https://dnb-frontend.vercel.app) · [Report a Bug](https://github.com/Deen-Bridge/dnb-ai/issues) · [Contribute](CONTRIBUTING.md)

</div>

---

## About

This service powers the AI assistant inside **Deen Bridge**, a platform for authentic Islamic education built on the **Stellar network** — courses and books are purchased with USDC, and creators are paid directly to their own Stellar wallets. The assistant wraps Google's Gemini model with an Islamic-knowledge system prompt, content safety filters, and per-session conversation history, exposing a simple chat API consumed by the web app.

On the roadmap: further Stellar-aware assistance beyond zakat and purchase Q&A
(see open issues). Zakat on a wallet's on-chain USDC balance and factual answers
about the signed-in user's Stellar course/book purchases are already supported.

The platform is composed of three services:

| Repository                                                  | Role                                       | Live                                                                 |
| ----------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------- |
| [dnb-frontend](https://github.com/Deen-Bridge/dnb-frontend) | Next.js web application                    | [dnb-frontend.vercel.app](https://dnb-frontend.vercel.app)           |
| [dnb-backend](https://github.com/Deen-Bridge/dnb-backend)   | REST API — auth, content, Stellar payments | [dnb-backend-api.onrender.com](https://dnb-backend-api.onrender.com) |
| **dnb-ai** (this repo)                                      | FastAPI service for the AI assistant       | [dnb-ai.onrender.com](https://dnb-ai.onrender.com)                   |

## ✨ Features

- 🤖 **Islamic context-aware responses** grounded in a curated system prompt
- 🌍 **Multilingual support** — Arabic, English, Urdu, Malay, French, and more; always quotes Quran in Arabic script with translation
- 🧵 **Conversation history** per chat session
- 🛡️ **Content safety filters** on model output
- 🎚️ **Confidence-aware answers** — abstains or hedges instead of guessing, and routes doubtful religious answers to a scholar
- 🧠 **Per-user long-term memory** — user profiles (knowledge level, madhhab, topics studied, remembered facts) extracted from conversations and injected across sessions; privacy controls with GET/DELETE endpoints and `remember` opt-out per request
- 📋 **Conversation summarization** — compaction API ready for token-budget-triggered eviction; merges and recompresses summaries when history exceeds budget
- 🗺️ **Personalized learning paths** — an ordered, justified "what to study next" drawn strictly from a caller-supplied course catalog, with grounding enforced in code so the model can never recommend a non-catalog or already-completed course
- 📖 **Tafsir-grounded ayah explanations** — retrieved from named classical works, never paraphrased from model memory
- 📚 **Structured citations** — Quran and Hadith references returned as validated, typed objects on every answer, bounds-checked against the 114-surah index
- ⚡ **FastAPI** with automatic OpenAPI docs at `/docs`

## 🔗 API

All chat endpoints require an `X-API-Key` header (see [Authentication & Rate Limiting](#authentication--rate-limiting)).

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/chat` | Start or continue a chat session |
| `POST` | `/chat/stream` | Streaming variant of `/chat` using Server-Sent Events |
| `DELETE` | `/chat/{chat_id}` | Delete a chat session |
| `GET` | `/memory/{user_id}` | Retrieve a stored user profile (transparency) |
| `DELETE` | `/memory/{user_id}` | Completely erase a stored user profile |
| `GET` | `/ping` | Trivial liveness check (always returns 200) |
| `GET` | `/health` | Structured health check - status, version and dependency checks. Returns 200 if all checks pass, 503 otherwise |
| `GET` | `/cache/stats` | Semantic cache metrics (hits, misses, hit rate, etc.) |
| `POST` | `/learning-path` | Personalized, catalog-grounded study path from a learner profile + progress (see [Learning-path contract](#learning-path-contract-for-dnb-backend)) |
| `POST` | `/tafsir` | Ayah explanation from named tafsir works, with attribution |
| `GET` | `/tafsir/sources` | Tafsir works available for retrieval, and their languages |
| `GET` | `/confidence/policy` | Active confidence thresholds and review-queue depth |
| `GET` | `/review/pending` | Answers awaiting a scholar's verdict (reviewer token) |
| `GET` | `/review/reviewed` | Answers that already carry a verdict (reviewer token) |
| `GET` | `/review/{id}` | A single review item (reviewer token) |
| `POST` | `/review/{id}/verdict` | Record approve / correct / reject (reviewer token) |
| `POST` | `/feedback` | Rate a specific answer and flag failure categories |
| `GET` | `/feedback/stats` | Aggregate answer-quality metrics (admin token) |
| `GET` | `/feedback/records` | Browse flagged records, filterable (admin token) |

### Learning-path contract (for `dnb-backend`)

`POST /learning-path` returns a personalized, ordered study path for a learner.
It is a companion to `/study/generate`: same structured-output machinery
(Gemini JSON mode, schema-validated, bounded retry-to-`502`), plus **deterministic
grounding guardrails** on top.

**This service is stateless about the catalog.** Course and book data lives in
[`dnb-backend`](https://github.com/Deen-Bridge/dnb-backend), not here — this AI
service only ever sees purchase *metadata* and holds no catalog or database. So
**the caller (`dnb-backend`) supplies the candidate courses in the request body**,
and that `catalog` is the single source of truth for what may be recommended. The
service **never invents course ids** and never recommends a course absent from the
submitted catalog.

Request body:

| Field | Type | Notes |
|-------|------|-------|
| `profile.level` | `beginner \| intermediate \| advanced` | required |
| `profile.goals` | `[goal]` | required, 1–10; enum: `quran_reading`, `tajweed`, `memorization`, `arabic`, `fiqh_basics`, `aqeedah`, `seerah`, `hadith`, `tafsir` |
| `profile.time_per_week_hours` | `float` | optional, `0 < h ≤ 168` |
| `profile.notes` | `string` | optional, ≤ 1000 chars |
| `progress[]` | `{course_id, title, category, level, completion_pct, quiz_scores?}` | what the learner has studied (`completion_pct` 0–100) |
| `catalog[]` | `{course_id, title, category, level, prerequisites[], description}` | **required, 1–200 items, unique ids** — the courses the learner may be recommended |

Response (`LearningPath`): ordered `steps[]`, each
`{course_id, title, order, reason, prerequisites_satisfied, estimated_weeks}`,
plus a path-level `summary` and a `scholarly_note`.

**Grounding is enforced in code, not just the prompt.** After the model responds,
any step is dropped when its `course_id` is not in `catalog`, when the course is
already completed (`completion_pct ≥ 90`), or when its prerequisites are not yet
satisfied; survivors are renumbered contiguously from 1. If nothing grounded
survives, the request is retried with the violations fed back, and exhaustion
returns `502`. Empty catalog and oversized inputs return `422` **before** any
model call.

```bash
curl -s -X POST "http://localhost:8000/learning-path" \
  -H "Content-Type: application/json" \
  -d '{
        "profile": {"level": "beginner", "goals": ["quran_reading", "tajweed"], "time_per_week_hours": 5},
        "progress": [{"course_id": "arabic-101", "title": "Arabic Alphabet", "category": "arabic", "level": "beginner", "completion_pct": 100}],
        "catalog": [
          {"course_id": "quran-101", "title": "Quran Reading Basics", "category": "quran", "level": "beginner", "prerequisites": ["arabic-101"], "description": "Read from the mushaf."},
          {"course_id": "tajweed-201", "title": "Introduction to Tajweed", "category": "quran", "level": "intermediate", "prerequisites": ["quran-101"], "description": "Rules of recitation."}
        ]
      }'
```

Offline demo (no API key, model mocked): `pytest -q tests/test_learning.py`.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Google Gemini API key](https://ai.google.dev/)

### Setup

```bash
git clone https://github.com/Deen-Bridge/dnb-ai.git
cd dnb-ai

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Copy environment template and add your API key
cp .env.example .env

echo "GEMINI_API_KEY=your_api_key_here" > .env

uvicorn main:app --reload
```

The API runs at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

### Docker

The included `Dockerfile` produces a production-ready image with Python 3.12,
non-root user, cached dependency layer, and a health-check on `/ping`.

```bash
# Build
docker build -t deenbridge-ai .

# Run — pass your Gemini API key via .env file
docker run --env-file .env -p 8000:8000 deenbridge-ai

# …or inline
docker run -e GEMINI_API_KEY=your_key_here -p 8000:8000 deenbridge-ai
```

The container listens on `PORT` (default `8000`), matching Render's runtime
behaviour.  A `HEALTHCHECK` hits `GET /ping` every 30 seconds.

To switch Render from the Python buildpack to Docker, change `render.yaml`:

```yaml
services:
  - type: web
    name: deenbridge-ai
    runtime: docker          # was: env: python
    envVars:
      - key: GEMINI_API_KEY
        sync: false
```

> **Note:** Production currently runs on the Python buildpack. To deploy the
> Docker image instead, set `runtime: docker` in `render.yaml` as shown above.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `SERVICE_API_KEY` | Shared secret for API-key auth; required in production. Clients must send `X-API-Key` header. | — |
| `AUTH_DISABLED` | Set to `true` to skip API-key auth (local development only) | `false` |
| `MODEL_NAME` | Gemini model used by the application | gemini-2.5-flash |
| `TEMPERATURE` | Model temperature | 0.7 |
| `TOP_P` | Nucleus sampling value | 0.8 |
| `TOP_K` | Top-K sampling value | 40 |
| `MAX_OUTPUT_TOKENS` | Maximum response tokens | 2048 |
| `PORT` | Server port used by Uvicorn | 8000 |
| `SEMANTIC_CACHE_ENABLED` | Enable semantic response cache (`1`/`true`/`yes`) | `0` (disabled) |
| `SEMANTIC_CACHE_THRESHOLD` | Minimum cosine similarity for a cache hit | `0.95` |
| `SEMANTIC_CACHE_TTL_SECONDS` | Entry time-to-live in seconds | `86400` (24h) |
| `SEMANTIC_CACHE_MAX_ENTRIES` | Maximum cache entries (LRU eviction) | `1000` |
| `RETRIEVAL_INDEX_PATH` | SQLite path for the persistent retrieval vector index; unset uses the in-memory fallback | — (in-memory) |
| `SAFETY_PIPELINE_ENABLED` | Layered policy enforcement; defaults to `true` | `true` |
| `CONFIDENCE_LOW_THRESHOLD` | Below this score the service abstains | `0.40` |
| `CONFIDENCE_HIGH_THRESHOLD` | At or above this score it answers with no caveat | `0.70` |
| `SCHOLAR_QUEUE_THRESHOLD` | Religious answers at or below this score are queued for review | `0.40` |
| `CONFIDENCE_HIGH_STAKES_PENALTY` | Score multiplier applied to high-stakes rulings | `0.15` |
| `CONFIDENCE_NO_SIGNAL_PRIOR` | Score when no signal is available | `0.55` |
| `CONFIDENCE_UNVERIFIED_CEILING` | Cap when nothing external corroborated the answer | `0.65` |
| `SCHOLAR_REVIEW_TOKEN` | Enables the reviewer endpoints; required as `X-Review-Token` | — (endpoints disabled) |
| `REVIEW_EXPORT_PATH` | JSONL export of reviewed answers | `data/review/reviewed.jsonl` |
| `REDIS_URL` | Makes the scholar-review queue and memory store durable across restarts | — (in-memory) |
| `MEMORY_TTL_DAYS` | Time-to-live for stored user profiles and chat summaries in days | `90` |
| `MEMORY_EXTRACTION_ENABLED` | Background memory extraction from conversation turns | `true` |
| `STELLAR_NETWORK` | Stellar network for zakat lookups (`testnet` or `public`) | `testnet` |
| `ZAKAT_NISAB_USD` | Fallback nisab when no gold price can be fetched | `6000` |
| `NISAB_CACHE_TTL_SECONDS` | How long a fetched gold price is reused | `21600` (6h) |
| `GOLD_PRICE_TIMEOUT` | Gold price request timeout in seconds | `8` |
| `ADMIN_TOKEN` | Enables the feedback admin endpoints; required as `X-Admin-Token` | — (endpoints disabled) |
| `FEEDBACK_DB_PATH` | SQLite path for the feedback store (when Redis is not used) | `feedback.db` |
| `FEEDBACK_RATE_LIMIT_MAX` | Max feedback submissions per IP per window | `20` |
| `FEEDBACK_RATE_LIMIT_WINDOW` | Feedback rate-limit window in seconds | `60` |
| `QURAN_API_BASE` | Base URL for tafsir/ayah retrieval | `https://api.quran.com/api/v4` |
| `QURAN_API_TIMEOUT` | Tafsir request timeout in seconds | `15` |
| `TAFSIR_MAX_AYAT` | Maximum ayat per `/tafsir` request | `10` |
| `TAFSIR_CHAT_EXCERPT_CHARS` | Tafsir characters per work handed to the model in `/chat` | `2500` |
| `TAFSIR_CHAT_TIMEOUT` | Wall-clock budget for tafsir retrieval inside a `/chat` turn | `20` (seconds) |

### Multilingual support (language field)

Pass a `language` field (BCP-47 code) in `ChatRequest` to get a response in that
language. When omitted, the model auto-detects and responds in the user's
language.

```jsonc
{
  "prompt": "ما هي أركان الإسلام؟",
  "language": "ar"
}
```

Quran quotations are always rendered in **Arabic script** with a translation in
the response language and a `surah:ayah` reference, regardless of which language
the response is in.

| Code | Language |
|------|----------|
| `ar` | Arabic |
| `en` | English |
| `ur` | Urdu |
| `ms` | Malay |
| `fr` | French |
| `tr` | Turkish |
| `id` | Indonesian |
| `bn` | Bengali |
| `fa` | Persian |
| `ha` | Hausa |
| `sw` | Swahili |
| `tl` | Tagalog |

An unrecognized code falls back to auto-detection (warns in logs, never 422).
The effective language is echoed in `ChatResponse.language` so the frontend can
set `dir="rtl"` correctly.

### Confidence, abstention, and scholar review

Every chat answer carries a documented 0–1 confidence score, and the service
acts on it rather than answering everything with equal certainty.

**The score** (one formula, in [`confidence.py`](confidence.py)) is a weighted
mean over whatever signals ran for that turn — a component that did not run
drops out of the average instead of being guessed at:

| Signal | Weight | Produced by |
|--------|--------|-------------|
| `self_consistency` | 0.40 | the self-consistency work (#ai-18) — **passed in, never recomputed here** |
| `citation_verification` | 0.30 | [structured citation extraction](citations.py) (#15) — the share of a turn's citations that validated |
| `expressed_certainty` | 0.30 | derived here from the answer's own hedging language |

```
base   = Σ(wᵢ · sᵢ) / Σ(wᵢ)                     over signals present
capped = min(base, UNVERIFIED_CEILING)          if no external signal ran
score  = capped · (1 − HIGH_STAKES_PENALTY)     if the question is a high-stakes ruling
```

Two deliberate choices worth knowing:

- **High stakes is a multiplier, not a fourth signal.** It comes from intent
  classification and applies once — the same evidence should support less
  confidence when being wrong means issuing a wrong ruling. Counting it as both
  a signal and a modifier would double-count it.
- **Self-reported certainty cannot certify itself.** With no external
  corroboration the score is capped below the confident band, so a fluent answer
  that nothing checked gets hedged rather than waved through.

**The bands**, all configurable:

| Band | Score | Behaviour |
|------|-------|-----------|
| abstain | `< CONFIDENCE_LOW_THRESHOLD` | No answer. A pointer to a qualified scholar and authenticated sources. |
| uncertain | `< CONFIDENCE_HIGH_THRESHOLD` | Answers, with an explicit "please verify this" note attached. |
| confident | otherwise | Answers normally. |

**Scholar review.** Religious answers that land in the abstain band are
persisted to a durable queue (Redis when `REDIS_URL` is set — the same store
shape session persistence uses — in-memory otherwise, and **never** with a TTL:
a question waiting on a scholar must not expire unanswered). Low-confidence
*non-religious* answers are hedged but never queued; a scholar's time is for
religious content.

Reviewers list the queue and record a verdict:

```bash
curl -H "X-Review-Token: $SCHOLAR_REVIEW_TOKEN" localhost:8000/review/pending

curl -X POST localhost:8000/review/$ID/verdict \
  -H "X-Review-Token: $SCHOLAR_REVIEW_TOKEN" -H 'Content-Type: application/json' \
  -d '{"verdict": "correct", "corrected_answer": "…", "reviewer": "Shaykh …"}'
```

If Redis is configured but becomes unreachable, the queue keeps accepting items
into an in-process fallback and reports `degraded: true` from `/review/stats`
rather than failing chat turns — the loss of durability is made visible instead
of silent. Verdicts are claimed atomically, so two concurrent reviewers cannot
both record one and silently overwrite each other.

The reviewer endpoints are **closed by default** — without `SCHOLAR_REVIEW_TOKEN`
they return 503 rather than exposing users' pending questions.

Approved and corrected answers flow back through the two sinks that already
exist, not a new pipeline: the semantic cache (#27), and a JSONL export in an
eval-case shape at `REVIEW_EXPORT_PATH` for the eval set (#16) and feedback loop
(#43). Rejected answers are exported too — an answer a scholar caught is a
valuable eval case.

`ChatResponse` gains an optional `confidence: {score, band, abstained, queued,
signals, review_id}` block. It is additive; existing clients are unaffected.

### Structured Quran & Hadith citations

Every chat answer carries a `citations` list of **validated, typed** references --
`2:153` arrives as a surah number, an ayah number, and the surah's name from the
index, not as a substring the client has to parse back out of prose.

```jsonc
{
  "response": "Allah counsels the believers to seek help in patience and prayer...",
  "citations": [
    {"type": "quran", "surah": 2, "ayah_start": 153, "ayah_end": null, "surah_name": "Al-Baqarah"},
    {"type": "hadith", "collection": "Sahih al-Bukhari", "number": 1, "grading": "sahih"},
    {"type": "scholarly", "work": "Riyad as-Salihin", "author": "Al-Nawawi", "detail": "Book of Patience"}
  ]
}
```

**How the model is asked.** Whole-response JSON was rejected deliberately: it
degrades prose quality, and one malformed brace loses the entire answer. The
model instead appends a single delimited block after its normal prose, which is
parsed off and never shown to the user:

```
<<<CITATIONS>>>
{"citations": [{"type": "quran", "surah": 2, "ayah_start": 153}]}
<<<END_CITATIONS>>>
```

On `/chat/stream` the block is withheld from the SSE deltas by a small hold-back
filter, so a half-emitted marker never flickers into the UI; the finished
`citations` array arrives on the terminal `done` event.

**Parsing is total.** A malformed, truncated, or absent block yields an empty
list and the prose is still returned -- nothing in this path can fail a chat
turn. A block cut off by `max_output_tokens` is stripped from the prose anyway:
half a JSON object is worse than no citations at all.

**Validation.** Quran references are bounds-checked against
[`data/quran/surah_index.json`](data/quran/surah_index.json) through the same
114-surah index the tafsir layer validates against, so the two cannot drift
apart -- `2:300` is refused against Al-Baqarah's real 286 ayat. `surah_name` is
always taken from that index and never from the model, which is what makes the
field trustworthy. Hadith collections are normalised through
`hadith.normalize_collection`'s existing alias table (Bukhari, Muslim, Tirmidhi,
Abu Dawud, Nasa'i, Ibn Majah and their variants) and gradings are read from the
bundled grading dataset rather than from the model's claim. A citation naming an
unrecognised collection is rejected, not echoed back. Citations are deduplicated
and capped at 24 per answer.

**It feeds confidence.** The share of a turn's attempted citations that validated
is exactly the `citation_verification` signal [`confidence.py`](confidence.py)
already reserves at weight 0.30 -- this completes machinery that was previously
declared and never fed. A well-cited answer is no longer capped by
`CONFIDENCE_UNVERIFIED_CEILING` (0.65); an answer that cites nothing produces no
signal at all and is not penalised for it.

#### Offline evaluation

`scripts/eval_citations.py` scores a 15-case set at
`data/eval/citations_eval.jsonl` with no API key and no network, and runs in CI:

```bash
python scripts/eval_citations.py --verbose               # offline, canned answers
python scripts/eval_citations.py --live --url http://localhost:8000
```

Three metrics, each with a floor the script enforces:

| Metric | Floor | Meaning |
|--------|-------|---------|
| extraction rate | 90% | answers that should have cited, and did |
| validity rate | 90% | genuine citations that survived validation |
| rejection rate | 100% | planted fabrications that were refused |

Four of the fifteen cases plant fabrications -- an out-of-range surah, an ayah
past a surah's real bound, an unknown collection, and a block mixing one real
citation with one invented one. They are excluded from the validity denominator
and scored only by the rejection rate, so a correct parser is never punished for
refusing them, and a parser that accepts everything cannot pass.

**Known limitation.** A semantic-cache hit replays stored prose with an empty
citation list, because the cache stores text only. No markers leak, and
re-asking with `X-Cache-Bypass` returns citations. Widening the cache record
shape is out of scope for this change.

### Streaming chat responses (Server-Sent Events)

`POST /chat/stream` is a streaming variant of the chat endpoint that returns
tokens incrementally via Server-Sent Events (SSE). Rather than waiting for the
entire Gemini generation to finish, text deltas are forwarded as they arrive
— a modern chat UI shows text appearing in real-time instead of a spinner.

#### Request schema

The same request body as `POST /chat`:

```json
{
  "prompt": "What does Islam say about patience?",
  "chat_id": "optional-existing-session-id",
  "context": "optional-additional-context",
  "madhhab": "hanafi",
  "language": "en"
}
```

#### SSE event protocol

Each line is a `data:` event conforming to the SSE spec. Events are separated
by double newlines (`\n\n`):

1. **Metadata** — emitted first, carries the `chat_id`:
   ```
   data: {"type": "metadata", "chat_id": "<uuid>"}
   ```

2. **Content deltas** — one or more events carrying incremental text. Each
   `delta` is a newly generated token fragment that should be appended to
   whatever the client has already received:
   ```
   data: {"type": "content", "delta": "In "}
   data: {"type": "content", "delta": "the "}
   data: {"type": "content", "delta": "name "}
   ```

3. **Done** — terminal event with the complete response, chat history, and
   metadata (confidence, hadith references, fiqh info, tafsir info, zakat info):
   ```json
   data: {"type": "done", "chat_id": "<uuid>", "history": [...], "text": "...", "confidence": {...}}
   ```

4. **Error** — if an upstream error occurs mid-stream, a terminal error event
   is emitted instead of silently truncating:
   ```json
   data: {"type": "error", "message": "An error occurred during response generation."}
   ```

#### Session behaviour

Streamed turns are stored in the same `active_chats` session store as
non-streamed ones. A follow-up `POST /chat` or `POST /chat/stream` request
with the same `chat_id` will see the streamed answer as context.

#### Example

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What does Islam say about patience?"}'
```

The `-N` flag disables curl's output buffering so text appears incrementally.

#### Safety, telemetry, and confidence

The streaming endpoint applies the same safety pipeline (InputGate before
generation, OutputCheck on the final text), telemetry (Trace spans, model call
recording), hadith authenticity grading, and confidence / abstention assessment
as the non-streaming `/chat` endpoint. All metadata is included in the terminal
`done` event.

### Tafsir (ayah explanation)

`POST /tafsir` explains an ayah from **named** tafsir works instead of from the
model's memory. Every passage is returned with the work, its author, and the
language the text is actually in — attribution comes from the source's own
response, never from the service's recollection of who wrote what.

```bash
curl -X POST http://localhost:8000/tafsir \
  -H 'Content-Type: application/json' \
  -d '{"reference": "103:1-3", "tafsirs": ["ibn-kathir", "tabari", "saadi"], "language": "en"}'
```

```jsonc
{
  "reference": "103:1-3",
  "language": "en",
  "ayat": [
    {
      "ayah": "103:1",
      "surah_name": "Al-'Asr",
      "arabic": "وَٱلْعَصْرِ",
      "translation": "By time,",
      "tafsirs": [
        {"key": "ibn-kathir", "name": "Ibn Kathir (Abridged)", "author": "Ibn Kathir (d. 774 AH)",
         "language": "english", "text": "…", "verse_range": "103:1-3"}
      ],
      "unavailable": [
        {"key": "qurtubi", "name": "Al-Jami' li-Ahkam al-Qur'an (Tafsir al-Qurtubi)",
         "author": "Al-Qurtubi (d. 671 AH)", "reason": "No entry for 103:1 in this tafsir."}
      ]
    }
  ],
  "disclaimer": "Tafsir text is retrieved verbatim from the works named above and is presented for study. …"
}
```

- **References** accept `103:1`, a range `103:1-3`, or a surah name (`Al-Asr 1-3`).
  Bounds are checked offline against [`data/quran/surah_index.json`](data/quran/surah_index.json),
  so `2:300` is a `400` naming Al-Baqarah's 286 ayat — never an invented verse.
- **Language**: tafsirs published in the requested language are served in it. A
  work with no such edition falls back to its original language and is labelled
  with it (set `allow_language_fallback: false` to omit it instead).
- **Degradation**: a work with no entry for the ayah appears under `unavailable`
  with a reason; the rest of the response is unaffected.
- **Latency**: ayat, and the works within an ayah, are fetched concurrently, and
  retrieval inside `/chat` is bounded by `TAFSIR_CHAT_TIMEOUT` — a slow upstream
  costs the turn its grounding, never its response.
- **Caching**: tafsir text is immutable per ayah, so it is cached by exact ayah
  key through `semantic_cache.KeyedCache` — the keyed sibling of the semantic
  response cache, sharing its TTL and eviction settings rather than adding a
  second cache system.

In `/chat`, a verse-explanation question ("what does Surah al-'Asr mean?",
"explain 2:255") is detected offline and answered from the same retrieved
passages, with the model instructed to attribute each claim to a named mufassir
and to surface — not flatten — points where the mufassirun differ. The response
carries a `tafsir` block naming the works whose text actually backed the answer.

### Zakat (on-chain, with a live nisab)

`POST /zakat` computes zakat on a wallet's on-chain USDC balance — 2.5% of the
whole balance once it reaches the nisab, nothing below it.

```bash
curl -sX POST http://localhost:8000/zakat \
  -H 'Content-Type: application/json' \
  -d '{"public_key": "GABC..."}'
```

**The nisab is live.** It is the value of **85g of gold**, so it moves with the
gold market and a hardcoded figure quietly goes wrong in both directions. It is
derived from a spot price fetched from [gold-api.com](https://api.gold-api.com/price/XAU),
falling back to CoinGecko's [PAX Gold](https://www.coingecko.com/en/coins/pax-gold)
price (PAXG is redeemable one-for-one for a troy ounce of allocated gold) and
then to `ZAKAT_NISAB_USD`. Neither source needs an API key. The price is cached
for `NISAB_CACHE_TTL_SECONDS`, and a price outside a plausible range is refused
rather than used — a source that changes its units should degrade, not produce a
nisab off by an order of magnitude.

Every response reports where its threshold came from, so a live figure is never
mistaken for a stale default:

```jsonc
{
  "usdc_balance": "10000.0000000",
  "nisab_usd": "11112.17",
  "zakat_due": "0",
  "message": "Your USDC balance of 10000.0000000 is below the nisab threshold of 11112.17 USD (live gold price via gold-api.com), so no zakat is due on this balance alone.",
  "nisab": {
    "live": true,
    "source": "gold-api.com",
    "basis": "85g gold",
    "gold_price_usd_per_ounce": "4066.199951",
    "as_of": "2026-07-24T15:00:00+00:00"
  },
  "disclaimer": "This is an automated estimate based on your on-chain USDC balance only. ..."
}
```

Pass `nisab_usd` to override the threshold; the response then reports it as a
caller override rather than a market figure.

**In chat.** Asking a zakat question with a public key — "how much zakat do I
owe on my wallet GABC…?" — reads the real balance and answers with the actual
figures, keeping the scholar disclaimer. The key may also arrive in the request's
`context` field. Asking without a key gets an explanation of how zakat is
calculated and an invitation to share a **public** key. Detection is offline
(keywords plus a key-shaped match), so an ordinary message never touches Horizon
or the price API, and a zakat answer is never written to the response cache —
it contains one user's real balance. The answer carries a `zakat` block with the
figures used.

**Strictly read-only.** Only public keys are ever accepted; secret keys fail
validation like any other malformed input, so one can never reach Horizon. If a
message looks like it contains a secret key, the assistant refuses to use it and
warns the user to treat it as compromised — without repeating it back.

**Purchase history in chat.** A signed-in frontend can pass a short
`transactions` summary (hash, amount, status, item title, date, optional memo)
on `POST /chat`, or an `auth_token` so this service fetches
`/api/stellar/payment/transactions` from dnb-backend. Purchase questions then
get factual answers with a [stellar.expert](https://stellar.expert) explorer
link per hash. Without history the assistant says it cannot see purchases;
memos are treated as untrusted data so injection text cannot change behavior.
Purchase answers are never written to the semantic cache and include a
`purchases` block on the response.

### Answer feedback & the quality loop

Every chat answer carries a stable `message_id`, so the frontend can rate a
specific turn. `POST /feedback` attaches an up/down rating and optional failure
categories to that answer:

```bash
curl -sX POST http://localhost:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{"chat_id": "…", "message_id": "…", "rating": "down",
       "categories": ["wrong_or_missing_citation"], "comment": "Ayah number is off"}'
```

- **Snapshot resolution.** The prompt and the *displayed* answer (after any
  safety, hadith, or confidence shaping) are resolved server-side from what the
  user actually saw — the client never has to be trusted for them. On a
  free-tier restart the snapshot is gone; the client then supplies `prompt` and
  `answer`, and a request missing both is a `422`.
- **Validation.** `rating` must be `up`/`down`, categories are checked against a
  fixed taxonomy, and `comment` is length-capped — bad input is a `422`.
- **Idempotent.** Resubmitting for the same `(chat_id, message_id)` overwrites,
  so a user changing their mind never double-counts.
- **Rate-limited** per IP (in-process sliding window), and durably stored in
  SQLite locally or Redis when `REDIS_URL` is set — the same store direction the
  session and scholar-review work use, not a parallel one.

Maintainers read the aggregate signal through two **admin** endpoints, gated on
`X-Admin-Token` and disabled entirely until `ADMIN_TOKEN` is set:

```bash
curl -s http://localhost:8000/feedback/stats   -H "X-Admin-Token: $ADMIN_TOKEN"
curl -s "http://localhost:8000/feedback/records?rating=down&category=too_vague" \
     -H "X-Admin-Token: $ADMIN_TOKEN"
```

#### Eval-candidate export

Down-rated answers become candidates for the evaluation dataset (issue #16
format). Each carries `needs_review: true` and an `answer_draft` for the
reviewer to judge — the script **never** fabricates an expected answer for
religious content:

```bash
# Reads whichever store the service is configured to use (Redis or SQLite):
python scripts/export_eval_candidates.py --output candidates.jsonl
REDIS_URL=redis://localhost:6379 python scripts/export_eval_candidates.py --output candidates.jsonl

# …or force a specific SQLite file:
python scripts/export_eval_candidates.py --db feedback.db --output candidates.jsonl
```

Near-duplicate prompts are deduplicated; approved candidates feed the harness
and, via #56, the semantic cache.

### Retrieval infrastructure (chunking, embedding & vector index)

The [`retrieval/`](retrieval/) package is the shared foundation the RAG epic
builds on — a document **chunking** strategy, an **embedding** step, a
**persistent vector store**, and an **incremental reindex + backfill** pipeline
that keeps the index in sync as content changes. It ships the pipeline only; the
product layers on top (personal-context #1, public-knowledge #2, access-scoped
retrieval #3, hybrid + reranking #5) build against it.

- **Chunking** ([`retrieval/chunking.py`](retrieval/chunking.py)) is
  deterministic and content-type-aware: ayah and hadith records stay **atomic**,
  while long prose is windowed by a token budget with overlap. Every chunk
  carries stable `source` / `source_id` / `content_hash` metadata plus the
  `scope` / `published` fields access-scoped retrieval (#3) filters on.
- **Embedding** reuses the existing `text-embedding-004` seam
  (`semantic_cache.embed_text`, offline-swappable via `set_fake_embedding`) and
  **dedupes by `content_hash`**, so unchanged content is never re-embedded.
- **Vector store** ([`retrieval/index.py`](retrieval/index.py)) is chosen by
  `create_vector_store()` the way `create_session_store` picks its backend: a
  durable **SQLite** store when `RETRIEVAL_INDEX_PATH` is set, or an in-memory
  fallback that keeps local dev and CI offline and restart-free.
- **Incremental sync** upserts changed chunks and deletes removed `source_id`s;
  the **semantic cache now runs on this shared store** (its former linear scan is
  retired).

Rebuild the full index from the bundled corpora — idempotent, so re-running
re-embeds nothing:

```bash
# Durable SQLite index (real embeddings via text-embedding-004):
RETRIEVAL_INDEX_PATH=data/retrieval_index.db python scripts/build_index.py

# Offline demo / CI — deterministic fake embeddings, no network or API key:
python scripts/build_index.py --index-path /tmp/idx.db --fake-embeddings
```

The full schema and design notes live in [`docs/retrieval.md`](docs/retrieval.md).

### Structured logging & request correlation

Logs are **newline-delimited JSON on stdout** ([`logging_config.py`](logging_config.py)),
so Render's log search — or any aggregator added later — can filter by field
instead of grepping prose. Every record carries `timestamp`, `level`, `logger`,
`message` and `request_id`, plus whatever fields the call site attached:

```json
{"timestamp":"2026-08-19T14:02:11.418+00:00","level":"INFO","logger":"main","message":"chat request received","request_id":"2d0f8912b04748c9bc0203d4756f9084","chat_id":"431b0bee-f63a-40c8-a963-1a38d719cdce","new_session":true,"prompt_chars":57,"context_chars":0}
{"timestamp":"2026-08-19T14:02:11.421+00:00","level":"INFO","logger":"telemetry","message":"model call completed","request_id":"2d0f8912b04748c9bc0203d4756f9084","stage":"generation","model":"gemini-2.5-flash","total_tokens":812,"cost_usd":0.00019,"latency_ms":1843.2}
{"timestamp":"2026-08-19T14:02:11.423+00:00","level":"INFO","logger":"request","message":"request completed","request_id":"2d0f8912b04748c9bc0203d4756f9084","http_method":"POST","path":"/chat","status_code":200,"duration_ms":1906.4}
```

**One id per request, everywhere.** A `contextvars`-backed filter stamps the id
onto every record emitted while a request is served — including from modules
that know nothing about HTTP — and the same value goes back on the
**`X-Request-ID`** response header. An inbound `X-Request-ID` is honoured (after
validation, since it is echoed into a header and into logs), so a trace started
in `dnb-backend` continues here. It is also the telemetry trace id, so
`X-Trace-Id` and `X-Request-ID` are the same value: one id links a log search, a
response header and a `/metrics` trace.

**Prompt content is never logged by default.** These are religious questions —
sensitive personal content. Log records carry `prompt_chars` and `chat_id`, not
the prompt. Raw text requires an explicit `LOG_PROMPTS=true` opt-in.

**Failures carry their stack trace.** Exception handlers use `logger.exception`,
so a production 500 lands as one JSON record with a nested `exception` object
holding the type, message and full traceback.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOG_LEVEL` | `INFO` | Root log level |
| `LOG_JSON` | `true` | `false` prints plain text, friendlier on a local console |
| `LOG_PROMPTS` | `false` | `true` includes raw prompt text — sensitive, keep off |
| `LOG_ACCESS` | `false` | `true` restores uvicorn's access log (otherwise the `request completed` record replaces it, avoiding double-logging) |

Offline demo (no API key, model mocked): `pytest -q tests/test_structured_logging.py`.

### Content-safety testing

The versioned policy lives in [`safety/policy.yaml`](safety/policy.yaml), with
review guidance in [`safety/POLICY.md`](safety/POLICY.md). Run the API-key-free
red-team suite with `pytest -q tests/redteam`. A manual live classifier audit is
available with `SAFETY_LIVE_TESTS=1 GEMINI_API_KEY=... pytest -q tests/redteam/test_live.py`.

## ☁️ Deployment

Deployed on [Render](https://render.com) via [`render.yaml`](render.yaml). CI runs lint and syntax checks on every PR (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## 🌊 Contributing & Drips Wave

This repository participates in the **[Stellar Drips Wave](https://www.drips.network/wave/stellar)** bounty program — contributors earn Points (and real rewards) for resolving this repo's issues during a Wave, with complexity tiers set in the Drips Wave app.

- All pull requests target the **`dev`** branch (`main` is releases only)
- CI must pass before review
- One contributor per issue — request it through the campaign (Drips Wave / GrantFox OSS); the maintainer assigns it. Please don't open a PR for an issue you haven't been assigned.

Read **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full workflow, coding standards, and Wave rules.

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- 🌐 Website: [dnb-frontend.vercel.app](https://dnb-frontend.vercel.app)
- 🐦 X/Twitter: [@deen_bridge](https://x.com/deen_bridge)
- 🏢 Organization: [github.com/Deen-Bridge](https://github.com/Deen-Bridge)
