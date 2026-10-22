# Architecture

How a question becomes a cited, safety-checked answer in `dnb-ai`, and where to
make a change without breaking the pipeline.

The service is a single FastAPI app (`main.py`) that boots without any external
service: retrieval corpora are bundled, classification is local and
deterministic, and only generation, persistence and the memory store reach out
to the network. That property is what lets the test suite and CI import `main`
with no API key.

---

## 1. Request lifecycle for `POST /chat`

The handler is `chat()` in `main.py`. Work happens in the following order;
every stage is timed and recorded on the telemetry trace.

```
POST /chat
  │
  ├─ rate limit (slowapi) ─────────────── 429 on excess
  ├─ Pydantic validation of ChatRequest ─ 422
  │
  ├─ 1. Redaction
  ├─ 2. Classification  (fiqh, madhhab, language, intent, Swahili)
  ├─ 3. Retrieval       (tafsir, zakat, purchases, Stellar)
  ├─ 4. Memory load     (profile + chat summary, only with user_id)
  ├─ 5. Cache lookup    (exact → semantic; skippable with X-Cache-Bypass: 1)
  ├─ 6. Token quota     ───────────────── 429 when exhausted
  ├─ 7. Safety + generation (Gemini, inside the input/output gates)
  ├─ 8. Post-processing (citations, hadith grading, evidence verification)
  ├─ 9. Confidence / abstention / scholar queue
  ├─ 10. Cache write    (confident answers only)
  └─ 11. Response + background persistence and memory extraction
```

### 1.1 Redaction

`redact_secret_keys()` strips Stellar secret keys from both `prompt` and
`context` before anything else touches them. The redacted text is what reaches
the model, the caches and stored history. The zakat layer separately notices a
key was present and warns the user.

### 1.2 Classification (`trace.span("classification")`)

All offline and deterministic — no model calls:

| Concern | Module | Notes |
| --- | --- | --- |
| Fiqh question? | `fiqh.py` (`classify_fiqh`) | Drives ikhtilaf/madhhab system context and high-stakes confidence handling |
| Madhhab | `main.normalize_madhhab` | Injected as a lead-in instruction, never invented |
| Language | `main.normalize_language` | `en`/`ar`/`sw`/… |
| Intent | `intent.py` (`classify_intent`) | Multi-label; predictions recorded for accuracy tracking |
| Swahili | `swahili_processor.py` | Dialect, code-switching and cultural notes for East African users |

### 1.3 Retrieval (`trace.span("retrieval")`)

Four best-effort retrievers. Each is wrapped so a retrieval failure logs a
warning and answers without that context; retrieval can never fail the turn.

| Retriever | Source | What it produces |
| --- | --- | --- |
| `tafsir_retriever` | `tafsir.py` + bundled surah index | `TafsirContext` with passages for tafsir questions |
| `zakat_retriever` | `stellar.py`, `nisab.py`, `faraid.py` | `ZakatContext` incl. gold-price/nisab calculation |
| `purchase_retriever` | `stellar.py` | `PurchaseContext` from inline transactions or a JWT fetch |
| `build_chat_stellar_calculation_context` | `stellar.py` | Prompt block for Stellar/Soroban questions |

### 1.4 Memory load

With `user_id` set, `memory_store.get_profile()` and
`memory_store.get_chat_summary()` run. `render_user_context()` turns them into a
delimited DATA block appended to the system context.

### 1.5 Two-tier response cache

`semantic_cache.py` provides both tiers:

1. **Exact** — `get_chat_exact_cache()`: keyed by `scope:normalize_text(prompt)`.
2. **Semantic** — `semantic_cache.get(embedding, scope=…)`: local embedding
   (`embed_text`), cosine similarity, TTL and LRU eviction.

Only *new* chats without explicit `context` and without retrieved
tafsir/zakat/purchase/Stellar context are cacheable. Scope is `public` for
anonymous requests and `user:{id}` otherwise, so one user's answer can never be
replayed to another.

### 1.6 Quota, safety, generation

`token_quota_tracker.is_allowed()` runs before any LLM call. Generation then
happens **inside** `safety_pipeline.run_async(prompt, generate)`:

- `safety/input_gate.py` classifies and can refuse before the model is called.
- `generate()` builds the system context (`ISLAMIC_CONTEXT`,
  `HADITH_ADAB_CONTEXT`, `CITATION_BLOCK_CONTEXT`, plus fiqh/madhhab,
  retrieval, Swahili and memory blocks), starts or reuses the Gemini chat
  session for `chat_id`, and calls `send_message_with_retry()`.
- `safety/output_check.py` can block or rewrite the model output afterwards.
- Set `SAFETY_PIPELINE_ENABLED=0` to bypass the gates entirely.

`SafetyResult.generator_called` is what the cache write consults, so a
pre-model refusal is never cached as an answer.

### 1.7 Post-processing

Ordered, and the order matters:

1. `citations.extract_citations()` — removes the structured citation block
   before the user sees it, returns a `CitationExtraction` with a verification
   score.
2. `hadith.annotate_hadith()` — grades narrated hadith and appends caution
   notes for weak chains.
3. `verify_evidence_claims()` (`evidence_agent.py`) — best-effort; failure logs
   and continues.
4. `confidence.build_signals()` → `assess()` → `_calibrate_assessment()` —
   produces the `ConfidenceAssessment` (score, band, signals used).

### 1.8 Confidence and scholar escalation

`confidence.py` bands are `abstain`, `uncertain` and `confident` (see
`ConfidenceBand`); `uncertain` answers are delivered with an uncertainty note.
Notable rules:

- Without an external signal the score is capped at `UNVERIFIED_CEILING`
  (default 0.65) — this is why citation verification is important: it lifts the
  ceiling.
- Non-confident turns are queued via `review.enqueue_for_review()`
  (persisted by `review_store.py`); the queue is only attempted before reply
  shaping, so the user is told "sent to a scholar" only when it really was.
- `apply_policy()` shapes the final text (abstention message or hedge note).

### 1.9 Cache write, persistence, response

Only `CONFIDENT` answers are written, and only if the generator actually ran.
Both tiers are written with the post-processed text, so a cache hit replays
exactly what the first asker saw (citations stripped, hadith caution included).

The response carries `X-Trace-Id`, `X-LLM-Total-Tokens`, `X-LLM-Cost-USD`,
`X-Handler-Latency-Ms` and `X-Cache-Tier` (`exact` / `semantic` / `miss`).
Chat history persistence and memory extraction/summarization are scheduled with
`asyncio.create_task` after the response is built.

`POST /chat/stream` is the SSE twin: same `ChatRequest`, same safety pipeline,
same citation/hadith/confidence handling, streamed instead of returned at once.

---

## 2. Subsystem map

```
main.py  ── HTTP surface, /chat + /chat/stream handlers, lifespan wiring
 ├── classification:  fiqh.py, intent.py, arabic_dialect/, malay/, swahili/,
 │                    urdu_processor.py, swahili_processor.py, sentiment.py
 ├── retrieval:       tafsir.py, stellar.py, nisab.py, faraid.py, corpus.py,
 │                    vector_store.py, cross_reference.py, hadith*.py,
 │                    thematic_quran.py, quran_benchmark.py
 ├── orchestration:   orchestration.py (generic DAG engine),
 │                    research_orchestration.py (Quran/hadith/tafsir/fiqh agents),
 │                    agent_orchestrator.py (capability registry + scheduler),
 │                    agent_fallback.py (circuit breaker + retry + degradation)
 ├── generation:      Google Gemini via google-generativeai
 ├── grounding:       grounding.py (source overlap, synthesis, block rules)
 ├── evaluation:      confidence.py, citations.py, citation_verification.py,
 │                    verifier.py, hallucination.py, misinformation.py,
 │                    evidence_agent.py, llm_judge.py, quality_judge.py,
 │                    self_consistency.py, regression.py, review.py
 ├── safety:          safety/ (policy, input_gate, output_check, pipeline)
 ├── caching:         semantic_cache.py
 ├── memory:          memory/ (models, store, extraction) + history.py, store.py
 └── telemetry:       telemetry.py, errors.py
```

### 2.1 Grounding (`grounding.py`)

Pure local post-processing, no LLM: it builds a keyword `SourceIndex` over
retrieved passages, scores each generated sentence (`sentence_fidelity`),
detects unsupported sentences (`detect_hallucinations`), aggregates into a
`GroundingReport` (`compute_grounding`) and maps it to a
`GroundingAction` — `pass`, `warn` or `block` (`apply_grounding_rules`). It also
merges multiple agent responses (`synthesize_responses`) with deduplication,
contradiction resolution, terminology normalization and attribution merging.
Fail-open by design: bad input yields a zero score, never an exception.
`grounding.py` is not yet called from `/chat`; it is a library seam intended
to run after generation, next to `confidence.py`. Its behavioral contract is
pinned by `tests/test_grounding.py`.

### 2.2 Evaluation

- **Citation verification** (`citations.py`, `citation_verification.py`,
  `verifier.py`) — parses the structured block; validates Qur'an references
  against the surah index and hadith references against the collection table.
- **Hallucination scanning** (`hallucination.py`) — fabricated citations,
  misattributions, temporal errors, scholar-position errors; also a benchmark
  harness (`run_benchmark`).
- **Misinformation / adab** (`misinformation.py`) — misconception catalogue,
  quotation validation, respectful-disagreement enforcement.
- **Confidence** (`confidence.py`) — signal averaging, banding, abstention and
  scholar escalation policy.
- **Judges and regressions** (`llm_judge.py`, `quality_judge.py`,
  `self_consistency.py`, `regression.py`, `quran_benchmark.py`,
  `tests/redteam/`) — offline evaluation of answer quality.

### 2.3 Safety (`safety/`)

`policy.py` loads the moderation policy; `InputGate`/`OutputCheck` evaluate
prompts and generated text; `pipeline.py` is the single orchestration seam that
guarantees the generator is only invoked when the input gate allows it, and that
`stages_fired`, `category_id`, `action` and latency are reported back to the
handler. Gemini's own `safety_settings` (built by `get_safety_settings()`) is a
second, independent layer.

### 2.4 Agents and orchestration

- `orchestration.py` — provider-independent engine: `AgentRegistry` +
  `AgentSpec`, `QueryAnalyzer` → `ExecutionPlanBuilder`, DAG validation
  (`PlanValidationError`), parallel waves, per-agent `CircuitBreaker`,
  `AsyncRateLimiter`, `TraceSpan`/`Metrics`, `ResultSynthesizer`. No network at
  import time.
- `research_orchestration.py` — wires the engine to offline Quran/hadith/tafsir/
  fiqh research agents and exposes traces at
  `GET /orchestration/traces/{workflow_id}`.
- `agent_orchestrator.py` — alternative capability-based scheduler
  (`AgentCapability`, `ExecutionMode`, `DAGScheduler`,
  `MultiAgentOrchestrator`) with its own circuit breaker and result synthesis.
- `agent_fallback.py` — resilience primitives: `CircuitBreakerConfig`,
  `RetryConfig`, `CircuitBreaker` (closed → open → half-open), health
  (`AgentHealth`), alternative-agent routing and graceful degradation.
- `tool_reliability.py` — tool descriptions, selection and execution traces for
  tool-using agents.

### 2.5 Caching

`semantic_cache.py` has three consumers: the semantic response cache
(`SemanticCache`, `get_cache`), the exact-match response cache
(`get_chat_exact_cache` → `KeyedCache`), and namespaced keyed caches
(`get_keyed_cache`). Entries are scoped (`public` / `user:{id}`),
`invalidate_user_cache`/`invalidate_public_cache` allow targeted purges, and
`/cache/stats` exposes hit/miss counters. Tafsir answers are additionally cached
by ayah key inside `tafsir.py`.

### 2.6 Memory

`memory/models.py` defines `UserProfile` and `ChatSummary`; `memory/store.py`
picks Redis when `REDIS_URL` is set and an in-memory dict otherwise
(`create_memory_store()`); `memory/extraction.py` extracts durable facts from a
finished turn. Summaries are merged once a chat exceeds
`MAX_CHAT_HISTORY_TURNS` (20). Memory is exposed at `GET /memory/{user_id}`.
Chat *history* is separate: `store.py` / `history.py` and a Firestore → Redis →
in-memory session store selected at startup.

### 2.7 Telemetry

`telemetry.py` is standard-library only and content-free by construction: it
records token counts, cost, per-stage spans and request outcomes, never prompt
or answer text. `Trace` is a `ContextVar` per request; `record_model_call()`
captures each Gemini call; `registry` aggregates for `GET /metrics`.
`errors.py` provides the typed error shapes used by the JSON error responses.

---

## 3. Model routing, fallback and circuit breaking

### 3.1 Routing (`model_router.py`)

`ModelRouter` chooses the cheapest model that still meets a query's accuracy
and latency needs, deterministically and without network calls:

1. `classify_query()` → `QueryFeatures` (estimated tokens, Arabic script
   ratio, Islamic-knowledge domains, 0–1 complexity, band).
2. `route_query()` walks the `ModelProfile` registry, skips unavailable models
   and scores the rest on a weighted accuracy/latency/cost blend. The winner is
   returned with the remaining models as the fallback chain
   (`RoutingDecision.fallbacks`).
3. `bucket_strategy()` hashes the query to keep A/B strategy assignment stable
   for the same question (`RoutingStrategy`).
4. `record_feedback()` adjusts a rolling per-model quality score from real
   outcomes, gradually demoting disappointing models.

It is exposed as a JSON API under `/routing` (`/routing/route`,
`/routing/models`, `/routing/metrics`, `/routing/feedback`) and is importable as
a pure library. `/chat` itself currently pins one Gemini model
(`telemetry.GEMINI_MODEL` via `get_model()`), so routing decisions are advisory
for the chat path today; adopting them in `/chat` only requires reading the
decision instead of `get_model()`.

### 3.2 Fallback and circuit breaking

Three layers, each independent:

1. **Transport retries** — `send_message_with_retry()` in `main.py` retries
   `TimeoutError` / `ServiceUnavailable` / `DeadlineExceeded` twice (max 3
   attempts) with exponential backoff (`0.5s, 1s, …` and jitter), and rolls the
   Gemini chat history back to its pre-call length so a failed turn never leaves
   an unanswered user message in the conversation.
2. **Agent circuit breakers** — `orchestration.CircuitBreaker` and
   `agent_fallback.CircuitBreaker` implement closed → open → half-open with a
   failure threshold and recovery window. An open circuit raises
   `CircuitOpenError`/`AgentUnavailableError`, so the orchestrator routes to an
   alternative agent or degrades to partial results instead of retrying a dead
   dependency.
3. **HTTP error mapping** — the `/chat` handler converts upstream failures into
   typed `APIException`s with hints: 429 (rate limit/quota), 400
   (`InvalidArgument`), 503 (`ServiceUnavailable`), 504 (timeout), 500
   (unexpected, including an empty model response). Every error response keeps
   `X-Trace-Id` so a report can be correlated with logs.

---

## 4. Touchpoints at a glance

| Concern | Read | Write |
| --- | --- | --- |
| Response cache | step 5, before any LLM call | step 10, confident answers only |
| Memory | step 4, profile + summary | background extraction after response |
| Session history | on chat-session creation | background persist after response |
| Telemetry | every stage (`Trace` spans) | headers, `/metrics` registry |
| Scholar queue | — | step 9, non-confident answers |

## 5. Testing and quality gates

- `pytest` with shared fixtures in `conftest.py` and `tests/conftest.py`;
  endpoint tests swap Firestore for an in-memory store, and LLM/network calls
  are stubbed, so the suite is hermetic.
- `ruff` (line length 120, pyflakes/pyupgrade/bugbear/isort) and `mypy`
  (`explicit_package_bases`, pydantic plugin) gate changes.
- Every stage above is independently testable: classification and retrieval are
  offline, generation is behind `send_message_with_retry`, and post-processing
  (`grounding`, `citations`, `hadith`, `confidence`) is pure Python.

## 6. Where to make a change

| Change | Start here |
| --- | --- |
| New answer stage | a pure module + a call in `/chat` post-processing |
| New retrieval source | a `*_retriever` seam in `main.py` + prompt block |
| Prompt/context tuning | `ISLAMIC_CONTEXT` / `*_CONTEXT` constants in `main.py` |
| Safety policy | `safety/policy.py` and the gate implementations |
| Confidence thresholds | `confidence.py` env vars (`CONFIDENCE_*`) |
| Grounding thresholds | `grounding.py` env vars (`GROUNDING_*`) |
| Model choice / routing | `model_router.py` (`_default_profiles`) |
| Resilience | `agent_fallback.py`, `send_message_with_retry` |
