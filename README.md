<div align="center">

# 🕌 Deen Bridge — AI Service

**The FastAPI service behind Deen Bridge's Islamic-knowledge AI assistant, powered by Google Gemini.**

[![CI](https://github.com/Deen-Bridge/dnb-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Deen-Bridge/dnb-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](CONTRIBUTING.md)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab.svg)](https://www.python.org/)

[Live API](https://dnb-ai.onrender.com) · [Web App](https://dnb-frontend.vercel.app) · [Report a Bug](https://github.com/Deen-Bridge/dnb-ai/issues) · [Contribute](CONTRIBUTING.md)

</div>

---

## About

This service powers the AI assistant inside **Deen Bridge**, a platform for authentic Islamic education built on the **Stellar network** — courses and books are purchased with USDC, and creators are paid directly to their own Stellar wallets. The assistant wraps Google's Gemini model with an Islamic-knowledge system prompt, content safety filters, and per-session conversation history, exposing a simple chat API consumed by the web app.

On the roadmap: Stellar-aware assistance — zakat calculation from a wallet's on-chain USDC balance via Horizon, and answering questions about the user's on-chain purchases (see the open `wave:*` issues).

The platform is composed of three services:

| Repository                                                  | Role                                       | Live                                                                 |
| ----------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------- |
| [dnb-frontend](https://github.com/Deen-Bridge/dnb-frontend) | Next.js web application                    | [dnb-frontend.vercel.app](https://dnb-frontend.vercel.app)           |
| [dnb-backend](https://github.com/Deen-Bridge/dnb-backend)   | REST API — auth, content, Stellar payments | [dnb-backend-api.onrender.com](https://dnb-backend-api.onrender.com) |
| **dnb-ai** (this repo)                                      | FastAPI service for the AI assistant       | [dnb-ai.onrender.com](https://dnb-ai.onrender.com)                   |

## ✨ Features

- 🤖 **Islamic context-aware responses** grounded in a curated system prompt
- 🧵 **Conversation history** per chat session
- 🛡️ **Content safety filters** on model output
- 🎚️ **Confidence-aware answers** — abstains or hedges instead of guessing, and routes doubtful religious answers to a scholar
- ⚡ **FastAPI** with automatic OpenAPI docs at `/docs`

## 🔗 API

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/chat` | Start or continue a chat session |
| `DELETE` | `/chat/{chat_id}` | Delete a chat session |
| `GET` | `/ping` | Health check |
| `GET` | `/cache/stats` | Semantic cache metrics (hits, misses, hit rate, etc.) |
| `GET` | `/confidence/policy` | Active confidence thresholds and review-queue depth |
| `GET` | `/review/pending` | Answers awaiting a scholar's verdict (reviewer token) |
| `GET` | `/review/reviewed` | Answers that already carry a verdict (reviewer token) |
| `GET` | `/review/{id}` | A single review item (reviewer token) |
| `POST` | `/review/{id}/verdict` | Record approve / correct / reject (reviewer token) |

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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `SEMANTIC_CACHE_ENABLED` | Enable semantic response cache (`1`/`true`/`yes`) | `0` (disabled) |
| `SEMANTIC_CACHE_THRESHOLD` | Minimum cosine similarity for a cache hit | `0.95` |
| `SEMANTIC_CACHE_TTL_SECONDS` | Entry time-to-live in seconds | `86400` (24h) |
| `SEMANTIC_CACHE_MAX_ENTRIES` | Maximum cache entries (LRU eviction) | `1000` |
| `SAFETY_PIPELINE_ENABLED` | Layered policy enforcement; defaults to `true` | `true` |
| `CONFIDENCE_LOW_THRESHOLD` | Below this score the service abstains | `0.40` |
| `CONFIDENCE_HIGH_THRESHOLD` | At or above this score it answers with no caveat | `0.70` |
| `SCHOLAR_QUEUE_THRESHOLD` | Religious answers at or below this score are queued for review | `0.40` |
| `CONFIDENCE_HIGH_STAKES_PENALTY` | Score multiplier applied to high-stakes rulings | `0.15` |
| `CONFIDENCE_NO_SIGNAL_PRIOR` | Score when no signal is available | `0.55` |
| `CONFIDENCE_UNVERIFIED_CEILING` | Cap when nothing external corroborated the answer | `0.65` |
| `SCHOLAR_REVIEW_TOKEN` | Enables the reviewer endpoints; required as `X-Review-Token` | — (endpoints disabled) |
| `REVIEW_EXPORT_PATH` | JSONL export of reviewed answers | `data/review/reviewed.jsonl` |
| `REDIS_URL` | Makes the scholar-review queue durable across restarts | — (in-memory) |

### Confidence, abstention, and scholar review

Every chat answer carries a documented 0–1 confidence score, and the service
acts on it rather than answering everything with equal certainty.

**The score** (one formula, in [`confidence.py`](confidence.py)) is a weighted
mean over whatever signals ran for that turn — a component that did not run
drops out of the average instead of being guessed at:

| Signal | Weight | Produced by |
|--------|--------|-------------|
| `self_consistency` | 0.40 | the self-consistency work (#ai-18) — **passed in, never recomputed here** |
| `citation_verification` | 0.30 | citation verification (#40) — passed in the same way |
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
- One contributor per issue — comment to claim it first

Read **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full workflow, coding standards, and Wave rules.

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- 🌐 Website: [dnb-frontend.vercel.app](https://dnb-frontend.vercel.app)
- 🐦 X/Twitter: [@deen_bridge](https://x.com/deen_bridge)
- 🏢 Organization: [github.com/Deen-Bridge](https://github.com/Deen-Bridge)
