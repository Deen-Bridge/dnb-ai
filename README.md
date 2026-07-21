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

| Repository | Role | Live |
|------------|------|------|
| [dnb-frontend](https://github.com/Deen-Bridge/dnb-frontend) | Next.js web application | [dnb-frontend.vercel.app](https://dnb-frontend.vercel.app) |
| [dnb-backend](https://github.com/Deen-Bridge/dnb-backend) | REST API — auth, content, Stellar payments | [dnb-backend-api.onrender.com](https://dnb-backend-api.onrender.com) |
| **dnb-ai** (this repo) | FastAPI service for the AI assistant | [dnb-ai.onrender.com](https://dnb-ai.onrender.com) |

## ✨ Features

- 🤖 **Islamic context-aware responses** grounded in a curated system prompt
- 🧵 **Conversation history** per chat session
- 🛡️ **Content safety filters** on model output
- ⚡ **FastAPI** with automatic OpenAPI docs at `/docs`
- 👍 **User feedback** — per-message ratings and failure categories with durable storage
- 📊 **Quality dashboard** — aggregate stats and records endpoints for maintainers
- 🧪 **Eval-candidate export** — grow the evaluation harness from real user pain

## 🔗 API

| Method | Route | Auth | Purpose |
|--------|-------|------|---------|
| `POST` | `/chat` | — | Start or continue a chat session |
| `DELETE` | `/chat/{chat_id}` | — | Delete a chat session |
| `GET` | `/ping` | — | Health check |
| `POST` | `/feedback` | — | Submit a rating for a specific model answer |
| `GET` | `/feedback/stats` | `X-Admin-Token` | Aggregate quality metrics |
| `GET` | `/feedback/records` | `X-Admin-Token` | Recent flagged records (filterable) |

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

echo "GEMINI_API_KEY=your_api_key_here" > .env

uvicorn main:app --reload
```

The API runs at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | ✅ |
| `ADMIN_TOKEN` | Secret header value for `/feedback/stats` and `/feedback/records` | ✅ in prod |
| `REDIS_URL` | Redis connection URL for persistent feedback storage | optional (SQLite fallback) |
| `FEEDBACK_DB_PATH` | Path to the SQLite feedback database | optional (default: `feedback.db`) |
| `ZAKAT_NISAB_USD` | Nisab threshold in USD for zakat calculation | optional (default: 6000) |
| `STELLAR_NETWORK` | `testnet` or `public` | optional (default: `testnet`) |

### Feedback Taxonomy

The `categories` field on `/feedback` accepts any subset of these validated labels:

| Category | Meaning |
|----------|---------|
| `incorrect_information` | Factually wrong answer |
| `wrong_or_missing_citation` | Hadith/Quran reference is wrong or absent |
| `one_sided_fiqh_answer` | Only one scholarly opinion presented |
| `too_vague` | Answer lacks sufficient detail |
| `too_long` | Answer is unnecessarily verbose |
| `wrong_language` | Response in wrong language |
| `poor_adab` | Tone or etiquette inappropriate |
| `refused_unnecessarily` | Model declined a legitimate question |
| `other` | Doesn't fit any above category |

### Feedback API Contract

`POST /feedback` body:
```json
{
  "chat_id":    "<uuid from /chat>",
  "message_id": "<uuid from /chat response>",
  "rating":     "up" | "down",
  "categories": ["<taxonomy label>", ...],
  "comment":    "<optional, max 1000 chars>",
  "prompt":     "<user question — required when session is no longer live>",
  "answer":     "<model answer — required when session is no longer live>"
}
```

> **Session-gone fallback**: the free-tier Render instance restarts frequently.
> If the session is no longer in memory, the client **must** supply `prompt` and
> `answer` in the request body; otherwise a `422` is returned.

Rate limiting: 20 submissions per IP per 60 s (in-process sliding window —
stopgap until issue #9 provides real auth infrastructure).
Resubmitting for the same `(chat_id, message_id)` overwrites rather than duplicates.

### Eval-candidate Export

Convert down-rated records into evaluation-dataset candidates in the issue #16 harness format:

```bash
python scripts/export_eval_candidates.py --output candidates.jsonl
# Redis store:
REDIS_URL=redis://localhost:6379 python scripts/export_eval_candidates.py --output candidates.jsonl
```

Each emitted entry carries `needs_review: true`. A human curator must supply
`expected_answer` before any record enters the golden set — the script
**never fabricates expected answers** for religious content.

## ☁️ Deployment

Deployed on [Render](https://render.com) via [`render.yaml`](render.yaml). CI runs lint, syntax checks, and the full pytest suite on every PR (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## 🌊 Contributing & Drips Wave

This repository participates in the **[Stellar Drips Wave](https://www.drips.network/wave/stellar)** bounty program — contributors earn Points (and real rewards) for resolving this repo's issues during a Wave, with complexity tiers set in the Drips Wave app.

- All pull requests target the **`dev`** branch (`main` is releases only)
- CI must pass before review
- One contributor per issue — comment to claim it first

Read **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full workflow, coding standards, and Wave rules.

## 📜 License

[MIT](LICENSE) © Deen Bridge

## 🔗 Links

- 🌐 Website: [dnb-frontend.vercel.app](https://dnb-frontend.vercel.app)
- 🐦 X/Twitter: [@deen_bridge](https://x.com/deen_bridge)
- 🏢 Organization: [github.com/Deen-Bridge](https://github.com/Deen-Bridge)
