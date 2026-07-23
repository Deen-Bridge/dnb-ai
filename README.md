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
- 📖 **Tafsir-grounded ayah explanations** — retrieved from named classical works, never paraphrased from model memory
- ⚡ **FastAPI** with automatic OpenAPI docs at `/docs`

## 🔗 API

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/chat` | Start or continue a chat session |
| `DELETE` | `/chat/{chat_id}` | Delete a chat session |
| `GET` | `/ping` | Health check |
| `GET` | `/cache/stats` | Semantic cache metrics (hits, misses, hit rate, etc.) |
| `POST` | `/tafsir` | Ayah explanation from named tafsir works, with attribution |
| `GET` | `/tafsir/sources` | Tafsir works available for retrieval, and their languages |

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
| `QURAN_API_BASE` | Base URL for tafsir/ayah retrieval | `https://api.quran.com/api/v4` |
| `QURAN_API_TIMEOUT` | Tafsir request timeout in seconds | `15` |
| `TAFSIR_MAX_AYAT` | Maximum ayat per `/tafsir` request | `10` |
| `TAFSIR_CHAT_EXCERPT_CHARS` | Tafsir characters per work handed to the model in `/chat` | `2500` |
| `TAFSIR_CHAT_TIMEOUT` | Wall-clock budget for tafsir retrieval inside a `/chat` turn | `20` (seconds) |

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
