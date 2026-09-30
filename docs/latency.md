# Chat latency: streaming and the response cache

This document covers how the chat endpoints spend their time, what the semantic
cache changes on each of them, and how to reproduce the numbers below.

## The two chat paths

| Endpoint | Shape | Cache lookup | Cache write |
| --- | --- | --- | --- |
| `POST /chat` | one JSON response after the full generation | yes | yes |
| `POST /chat/stream` | SSE: `metadata` → `content` deltas → `done` | yes | yes |

Both paths use the same two tiers, the same `cache_scope`, and the same
normalized-prompt key, so an answer cached by either endpoint is served by
both:

1. **Exact** — a keyed lookup on `{scope}:{normalized prompt}`. Costs nothing
   and answers the common case of the same question asked twice.
2. **Semantic** — embedding similarity, consulted only when tier 1 misses, so
   a reworded question still hits.

`cache_scope` is `public` for anonymous callers and `user:<id>` for
authenticated ones, so one user's answers are never read by another.

## Where the time goes

An uncached turn is dominated by the model round-trip. Classification, tafsir
and zakat detection are offline (regex plus a bundled index) and cost
microseconds; retrieval only runs when the prompt calls for it.

Streaming attacks the *perceived* cost: the first delta leaves as soon as the
provider emits its first chunk, instead of after the last one. Caching attacks
the *actual* cost: a repeat question skips the provider entirely.

```text
uncached:  [request] ── model stream ────────────────────────▶ [done]
                       ▲ first delta

cached:    [request] ─▶ [deltas] [done]
                        ▲ first delta (in-memory replay)
```

## What a cache hit does on the streaming path

1. The input gate runs first, so a prompt it would refuse is never answered
   from cache. Its verdict is reused by the generator, so classification still
   happens once.
2. The exact tier is checked. Only if it misses is the prompt embedded — off
   the event loop, since the embedding call is blocking HTTP — and matched
   against the semantic tier.
3. On a hit, the answer is replayed as ordinary `content` deltas in
   `CACHE_REPLAY_CHUNK_CHARS`-sized slices, then a `done` event carrying
   `"cached": true` and the confidence block the original asker saw. No model
   call is made.
4. The chat session is seeded with the question and the replayed answer, so a
   follow-up turn in the same `chat_id` — streamed or not — still has context.

`X-Cache-Tier: exact | semantic | miss` and
`X-Semantic-Cache: hit | miss | bypass` are set on the response either way, and
`X-Cache-Bypass: 1` on the request forces a fresh generation.

### What is never cached

A turn is only eligible when all of the following hold. This is one predicate,
`cache_eligible()` in `main.py`, shared by both endpoints so they cannot drift
apart.

- **It is the first message of a chat** — decided from the session store, not
  from the process-local `active_chats` dict. That dict is empty after a
  restart and on every other worker, so trusting it would let a resumed
  conversation be answered from cache as though it were a fresh question.
- **No `context`**, and no tafsir, zakat, purchase or personal-memory
  retrieval — each grounds the answer in something that belongs to one asker.
- **No `language` or `madhhab`.** Both reshape the answer while leaving the
  prompt untouched, and the key is derived from the prompt alone. Folding them
  into the embedded text would not help: two strings differing by a short
  prefix sit far inside the similarity threshold. Serving them correctly needs
  a variant-keyed store, so until there is one they are left out.
- **The input gate returned a plain `allow`.** A refusal must never be answered
  from cache, and a guidance-shaped answer belongs to the prompt that earned
  the guidance.

On top of that, only non-abstained answers are written: an abstention must not
outlive the doubt that produced it.

## Measured results

[`scripts/bench_stream_cache.py`](../scripts/bench_stream_cache.py) serves the
real app under uvicorn on a loopback port and drives `POST /chat/stream` over
HTTP against a stub model that streams four chunks with a fixed delay each. The
stub stands in for the provider so the numbers are reproducible and offline —
what is measured is the effect of the cache, not the day's Gemini weather.

The measurement runs over a real socket on purpose: httpx's in-process ASGI
transport buffers the whole body before yielding a line, which collapses TTFB
onto total time and would report a streaming endpoint as though it did not
stream.

```console
$ python -m scripts.bench_stream_cache --rounds 8 --chunk-delay 0.4

rounds per arm: 8   stub chunk delay: 0.4s

              TTFB (ms)   total (ms)
uncached         408.73      1625.43
cached             9.23        12.24

TTFB reduction:  97.7%
total reduction: 99.2%
```

* **TTFB** is time to the first `content` delta — when the user first sees
  text. Uncached it tracks the provider's first chunk (~400 ms here); cached it
  is an in-memory replay.
* **Total** is time to the `done` event.

The uncached arm sends `X-Cache-Bypass: 1` on every request, so each one pays a
full round-trip; otherwise only the first would and the "before" figure would
be an average of one slow request and several fast ones.

`--min-reduction` (default 50 %, the issue's target) makes the script exit
non-zero if the cached arm fails to beat the uncached one by that margin, or if
any warm request was not actually served from cache. CI runs it that way, so a
change that quietly stops caching fails the build rather than the review.

Against the acceptance criteria: streaming already puts first text on screen
well inside 1–2 s (TTFB is one provider chunk, not a whole answer), and the
cache takes a repeat question from 1.6 s to 19 ms — far past the 50 % reduction
the issue asks for.

### Reproducing by hand

With a server running (`make run`) and `SEMANTIC_CACHE_ENABLED=1`. The
timings below are from a run against the same stub provider the benchmark
uses, so they are comparable with the table above; against live Gemini the
uncached figure is whatever the model takes that day, and the cached one is
unchanged.

```console
$ curl -sN -D - -X POST localhost:8000/chat/stream \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"What are the five daily prayers?"}' \
    -w '\ntotal=%{time_total}s\n'
HTTP/1.1 200 OK
x-cache-tier: miss
x-semantic-cache: miss
...
total=2.020285s

$ # the same question again — answered by the exact tier
x-cache-tier: exact
x-semantic-cache: hit
...
total=0.003869s

$ # reworded — misses tier 1, matched by embedding
x-cache-tier: semantic
x-semantic-cache: hit
```

`GET /cache/stats` reports hits, misses, bypasses, evictions and hit rate.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SEMANTIC_CACHE_ENABLED` | `0` | Master switch. Off, both endpoints always generate. |
| `SEMANTIC_CACHE_THRESHOLD` | `0.95` | Minimum cosine similarity for a hit. |
| `SEMANTIC_CACHE_TTL_SECONDS` | `86400` | How long an entry may be replayed. |
| `SEMANTIC_CACHE_MAX_ENTRIES` | `1000` | LRU capacity. |

The cache is in-memory by design: a stale answer must never outlive a restart.
