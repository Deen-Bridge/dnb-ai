# Async processing

The service keeps request I/O asynchronous and bounds work that can otherwise
consume the process under load.

## Request path

- Independent chat enrichments run concurrently with `asyncio.gather`. Each
  enrichment remains best-effort, so one unavailable source does not cancel the
  others or fail the chat turn.
- Requests for the same `chat_id` are serialized because the model SDK mutates
  session history. Different chats still run concurrently.
- Model generation is guarded by `LLM_MAX_CONCURRENCY`. Requests above the
  limit wait on an async semaphore instead of opening more upstream calls.
- Synchronous SDK and CPU-heavy calls must use `asyncio.to_thread` or
  `run_in_threadpool`; they must not run directly in an async endpoint.

## HTTP clients

`async_runtime.http_client_pool` owns one `httpx.AsyncClient` per event loop.
Tafsir, backend context, gold-price, and OpenAI-compatible provider calls reuse
that client so TLS handshakes and sockets are pooled.

Tune these together and validate them with the load test:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `HTTP_MAX_CONNECTIONS` | 200 | Total concurrent outbound connections |
| `HTTP_MAX_KEEPALIVE_CONNECTIONS` | 50 | Idle connections retained for reuse |
| `HTTP_KEEPALIVE_EXPIRY_SECONDS` | 30 | Idle keep-alive lifetime |

Do not create an `AsyncClient` inside a per-request `async with` block unless
the caller explicitly owns that short-lived client.

## Background work

Response-side persistence, memory extraction, and summarization use the
priority scheduler:

1. Chat persistence is high priority.
2. Memory extraction is normal priority.
3. History summarization is low priority.

The queue is bounded by `ASYNC_BACKGROUND_QUEUE_SIZE` and worker concurrency by
`ASYNC_BACKGROUND_WORKERS`. Queue depth, failures, rejections, queue wait, and
scheduling overhead are exposed under `async_runtime` on `/metrics`.

Application shutdown first allows work to drain for
`ASYNC_SHUTDOWN_GRACE_SECONDS`, then cancels remaining tasks. Task factories
must therefore handle `asyncio.CancelledError` through normal `finally` cleanup
and must not suppress cancellation.

## Validation

Run the async tests and the existing mock-upstream load budget:

```bash
pytest -q tests/test_async_runtime.py tests/test_chat_robustness.py

MOCK_UPSTREAMS=1 SAFETY_PIPELINE_ENABLED=false AUTH_DISABLED=true \
  uvicorn main:app --port 8000
locust -f loadtest/locustfile.py --headless -H http://127.0.0.1:8000 \
  -u 10 -r 2 -t 45s --csv=/tmp/locust
python loadtest/check_budget.py loadtest/budget.yaml /tmp/locust_stats.csv
```
