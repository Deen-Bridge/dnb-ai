"""Measure streaming-chat latency with and without the semantic cache (#1).

Serves the real app from a uvicorn server on a loopback port and drives
``POST /chat/stream`` over HTTP, against a stub model that emits an answer in
chunks with a fixed per-chunk delay standing in for the provider's token
stream. That keeps the numbers reproducible and offline: what is being
measured is the effect of the cache, not the day's Gemini weather.

A real socket matters here. httpx's in-process ASGI transport buffers the
whole response body before yielding a line, which collapses TTFB onto total
time and would report a streaming endpoint as though it did not stream.

Reported per run:

* **TTFB** — time to the first ``content`` delta, i.e. when the user first
  sees text. This is the number the issue's "within 1 to 2 seconds" target and
  its perceived-latency goal are about.
* **Total** — time to the terminal ``done`` event.

Usage::

    python -m scripts.bench_stream_cache
    python -m scripts.bench_stream_cache --rounds 10 --chunk-delay 0.25
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

# ---------------------------------------------------------------------------
# Stub model — emulates a provider that streams `chunks` with a delay each
# ---------------------------------------------------------------------------

ANSWER_CHUNKS = [
    "The five daily prayers are Fajr, Dhuhr, Asr, ",
    "Maghrib and Isha. They are obligatory upon every adult Muslim ",
    "and are established at fixed times.",
    '<<<CITATIONS>>>{"citations": [{"type": "quran", "surah": 4, "ayah_start": 103}]}<<<END_CITATIONS>>>',
]


@dataclass
class _Content:
    role: str
    text: str

    @property
    def parts(self) -> list[Any]:
        return [SimpleNamespace(text=self.text)]


class _StreamResponse:
    def __init__(self, chunks: list[str], delay: float) -> None:
        self._chunks = chunks
        self._delay = delay

    async def __aiter__(self):
        for chunk in self._chunks:
            await asyncio.sleep(self._delay)
            yield SimpleNamespace(text=chunk)

    async def resolve(self) -> None:
        return None


@dataclass
class _ChatSession:
    delay: float
    history: list[_Content] = field(default_factory=list)

    async def send_message_async(self, message: str, **kwargs):
        answer = "".join(ANSWER_CHUNKS)
        self.history.extend([_Content("user", message), _Content("model", answer)])
        if kwargs.get("stream"):
            return _StreamResponse(ANSWER_CHUNKS, self.delay)
        return SimpleNamespace(text=answer, candidates=[SimpleNamespace(finish_reason="STOP")], prompt_feedback=None)


class _Model:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def start_chat(self, history=None) -> _ChatSession:
        session = _ChatSession(self.delay)
        for content in history or []:
            session.history.append(_Content(content["role"], content["parts"][0]["text"]))
        return session


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class Timing:
    """One streamed request, timed from the client's side."""

    ttfb_ms: float
    total_ms: float
    cached: bool


@dataclass
class BenchResult:
    cold: list[Timing]
    warm: list[Timing]

    @staticmethod
    def _median(values: list[float]) -> float:
        return round(statistics.median(values), 2) if values else 0.0

    def summary(self) -> dict[str, Any]:
        cold_ttfb = self._median([t.ttfb_ms for t in self.cold])
        warm_ttfb = self._median([t.ttfb_ms for t in self.warm])
        cold_total = self._median([t.total_ms for t in self.cold])
        warm_total = self._median([t.total_ms for t in self.warm])
        return {
            "rounds": len(self.cold),
            "cold_ttfb_ms": cold_ttfb,
            "warm_ttfb_ms": warm_ttfb,
            "cold_total_ms": cold_total,
            "warm_total_ms": warm_total,
            "ttfb_reduction_pct": round((1 - warm_ttfb / cold_ttfb) * 100, 1) if cold_ttfb else 0.0,
            "total_reduction_pct": round((1 - warm_total / cold_total) * 100, 1) if cold_total else 0.0,
            # An empty arm is not a pass: with no warm requests there is no
            # evidence the cache served anything.
            "all_warm_cached": bool(self.warm) and all(t.cached for t in self.warm),
        }


@contextmanager
def serve(app: Any) -> Iterator[str]:
    """Run *app* under uvicorn on a free loopback port; yield its base URL."""
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start within 30s")
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30)


def _time_stream(client: Any, url: str, prompt: str, bypass: bool = False) -> Timing:
    headers = {"X-Cache-Bypass": "1"} if bypass else {}
    start = time.perf_counter()
    ttfb: float | None = None
    cached = False
    with client.stream("POST", url, json={"prompt": prompt}, headers=headers) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            if event["type"] == "content" and ttfb is None:
                ttfb = (time.perf_counter() - start) * 1000
            elif event["type"] == "done":
                cached = bool(event.get("cached"))
            elif event["type"] == "error":
                raise RuntimeError(f"stream failed: {event['message']}")
    total = (time.perf_counter() - start) * 1000
    if ttfb is None:
        raise RuntimeError("stream produced no content delta")
    return Timing(ttfb_ms=round(ttfb, 2), total_ms=round(total, 2), cached=cached)


def run_benchmark(rounds: int = 5, chunk_delay: float = 0.25) -> BenchResult:
    """Time *rounds* uncached and cached streamed answers to the same question.

    The uncached runs send ``X-Cache-Bypass: 1`` so every one of them pays the
    full model round-trip — otherwise only the first would, and the "before"
    figure would be an average of one slow request and four fast ones.
    """
    # No request leaves the process — the model is stubbed and embeddings are
    # faked — but Settings still requires a key to build the app. Safety
    # classification is off because it calls the provider, and because it runs
    # only on the generated path: leaving it on would flatter the cached arm.
    previous_api_key = os.environ.get("GEMINI_API_KEY")
    os.environ.setdefault("GEMINI_API_KEY", "bench-key")
    previous_safety = os.environ.get("SAFETY_PIPELINE_ENABLED")
    os.environ["SAFETY_PIPELINE_ENABLED"] = "false"

    import httpx

    import main
    import semantic_cache

    prompt = "What are the five daily prayers?"
    model = _Model(chunk_delay)

    cache = semantic_cache.get_cache()
    cache.clear()
    original_get_model = main.get_model
    original_chats = main.active_chats

    semantic_cache.set_fake_embedding(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    main.get_model = lambda: model  # type: ignore[assignment]
    main.active_chats = {}

    with (
        patch.object(main, "SEMANTIC_CACHE_ENABLED", True),
        patch.object(semantic_cache, "SEMANTIC_CACHE_ENABLED", True),
    ):
        try:
            with serve(main.app) as base_url, httpx.Client(timeout=60.0) as client:
                url = f"{base_url}/chat/stream"
                cold = [_time_stream(client, url, prompt, bypass=True) for _ in range(rounds)]
                # One uncached run with the bypass off populates the cache.
                _time_stream(client, url, prompt)
                warm = [_time_stream(client, url, prompt) for _ in range(rounds)]
        finally:
            main.get_model = original_get_model  # type: ignore[assignment]
            main.active_chats = original_chats
            semantic_cache.set_fake_embedding(None)
            cache.clear()
            # clear() drops the entries but keeps the counters. Zero them too,
            # so running this in-process (the suite does) leaves no residue for
            # the next caller's stats assertions.
            cache.hits = cache.misses = cache.bypasses = cache.evictions = 0
            if previous_safety is None:
                os.environ.pop("SAFETY_PIPELINE_ENABLED", None)
            else:
                os.environ["SAFETY_PIPELINE_ENABLED"] = previous_safety
            # The synthetic key must not outlive the run either, or a later
            # config-validation test in the same process silently passes.
            if previous_api_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = previous_api_key

    return BenchResult(cold=cold, warm=warm)


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=5, help="measured requests per arm (default: 5)")
    parser.add_argument(
        "--chunk-delay",
        type=float,
        default=0.25,
        help="seconds the stub model spends per streamed chunk (default: 0.25)",
    )
    parser.add_argument(
        "--min-reduction",
        type=float,
        default=50.0,
        help="fail if the cached arm does not cut total latency by this %% (default: 50, the issue's target)",
    )
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.chunk_delay < 0:
        parser.error("--chunk-delay must not be negative")

    summary = run_benchmark(rounds=args.rounds, chunk_delay=args.chunk_delay).summary()

    print(f"rounds per arm: {summary['rounds']}   stub chunk delay: {args.chunk_delay}s")
    print()
    print(f"{'':10} {'TTFB (ms)':>12} {'total (ms)':>12}")
    print(f"{'uncached':10} {summary['cold_ttfb_ms']:>12} {summary['cold_total_ms']:>12}")
    print(f"{'cached':10} {summary['warm_ttfb_ms']:>12} {summary['warm_total_ms']:>12}")
    print()
    print(f"TTFB reduction:  {summary['ttfb_reduction_pct']}%")
    print(f"total reduction: {summary['total_reduction_pct']}%")

    if not summary["all_warm_cached"]:
        print("\nFAIL: the second arm was not served from cache", file=sys.stderr)
        return 1
    if summary["total_reduction_pct"] < args.min_reduction:
        print(
            f"\nFAIL: total latency fell {summary['total_reduction_pct']}%, below the required {args.min_reduction}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
