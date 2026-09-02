"""End-to-end tests for the semantic cache on the streaming chat path (#1).

Every test drives the real ``POST /chat/stream`` route through FastAPI's
TestClient — no handler internals are called directly. The Gemini SDK is the
only thing faked, so the safety pipeline, citation filter, hadith annotator,
confidence assessment and cache all run for real.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
import semantic_cache
from main import app
from scripts.bench_stream_cache import run_benchmark

client = TestClient(app)

# A confident, citation-bearing answer. The citation block lifts the
# unverified-score ceiling so the assessment lands in the CONFIDENT band —
# the only band the cache is allowed to store.
ANSWER_PROSE = (
    "The five daily prayers are Fajr, Dhuhr, Asr, Maghrib and Isha. "
    "They are obligatory upon every adult Muslim and are established at fixed times."
)
CITATION_BLOCK = '<<<CITATIONS>>>{"citations": [{"type": "quran", "surah": 4, "ayah_start": 103}]}<<<END_CITATIONS>>>'
ANSWER_CHUNKS = [
    "The five daily prayers are Fajr, Dhuhr, Asr, ",
    "Maghrib and Isha. They are obligatory upon every adult Muslim ",
    "and are established at fixed times.",
    CITATION_BLOCK,
]

V_PROMPT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
V_OTHER = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


@dataclass
class FakePart:
    text: str


@dataclass
class FakeContent:
    role: str
    text: str

    @property
    def parts(self) -> list[FakePart]:
        return [FakePart(self.text)]


class FakeStreamResponse:
    """Async-iterable stand-in for a Gemini streaming response."""

    def __init__(self, chunks: list[str], delay: float = 0.0) -> None:
        self._chunks = chunks
        self._delay = delay
        self.resolved = False

    async def __aiter__(self):
        for chunk in self._chunks:
            if self._delay:
                await _sleep(self._delay)
            yield SimpleNamespace(text=chunk)

    async def resolve(self) -> None:
        self.resolved = True


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


@dataclass
class FakeChatSession:
    history: list[FakeContent] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=lambda: list(ANSWER_CHUNKS))
    delay: float = 0.0

    async def send_message_async(self, message: str, **kwargs):
        self.messages.append(message)
        answer = "".join(self.chunks)
        self.history.extend(
            [
                FakeContent(role="user", text=message),
                FakeContent(role="model", text=answer),
            ]
        )
        if kwargs.get("stream"):
            return FakeStreamResponse(self.chunks, self.delay)
        return SimpleNamespace(
            text=answer,
            candidates=[SimpleNamespace(finish_reason="STOP")],
            prompt_feedback=None,
        )


class FakeModel:
    """Records every generation so a test can prove the model was never called."""

    def __init__(self, chunks: list[str] | None = None, delay: float = 0.0) -> None:
        self.sessions: list[FakeChatSession] = []
        self.chunks = list(chunks) if chunks is not None else list(ANSWER_CHUNKS)
        self.delay = delay

    def start_chat(self, history=None) -> FakeChatSession:
        session = FakeChatSession(chunks=list(self.chunks), delay=self.delay)
        for content in history or []:
            text = content["parts"][0]["text"] if isinstance(content, dict) else content.parts[0].text
            role = content["role"] if isinstance(content, dict) else content.role
            session.history.append(FakeContent(role=role, text=text))
        self.sessions.append(session)
        return session

    @property
    def generation_count(self) -> int:
        return sum(len(session.messages) for session in self.sessions)


async def _empty_retriever(*args, **kwargs):
    return None


async def _empty_enqueue(*args, **kwargs):
    return None


@pytest.fixture
def cache_env(monkeypatch):
    """Enable the cache, silence the retrievers, and swap in the fake model."""
    cache = semantic_cache.get_cache()
    cache.clear()
    cache.hits = cache.misses = cache.bypasses = cache.evictions = 0
    # Tier 1 is a separate store with its own counters; leaving it populated
    # would serve one test's answer to the next.
    exact = semantic_cache.get_chat_exact_cache()
    exact.clear()
    exact.hits = exact.misses = exact.evictions = 0

    model = FakeModel()
    monkeypatch.setattr(main, "get_model", lambda: model)
    monkeypatch.setattr(main, "active_chats", {})
    monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", True)
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
    monkeypatch.setattr(main, "tafsir_retriever", _empty_retriever)
    monkeypatch.setattr(main, "zakat_retriever", _empty_retriever)
    monkeypatch.setattr(main, "purchase_retriever", _empty_retriever)
    monkeypatch.setattr(main, "personal_context_retriever", _empty_retriever)
    monkeypatch.setattr(main, "enqueue_for_review", _empty_enqueue)

    # Deterministic offline embeddings: the same prompt always maps to V_PROMPT.
    semantic_cache.set_fake_embedding(V_PROMPT)
    patcher = patch("semantic_cache.SEMANTIC_CACHE_ENABLED", True)
    patcher.start()
    try:
        yield model
    finally:
        patcher.stop()
        semantic_cache.set_fake_embedding(None)
        cache.clear()
        exact.clear()


def stream_chat(prompt: str, **body):
    """POST /chat/stream and return (response, parsed SSE events)."""
    payload = {"prompt": prompt, **body}
    with client.stream("POST", "/chat/stream", json=payload) as response:
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
        return response, events


def deltas(events) -> str:
    return "".join(e["delta"] for e in events if e["type"] == "content")


def done_event(events) -> dict:
    return next(e for e in events if e["type"] == "done")


# ---------------------------------------------------------------------------
# The headline: a repeated question is served from cache, with no model call
# ---------------------------------------------------------------------------


def test_second_identical_stream_is_served_from_cache(cache_env):
    model = cache_env

    first, first_events = stream_chat("What are the five daily prayers?")
    assert first.status_code == 200
    assert first.headers["X-Semantic-Cache"] == "miss"
    assert deltas(first_events) == ANSWER_PROSE
    assert done_event(first_events)["cached"] is False
    # The write only happens for a confident answer, so assert the band that
    # made this turn cacheable rather than inferring it from the next hit.
    assert done_event(first_events)["confidence"]["band"] == "confident"
    assert model.generation_count == 1

    second, second_events = stream_chat("What are the five daily prayers?")

    assert second.status_code == 200
    assert second.headers["X-Semantic-Cache"] == "hit"
    # An identical prompt is answered by the exact tier, ahead of embedding.
    assert second.headers["X-Cache-Tier"] == "exact"
    # Same prose, streamed as SSE deltas, with the citation block still stripped.
    assert deltas(second_events) == ANSWER_PROSE
    assert done_event(second_events)["cached"] is True
    # The whole point: the second request never reached the model.
    assert model.generation_count == 1


def test_paraphrase_is_served_from_the_semantic_tier(cache_env):
    """A reworded question misses tier 1 and is matched by embedding instead."""
    model = cache_env

    first, _ = stream_chat("What are the five daily prayers?")
    assert first.headers["X-Semantic-Cache"] == "miss"

    second, second_events = stream_chat("What are the 5 daily prayers?")

    assert second.headers["X-Semantic-Cache"] == "hit"
    assert second.headers["X-Cache-Tier"] == "semantic"
    assert deltas(second_events) == ANSWER_PROSE
    assert model.generation_count == 1
    assert semantic_cache.get_cache().hits == 1


def test_cache_hit_streams_progressively_and_carries_metadata(cache_env):
    stream_chat("What are the five daily prayers?")
    response, events = stream_chat("What are the five daily prayers?")

    assert response.headers["X-Semantic-Cache"] == "hit"
    assert events[0]["type"] == "metadata"
    assert events[0]["chat_id"]
    # A replay is delivered as several deltas, not one lump, so a client
    # renders it through the same progressive path as a live generation.
    content_events = [e for e in events if e["type"] == "content"]
    assert len(content_events) == len(main.replay_chunks(ANSWER_PROSE))
    assert len(content_events) > 1, "ANSWER_PROSE must exceed CACHE_REPLAY_CHUNK_CHARS for this to mean anything"
    assert all(e["delta"] for e in content_events)

    done = done_event(events)
    assert done["text"] == ANSWER_PROSE
    assert done["chat_id"] == events[0]["chat_id"]
    assert done["history"][-1]["content"] == ANSWER_PROSE


def test_cache_hit_replays_the_stored_confidence(cache_env):
    """The done event must not change shape depending on who served the turn."""
    _, first_events = stream_chat("What are the five daily prayers?")
    generated = done_event(first_events)["confidence"]

    _, second_events = stream_chat("What are the five daily prayers?")
    replayed = done_event(second_events)["confidence"]

    assert generated is not None
    assert replayed == generated
    assert replayed["band"] == "confident"


def test_cache_hit_is_dramatically_faster_than_generation(cache_env, monkeypatch):
    """The acceptance criterion, measured: a hit must beat a live generation."""
    slow_model = FakeModel(delay=0.15)
    monkeypatch.setattr(main, "get_model", lambda: slow_model)

    start = time.perf_counter()
    stream_chat("How is wudu performed?")
    generated_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    response, events = stream_chat("How is wudu performed?")
    cached_ms = (time.perf_counter() - start) * 1000

    assert response.headers["X-Semantic-Cache"] == "hit"
    assert done_event(events)["cached"] is True
    # Four 0.15s chunks vs an in-memory replay: far better than the 50%
    # perceived-latency reduction the issue asks for.
    assert cached_ms < generated_ms / 2


# ---------------------------------------------------------------------------
# Session continuity
# ---------------------------------------------------------------------------


def test_followup_after_cached_stream_keeps_the_answer_as_context(cache_env):
    stream_chat("What are the five daily prayers?")

    chat_id = str(uuid4())
    cached, _ = stream_chat("What are the five daily prayers?", chat_id=chat_id)
    assert cached.headers["X-Semantic-Cache"] == "hit"

    follow_up = client.post("/chat", json={"prompt": "And when is Asr?", "chat_id": chat_id})

    assert follow_up.status_code == 200
    history = follow_up.json()["history"]
    # The replayed answer is in the session the follow-up continues.
    assert any(m["content"] == ANSWER_PROSE for m in history)


def test_cache_is_shared_with_the_non_streaming_endpoint(cache_env):
    model = cache_env

    seeded = client.post("/chat", json={"prompt": "What are the five daily prayers?"})
    assert seeded.status_code == 200
    assert seeded.headers["X-Semantic-Cache"] == "miss"
    assert model.generation_count == 1

    response, events = stream_chat("What are the five daily prayers?")

    assert response.headers["X-Semantic-Cache"] == "hit"
    assert deltas(events) == ANSWER_PROSE
    assert model.generation_count == 1


# ---------------------------------------------------------------------------
# What must never be served from cache
# ---------------------------------------------------------------------------


def test_chat_hit_shows_the_callers_own_wording_and_confidence(cache_env):
    """Both endpoints derive a hit's history from the caller's own session.

    A match only has to clear the similarity threshold, not be word-for-word,
    so replaying the stored history would show this user a question phrased
    the way an earlier asker put it.
    """
    first = client.post("/chat", json={"prompt": "What are the five daily prayers?"})
    assert first.headers["X-Semantic-Cache"] == "miss"

    second = client.post("/chat", json={"prompt": "What are the 5 daily prayers?"})
    body = second.json()

    assert second.headers["X-Semantic-Cache"] == "hit"
    assert body["history"][0]["content"] == "What are the 5 daily prayers?"
    # The confidence block survives the round-trip, so the response shape does
    # not depend on whether the cache served the turn.
    assert body["confidence"]["band"] == "confident"


def test_bypass_header_forces_a_fresh_generation(cache_env):
    model = cache_env
    stream_chat("What are the five daily prayers?")

    with client.stream(
        "POST",
        "/chat/stream",
        json={"prompt": "What are the five daily prayers?"},
        headers={"X-Cache-Bypass": "1"},
    ) as response:
        events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]

    assert response.headers["X-Semantic-Cache"] == "bypass"
    assert done_event(events)["cached"] is False
    assert model.generation_count == 2
    assert semantic_cache.get_cache().bypasses == 1


def test_one_users_answer_is_never_served_to_another(cache_env):
    """Scope isolation, not exclusion, is what keeps users apart.

    An authenticated turn is cached under ``user:<id>``; an anonymous one under
    ``public``. Neither may read the other's entry, and two different users may
    not read each other's.
    """
    model = cache_env
    prompt = "What are the five daily prayers?"

    first, _ = stream_chat(prompt, user_id="user-123")
    assert first.headers["X-Semantic-Cache"] == "miss"

    # That user reads their own scope back.
    again, _ = stream_chat(prompt, user_id="user-123")
    assert again.headers["X-Semantic-Cache"] == "hit"
    assert model.generation_count == 1

    # A different user must not.
    other, _ = stream_chat(prompt, user_id="user-456")
    assert other.headers["X-Semantic-Cache"] == "miss"

    # Nor may an anonymous asker.
    anonymous, _ = stream_chat(prompt)
    assert anonymous.headers["X-Semantic-Cache"] == "miss"
    assert model.generation_count == 3


def test_low_confidence_answer_is_not_cached(cache_env, monkeypatch):
    """Without a verifiable citation the answer is hedged — and must not replay."""
    hedged = FakeModel(chunks=["It may possibly depend, and I am not certain about this."])
    monkeypatch.setattr(main, "get_model", lambda: hedged)

    first, first_events = stream_chat("Is coffee permissible?")
    assert first.headers["X-Semantic-Cache"] == "miss"
    assert done_event(first_events)["confidence"]["band"] != "confident"

    second, _ = stream_chat("Is coffee permissible?")
    assert second.headers["X-Semantic-Cache"] == "miss"
    assert hedged.generation_count == 2


def test_turn_carrying_extra_context_is_never_cached(cache_env):
    """An answer shaped by caller-supplied context must not replay to others."""
    model = cache_env
    prompt = "What are the five daily prayers?"

    first, _ = stream_chat(prompt, context="I am travelling and cannot stand to pray.")
    assert first.headers["X-Semantic-Cache"] == "miss"

    second, _ = stream_chat(prompt, context="I am travelling and cannot stand to pray.")
    assert second.headers["X-Semantic-Cache"] == "miss"
    assert model.generation_count == 2


def _grounded_context(retriever: str):
    """A context object shaped the way each retriever's consumers expect."""
    if retriever == "tafsir_retriever":
        # The tafsir summariser walks .ayat, so this one needs the real model.
        from tafsir import TafsirContext

        return TafsirContext(references=["2:153"], prompt_block="\nRetrieved tafsir.\n", ayat=[])
    return SimpleNamespace(prompt_block="\nGrounding for this asker only.\n", info=None)


@pytest.mark.parametrize(
    "retriever",
    ["tafsir_retriever", "zakat_retriever", "purchase_retriever", "personal_context_retriever"],
)
def test_retrieval_grounded_stream_is_never_cached(cache_env, monkeypatch, retriever):
    """Each retriever grounds the answer in data that belongs to one asker.

    Parametrised over all four rather than trusting one: if a clause is
    dropped from ``cache_eligible`` in a later refactor, that retriever's
    answers would start replaying to everyone, and this is what catches it.
    """
    model = cache_env

    async def _grounded(*args, **kwargs):
        return _grounded_context(retriever)

    monkeypatch.setattr(main, retriever, _grounded)

    first, _ = stream_chat("What are the five daily prayers?")
    assert first.headers["X-Semantic-Cache"] == "miss"

    second, _ = stream_chat("What are the five daily prayers?")
    assert second.headers["X-Semantic-Cache"] == "miss"
    assert model.generation_count == 2


@pytest.mark.parametrize("variant", [{"language": "ar"}, {"madhhab": "hanafi"}])
def test_response_variants_do_not_share_a_cache_entry(cache_env, variant):
    """language and madhhab reshape the answer but not the prompt.

    The key is derived from the prompt alone, so a variant request must stay
    out of the cache entirely rather than replay — or be replayed by — a turn
    that asked the same question with different settings.
    """
    model = cache_env
    prompt = "What are the five daily prayers?"

    plain, _ = stream_chat(prompt)
    assert plain.headers["X-Semantic-Cache"] == "miss"

    # The variant request must not be served the plain answer...
    varied, _ = stream_chat(prompt, **variant)
    assert varied.headers["X-Semantic-Cache"] == "miss"
    assert model.generation_count == 2

    # ...nor may it have written one that a later plain asker would receive.
    again, _ = stream_chat(prompt, **variant)
    assert again.headers["X-Semantic-Cache"] == "miss"
    assert model.generation_count == 3


def test_resumed_conversation_is_not_treated_as_a_new_chat(cache_env, monkeypatch):
    """A restart empties active_chats but not the session store.

    Without consulting the store, a returning chat_id looks new, and the cache
    would answer a follow-up with a standalone reply that ignores the context.
    """
    model = cache_env
    prompt = "What are the five daily prayers?"

    # Warm the cache with an ordinary standalone turn.
    first, _ = stream_chat(prompt)
    assert first.headers["X-Semantic-Cache"] == "miss"

    # A conversation that exists only in the store — the state after a restart,
    # or on any other worker.
    resumed_id = str(uuid4())
    asyncio.run(
        main.session_store.save_history(
            resumed_id,
            [{"role": "user", "text": "Earlier question"}, {"role": "model", "text": "Earlier answer"}],
        )
    )

    resumed, _ = stream_chat(prompt, chat_id=resumed_id)

    assert resumed.headers["X-Semantic-Cache"] == "miss"
    assert model.generation_count == 2


def test_refused_prompt_is_never_answered_from_cache(cache_env, monkeypatch):
    """The input gate runs before the lookup, so a refusal cannot be replayed."""
    model = cache_env
    prompt = "What are the five daily prayers?"

    warm, _ = stream_chat(prompt)
    assert warm.headers["X-Semantic-Cache"] == "miss"

    # Turn the gate on and make it refuse everything.
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "true")

    async def _refuse(_prompt):
        return SimpleNamespace(
            action="refuse",
            refusal="No.",
            guidance=None,
            stages_fired=["test"],
            category_id="test",
        )

    monkeypatch.setattr(main.safety_pipeline.input_gate, "evaluate_async", _refuse)

    response, events = stream_chat(prompt)

    assert response.headers["X-Semantic-Cache"] == "miss"
    assert not [e for e in events if e["type"] == "content"]
    assert events[-1]["type"] == "error"
    assert model.generation_count == 1


def test_unrelated_question_misses(cache_env):
    model = cache_env
    stream_chat("What are the five daily prayers?")

    semantic_cache.set_fake_embedding(V_OTHER)
    response, _ = stream_chat("How is zakat calculated on gold?")

    assert response.headers["X-Semantic-Cache"] == "miss"
    assert model.generation_count == 2


def test_replay_chunks_splits_without_losing_text():
    text = "abcdefghij"
    assert main.replay_chunks(text, 4) == ["abcd", "efgh", "ij"]
    assert "".join(main.replay_chunks(text, 4)) == text
    assert main.replay_chunks("", 4) == []
    with pytest.raises(ValueError):
        main.replay_chunks(text, 0)


def test_benchmark_script_reports_a_faster_cached_path():
    """scripts/bench_stream_cache.py is the measurement the issue asks for.

    Runs it end-to-end (real uvicorn server, real route) at a short delay, so
    the numbers quoted in docs/latency.md come from code that is exercised by
    the suite rather than from a one-off run nobody can reproduce.
    """
    # 0.2s per chunk gives the reduction assertion headroom against runner
    # jitter, and two rounds give each arm a median rather than one sample.
    summary = run_benchmark(rounds=2, chunk_delay=0.2).summary()

    assert summary["all_warm_cached"] is True
    assert summary["warm_ttfb_ms"] < summary["cold_ttfb_ms"]
    assert summary["ttfb_reduction_pct"] > 50.0
