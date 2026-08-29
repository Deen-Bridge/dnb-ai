import asyncio
import contextvars
import uuid

import pytest
from fastapi import Request, Response

from async_runtime import AsyncTaskScheduler, HttpClientPool, KeyedLockPool, TaskPriority


@pytest.mark.asyncio
async def test_priority_scheduler_runs_high_priority_work_first():
    scheduler = AsyncTaskScheduler(max_workers=1, max_queue_size=10)
    await scheduler.start()
    started: list[str] = []
    gate = asyncio.Event()

    scheduler.submit(lambda: _record_after_gate(started, "first", gate), name="first")
    await asyncio.sleep(0)
    scheduler.submit(lambda: _record_after_gate(started, "low", gate), priority=TaskPriority.LOW, name="low")
    scheduler.submit(
        lambda: _record_after_gate(started, "critical", gate),
        priority=TaskPriority.CRITICAL,
        name="critical",
    )

    await asyncio.sleep(0)
    gate.set()
    await asyncio.wait_for(scheduler.stop(grace_seconds=1), timeout=2)

    assert started == ["first", "critical", "low"]
    assert scheduler.stats()["completed"] == 3


async def _record_after_gate(target: list[str], value: str, gate: asyncio.Event) -> None:
    await gate.wait()
    target.append(value)


@pytest.mark.asyncio
async def test_scheduler_limits_active_work_and_cleans_up_on_shutdown():
    scheduler = AsyncTaskScheduler(max_workers=2, max_queue_size=10)
    await scheduler.start()
    active = 0
    peak = 0

    async def work() -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1

    for _ in range(8):
        assert scheduler.submit(work)

    await scheduler.stop(grace_seconds=1)
    assert peak == 2
    assert scheduler.stats()["active"] == 0
    assert scheduler.stats()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_scheduler_cancels_running_work_after_grace_period():
    scheduler = AsyncTaskScheduler(max_workers=1, max_queue_size=2)
    await scheduler.start()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def work() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    assert scheduler.submit(work)
    await started.wait()
    await scheduler.stop(grace_seconds=0)

    assert cancelled.is_set()
    assert scheduler.stats()["active"] == 0


@pytest.mark.asyncio
async def test_scheduler_tracks_work_without_lifespan_start():
    scheduler = AsyncTaskScheduler(max_workers=1)
    completed = asyncio.Event()

    async def work() -> None:
        completed.set()

    assert scheduler.submit(work, name="direct-work")
    await asyncio.wait_for(completed.wait(), timeout=1)
    await scheduler.stop(grace_seconds=1)

    assert scheduler.stats()["submitted"] == 1
    assert scheduler.stats()["completed"] == 1


@pytest.mark.asyncio
async def test_scheduler_preserves_submission_context_for_task_factory():
    scheduler = AsyncTaskScheduler(max_workers=1)
    request_id = contextvars.ContextVar("request_id", default="missing")
    factory_context: list[str] = []
    task_context: list[str] = []

    async def work() -> None:
        task_context.append(request_id.get())

    def factory():
        factory_context.append(request_id.get())
        return work()

    token = request_id.set("submitted")
    assert scheduler.submit(factory, name="context-work")
    request_id.reset(token)
    await scheduler.stop(grace_seconds=1)

    assert factory_context == ["submitted"]
    assert task_context == ["submitted"]


@pytest.mark.asyncio
async def test_scheduler_survives_task_factory_failure():
    scheduler = AsyncTaskScheduler(max_workers=1)
    await scheduler.start()
    completed = asyncio.Event()

    def failing_factory():
        raise RuntimeError("factory failed")

    async def succeeding_work() -> None:
        completed.set()

    assert scheduler.submit(failing_factory, name="failing-factory")
    assert scheduler.submit(succeeding_work, name="succeeding-work")
    await asyncio.wait_for(completed.wait(), timeout=1)
    await scheduler.stop(grace_seconds=1)

    assert scheduler.stats()["failed"] == 1
    assert scheduler.stats()["completed"] == 1
    assert scheduler.stats()["active"] == 0


@pytest.mark.asyncio
async def test_keyed_locks_serialize_same_key_but_parallelize_other_keys():
    locks = KeyedLockPool()
    active = 0
    peak = 0

    async def work(key: str) -> None:
        nonlocal active, peak
        async with locks.hold(key):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(work("same"), work("same"), work("other"))
    assert peak == 2


@pytest.mark.asyncio
async def test_chat_wrapper_serializes_same_chat_and_parallelizes_other_chats(monkeypatch):
    import main

    active = 0
    peak = 0

    async def fake_chat(request, _http_request, _fastapi_response, chat_id):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return chat_id

    monkeypatch.setattr(main, "_chat", fake_chat)
    first = main.ChatRequest(prompt="one", chat_id=uuid.uuid4())
    same = main.ChatRequest(prompt="two", chat_id=first.chat_id)
    other = main.ChatRequest(prompt="three", chat_id=uuid.uuid4())

    scope = {"type": "http", "path": "/chat", "headers": [], "client": ("127.0.0.1", 12345)}
    results = await asyncio.gather(
        main.chat(first, Request(scope), Response()),
        main.chat(same, Request(scope), Response()),
        main.chat(other, Request(scope), Response()),
    )

    assert results == [str(first.chat_id), str(first.chat_id), str(other.chat_id)]
    assert peak == 2


@pytest.mark.asyncio
async def test_http_client_pool_reuses_client_and_closes_it():
    pool = HttpClientPool()
    first = pool.get()
    second = pool.get()
    assert first is second
    assert not first.is_closed

    await pool.aclose()
    assert first.is_closed


@pytest.mark.asyncio
async def test_retrieval_fanout_does_not_wait_for_siblings_serially(monkeypatch):
    import main

    request = main.ChatRequest(prompt="question")
    started: list[str] = []

    async def retrieve(name: str) -> None:
        started.append(name)
        await asyncio.sleep(0.03)

    monkeypatch.setattr(main, "tafsir_retriever", lambda *args: retrieve("tafsir"))
    monkeypatch.setattr(main, "zakat_retriever", lambda *args: retrieve("zakat"))
    monkeypatch.setattr(main, "purchase_retriever", lambda *args: retrieve("purchase"))
    monkeypatch.setattr(main, "personal_context_retriever", lambda *args: retrieve("personal"))

    started_at = asyncio.get_running_loop().time()
    result = await main.retrieve_chat_contexts(request, request.prompt)
    elapsed = asyncio.get_running_loop().time() - started_at

    assert result == (None, None, None, None)
    assert set(started) == {"tafsir", "zakat", "purchase", "personal"}
    assert elapsed < 0.08
