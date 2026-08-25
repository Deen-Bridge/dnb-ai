"""Shared async runtime primitives for bounded, observable concurrency."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import time
import weakref
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _percentile(samples: deque[float], percentile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * percentile)
    return round(ordered[index], 3)


class TaskPriority(IntEnum):
    """Lower values run first."""

    CRITICAL = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75


TaskFactory = Callable[[], Coroutine[Any, Any, Any]]


@dataclass(order=True)
class _QueuedTask:
    priority: int
    sequence: int
    queued_at: float = field(compare=False)
    name: str = field(compare=False)
    factory: TaskFactory = field(compare=False)
    context: contextvars.Context = field(compare=False)


class AsyncTaskScheduler:
    """Bounded priority scheduler for response-side work.

    Workers are started by the FastAPI lifespan. A direct tracked-task fallback
    keeps endpoint tests and embedded ASGI callers correct when lifespan events
    are not emitted.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        max_queue_size: int | None = None,
        sample_size: int = 1024,
    ) -> None:
        self.max_workers = max_workers or _env_int("ASYNC_BACKGROUND_WORKERS", 4)
        self.max_queue_size = max_queue_size or _env_int("ASYNC_BACKGROUND_QUEUE_SIZE", 1000)
        self._sample_size = sample_size
        self._queue: asyncio.PriorityQueue[_QueuedTask] | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._direct_tasks: set[asyncio.Task[Any]] = set()
        self._direct_limits: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._accepting = True
        self._sequence = 0
        self._active = 0
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._rejected = 0
        self._queue_wait_ms: deque[float] = deque(maxlen=sample_size)
        self._schedule_ms: deque[float] = deque(maxlen=sample_size)

    async def start(self) -> None:
        """Start priority workers. Safe to call more than once."""
        loop = asyncio.get_running_loop()
        if self._running and self._loop is loop:
            return
        if self._running:
            await self.stop(grace_seconds=0)
        self._queue = asyncio.PriorityQueue(maxsize=self.max_queue_size)
        self._loop = loop
        self._running = True
        self._accepting = True
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"background-worker-{index}")
            for index in range(self.max_workers)
        ]

    def submit(
        self,
        factory: TaskFactory,
        *,
        priority: TaskPriority = TaskPriority.NORMAL,
        name: str = "background-task",
    ) -> bool:
        """Schedule work without creating its coroutine until capacity is available."""
        started = time.perf_counter()
        if not self._accepting:
            self._rejected += 1
            return False

        self._sequence += 1
        item = _QueuedTask(
            priority=int(priority),
            sequence=self._sequence,
            queued_at=time.perf_counter(),
            name=name,
            factory=factory,
            context=contextvars.copy_context(),
        )

        loop = asyncio.get_running_loop()
        if self._running and self._loop is loop and self._queue is not None:
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                self._rejected += 1
                logger.warning("Background queue is full; rejected %s", name)
                return False
        else:
            task = asyncio.create_task(
                self._run_direct(item),
                name=f"direct-{name}",
            )
            self._direct_tasks.add(task)
            task.add_done_callback(self._direct_tasks.discard)

        self._submitted += 1
        self._schedule_ms.append((time.perf_counter() - started) * 1000.0)
        return True

    async def _run_direct(self, item: _QueuedTask) -> None:
        loop = asyncio.get_running_loop()
        limiter = self._direct_limits.get(loop)
        if limiter is None:
            limiter = asyncio.Semaphore(self.max_workers)
            self._direct_limits[loop] = limiter
        async with limiter:
            await self._execute(item)

    async def _worker(self, worker_id: int) -> None:
        queue = self._queue
        assert queue is not None
        try:
            while True:
                item = await queue.get()
                try:
                    await self._execute(item)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            logger.debug("Background worker %d cancelled", worker_id)
            raise

    async def _execute(self, item: _QueuedTask) -> None:
        self._queue_wait_ms.append((time.perf_counter() - item.queued_at) * 1000.0)
        self._active += 1
        task: asyncio.Task[Any] | None = None
        try:
            coroutine = item.context.run(item.factory)
            task = asyncio.create_task(coroutine, name=item.name, context=item.context)
            await task
        except asyncio.CancelledError:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self._cancelled += 1
            raise
        except Exception:
            self._failed += 1
            logger.exception("Background task failed", extra={"task_name": item.name})
        else:
            self._completed += 1
        finally:
            self._active -= 1

    async def stop(self, *, grace_seconds: float | None = None) -> None:
        """Drain queued work, then cancel anything still running."""
        self._accepting = False
        grace = (
            float(os.getenv("ASYNC_SHUTDOWN_GRACE_SECONDS", "10")) if grace_seconds is None else max(0.0, grace_seconds)
        )

        if grace > 0:
            waiters: list[Awaitable[Any]] = []
            if self._queue is not None:
                waiters.append(self._queue.join())
            if self._direct_tasks:
                waiters.append(asyncio.gather(*self._direct_tasks, return_exceptions=True))
            try:
                if waiters:
                    await asyncio.wait_for(asyncio.gather(*waiters), timeout=grace)
            except TimeoutError:
                logger.warning("Background tasks exceeded %.1fs shutdown grace", grace)

        for worker in self._workers:
            worker.cancel()
        for task in list(self._direct_tasks):
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        if self._direct_tasks:
            await asyncio.gather(*self._direct_tasks, return_exceptions=True)

        if self._queue is not None:
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
                self._cancelled += 1

        self._workers.clear()
        self._direct_tasks.clear()
        self._queue = None
        self._loop = None
        self._running = False

    def stats(self) -> dict[str, Any]:
        queue_depth = self._queue.qsize() if self._queue is not None else 0
        return {
            "running": self._running,
            "workers": len(self._workers),
            "active": self._active,
            "queue_depth": queue_depth,
            "queue_capacity": self.max_queue_size,
            "submitted": self._submitted,
            "completed": self._completed,
            "failed": self._failed,
            "cancelled": self._cancelled,
            "rejected": self._rejected,
            "queue_wait_p95_ms": _percentile(self._queue_wait_ms, 0.95),
            "schedule_p95_ms": _percentile(self._schedule_ms, 0.95),
        }


@dataclass
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


class KeyedLockPool:
    """Serialize work for one key while allowing unrelated keys to run."""

    def __init__(self) -> None:
        self._entries: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _LockEntry]] = (
            weakref.WeakKeyDictionary()
        )

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        entries = self._entries.setdefault(loop, {})
        entry = entries.get(key)
        if entry is None:
            entry = _LockEntry(asyncio.Lock())
            entries[key] = entry
        entry.users += 1
        try:
            await entry.lock.acquire()
        except BaseException:
            entry.users -= 1
            if entry.users == 0:
                entries.pop(key, None)
            raise
        try:
            yield
        finally:
            entry.lock.release()
            entry.users -= 1
            if entry.users == 0:
                entries.pop(key, None)


class CapacityLimiter:
    """Loop-local semaphore with lightweight utilization metrics."""

    def __init__(self, env_name: str, default: int) -> None:
        self.limit = _env_int(env_name, default)
        self._semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )
        self.active = 0
        self.waiting = 0
        self.peak_active = 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.limit)
            self._semaphores[loop] = semaphore
        self.waiting += 1
        try:
            await semaphore.acquire()
        finally:
            self.waiting -= 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            yield
        finally:
            self.active -= 1
            semaphore.release()

    def stats(self) -> dict[str, int]:
        return {
            "limit": self.limit,
            "active": self.active,
            "waiting": self.waiting,
            "peak_active": self.peak_active,
        }


class HttpClientPool:
    """One keep-alive HTTP client per event loop."""

    def __init__(self) -> None:
        self.max_connections = _env_int("HTTP_MAX_CONNECTIONS", 200)
        self.max_keepalive_connections = _env_int("HTTP_MAX_KEEPALIVE_CONNECTIONS", 50)
        self.keepalive_expiry = float(os.getenv("HTTP_KEEPALIVE_EXPIRY_SECONDS", "30"))
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )

    def get(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_keepalive_connections,
                    keepalive_expiry=self.keepalive_expiry,
                )
            )
            self._clients[loop] = client
        return client

    async def aclose(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        if clients:
            await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)

    def stats(self) -> dict[str, int | float]:
        return {
            "open_clients": sum(not client.is_closed for client in self._clients.values()),
            "max_connections": self.max_connections,
            "max_keepalive_connections": self.max_keepalive_connections,
            "keepalive_expiry_seconds": self.keepalive_expiry,
        }


background_tasks = AsyncTaskScheduler()
chat_locks = KeyedLockPool()
llm_limiter = CapacityLimiter("LLM_MAX_CONCURRENCY", 100)
http_client_pool = HttpClientPool()
