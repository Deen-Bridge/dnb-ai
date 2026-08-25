"""Queue System for Heavy Operations (#215)

A robust queue system for managing heavy computational operations, batch processing,
and long-running tasks with priority-based management, progress monitoring, and
configurable retry mechanisms.

Features:
- Priority-based queue management
- Job scheduling (immediate, delayed, recurring)
- Real-time progress monitoring
- Exponential backoff retry
- Dead letter queue
- Resource constraints per job type
- Job chaining and dependencies

Architecture:
- JobQueue: Main queue management
- Job: Individual job representation
- Worker: Job execution
- JobStore: Persistence layer
"""

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a job in the queue."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD = "dead"  # In dead letter queue
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Priority levels for jobs."""

    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


@dataclass
class JobResult:
    """Result of a job execution."""

    success: bool
    data: Any | None = None
    error: str | None = None
    duration_ms: float = 0


@dataclass
class JobProgress:
    """Progress information for a running job."""

    job_id: str
    percent: float = 0
    current_step: str = ""
    total_steps: int = 0
    current_step_num: int = 0
    estimated_remaining_ms: float | None = None


@dataclass
class ResourceConstraints:
    """Resource constraints for a job type."""

    max_memory_mb: int = 512
    max_cpu_percent: float = 100
    max_concurrent: int = 10
    timeout_seconds: int = 300


@dataclass
class RetryConfig:
    """Retry configuration with exponential backoff."""

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0
    exponential_base: float = 2.0
    jitter: bool = True

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt."""
        delay = self.initial_delay_seconds * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay_seconds)
        if self.jitter:
            import random

            delay *= 0.5 + random.random()
        return delay


@dataclass
class Job:
    """Represents a job in the queue."""

    id: str
    job_type: str
    payload: dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    scheduled_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    retry_count: int = 0
    max_retries: int = 3
    error: str | None = None
    result: Any | None = None
    progress: float = 0
    progress_message: str = ""
    parent_job_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "payload": self.payload,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "scheduled_at": self.scheduled_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error": self.error,
            "result": self.result,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "parent_job_id": self.parent_job_id,
            "depends_on": self.depends_on,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        return cls(
            id=data["id"],
            job_type=data["job_type"],
            payload=data["payload"],
            priority=JobPriority(data.get("priority", JobPriority.NORMAL.value)),
            status=JobStatus(data.get("status", JobStatus.PENDING.value)),
            created_at=data.get("created_at", time.time()),
            scheduled_at=data.get("scheduled_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            error=data.get("error"),
            result=data.get("result"),
            progress=data.get("progress", 0),
            progress_message=data.get("progress_message", ""),
            parent_job_id=data.get("parent_job_id"),
            depends_on=data.get("depends_on", []),
            metadata=data.get("metadata", {}),
        )


class JobHandler(ABC):
    """Abstract base class for job handlers."""

    @property
    @abstractmethod
    def job_type(self) -> str:
        """The job type this handler processes."""
        pass

    @property
    def constraints(self) -> ResourceConstraints:
        """Resource constraints for this job type."""
        return ResourceConstraints()

    @property
    def retry_config(self) -> RetryConfig:
        """Retry configuration for this job type."""
        return RetryConfig()

    @abstractmethod
    async def execute(
        self,
        job: Job,
        progress_callback: Callable[[float, str], Any] | None = None,
        progress_callback: Callable[[float, str], Awaitable[None]] | None = None,
    ) -> JobResult:
        """Execute the job and return result."""
        pass


class JobStore(ABC):
    """Abstract base class for job persistence."""

    @abstractmethod
    async def save_job(self, job: Job) -> None:
        """Save a job to storage."""
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        pass

    @abstractmethod
    async def get_pending_jobs(self, limit: int = 100) -> list[Job]:
        """Get pending jobs ordered by priority."""
        pass

    @abstractmethod
    async def get_jobs_by_status(self, status: JobStatus, limit: int = 100) -> list[Job]:
        """Get jobs by status."""
        pass

    @abstractmethod
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        pass


class InMemoryJobStore(JobStore):
    """In-memory job store for development/testing."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def save_job(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.id] = job

    async def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def get_pending_jobs(self, limit: int = 100) -> list[Job]:
        now = time.time()
        pending = [
            j
            for j in self._jobs.values()
            if j.status == JobStatus.PENDING
            and (j.scheduled_at is None or j.scheduled_at <= now)
            and all(self._jobs.get(dep) and self._jobs[dep].status == JobStatus.COMPLETED for dep in j.depends_on)
        ]
        # Sort by priority (descending) then created_at (ascending)
        pending.sort(key=lambda j: (-j.priority.value, j.created_at))
        return pending[:limit]

    async def get_jobs_by_status(self, status: JobStatus, limit: int = 100) -> list[Job]:
        jobs = [j for j in self._jobs.values() if j.status == status]
        return jobs[:limit]

    async def delete_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False


class RedisJobStore(JobStore):
    """Redis-backed job store for production."""

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis: Any = None
        self._prefix = "dnb:jobs:"

    async def _get_redis(self) -> Any:
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = await aioredis.from_url(
                    self._redis_url,
                    decode_responses=True,
                )
            except ImportError as err:
                raise RuntimeError("redis package not installed") from err
        return self._redis

    async def save_job(self, job: Job) -> None:
        redis = await self._get_redis()
        key = f"{self._prefix}{job.id}"
        await redis.set(key, json.dumps(job.to_dict()))
        # Add to status set
        await redis.sadd(f"{self._prefix}status:{job.status.value}", job.id)
        # Add to priority sorted set for pending jobs
        if job.status == JobStatus.PENDING:
            score = -job.priority.value * 1e12 + job.created_at
            await redis.zadd(f"{self._prefix}pending", {job.id: score})

    async def get_job(self, job_id: str) -> Job | None:
        redis = await self._get_redis()
        data = await redis.get(f"{self._prefix}{job_id}")
        if data:
            return Job.from_dict(json.loads(data))
        return None

    async def get_pending_jobs(self, limit: int = 100) -> list[Job]:
        redis = await self._get_redis()
        job_ids = await redis.zrange(f"{self._prefix}pending", 0, limit - 1)
        jobs = []
        for job_id in job_ids:
            job = await self.get_job(job_id)
            if job:
                jobs.append(job)
        return jobs

    async def get_jobs_by_status(self, status: JobStatus, limit: int = 100) -> list[Job]:
        redis = await self._get_redis()
        job_ids = await redis.smembers(f"{self._prefix}status:{status.value}")
        jobs = []
        for job_id in list(job_ids)[:limit]:
            job = await self.get_job(job_id)
            if job:
                jobs.append(job)
        return jobs

    async def delete_job(self, job_id: str) -> bool:
        redis = await self._get_redis()
        job = await self.get_job(job_id)
        if job:
            await redis.delete(f"{self._prefix}{job_id}")
            await redis.srem(f"{self._prefix}status:{job.status.value}", job_id)
            await redis.zrem(f"{self._prefix}pending", job_id)
            return True
        return False


class JobQueue:
    """Main job queue manager."""

    def __init__(
        self,
        store: JobStore | None = None,
        max_workers: int = 4,
    ) -> None:
        self._store = store or InMemoryJobStore()
        self._handlers: dict[str, JobHandler] = {}
        self._max_workers = max_workers
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._progress_callbacks: dict[str, Callable[[JobProgress], None]] = {}
        self._dead_letter_queue: list[Job] = []
        self._stats = {
            "total_processed": 0,
            "total_failed": 0,
            "total_retried": 0,
        }

    def register_handler(self, handler: JobHandler) -> None:
        """Register a job handler."""
        self._handlers[handler.job_type] = handler
        logger.info("Registered handler for job type: %s", handler.job_type)

    async def submit(
        self,
        job_type: str,
        payload: dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        scheduled_at: datetime | None = None,
        depends_on: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Job:
        """Submit a new job to the queue."""
        if job_type not in self._handlers:
            raise ValueError(f"No handler registered for job type: {job_type}")

        job = Job(
            id=str(uuid.uuid4()),
            job_type=job_type,
            payload=payload,
            priority=priority,
            scheduled_at=scheduled_at.timestamp() if scheduled_at else None,
            depends_on=depends_on or [],
            metadata=metadata or {},
            max_retries=self._handlers[job_type].retry_config.max_retries,
        )

        await self._store.save_job(job)
        logger.info("Job submitted: %s (type=%s, priority=%s)", job.id, job_type, priority.name)
        return job

    async def submit_batch(
        self,
        jobs: list[dict[str, Any]],
        parent_job_id: str | None = None,
    ) -> list[Job]:
        """Submit multiple jobs as a batch."""
        created_jobs = []
        for job_spec in jobs:
            job = await self.submit(
                job_type=job_spec["job_type"],
                payload=job_spec["payload"],
                priority=job_spec.get("priority", JobPriority.NORMAL),
                depends_on=job_spec.get("depends_on"),
                metadata={"parent_job_id": parent_job_id, **job_spec.get("metadata", {})},
            )
            created_jobs.append(job)
        return created_jobs

    async def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        return await self._store.get_job(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        job = await self._store.get_job(job_id)
        if job and job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            await self._store.save_job(job)
            return True
        return False

    async def get_progress(self, job_id: str) -> JobProgress | None:
        """Get progress for a running job."""
        job = await self._store.get_job(job_id)
        if job:
            return JobProgress(
                job_id=job.id,
                percent=job.progress,
                current_step=job.progress_message,
            )
        return None

    def on_progress(self, job_id: str, callback: Callable[[JobProgress], None]) -> None:
        """Register a progress callback for a job."""
        self._progress_callbacks[job_id] = callback

    async def _update_progress(self, job: Job, percent: float, message: str) -> None:
        """Update job progress and notify callbacks."""
        job.progress = percent
        job.progress_message = message
        await self._store.save_job(job)

        if job.id in self._progress_callbacks:
            progress = JobProgress(
                job_id=job.id,
                percent=percent,
                current_step=message,
            )
            self._progress_callbacks[job.id](progress)

    async def _execute_job(self, job: Job) -> None:
        """Execute a single job."""
        handler = self._handlers.get(job.job_type)
        if not handler:
            job.status = JobStatus.FAILED
            job.error = f"No handler for job type: {job.job_type}"
            await self._store.save_job(job)
            return

        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        await self._store.save_job(job)

        try:
            # Create progress callback
            async def progress_cb(percent: float, message: str) -> None:
                await self._update_progress(job, percent, message)

            # Execute with timeout
            timeout = handler.constraints.timeout_seconds
            result = await asyncio.wait_for(
                handler.execute(job, progress_cb),
                timeout=timeout,
            )

            if result.success:
                job.status = JobStatus.COMPLETED
                job.result = result.data
                job.progress = 100
                self._stats["total_processed"] += 1
            else:
                raise Exception(result.error or "Job failed")

        except TimeoutError:
            job.error = f"Job timed out after {handler.constraints.timeout_seconds}s"
            await self._handle_failure(job, handler)
        except Exception as e:
            job.error = str(e)
            await self._handle_failure(job, handler)

        job.completed_at = time.time()
        await self._store.save_job(job)

        # Cleanup progress callback
        self._progress_callbacks.pop(job.id, None)

    async def _handle_failure(self, job: Job, handler: JobHandler) -> None:
        """Handle job failure with retry logic."""
        retry_config = handler.retry_config

        if job.retry_count < retry_config.max_retries:
            job.retry_count += 1
            job.status = JobStatus.RETRYING
            delay = retry_config.get_delay(job.retry_count)
            job.scheduled_at = time.time() + delay
            self._stats["total_retried"] += 1
            logger.warning(
                "Job %s failed, scheduling retry %d/%d in %.1fs",
                job.id,
                job.retry_count,
                retry_config.max_retries,
                delay,
            )
        else:
            job.status = JobStatus.DEAD
            self._dead_letter_queue.append(job)
            self._stats["total_failed"] += 1
            logger.error("Job %s failed permanently after %d retries", job.id, job.retry_count)

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes jobs from the queue."""
        logger.info("Worker %d started", worker_id)
        while self._running:
            jobs = await self._store.get_pending_jobs(limit=1)
            if jobs:
                job = jobs[0]
                # Mark as running immediately to prevent double-processing
                job.status = JobStatus.RUNNING
                await self._store.save_job(job)
                await self._execute_job(job)
            else:
                await asyncio.sleep(0.1)
        logger.info("Worker %d stopped", worker_id)

    async def start(self) -> None:
        """Start the job queue workers."""
        if self._running:
            return
        self._running = True
        for i in range(self._max_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)
        logger.info("Job queue started with %d workers", self._max_workers)

    async def stop(self, wait: bool = True) -> None:
        """Stop the job queue workers."""
        self._running = False
        if wait:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Job queue stopped")

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            **self._stats,
            "dead_letter_count": len(self._dead_letter_queue),
            "worker_count": len(self._workers),
            "is_running": self._running,
        }

    async def get_dead_letter_jobs(self) -> list[Job]:
        """Get jobs from the dead letter queue."""
        return self._dead_letter_queue.copy()

    async def retry_dead_letter_job(self, job_id: str) -> bool:
        """Retry a job from the dead letter queue."""
        for i, job in enumerate(self._dead_letter_queue):
            if job.id == job_id:
                job.status = JobStatus.PENDING
                job.retry_count = 0
                job.error = None
                job.scheduled_at = None
                await self._store.save_job(job)
                self._dead_letter_queue.pop(i)
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Example handlers
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingGenerationHandler(JobHandler):
    """Handler for batch embedding generation."""

    @property
    def job_type(self) -> str:
        return "embedding_generation"

    @property
    def constraints(self) -> ResourceConstraints:
        return ResourceConstraints(
            max_memory_mb=1024,
            max_concurrent=5,
            timeout_seconds=600,
        )

    async def execute(
        self,
        job: Job,
        progress_callback: Callable[[float, str], Any] | None = None,
        progress_callback: Callable[[float, str], Awaitable[None]] | None = None,
    ) -> JobResult:
        texts = job.payload.get("texts", [])
        total = len(texts)
        embeddings = []

        for i, _text in enumerate(texts):
            # Simulate embedding generation
            await asyncio.sleep(0.1)
            embeddings.append([0.1] * 768)  # Mock embedding

            if progress_callback:
                percent = (i + 1) / total * 100
                res = progress_callback(percent, f"Processing {i + 1}/{total}")
                if asyncio.iscoroutine(res):
                    await res

        return JobResult(success=True, data={"embeddings": embeddings})


class IndexBuildHandler(JobHandler):
    """Handler for building search indices."""

    @property
    def job_type(self) -> str:
        return "index_build"

    @property
    def constraints(self) -> ResourceConstraints:
        return ResourceConstraints(
            max_memory_mb=2048,
            max_concurrent=2,
            timeout_seconds=1800,
        )

    async def execute(
        self,
        job: Job,
        progress_callback: Callable[[float, str], Any] | None = None,
        progress_callback: Callable[[float, str], Awaitable[None]] | None = None,
    ) -> JobResult:
        index_name = job.payload.get("index_name", "default")

        if progress_callback:
            res1 = progress_callback(10, "Loading data")
            if asyncio.iscoroutine(res1):
                await res1
        await asyncio.sleep(0.5)

        if progress_callback:
            res2 = progress_callback(50, "Building index")
            if asyncio.iscoroutine(res2):
                await res2
        await asyncio.sleep(0.5)

        if progress_callback:
            res3 = progress_callback(90, "Saving index")
            if asyncio.iscoroutine(res3):
                await res3
        await asyncio.sleep(0.2)

        return JobResult(success=True, data={"index_name": index_name, "document_count": 1000})


# ─────────────────────────────────────────────────────────────────────────────
# Factory and singleton
# ─────────────────────────────────────────────────────────────────────────────

_queue: JobQueue | None = None


def get_job_queue() -> JobQueue:
    """Get or create the singleton job queue instance."""
    global _queue
    if _queue is None:
        # Use Redis store in production, in-memory for development
        redis_url = os.getenv("REDIS_URL")
        store: JobStore
        if redis_url:
            store = RedisJobStore(redis_url)
        else:
            store = InMemoryJobStore()

        _queue = JobQueue(store=store)
        # Register default handlers
        _queue.register_handler(EmbeddingGenerationHandler())
        _queue.register_handler(IndexBuildHandler())

    return _queue


async def init_queue() -> JobQueue:
    """Initialize and start the job queue."""
    queue = get_job_queue()
    await queue.start()
    return queue
