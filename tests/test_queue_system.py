"""Tests for queue system (#215)."""

import asyncio

import pytest

from queue_system import (
    InMemoryJobStore,
    Job,
    JobHandler,
    JobPriority,
    JobProgress,
    JobQueue,
    JobResult,
    JobStatus,
    ResourceConstraints,
    RetryConfig,
)


class TestJob:
    """Tests for Job dataclass."""

    def test_job_creation(self):
        job = Job(
            id="test-123",
            job_type="test_job",
            payload={"key": "value"},
        )
        assert job.id == "test-123"
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.NORMAL
        assert job.retry_count == 0

    def test_job_to_dict(self):
        job = Job(
            id="test-123",
            job_type="test_job",
            payload={"key": "value"},
            priority=JobPriority.HIGH,
        )
        d = job.to_dict()
        assert d["id"] == "test-123"
        assert d["job_type"] == "test_job"
        assert d["priority"] == JobPriority.HIGH.value

    def test_job_from_dict(self):
        data = {
            "id": "test-123",
            "job_type": "test_job",
            "payload": {"key": "value"},
            "priority": JobPriority.HIGH.value,
            "status": JobStatus.RUNNING.value,
        }
        job = Job.from_dict(data)
        assert job.id == "test-123"
        assert job.priority == JobPriority.HIGH
        assert job.status == JobStatus.RUNNING


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay_seconds == 1.0

    def test_exponential_backoff(self):
        config = RetryConfig(
            initial_delay_seconds=1.0,
            exponential_base=2.0,
            jitter=False,
        )
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0

    def test_max_delay_cap(self):
        config = RetryConfig(
            initial_delay_seconds=100,
            max_delay_seconds=60,
            jitter=False,
        )
        assert config.get_delay(5) == 60


class TestInMemoryJobStore:
    """Tests for InMemoryJobStore."""

    @pytest.fixture
    def store(self):
        return InMemoryJobStore()

    @pytest.mark.asyncio
    async def test_save_and_get_job(self, store):
        job = Job(id="test-1", job_type="test", payload={})
        await store.save_job(job)
        retrieved = await store.get_job("test-1")
        assert retrieved is not None
        assert retrieved.id == "test-1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, store):
        job = await store.get_job("nonexistent")
        assert job is None

    @pytest.mark.asyncio
    async def test_get_pending_jobs_ordered_by_priority(self, store):
        low = Job(id="low", job_type="test", payload={}, priority=JobPriority.LOW)
        high = Job(id="high", job_type="test", payload={}, priority=JobPriority.HIGH)
        normal = Job(id="normal", job_type="test", payload={}, priority=JobPriority.NORMAL)

        await store.save_job(low)
        await store.save_job(normal)
        await store.save_job(high)

        pending = await store.get_pending_jobs()
        assert len(pending) == 3
        assert pending[0].id == "high"
        assert pending[1].id == "normal"
        assert pending[2].id == "low"

    @pytest.mark.asyncio
    async def test_delete_job(self, store):
        job = Job(id="test-1", job_type="test", payload={})
        await store.save_job(job)

        deleted = await store.delete_job("test-1")
        assert deleted is True

        retrieved = await store.get_job("test-1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_jobs_by_status(self, store):
        pending = Job(id="pending", job_type="test", payload={}, status=JobStatus.PENDING)
        running = Job(id="running", job_type="test", payload={}, status=JobStatus.RUNNING)

        await store.save_job(pending)
        await store.save_job(running)

        pending_jobs = await store.get_jobs_by_status(JobStatus.PENDING)
        assert len(pending_jobs) == 1
        assert pending_jobs[0].id == "pending"


class MockHandler(JobHandler):
    """Mock handler for testing."""

    def __init__(self, should_fail: bool = False, execution_time: float = 0.01):
        self._should_fail = should_fail
        self._execution_time = execution_time

    @property
    def job_type(self) -> str:
        return "mock_job"

    @property
    def constraints(self) -> ResourceConstraints:
        return ResourceConstraints(timeout_seconds=1)

    @property
    def retry_config(self) -> RetryConfig:
        return RetryConfig(max_retries=2)

    async def execute(self, job, progress_callback=None):
        await asyncio.sleep(self._execution_time)
        if progress_callback:
            await progress_callback(50, "Halfway done")
        await asyncio.sleep(self._execution_time)
        if self._should_fail:
            return JobResult(success=False, error="Mock failure")
        return JobResult(success=True, data={"result": "success"})


class TestJobQueue:
    """Tests for JobQueue."""

    @pytest.fixture
    def queue(self):
        q = JobQueue(max_workers=2)
        q.register_handler(MockHandler())
        return q

    @pytest.mark.asyncio
    async def test_submit_job(self, queue):
        job = await queue.submit(
            job_type="mock_job",
            payload={"test": "data"},
        )
        assert job.id is not None
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_submit_with_priority(self, queue):
        job = await queue.submit(
            job_type="mock_job",
            payload={},
            priority=JobPriority.HIGH,
        )
        assert job.priority == JobPriority.HIGH

    @pytest.mark.asyncio
    async def test_submit_unknown_job_type(self, queue):
        with pytest.raises(ValueError) as exc:
            await queue.submit(job_type="unknown", payload={})
        assert "No handler registered" in str(exc.value)

    @pytest.mark.asyncio
    async def test_get_job(self, queue):
        job = await queue.submit(job_type="mock_job", payload={})
        retrieved = await queue.get_job(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, queue):
        job = await queue.submit(job_type="mock_job", payload={})
        cancelled = await queue.cancel_job(job.id)
        assert cancelled is True

        updated = await queue.get_job(job.id)
        assert updated.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_job_execution(self, queue):
        await queue.start()
        try:
            job = await queue.submit(job_type="mock_job", payload={})

            # Wait for job to complete
            for _ in range(50):
                await asyncio.sleep(0.1)
                updated = await queue.get_job(job.id)
                if updated.status == JobStatus.COMPLETED:
                    break

            assert updated.status == JobStatus.COMPLETED
            assert updated.result == {"result": "success"}
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_job_failure_and_retry(self, queue):
        queue.register_handler(MockHandler(should_fail=True))
        await queue.start()
        try:
            job = await queue.submit(job_type="mock_job", payload={})

            # Wait for job to fail and retry
            for _ in range(100):
                await asyncio.sleep(0.1)
                updated = await queue.get_job(job.id)
                if updated.status == JobStatus.DEAD:
                    break

            # Should have tried 3 times (1 initial + 2 retries)
            assert updated.status == JobStatus.DEAD
            assert updated.retry_count == 2
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_submit_batch(self, queue):
        jobs = await queue.submit_batch(
            [
                {"job_type": "mock_job", "payload": {"i": 1}},
                {"job_type": "mock_job", "payload": {"i": 2}},
                {"job_type": "mock_job", "payload": {"i": 3}},
            ]
        )
        assert len(jobs) == 3
        assert all(j.status == JobStatus.PENDING for j in jobs)

    @pytest.mark.asyncio
    async def test_get_stats(self, queue):
        stats = queue.get_stats()
        assert "total_processed" in stats
        assert "total_failed" in stats
        assert "is_running" in stats

    @pytest.mark.asyncio
    async def test_progress_callback(self, queue):
        progress_updates = []

        def on_progress(progress: JobProgress):
            progress_updates.append(progress)

        await queue.start()
        try:
            job = await queue.submit(job_type="mock_job", payload={})
            queue.on_progress(job.id, on_progress)

            # Wait for job to complete
            for _ in range(50):
                await asyncio.sleep(0.1)
                updated = await queue.get_job(job.id)
                if updated.status == JobStatus.COMPLETED:
                    break

            assert len(progress_updates) > 0
        finally:
            await queue.stop()
