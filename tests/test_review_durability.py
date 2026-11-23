"""Failure-injection tests for scholar review queue durability under Redis outage,
process restart, recovery reconciliation, and multi-replica concurrency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import review
import review_store
from main import app
from review_store import (
    ITEM_KEY,
    PENDING_INDEX,
    REVIEWED_INDEX,
    ReviewItem,
    ReviewSpoolFullError,
    ReviewStatus,
    ReviewStore,
    Verdict,
)

TOKEN = "test-durability-review-token"


def run(coro):
    return asyncio.run(coro)


class FakePipeline:
    def __init__(self, redis: FakeAsyncRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple[Any, ...]]] = []

    def set(self, key: str, val: str) -> None:
        self._commands.append(("set", (key, val)))

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        self._commands.append(("zadd", (key, mapping)))

    def zrem(self, key: str, member: str) -> None:
        self._commands.append(("zrem", (key, member)))

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        for cmd, args in self._commands:
            if cmd == "set":
                results.append(await self._redis.set(*args))
            elif cmd == "zadd":
                results.append(await self._redis.zadd(*args))
            elif cmd == "zrem":
                results.append(await self._redis.zrem(*args))
        return results

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class FakeAsyncRedis:
    """In-memory async Redis simulation for testing durable replication and recovery."""

    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.is_connected = True

    async def ping(self) -> bool:
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        return True

    async def get(self, key: str) -> str | None:
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        return self.strings.get(key)

    async def set(self, key: str, val: str) -> bool:
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        self.strings[key] = val
        return True

    async def delete(self, *keys: str) -> int:
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        deleted = 0
        for k in keys:
            if k in self.strings:
                del self.strings[k]
                deleted += 1
            if k in self.zsets:
                del self.zsets[k]
                deleted += 1
        return deleted

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        if key not in self.zsets:
            self.zsets[key] = {}
        added = 0
        for member, score in mapping.items():
            if member not in self.zsets[key]:
                added += 1
            self.zsets[key][member] = score
        return added

    async def zrem(self, key: str, member: str) -> int:
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        if key in self.zsets and member in self.zsets[key]:
            del self.zsets[key][member]
            return 1
        return 0

    async def zcard(self, key: str) -> int:
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        return len(self.zsets.get(key, {}))

    async def zrange(self, key: str, start: int, stop: int) -> list[str]:
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        members = sorted(self.zsets.get(key, {}).items(), key=lambda x: x[1])
        slice_members = members[start : stop + 1 if stop != -1 else None]
        return [m[0] for m in slice_members]

    async def zrevrange(self, key: str, start: int, stop: int) -> list[str]:
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        members = sorted(self.zsets.get(key, {}).items(), key=lambda x: x[1], reverse=True)
        slice_members = members[start : stop + 1 if stop != -1 else None]
        return [m[0] for m in slice_members]

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        return FakePipeline(self)


def make_item(idx: int = 1, **overrides: Any) -> ReviewItem:
    data: dict[str, Any] = {
        "question": f"Is transaction #{idx} permissible under Islamic fiqh?",
        "answer": f"Initial uncertain answer #{idx}",
        "confidence": 0.25,
        "band": "abstain",
        "signals": {"theological_controversy": 0.8},
        "chat_id": f"chat-{idx}",
    }
    data.update(overrides)
    return ReviewItem(**data)


class TestScholarReviewDurability:
    def test_outage_spools_items_to_disk_durably(self, tmp_path: Path):
        """When Redis fails on add, the item is persisted to disk and marked degraded."""
        spool_dir = tmp_path / "spool"
        fake_redis = FakeAsyncRedis()
        fake_redis.is_connected = False  # Inject outage

        store = ReviewStore(spool_dir=spool_dir)
        store._redis = fake_redis
        store._use_redis = True

        item = make_item(1)
        run(store.add(item))

        # Check in-memory state
        assert store.durable is False
        assert store._degraded is True
        assert item.id in store._local

        # Check disk spool file
        spool_file = spool_dir / f"item_{item.id}.json"
        assert spool_file.exists(), "Item must be written to disk spool file"
        content = json.loads(spool_file.read_text(encoding="utf-8"))
        assert content["id"] == item.id
        assert content["question"] == item.question

        # Stats reflect spooled item
        stats = run(store.stats())
        assert stats["durable"] is False
        assert stats["degraded"] is True
        assert stats["spool"]["size"] == 1
        assert stats["spool"]["total_spooled"] == 1

    def test_process_restart_preserves_spooled_items(self, tmp_path: Path):
        """A process restart or container replacement must recover all spooled items."""
        spool_dir = tmp_path / "spool"
        fake_redis = FakeAsyncRedis()
        fake_redis.is_connected = False

        # 1. Instance 1 writes 3 items while Redis is down
        store1 = ReviewStore(spool_dir=spool_dir)
        store1._redis = fake_redis
        store1._use_redis = True

        items = [make_item(i) for i in range(1, 4)]
        for it in items:
            run(store1.add(it))

        assert store1._spool.size() == 3

        # 2. Simulate process crash / replacement: discard store1 and instantiate store2
        del store1

        store2 = ReviewStore(spool_dir=spool_dir)
        store2._redis = fake_redis
        store2._use_redis = True

        # Store 2 loads all 3 items automatically from the spool directory
        assert store2._degraded is True
        assert len(store2._local) == 3
        for it in items:
            loaded = run(store2.get(it.id))
            assert loaded is not None
            assert loaded.question == it.question
            assert loaded.confidence == it.confidence

    def test_recovery_reconciles_spool_to_redis_without_duplicates(self, tmp_path: Path):
        """When Redis recovers, spooled items are reconciled into Redis with zero duplicates."""
        spool_dir = tmp_path / "spool"
        fake_redis = FakeAsyncRedis()
        fake_redis.is_connected = False

        store = ReviewStore(spool_dir=spool_dir)
        store._redis = fake_redis
        store._use_redis = True

        item1 = make_item(1)
        item2 = make_item(2)
        run(store.add(item1))
        run(store.add(item2))
        assert store._spool.size() == 2

        # Redis comes back online!
        fake_redis.is_connected = True

        reconciled = run(store.reconcile_spool())
        assert reconciled == 2
        assert store._spool.size() == 0, "Spool files must be deleted after reconciliation"
        assert store._degraded is False
        assert store.durable is True

        # Verify items exist in Redis
        assert fake_redis.strings[ITEM_KEY.format(item_id=item1.id)] is not None
        assert fake_redis.strings[ITEM_KEY.format(item_id=item2.id)] is not None
        assert len(fake_redis.zsets[PENDING_INDEX]) == 2

        # Second reconciliation is idempotent
        reconciled_again = run(store.reconcile_spool())
        assert reconciled_again == 0
        assert len(fake_redis.zsets[PENDING_INDEX]) == 2

    def test_multi_replica_reconciliation_zero_duplicates(self, tmp_path: Path):
        """Multiple replicas spooling independently reconcile to shared Redis without duplicate entries."""
        spool_dir_1 = tmp_path / "replica_1"
        spool_dir_2 = tmp_path / "replica_2"

        shared_redis = FakeAsyncRedis()
        shared_redis.is_connected = False  # Both replicas experience outage

        replica1 = ReviewStore(spool_dir=spool_dir_1)
        replica1._redis = shared_redis
        replica1._use_redis = True

        replica2 = ReviewStore(spool_dir=spool_dir_2)
        replica2._redis = shared_redis
        replica2._use_redis = True

        item_a = make_item(1)
        item_b = make_item(2)
        item_c = make_item(3)

        # Replica 1 receives A and B
        run(replica1.add(item_a))
        run(replica1.add(item_b))

        # Replica 2 receives B (retry) and C
        run(replica2.add(item_b))
        run(replica2.add(item_c))

        assert replica1._spool.size() == 2
        assert replica2._spool.size() == 2

        # Redis recovers
        shared_redis.is_connected = True

        # Both replicas reconcile to the shared Redis
        reconciled1 = run(replica1.reconcile_spool())
        reconciled2 = run(replica2.reconcile_spool())

        assert reconciled1 == 2
        assert reconciled2 == 2

        # The pending index must have exactly 3 unique members
        pending_ids = run(shared_redis.zrange(PENDING_INDEX, 0, -1))
        assert len(pending_ids) == 3
        assert set(pending_ids) == {item_a.id, item_b.id, item_c.id}

        # Both local spools are now empty
        assert replica1._spool.size() == 0
        assert replica2._spool.size() == 0

    def test_bounded_spool_capacity_rejection_policy(self, tmp_path: Path, monkeypatch: Any):
        """When capacity is reached under spool_or_reject, subsequent items raise ReviewSpoolFullError."""
        spool_dir = tmp_path / "spool"
        fake_redis = FakeAsyncRedis()
        fake_redis.is_connected = False

        store = ReviewStore(
            spool_dir=spool_dir,
            max_spool_size=2,
            degraded_policy="spool_or_reject",
        )
        store._redis = fake_redis
        store._use_redis = True

        it1 = make_item(1)
        it2 = make_item(2)
        it3 = make_item(3)

        run(store.add(it1))
        run(store.add(it2))
        assert store._spool.size() == 2

        # Adding 3rd item exceeds capacity
        with pytest.raises(ReviewSpoolFullError) as exc_info:
            run(store.add(it3))
        assert exc_info.value.size == 2
        assert exc_info.value.max_size == 2
        assert store.total_rejected == 1

        stats = run(store.stats())
        assert stats["spool"]["total_rejected"] == 1
        assert stats["alerts"]["drops_or_rejections"] is True

    def test_bounded_spool_capacity_drop_policy(self, tmp_path: Path):
        """When capacity is reached under spool_or_drop, oldest item is evicted with drop counter incremented."""
        spool_dir = tmp_path / "spool"
        fake_redis = FakeAsyncRedis()
        fake_redis.is_connected = False

        store = ReviewStore(
            spool_dir=spool_dir,
            max_spool_size=2,
            degraded_policy="spool_or_drop",
        )
        store._redis = fake_redis
        store._use_redis = True

        it1 = make_item(1, created_at=100.0)
        it2 = make_item(2, created_at=200.0)
        it3 = make_item(3, created_at=300.0)

        run(store.add(it1))
        run(store.add(it2))
        assert store._spool.size() == 2

        # Adding 3rd item evicts it1
        run(store.add(it3))
        assert store._spool.size() == 2
        assert store.total_dropped == 1

        # it1 is gone from memory and disk; it2 and it3 remain
        assert run(store.get(it1.id)) is None
        assert not (spool_dir / f"item_{it1.id}.json").exists()
        assert run(store.get(it2.id)) is not None
        assert run(store.get(it3.id)) is not None

        stats = run(store.stats())
        assert stats["spool"]["total_dropped"] == 1
        assert stats["alerts"]["drops_or_rejections"] is True

    def test_alert_on_queue_depth(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        """Depth exceeding threshold emits a warning and triggers alerts.high_depth."""
        spool_dir = tmp_path / "spool"
        fake_redis = FakeAsyncRedis()
        fake_redis.is_connected = False

        store = ReviewStore(
            spool_dir=spool_dir,
            alert_depth_threshold=2,
        )
        store._redis = fake_redis
        store._use_redis = True

        with caplog.at_level(logging.WARNING):
            run(store.add(make_item(1)))
            assert store._evaluate_alerts()["high_depth"] is False

            run(store.add(make_item(2)))
            assert store._evaluate_alerts()["high_depth"] is True

        assert "REVIEW_QUEUE_ALERT alert=depth_exceeded depth=2 threshold=2" in caplog.text

    def test_alert_on_queue_age(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        """Age of oldest item exceeding threshold triggers alerts.high_age."""
        spool_dir = tmp_path / "spool"
        fake_redis = FakeAsyncRedis()
        fake_redis.is_connected = False

        store = ReviewStore(
            spool_dir=spool_dir,
            alert_age_threshold_seconds=10.0,
        )
        store._redis = fake_redis
        store._use_redis = True

        # Old item created 25 seconds ago
        old_item = make_item(1, created_at=time.time() - 25.0)

        with caplog.at_level(logging.WARNING):
            run(store.add(old_item))
            alerts = store._evaluate_alerts()
            assert alerts["high_age"] is True

        assert "REVIEW_QUEUE_ALERT alert=age_exceeded" in caplog.text

    def test_verdict_recorded_during_outage_persists_and_reconciles(self, tmp_path: Path):
        """Verdicts recorded while Redis is down update the local spool and reconcile to reviewed index on recovery."""
        spool_dir = tmp_path / "spool"
        fake_redis = FakeAsyncRedis()
        fake_redis.is_connected = False

        store = ReviewStore(spool_dir=spool_dir)
        store._redis = fake_redis
        store._use_redis = True

        item = make_item(1)
        run(store.add(item))

        # Scholar approves item while Redis is still down
        decided = run(store.record_verdict(item.id, Verdict.APPROVE, reviewer="Shaykh B"))
        assert decided is not None
        assert decided.status is ReviewStatus.APPROVED

        # Verify spool file on disk was updated
        spool_file = spool_dir / f"item_{item.id}.json"
        content = json.loads(spool_file.read_text(encoding="utf-8"))
        assert content["status"] == "approved"
        assert content["reviewer"] == "Shaykh B"

        # Redis recovers
        fake_redis.is_connected = True
        run(store.reconcile_spool())

        # Item should be in review:reviewed index, NOT review:pending
        assert run(fake_redis.zcard(REVIEWED_INDEX)) == 1
        assert run(fake_redis.zcard(PENDING_INDEX)) == 0

    def test_reconcile_endpoint(self, tmp_path: Path, monkeypatch: Any):
        """POST /review/reconcile endpoint triggers spool reconciliation for operators."""
        spool_dir = tmp_path / "spool"
        fake_redis = FakeAsyncRedis()
        fake_redis.is_connected = False

        store = ReviewStore(spool_dir=spool_dir)
        store._redis = fake_redis
        store._use_redis = True

        item = make_item(1)
        run(store.add(item))

        fake_redis.is_connected = True

        monkeypatch.setattr(review, "SCHOLAR_REVIEW_TOKEN", TOKEN)
        monkeypatch.setattr(review_store, "_store", store)

        client = TestClient(app)
        res = client.post("/review/reconcile", headers={"X-Review-Token": TOKEN})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["reconciled"] == 1
        assert data["remaining_spool_size"] == 0
        assert data["durable"] is True
