"""Tests for the scholar-review queue: persistence, reviewer endpoints,
verdict round-trip, authorization, and feedback into the knowledge base.

All tests use the in-memory backend and a temporary export path — no Redis,
no live API calls.
"""

import asyncio
import json

import pytest
from fastapi import HTTPException

import review
import review_store
from confidence import ConfidenceSignals, assess
from review import (
    VerdictRequest,
    enqueue_for_review,
    export_reviewed_item,
    get_item,
    list_pending,
    list_reviewed,
    record_verdict,
    require_reviewer,
    review_stats,
)
from review_store import (
    AlreadyReviewedError,
    ReviewItem,
    ReviewStatus,
    ReviewStore,
    Verdict,
)

TOKEN = "test-review-token"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def store(monkeypatch):
    """A fresh in-memory store and a configured reviewer token per test."""
    fresh = ReviewStore()
    fresh._use_redis = False
    fresh._local = {}
    monkeypatch.setattr(review_store, "_store", fresh)
    monkeypatch.setattr(review, "SCHOLAR_REVIEW_TOKEN", TOKEN)
    return fresh


@pytest.fixture
def export_path(tmp_path, monkeypatch):
    path = tmp_path / "reviewed.jsonl"
    monkeypatch.setattr(review, "REVIEW_EXPORT_PATH", str(path))
    return path


def make_item(**overrides) -> ReviewItem:
    fields = {
        "question": "Is this transaction riba?",
        "answer": "I think it might be, but I'm not sure.",
        "confidence": 0.2,
        "band": "abstain",
        "signals": {"expressed_certainty": 0.0},
        "chat_id": "chat-1",
    }
    fields.update(overrides)
    return ReviewItem(**fields)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestReviewStore:
    def test_add_and_get_roundtrip(self, store):
        item = run(store.add(make_item()))
        loaded = run(store.get(item.id))
        assert loaded is not None
        assert loaded.question == "Is this transaction riba?"
        assert loaded.status is ReviewStatus.PENDING

    def test_get_unknown_returns_none(self, store):
        assert run(store.get("no-such-item")) is None

    def test_pending_is_oldest_first(self, store):
        first = run(store.add(make_item(question="first", created_at=100.0)))
        second = run(store.add(make_item(question="second", created_at=200.0)))
        pending = run(store.list_pending())
        assert [item.id for item in pending] == [first.id, second.id]

    def test_verdict_removes_from_pending(self, store):
        item = run(store.add(make_item()))
        run(store.record_verdict(item.id, Verdict.APPROVE, reviewer="Shaykh A"))
        assert run(store.list_pending()) == []
        reviewed = run(store.list_reviewed())
        assert [i.id for i in reviewed] == [item.id]

    def test_verdict_on_unknown_item_returns_none(self, store):
        assert run(store.record_verdict("nope", Verdict.APPROVE)) is None

    def test_approve_keeps_the_original_answer(self, store):
        item = run(store.add(make_item(answer="Original answer.")))
        decided = run(store.record_verdict(item.id, Verdict.APPROVE))
        assert decided.status is ReviewStatus.APPROVED
        assert decided.final_answer == "Original answer."

    def test_correct_replaces_the_answer(self, store):
        item = run(store.add(make_item(answer="Wrong answer.")))
        decided = run(store.record_verdict(
            item.id, Verdict.CORRECT, corrected_answer="The correct answer."
        ))
        assert decided.status is ReviewStatus.CORRECTED
        assert decided.final_answer == "The correct answer."
        assert decided.answer == "Wrong answer."

    def test_reject_yields_no_usable_answer(self, store):
        item = run(store.add(make_item()))
        decided = run(store.record_verdict(item.id, Verdict.REJECT))
        assert decided.status is ReviewStatus.REJECTED
        assert decided.final_answer is None

    def test_second_verdict_raises_already_reviewed(self, store):
        item = run(store.add(make_item()))
        run(store.record_verdict(item.id, Verdict.APPROVE))
        with pytest.raises(AlreadyReviewedError) as exc:
            run(store.record_verdict(item.id, Verdict.REJECT))
        assert exc.value.item.status is ReviewStatus.APPROVED

    def test_concurrent_verdicts_do_not_both_win(self, store):
        """A scholar's verdict is final — a concurrent one must not overwrite it."""
        item = run(store.add(make_item()))

        async def race():
            return await asyncio.gather(
                store.record_verdict(item.id, Verdict.APPROVE, reviewer="A"),
                store.record_verdict(item.id, Verdict.REJECT, reviewer="B"),
                return_exceptions=True,
            )

        results = run(race())
        succeeded = [r for r in results if isinstance(r, ReviewItem)]
        conflicts = [r for r in results if isinstance(r, AlreadyReviewedError)]
        assert len(succeeded) == 1
        assert len(conflicts) == 1
        # And the stored item matches whoever won, not a mix of the two.
        stored = run(store.get(item.id))
        assert stored.reviewer == succeeded[0].reviewer
        assert stored.status is succeeded[0].status

    def test_pagination(self, store):
        for i in range(5):
            run(store.add(make_item(question=f"q{i}", created_at=float(i))))
        page = run(store.list_pending(limit=2, offset=2))
        assert [item.question for item in page] == ["q2", "q3"]

    def test_stats_counts_queue_depth(self, store):
        a = run(store.add(make_item()))
        run(store.add(make_item()))
        run(store.record_verdict(a.id, Verdict.APPROVE))
        stats = run(store.stats())
        assert stats["pending"] == 1
        assert stats["reviewed"] == 1

    def test_in_memory_store_reports_it_is_not_durable(self, store):
        assert store.durable is False
        assert run(store.stats())["durable"] is False

    def test_items_have_no_ttl(self, store):
        """A question waiting on a scholar must not expire unanswered."""
        assert not hasattr(store, "_ttl")
        item = run(store.add(make_item(created_at=0.0)))
        assert run(store.get(item.id)) is not None


# ---------------------------------------------------------------------------
# Redis outage behaviour
#
# from_url() builds the client lazily, so an unreachable Redis only shows up on
# the first real operation — long after the store decided it was durable. These
# use a stub client that always raises, which is what that looks like.
# ---------------------------------------------------------------------------


class ExplodingRedis:
    """Stands in for a Redis that has gone away mid-flight."""

    def __init__(self):
        self.calls = 0

    def _boom(self, *args, **kwargs):
        self.calls += 1
        raise ConnectionError("redis is unreachable")

    get = set = zadd = zrem = zrange = zrevrange = zcard = delete = _boom

    def pipeline(self, transaction=True):
        raise ConnectionError("redis is unreachable")


@pytest.fixture
def redis_down(store):
    store._use_redis = True
    store._redis = ExplodingRedis()
    return store


class TestRedisOutage:
    def test_add_keeps_the_item_instead_of_raising(self, redis_down):
        """A queue write must never be the reason a chat turn fails."""
        item = run(redis_down.add(make_item()))
        assert run(redis_down.get(item.id)) is not None
        assert redis_down._degraded is True

    def test_outage_is_reported_as_not_durable(self, redis_down):
        run(redis_down.add(make_item()))
        assert redis_down.durable is False
        stats = run(redis_down.stats())
        assert stats["degraded"] is True
        assert stats["durable"] is False

    def test_listing_degrades_to_empty_rather_than_500(self, redis_down):
        assert run(redis_down.list_pending(limit=50, offset=0)) == []
        assert run(redis_down.list_reviewed(limit=50, offset=0)) == []

    def test_verdict_still_records_on_a_locally_held_item(self, redis_down):
        item = run(redis_down.add(make_item()))
        decided = run(redis_down.record_verdict(item.id, Verdict.APPROVE))
        assert decided.status is ReviewStatus.APPROVED

    def test_healthy_store_is_not_marked_degraded(self, store):
        run(store.add(make_item()))
        assert store._degraded is False


# ---------------------------------------------------------------------------
# Enqueue from the chat path
# ---------------------------------------------------------------------------


class TestEnqueue:
    def test_enqueue_stores_the_original_answer(self, store):
        assessment = assess(
            ConfidenceSignals(self_consistency=0.1, is_religious=True)
        )
        item = run(enqueue_for_review(
            question="Is X halal?",
            answer="The original doubtful answer.",
            score=assessment.score,
            band=assessment.band.value,
            signals=assessment.signals,
            chat_id="chat-9",
        ))
        stored = run(store.get(item.id))
        # The reviewer must see what the model produced, not the abstention
        # message the user was shown.
        assert stored.answer == "The original doubtful answer."
        assert stored.confidence == assessment.score
        assert stored.chat_id == "chat-9"
        assert stored.status is ReviewStatus.PENDING


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


class TestAuthorization:
    def test_valid_token_passes(self):
        require_reviewer(TOKEN)

    @pytest.mark.parametrize("token", [None, "", "wrong-token"])
    def test_bad_token_is_401(self, token):
        with pytest.raises(HTTPException) as exc:
            require_reviewer(token)
        assert exc.value.status_code == 401

    def test_unconfigured_token_disables_the_endpoints(self, monkeypatch):
        """Closed by default — an unset token must not mean 'open to all'."""
        monkeypatch.setattr(review, "SCHOLAR_REVIEW_TOKEN", "")
        with pytest.raises(HTTPException) as exc:
            require_reviewer("anything")
        assert exc.value.status_code == 503

    def test_endpoints_require_the_token(self, store):
        for call in (
            lambda: list_pending(limit=50, offset=0, x_review_token="wrong"),
            lambda: list_reviewed(limit=50, offset=0, x_review_token="wrong"),
            lambda: review_stats(x_review_token="wrong"),
            lambda: get_item("any", x_review_token="wrong"),
        ):
            with pytest.raises(HTTPException) as exc:
                run(call())
            assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# Reviewer endpoints
# ---------------------------------------------------------------------------


class TestReviewerEndpoints:
    def test_pending_lists_queued_items(self, store):
        run(store.add(make_item(question="q1")))
        response = run(list_pending(limit=50, offset=0, x_review_token=TOKEN))
        assert response.count == 1
        assert response.items[0].question == "q1"

    def test_get_unknown_item_is_404(self, store):
        with pytest.raises(HTTPException) as exc:
            run(get_item("missing", x_review_token=TOKEN))
        assert exc.value.status_code == 404

    def test_verdict_roundtrip_approve(self, store, export_path):
        item = run(store.add(make_item()))
        response = run(record_verdict(
            item.id,
            VerdictRequest(verdict=Verdict.APPROVE, reviewer="Shaykh A"),
            x_review_token=TOKEN,
        ))
        assert response.item.status is ReviewStatus.APPROVED
        assert response.item.reviewer == "Shaykh A"
        assert response.item.reviewed_at is not None
        assert response.exported is True

        # And it is out of the pending queue afterwards.
        assert run(list_pending(limit=50, offset=0, x_review_token=TOKEN)).count == 0

    def test_verdict_roundtrip_correct(self, store, export_path):
        item = run(store.add(make_item()))
        response = run(record_verdict(
            item.id,
            VerdictRequest(
                verdict=Verdict.CORRECT,
                corrected_answer="The scholar-vetted answer.",
                reviewer="Shaykh B",
                note="Missing the hanafi position.",
            ),
            x_review_token=TOKEN,
        ))
        assert response.item.status is ReviewStatus.CORRECTED
        assert response.item.final_answer == "The scholar-vetted answer."
        assert response.item.reviewer_note == "Missing the hanafi position."

    def test_verdict_roundtrip_reject(self, store, export_path):
        item = run(store.add(make_item()))
        response = run(record_verdict(
            item.id, VerdictRequest(verdict=Verdict.REJECT), x_review_token=TOKEN
        ))
        assert response.item.status is ReviewStatus.REJECTED
        assert response.item.final_answer is None

    def test_verdict_on_unknown_item_is_404(self, store):
        with pytest.raises(HTTPException) as exc:
            run(record_verdict(
                "missing", VerdictRequest(verdict=Verdict.APPROVE), x_review_token=TOKEN
            ))
        assert exc.value.status_code == 404

    def test_second_verdict_is_rejected(self, store, export_path):
        item = run(store.add(make_item()))
        run(record_verdict(
            item.id, VerdictRequest(verdict=Verdict.APPROVE), x_review_token=TOKEN
        ))
        with pytest.raises(HTTPException) as exc:
            run(record_verdict(
                item.id, VerdictRequest(verdict=Verdict.REJECT), x_review_token=TOKEN
            ))
        assert exc.value.status_code == 409

    def test_correction_requires_an_answer(self):
        with pytest.raises(ValueError, match="corrected_answer is required"):
            VerdictRequest(verdict=Verdict.CORRECT)

    def test_corrected_answer_rejected_for_other_verdicts(self):
        with pytest.raises(ValueError, match="only accepted"):
            VerdictRequest(verdict=Verdict.APPROVE, corrected_answer="something")

    def test_unknown_verdict_is_rejected(self):
        with pytest.raises(ValueError):
            VerdictRequest(verdict="maybe")


# ---------------------------------------------------------------------------
# Feedback into the knowledge base
# ---------------------------------------------------------------------------


class TestFeedbackExport:
    def test_export_writes_one_jsonl_record(self, store, export_path):
        item = run(store.add(make_item()))
        decided = run(store.record_verdict(
            item.id, Verdict.CORRECT, corrected_answer="Vetted.", reviewer="Shaykh C"
        ))
        assert export_reviewed_item(decided) is True

        lines = export_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["question"] == "Is this transaction riba?"
        assert record["answer"] == "Vetted."
        assert record["original_answer"] == item.answer
        assert record["verdict"] == "correct"
        assert record["source"] == "scholar_review"

    def test_rejected_answers_are_exported_too(self, store, export_path):
        """A wrong answer a scholar caught is a valuable eval case."""
        item = run(store.add(make_item()))
        decided = run(store.record_verdict(item.id, Verdict.REJECT))
        assert export_reviewed_item(decided) is True
        record = json.loads(export_path.read_text(encoding="utf-8").strip())
        assert record["verdict"] == "reject"
        assert record["answer"] == item.answer

    def test_export_appends_rather_than_overwrites(self, store, export_path):
        for i in range(3):
            item = run(store.add(make_item(question=f"q{i}")))
            decided = run(store.record_verdict(item.id, Verdict.APPROVE))
            export_reviewed_item(decided)
        lines = export_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_export_failure_does_not_raise(self, store, tmp_path, monkeypatch):
        """A verdict must record even when the export cannot be written."""
        blocked = tmp_path / "afile"
        blocked.write_text("not a directory")
        monkeypatch.setattr(review, "REVIEW_EXPORT_PATH", str(blocked / "out.jsonl"))
        item = run(store.add(make_item()))
        decided = run(store.record_verdict(item.id, Verdict.APPROVE))
        assert export_reviewed_item(decided) is False

    def test_cache_write_is_skipped_when_cache_disabled(self, store):
        item = run(store.add(make_item()))
        decided = run(store.record_verdict(item.id, Verdict.APPROVE))
        # SEMANTIC_CACHE_ENABLED is off by default in tests.
        assert review.cache_reviewed_answer(decided) is False

    def test_rejected_answer_is_never_cached(self, store):
        item = run(store.add(make_item()))
        decided = run(store.record_verdict(item.id, Verdict.REJECT))
        assert review.cache_reviewed_answer(decided) is False

    def test_approved_answer_reaches_the_cache(self, store, monkeypatch):
        import numpy as np

        import semantic_cache

        monkeypatch.setattr(semantic_cache, "SEMANTIC_CACHE_ENABLED", True)
        semantic_cache.set_fake_embedding(np.array([1.0, 0.0], dtype=np.float32))
        cache = semantic_cache.get_cache()
        cache.clear()
        try:
            item = run(store.add(make_item()))
            decided = run(store.record_verdict(
                item.id, Verdict.CORRECT, corrected_answer="Vetted answer."
            ))
            assert review.cache_reviewed_answer(decided) is True
            hit = cache.get(np.array([1.0, 0.0], dtype=np.float32))
            assert hit is not None
            assert hit.response == "Vetted answer."
        finally:
            semantic_cache.set_fake_embedding(None)
            cache.clear()
