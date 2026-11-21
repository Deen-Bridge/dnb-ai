import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

import video_analysis
from video_analysis import build_search_index, extract_quran_references, segment_topics


@pytest.fixture(autouse=True)
def _clean_jobs():
    video_analysis._jobs.clear()
    yield
    video_analysis._jobs.clear()


def _job(status, **extra):
    video_analysis._jobs["j"] = {"id": "j", "status": status, "created_at": datetime.now(UTC).isoformat(), **extra}


def test_cancel_queued_job_and_idempotent():
    _job("queued")
    assert asyncio.run(video_analysis.cancel_video_analysis("j"))["status"] == "cancelled"
    assert asyncio.run(video_analysis.cancel_video_analysis("j"))["status"] == "cancelled"


def test_cancel_completed_job_conflicts_and_unknown_404():
    _job("completed")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(video_analysis.cancel_video_analysis("j"))
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        asyncio.run(video_analysis.cancel_video_analysis("missing"))
    assert exc.value.status_code == 404


def test_cancelled_job_is_not_processed(monkeypatch):
    _job("cancelled")
    monkeypatch.setattr(video_analysis, "analyze_video", lambda *a: pytest.fail("should not run"))
    asyncio.run(video_analysis._process("j", b"x", ".mp4", ""))
    assert video_analysis._jobs["j"]["status"] == "cancelled"


def test_cancel_during_processing_discards_result(monkeypatch):
    _job("queued")

    def fake(*_args):
        video_analysis._jobs["j"]["status"] = "cancelled"
        return {"transcript": "secret"}

    monkeypatch.setattr(video_analysis, "analyze_video", fake)
    asyncio.run(video_analysis._process("j", b"x", ".mp4", ""))
    assert video_analysis._jobs["j"]["status"] == "cancelled"
    assert "result" not in video_analysis._jobs["j"]


def test_failure_message_is_sanitized(monkeypatch):
    _job("queued")

    def boom(*_args):
        raise RuntimeError("/secret/path leaked")

    monkeypatch.setattr(video_analysis, "analyze_video", boom)
    asyncio.run(video_analysis._process("j", b"x", ".mp4", ""))
    assert video_analysis._jobs["j"]["status"] == "failed"
    assert "secret" not in video_analysis._jobs["j"]["error"]


def test_expired_terminal_jobs_are_purged():
    _job("completed", finished_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat())
    assert video_analysis.purge_expired_jobs() == 1
    assert "j" not in video_analysis._jobs


def test_extracts_quran_references_without_duplicates():
    assert extract_quran_references("Read 2:255, then return to 2:255 and 36:1.") == ["2:255", "36:1"]


def test_segments_transcript_into_searchable_topics():
    topics = segment_topics("First lesson. Second lesson about charity.")
    assert len(topics) == 2
    assert topics[0]["start_seconds"] == 0
    assert topics[1]["start_seconds"] == 30


def test_search_index_maps_transcript_tokens_to_positions():
    index = build_search_index("Arabic charity lecture", [])
    assert index["charity"] == [1]
