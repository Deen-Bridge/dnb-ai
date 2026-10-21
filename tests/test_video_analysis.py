import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from video_analysis import (
    build_search_index,
    extract_quran_references,
    router,
    segment_topics,
)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_extracts_quran_references_without_duplicates():
    assert extract_quran_references("Read 2:255, then return to 2:255 and 36:1.") == ["2:255", "36:1"]


def test_extracts_quran_references_empty():
    assert extract_quran_references("No verses here") == []


def test_segments_transcript_into_searchable_topics():
    topics = segment_topics("First lesson. Second lesson about charity.")
    assert len(topics) == 2
    assert topics[0]["start_seconds"] == 0
    assert topics[1]["start_seconds"] == 30


def test_segments_empty_transcript():
    assert segment_topics("") == []


def test_search_index_maps_transcript_tokens_to_positions():
    index = build_search_index("Arabic charity lecture", [])
    assert index["charity"] == [1]


def test_get_nonexistent_job_returns_404(client: TestClient):
    response = client.get("/video-analysis/jobs/nonexistent-uuid")
    assert response.status_code == 404
    assert "Analysis job not found" in response.json()["detail"]


def test_submit_unsupported_extension_returns_415(client: TestClient):
    file_bytes = io.BytesIO(b"dummy audio content")
    response = client.post(
        "/video-analysis/analyze",
        files={"file": ("lecture.mp3", file_bytes, "audio/mpeg")},
        data={"transcript": "Sample transcript"},
    )
    assert response.status_code == 415
    assert "Only MP4, MOV, and WebM" in response.json()["detail"]


def test_submit_valid_video_returns_202_and_job_is_retrievable(client: TestClient):
    file_bytes = io.BytesIO(b"fake video payload")
    response = client.post(
        "/video-analysis/analyze",
        files={"file": ("lecture.mp4", file_bytes, "video/mp4")},
        data={"transcript": "Lecture on patience and gratitude."},
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    job_id = data["job_id"]

    # Verify job endpoint returns the job
    job_response = client.get(f"/video-analysis/jobs/{job_id}")
    assert job_response.status_code == 200
    job_data = job_response.json()
    assert job_data["id"] == job_id
    assert job_data["filename"] == "lecture.mp4"
