"""Tests for media upload limits, magic-byte sniffing, and resource guardrails (#853)."""

from __future__ import annotations

import io
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from main import app
from media_upload_limits import (
    MAX_AUDIO_BYTES,
    MAX_IMAGE_BYTES,
    MAX_PAGE_BLOCKS,
    MAX_PAGE_DIMENSION,
    MAX_VIDEO_BYTES,
    BoundedJobManager,
    ConfidenceBand,
    evaluate_confidence_and_fallback,
    sniff_audio_format,
    sniff_image_format,
    sniff_video_format,
    validate_audio_upload,
    validate_image_upload,
    validate_page_layout_input,
    validate_video_upload,
)

# Sample magic byte payloads
SAMPLE_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
SAMPLE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00"
SAMPLE_MP4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free"
SAMPLE_WEBM = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08"
SAMPLE_MP3 = b"ID3\x04\x00\x00\x00\x00\x00#TSSE\x00\x00\x00\x0f\x00\x00\x03Lavf58.76.100\x00"
SAMPLE_WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00"


# ---------------------------------------------------------------------------
# Magic-Byte Sniffing Tests
# ---------------------------------------------------------------------------


class TestMagicByteSniffing:
    def test_sniff_image_signatures(self):
        assert sniff_image_format(SAMPLE_PNG) == "png"
        assert sniff_image_format(SAMPLE_JPEG) == "jpeg"
        assert sniff_image_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
        assert sniff_image_format(b"II*\x00\x08\x00\x00\x00") == "tiff"
        assert sniff_image_format(b"MM\x00*\x00\x00\x00\x08") == "tiff"
        assert sniff_image_format(b"%PDF-1.7\n") == "pdf"
        assert sniff_image_format(b"plain text file") is None

    def test_sniff_audio_signatures(self):
        assert sniff_audio_format(SAMPLE_MP3) == "mp3"
        assert sniff_audio_format(SAMPLE_WAV) == "wav"
        assert sniff_audio_format(b"OggS\x00\x02\x00\x00\x00\x00") == "ogg"
        assert sniff_audio_format(b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00") == "m4a"
        assert sniff_audio_format(b"not audio") is None

    def test_sniff_video_signatures(self):
        assert sniff_video_format(SAMPLE_MP4) == "mp4"
        assert sniff_video_format(SAMPLE_WEBM) == "webm"
        assert sniff_video_format(b"RIFF\x00\x00\x00\x00AVI LIST") == "avi"
        assert sniff_video_format(b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00") == "mov"
        assert sniff_video_format(b"\x00\x00\x00\x08moov") == "mov"
        assert sniff_video_format(b"echo 'hello'") is None


# ---------------------------------------------------------------------------
# Media Upload Validation Tests
# ---------------------------------------------------------------------------


class TestMediaUploadValidation:
    def test_video_upload_happy_path(self):
        mime, fmt = validate_video_upload(SAMPLE_MP4, "lecture.mp4", duration_seconds=120.0)
        assert mime == "video/mp4"
        assert fmt == "mp4"

    def test_video_upload_empty_fails(self):
        with pytest.raises(HTTPException) as exc:
            validate_video_upload(b"", "empty.mp4")
        assert exc.value.status_code == 400
        assert "empty" in exc.value.detail.lower()

    def test_video_upload_missing_consent_fails(self):
        with pytest.raises(HTTPException) as exc:
            validate_video_upload(SAMPLE_MP4, "lecture.mp4", consent=False)
        assert exc.value.status_code == 400
        assert "consent" in exc.value.detail.lower()

    def test_video_upload_spoofed_extension_fails(self):
        fake_payload = b"<html><body>Not a video</body></html>"
        with pytest.raises(HTTPException) as exc:
            validate_video_upload(fake_payload, "malicious.mp4")
        assert exc.value.status_code == 415

    def test_video_upload_oversized_fails(self):
        oversized = SAMPLE_MP4 + b"x" * (MAX_VIDEO_BYTES + 10)
        with pytest.raises(HTTPException) as exc:
            validate_video_upload(oversized, "huge.mp4")
        assert exc.value.status_code == 413

    def test_video_upload_duration_limits(self):
        # Negative duration
        with pytest.raises(HTTPException) as exc:
            validate_video_upload(SAMPLE_MP4, "clip.mp4", duration_seconds=-5.0)
        assert exc.value.status_code == 422

        # Exceeds max 1800s (30m)
        with pytest.raises(HTTPException) as exc:
            validate_video_upload(SAMPLE_MP4, "long.mp4", duration_seconds=1801.0)
        assert exc.value.status_code == 422
        assert "duration" in exc.value.detail.lower()

    def test_audio_upload_happy_path(self):
        mime, fmt = validate_audio_upload(SAMPLE_MP3, "hadith.mp3", duration_seconds=60.0)
        assert mime == "audio/mpeg"
        assert fmt == "mp3"

    def test_audio_upload_duration_limits(self):
        with pytest.raises(HTTPException) as exc:
            validate_audio_upload(SAMPLE_MP3, "recitation.mp3", duration_seconds=3601.0)
        assert exc.value.status_code == 422

    def test_audio_upload_oversized_fails(self):
        oversized = SAMPLE_MP3 + b"x" * (MAX_AUDIO_BYTES + 10)
        with pytest.raises(HTTPException) as exc:
            validate_audio_upload(oversized, "huge.mp3")
        assert exc.value.status_code == 413

    def test_image_upload_happy_path(self):
        mime, fmt = validate_image_upload(SAMPLE_PNG, "manuscript.png")
        assert mime == "image/png"
        assert fmt == "png"

    def test_image_upload_oversized_fails(self):
        oversized = SAMPLE_PNG + b"x" * (MAX_IMAGE_BYTES + 10)
        with pytest.raises(HTTPException) as exc:
            validate_image_upload(oversized, "large.png")
        assert exc.value.status_code == 413

    def test_page_layout_validation(self):
        # Valid page
        validate_page_layout_input(width=800, height=1200, blocks_count=10)

        # Missing consent
        with pytest.raises(HTTPException) as exc:
            validate_page_layout_input(width=800, height=1200, blocks_count=10, consent=False)
        assert exc.value.status_code == 400

        # Excessive dimensions
        with pytest.raises(HTTPException) as exc:
            validate_page_layout_input(width=MAX_PAGE_DIMENSION + 1, height=1200, blocks_count=10)
        assert exc.value.status_code == 422

        # Excessive blocks
        with pytest.raises(HTTPException) as exc:
            validate_page_layout_input(width=800, height=1200, blocks_count=MAX_PAGE_BLOCKS + 1)
        assert exc.value.status_code == 413


# ---------------------------------------------------------------------------
# Confidence Bands & Safe Fallback Tests
# ---------------------------------------------------------------------------


class TestConfidenceAndFallback:
    def test_high_confidence(self):
        band, fallback, review = evaluate_confidence_and_fallback(0.85)
        assert band == ConfidenceBand.HIGH
        assert fallback is False
        assert review is False

    def test_medium_confidence(self):
        band, fallback, review = evaluate_confidence_and_fallback(0.55)
        assert band == ConfidenceBand.MEDIUM
        assert fallback is False
        assert review is True

    def test_low_confidence_triggers_fallback(self):
        band, fallback, review = evaluate_confidence_and_fallback(0.20)
        assert band == ConfidenceBand.LOW
        assert fallback is True
        assert review is True

    def test_none_confidence_treated_defensively(self):
        band, fallback, review = evaluate_confidence_and_fallback(None)
        assert band == ConfidenceBand.MEDIUM
        assert fallback is False
        assert review is True


# ---------------------------------------------------------------------------
# Bounded Job Manager Tests
# ---------------------------------------------------------------------------


class TestBoundedJobManager:
    def test_register_and_retrieve_job(self):
        mgr = BoundedJobManager()
        job = mgr.register_job("job-1", "user/path/lecture.mp4")
        assert job["id"] == "job-1"
        assert job["filename"] == "lecture.mp4"
        assert job["status"] == "queued"

        retrieved = mgr.get_job("job-1")
        assert retrieved is not None
        assert retrieved["id"] == "job-1"

    def test_path_traversal_sanitized(self):
        mgr = BoundedJobManager()
        job = mgr.register_job("job-safe", "../../../etc/passwd")
        assert ".." not in job["filename"]
        assert "/" not in job["filename"]

    def test_active_capacity_ceiling(self):
        mgr = BoundedJobManager(max_active=3)
        mgr.register_job("j1")
        mgr.register_job("j2")
        mgr.register_job("j3")
        with pytest.raises(HTTPException) as exc:
            mgr.register_job("j4")
        assert exc.value.status_code == 429

    def test_cancel_job(self):
        mgr = BoundedJobManager()
        mgr.register_job("j-cancel")
        cancelled = mgr.cancel_job("j-cancel")
        assert cancelled["status"] == "cancelled"

        # Repeating cancel is safe
        cancelled_again = mgr.cancel_job("j-cancel")
        assert cancelled_again["status"] == "cancelled"

    def test_cancel_completed_job_raises_409(self):
        mgr = BoundedJobManager()
        mgr.register_job("j-done")
        mgr.update_job("j-done", status="completed")
        with pytest.raises(HTTPException) as exc:
            mgr.cancel_job("j-done")
        assert exc.value.status_code == 409

    def test_ttl_purges_expired_jobs(self):
        mgr = BoundedJobManager(ttl_seconds=0.05)
        mgr.register_job("j-expiring")
        time.sleep(0.06)
        purged = mgr.purge_expired_jobs()
        assert purged >= 1
        assert mgr.get_job("j-expiring") is None


# ---------------------------------------------------------------------------
# API Endpoints Integration Tests
# ---------------------------------------------------------------------------


class TestAPIEndpointsIntegration:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = TestClient(app)

    def test_video_analyze_spoofed_file_rejected(self):
        files = {"file": ("malicious.mp4", io.BytesIO(b"not video data"), "video/mp4")}
        response = self.client.post("/video-analysis/analyze", files=files)
        assert response.status_code == 415

    def test_video_analyze_empty_file_rejected(self):
        files = {"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")}
        response = self.client.post("/video-analysis/analyze", files=files)
        assert response.status_code == 400

    def test_video_analyze_missing_consent_rejected(self):
        files = {"file": ("test.mp4", io.BytesIO(SAMPLE_MP4), "video/mp4")}
        data = {"consent_to_process": "false"}
        response = self.client.post("/video-analysis/analyze", files=files, data=data)
        assert response.status_code == 400

    def test_video_analyze_valid_upload_queued_and_cancellable(self):
        from video_analysis import job_manager

        files = {"file": ("lecture.mp4", io.BytesIO(SAMPLE_MP4), "video/mp4")}
        data = {"consent_to_process": "true"}
        response = self.client.post("/video-analysis/analyze", files=files, data=data)
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        job_id = body["job_id"]

        # In TestClient, background tasks execute synchronously before return,
        # so this completed job returns 409 conflict when cancelled.
        completed_cancel = self.client.post(f"/video-analysis/jobs/{job_id}/cancel")
        assert completed_cancel.status_code == 409

        # Register a pending queued job to test successful 200 cancellation endpoint
        queued_id = "test-cancellable-queued"
        job_manager.register_job(queued_id, "pending_video.mp4")
        cancel_resp = self.client.post(f"/video-analysis/jobs/{queued_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

    def test_audio_hadith_consent_and_duration_limits(self):
        # Missing consent
        payload = {
            "video_ref": "Bukhari 1: Deeds are by intentions",
            "consent_to_process": False,
        }
        resp = self.client.post("/audio-hadith/lecture-transcript", json=payload)
        assert resp.status_code == 400

        # Duration exceeded
        payload_long = {
            "video_ref": "Bukhari 1: Deeds are by intentions",
            "duration_seconds": 1801.0,
            "consent_to_process": True,
        }
        resp_long = self.client.post("/audio-hadith/lecture-transcript", json=payload_long)
        assert resp_long.status_code == 422

        # Valid request
        payload_valid = {
            "video_ref": "Bukhari 1: Deeds are by intentions",
            "duration_seconds": 300.0,
            "consent_to_process": True,
        }
        resp_valid = self.client.post("/audio-hadith/lecture-transcript", json=payload_valid)
        assert resp_valid.status_code == 200

    def test_quran_recitation_consent_and_fallback(self):
        # Missing consent
        resp = self.client.post(
            "/quran-recitation/analyze",
            json={"transcript": "بسم الله الرحمن الرحيم", "consent_to_process": False},
        )
        assert resp.status_code == 400

        # Duration exceeded
        resp_dur = self.client.post(
            "/quran-recitation/analyze",
            json={
                "transcript": "بسم الله الرحمن الرحيم",
                "duration_seconds": 3601.0,
                "consent_to_process": True,
            },
        )
        assert resp_dur.status_code == 422

        # Low ASR confidence activates fallback note
        resp_low = self.client.post(
            "/quran-recitation/analyze",
            json={
                "transcript": "بسم الله الرحمن الرحيم الحمد لله رب العالمين",
                "asr_confidence": 0.25,
                "consent_to_process": True,
            },
        )
        assert resp_low.status_code == 200
        data = resp_low.json()
        assert data["confidence_band"] == "low"
        assert data["fallback_recommended"] is True
        assert "fallback" in data["note"].lower()

    def test_page_analysis_consent_and_bounds(self):
        page_no_consent = {
            "width": 1000,
            "height": 1500,
            "blocks": [],
            "consent_to_process": False,
        }
        resp = self.client.post("/page-analysis/analyze", json=page_no_consent)
        assert resp.status_code == 400

        page_oversized_dim = {
            "width": 25000,
            "height": 1500,
            "blocks": [],
            "consent_to_process": True,
        }
        resp_dim = self.client.post("/page-analysis/analyze", json=page_oversized_dim)
        assert resp_dim.status_code == 422

    def test_calligraphy_consent(self):
        resp = self.client.post(
            "/calligraphy/classify",
            json={"angularity": 0.5, "consent_to_process": False},
        )
        assert resp.status_code == 400

        resp_valid = self.client.post(
            "/calligraphy/classify",
            json={"angularity": 0.5, "consent_to_process": True},
        )
        assert resp_valid.status_code == 200

    def test_arabic_ocr_process_file_validation(self):
        # Empty file
        files_empty = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
        resp_empty = self.client.post("/arabic-ocr/process-file", files=files_empty)
        assert resp_empty.status_code == 400

        # Fake file
        files_fake = {"file": ("fake.png", io.BytesIO(b"hello world text"), "image/png")}
        resp_fake = self.client.post("/arabic-ocr/process-file", files=files_fake)
        assert resp_fake.status_code == 415

        # Missing consent
        files_valid = {"file": ("valid.png", io.BytesIO(SAMPLE_PNG), "image/png")}
        data_no_consent = {"consent_to_process": "false"}
        resp_consent = self.client.post("/arabic-ocr/process-file", files=files_valid, data=data_no_consent)
        assert resp_consent.status_code == 400
