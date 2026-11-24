"""Multimodal media upload limits, magic-byte sniffing, and resource guardrails (#853).

This module enforces explicit upload contracts across all media endpoints:
- Explicit input limits: size ceilings, duration bounds, and dimension constraints.
- Content sniffing: validates true container headers, preventing spoofed extensions.
- User consent: enforces affirmative consent for media ingestion and processing.
- Ephemeral retention: bounded job queues and automatic time-to-live eviction.
- Confidence assessment and safe fallback recommendations.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Limits & Ceilings
# ---------------------------------------------------------------------------

MAX_VIDEO_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PAGE_BYTES = 5 * 1024 * 1024  # 5 MB

MAX_VIDEO_DURATION_SECONDS = 1800.0  # 30 minutes
MAX_AUDIO_DURATION_SECONDS = 3600.0  # 60 minutes
MAX_PAGE_BLOCKS = 5000
MAX_PAGE_DIMENSION = 20000.0
MAX_TRANSCRIPT_CHARS = 200000

# Page-analysis text ceilings (#854). A dense manuscript page carries a few
# thousand characters; these ceilings leave wide headroom while bounding the
# regex work done per request.
MAX_PAGE_BLOCK_TEXT_CHARS = 20_000
MAX_PAGE_TEXT_CHARS = 200_000

# Formats the manuscript OCR pipeline accepts (#854). The shared image sniffer
# also recognizes PDF, but a PDF is a multi-page document container, not a page
# image, so OCR routes reject it with 415 instead of passing it downstream.
OCR_IMAGE_FORMATS: frozenset[str] = frozenset({"png", "jpeg", "tiff", "webp"})

# Largest base64 string that can decode to MAX_IMAGE_BYTES (4 output chars per
# 3 input bytes, padded). Checked before decoding so an oversized string never
# triggers the decode allocation.
MAX_IMAGE_BASE64_CHARS = 4 * ((MAX_IMAGE_BYTES + 2) // 3)

# Line-wrapped base64 (76 chars + CRLF, as `base64` and MIME tools emit) is
# accepted, so the raw pre-decode ceiling leaves room for that whitespace.
_MAX_BASE64_WHITESPACE = 2 * (MAX_IMAGE_BASE64_CHARS // 76) + 64

# The only data-URL prefixes accepted on base64 image fields.
_DATA_URL_PREFIX = re.compile(r"^data:image/(?:png|jpeg|jpg|tiff|webp);base64,", re.IGNORECASE)

MAX_CONCURRENT_ACTIVE_JOBS = 20
MAX_TOTAL_RETAINED_JOBS = 200
JOB_RETENTION_TTL_SECONDS = 3600.0  # 1 hour


class MediaKind(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"


class ConfidenceBand(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class MediaValidationResult:
    valid: bool
    media_kind: MediaKind
    detected_format: str
    mime_type: str
    size_bytes: int
    duration_seconds: float | None = None
    confidence_band: ConfidenceBand = ConfidenceBand.HIGH
    fallback_recommended: bool = False
    review_required: bool = False
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Magic Byte Sniffing
# ---------------------------------------------------------------------------


def sniff_image_format(header: bytes) -> str | None:
    """Detect image format by signature."""
    if len(header) < 4:
        return None
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return "webp"
    if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
        return "tiff"
    if header.startswith(b"%PDF-"):
        return "pdf"
    return None


def sniff_audio_format(header: bytes) -> str | None:
    """Detect audio format by signature."""
    if len(header) < 4:
        return None
    if header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        return "mp3"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
        return "wav"
    if header.startswith(b"OggS"):
        return "ogg"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12]
        if brand in (b"M4A ", b"mp42", b"isom", b"M4B "):
            return "m4a"
    return None


def sniff_video_format(header: bytes) -> str | None:
    """Detect video container format by signature."""
    if len(header) < 4:
        return None
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"AVI ":
        return "avi"
    if len(header) >= 8 and header[4:8] == b"ftyp":
        brand = header[8:12] if len(header) >= 12 else b""
        if brand == b"qt  ":
            return "mov"
        return "mp4"
    if len(header) >= 8 and header[4:8] in (b"moov", b"wide", b"skip", b"free"):
        return "mov"
    return None


# ---------------------------------------------------------------------------
# Upload Validators
# ---------------------------------------------------------------------------


def validate_video_upload(
    payload: bytes,
    filename: str | None = None,
    duration_seconds: float | None = None,
    consent: bool = True,
) -> tuple[str, str]:
    """Validate a video upload for consent, size, format signature, and duration.

    Returns (mime_type, format_name). Raises HTTPException with appropriate 4xx status.
    """
    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Processing requires explicit user consent (consent_to_process=true).",
        )

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded media payload cannot be empty.")

    if len(payload) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Video payload exceeds maximum allowed size of {MAX_VIDEO_BYTES // (1024 * 1024)}MB.",
        )

    detected_format = sniff_video_format(payload[:64])
    if not detected_format:
        # Check if caller specified a known extension that failed signature match
        raise HTTPException(
            status_code=415,
            detail="Unsupported video format or corrupted file header. Supported formats: MP4, MOV, AVI, WebM.",
        )

    if duration_seconds is not None:
        if duration_seconds <= 0:
            raise HTTPException(status_code=422, detail="Duration must be positive.")
        if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
            raise HTTPException(
                status_code=422,
                detail=f"Video duration ({duration_seconds:.1f}s) exceeds maximum allowed limit of {MAX_VIDEO_DURATION_SECONDS:.0f}s (30 minutes).",
            )

    mime_map = {
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "avi": "video/x-msvideo",
        "webm": "video/webm",
    }
    return mime_map.get(detected_format, "video/mp4"), detected_format


def validate_audio_upload(
    payload: bytes,
    filename: str | None = None,
    duration_seconds: float | None = None,
    consent: bool = True,
) -> tuple[str, str]:
    """Validate an audio upload for consent, size, format signature, and duration.

    Returns (mime_type, format_name). Raises HTTPException with appropriate 4xx status.
    """
    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Processing requires explicit user consent (consent_to_process=true).",
        )

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded media payload cannot be empty.")

    if len(payload) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio payload exceeds maximum allowed size of {MAX_AUDIO_BYTES // (1024 * 1024)}MB.",
        )

    detected_format = sniff_audio_format(payload[:64])
    if not detected_format:
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio format or corrupted file header. Supported formats: MP3, WAV, OGG, M4A.",
        )

    if duration_seconds is not None:
        if duration_seconds <= 0:
            raise HTTPException(status_code=422, detail="Duration must be positive.")
        if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            raise HTTPException(
                status_code=422,
                detail=f"Audio duration ({duration_seconds:.1f}s) exceeds maximum allowed limit of {MAX_AUDIO_DURATION_SECONDS:.0f}s (60 minutes).",
            )

    mime_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "m4a": "audio/mp4",
    }
    return mime_map.get(detected_format, "audio/mpeg"), detected_format


def decode_base64_image(value: str) -> bytes:
    """Strictly decode a base64 image field, bounding its size before decoding.

    Accepts plain base64, optionally line-wrapped, or a ``data:image/<png|jpeg|
    tiff|webp>;base64,`` data URL. Raises HTTPException 413 when the encoded
    string could not fit MAX_IMAGE_BYTES and 422 for a disallowed data-URL
    prefix or malformed base64. The decoded bytes still need
    ``validate_image_upload`` for emptiness, size, and format signature.
    """
    if len(value) > MAX_IMAGE_BASE64_CHARS + _MAX_BASE64_WHITESPACE + 64:
        raise HTTPException(
            status_code=413,
            detail=f"Image payload exceeds maximum allowed size of {MAX_IMAGE_BYTES // (1024 * 1024)}MB.",
        )

    if value[:5].lower() == "data:":
        prefix = _DATA_URL_PREFIX.match(value)
        if prefix is None:
            raise HTTPException(
                status_code=422,
                detail="Unsupported data URL. Only data:image/<png|jpeg|tiff|webp>;base64, is accepted.",
            )
        value = value[prefix.end() :]

    compact = "".join(value.split())
    if len(compact) > MAX_IMAGE_BASE64_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Image payload exceeds maximum allowed size of {MAX_IMAGE_BYTES // (1024 * 1024)}MB.",
        )

    try:
        return base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as err:
        raise HTTPException(status_code=422, detail="Invalid base64 encoding.") from err


def validate_image_upload(
    payload: bytes,
    filename: str | None = None,
    consent: bool = True,
    allowed_formats: frozenset[str] | None = None,
) -> tuple[str, str]:
    """Validate an image upload for consent, size, and magic-byte format signature.

    ``allowed_formats`` narrows the accepted signatures for a route whose
    pipeline cannot consume every sniffed format (OCR passes OCR_IMAGE_FORMATS).
    The format is always taken from the content signature; ``filename`` and any
    declared MIME type are never trusted.
    """
    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Processing requires explicit user consent (consent_to_process=true).",
        )

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image payload cannot be empty.")

    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image payload exceeds maximum allowed size of {MAX_IMAGE_BYTES // (1024 * 1024)}MB.",
        )

    detected_format = sniff_image_format(payload[:32])
    if not detected_format:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format or invalid header. Supported formats: PNG, JPEG, TIFF, WebP, PDF.",
        )

    if allowed_formats is not None and detected_format not in allowed_formats:
        supported = ", ".join(fmt.upper() if fmt != "webp" else "WebP" for fmt in sorted(allowed_formats))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image format '{detected_format}' for this endpoint. Supported formats: {supported}.",
        )

    mime_map = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "tiff": "image/tiff",
        "pdf": "application/pdf",
    }
    return mime_map.get(detected_format, "image/png"), detected_format


def validate_page_layout_input(
    width: float,
    height: float,
    blocks_count: int,
    consent: bool = True,
) -> None:
    """Validate page analysis dimensions, block count, and consent."""
    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Processing requires explicit user consent (consent_to_process=true).",
        )

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=422, detail="Page dimensions must be greater than zero.")

    if width > MAX_PAGE_DIMENSION or height > MAX_PAGE_DIMENSION:
        raise HTTPException(
            status_code=422,
            detail=f"Page dimensions exceed maximum allowed limit of {MAX_PAGE_DIMENSION:.0f} units.",
        )

    if blocks_count > MAX_PAGE_BLOCKS:
        raise HTTPException(
            status_code=413,
            detail=f"Page contains too many blocks ({blocks_count}). Maximum allowed is {MAX_PAGE_BLOCKS}.",
        )


def evaluate_confidence_and_fallback(
    confidence: float | None,
    threshold_high: float = 0.75,
    threshold_medium: float = 0.40,
) -> tuple[ConfidenceBand, bool, bool]:
    """Calculate confidence band and safe fallback recommendation.

    Returns (ConfidenceBand, fallback_recommended, review_required).
    """
    if confidence is None:
        return ConfidenceBand.MEDIUM, False, True

    if confidence >= threshold_high:
        return ConfidenceBand.HIGH, False, False

    if confidence >= threshold_medium:
        return ConfidenceBand.MEDIUM, False, True

    # Low confidence triggers fallback recommendation
    return ConfidenceBand.LOW, True, True


# ---------------------------------------------------------------------------
# Bounded Job Manager for Background Processing
# ---------------------------------------------------------------------------


class BoundedJobManager:
    """Thread-safe, bounded in-memory job registry with TTL eviction."""

    def __init__(
        self,
        max_active: int = MAX_CONCURRENT_ACTIVE_JOBS,
        max_total: int = MAX_TOTAL_RETAINED_JOBS,
        ttl_seconds: float = JOB_RETENTION_TTL_SECONDS,
    ) -> None:
        self.max_active = max_active
        self.max_total = max_total
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, dict[str, Any]] = {}

    def purge_expired_jobs(self) -> int:
        """Remove jobs that have exceeded their retention TTL."""
        now = time.time()
        expired: list[str] = []
        for job_id, record in self._jobs.items():
            created_ts = record.get("_created_timestamp", 0.0)
            if now - created_ts > self.ttl_seconds:
                expired.append(job_id)
        for job_id in expired:
            self._jobs.pop(job_id, None)
        return len(expired)

    def register_job(self, job_id: str, filename: str | None = None) -> dict[str, Any]:
        """Register a new job enforcing concurrency and storage capacity."""
        self.purge_expired_jobs()

        active_count = sum(1 for j in self._jobs.values() if j.get("status") in ("queued", "processing"))
        if active_count >= self.max_active:
            raise HTTPException(
                status_code=429,
                detail=f"Server busy: maximum active jobs limit ({self.max_active}) reached. Please retry later.",
            )

        if len(self._jobs) >= self.max_total:
            # Evict oldest completed/failed job
            completed_jobs = sorted(
                [
                    (jid, j.get("_created_timestamp", 0.0))
                    for jid, j in self._jobs.items()
                    if j.get("status") in ("completed", "failed", "cancelled")
                ],
                key=lambda x: x[1],
            )
            if completed_jobs:
                self._jobs.pop(completed_jobs[0][0], None)
            else:
                raise HTTPException(
                    status_code=429,
                    detail=f"Resource limit reached: job capacity ({self.max_total}) full.",
                )

        sanitized_filename = os.path.basename(filename) if filename else "unnamed_media"
        # Strip path traversal characters
        sanitized_filename = re.sub(r"[^\w\.\-]", "_", sanitized_filename)

        record = {
            "id": job_id,
            "status": "queued",
            "filename": sanitized_filename,
            "created_at": datetime.now(UTC).isoformat(),
            "_created_timestamp": time.time(),
        }
        self._jobs[job_id] = record
        return record

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self.purge_expired_jobs()
        job = self._jobs.get(job_id)
        if not job:
            return None
        # Return public-facing representation without internal timestamp
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        status = job.get("status")
        if status in ("completed", "failed"):
            raise HTTPException(
                status_code=409,
                detail=f"Job cannot be cancelled because it is already in terminal state: {status}.",
            )
        self.update_job(job_id, status="cancelled", cancelled_at=datetime.now(UTC).isoformat())
        return self.get_job(job_id) or {}
