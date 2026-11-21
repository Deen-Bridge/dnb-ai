"""Progressive screen-recording analysis for Islamic lectures (#146).

The pipeline is deliberately dependency-tolerant: ffmpeg is used when present
for audio/frame extraction, while ASR and OCR adapters are optional. A request
therefore produces useful metadata and a searchable result even on a worker
that has not installed heavyweight model packages.

Production contract (#858, PROD-AUDIT AI-082)
---------------------------------------------
* Everything this pipeline extracts (ASR transcript, frame OCR, and the
  optional user-supplied ``transcript``) is untrusted data. Results carry a
  ``content_trust`` block (see ``untrusted_content.py``); no extracted text is
  sent to a model, used to pick a tool, or passed to a shell. The "summary" is
  extractive (the first transcript segments verbatim) and references are
  unverified pattern matches, and both are labelled as such.
* Work is bounded: uploads are streamed to a temporary file and capped at
  ``VIDEO_MAX_BYTES``; ffmpeg processes at most ``VIDEO_MAX_DURATION_SECONDS``
  of media and ``VIDEO_MAX_FRAMES`` frames, each call under
  ``VIDEO_FFMPEG_TIMEOUT_SECONDS``; at most ``VIDEO_MAX_CONCURRENT_JOBS`` run at
  once and at most ``VIDEO_MAX_PENDING_JOBS`` may be queued or running.
* Retention: the uploaded file and all extracted audio/frames are deleted when
  processing ends, on success or failure. Job results (which contain the
  transcript) stay in process memory for ``VIDEO_JOB_RETENTION_SECONDS`` after
  completion, at most ``VIDEO_MAX_STORED_JOBS`` jobs, and are lost on restart.
* Failures surface as a stable ``error_code`` with a safe message. Exception
  text, ffmpeg output, paths and media content are never returned or logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from untrusted_content import FLAG_TRUNCATED, ContentSource, describe_untrusted, truncate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video-analysis", tags=["video-analysis"])


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s; using %d", name, default)
        return default
    return max(minimum, value)


MAX_VIDEO_BYTES = _env_int("VIDEO_MAX_BYTES", 100 * 1024 * 1024)
MAX_TRANSCRIPT_CHARS = _env_int("VIDEO_MAX_TRANSCRIPT_CHARS", 200_000)
MAX_EXTRACTED_CHARS = _env_int("VIDEO_MAX_EXTRACTED_CHARS", 200_000)
MAX_DURATION_SECONDS = _env_int("VIDEO_MAX_DURATION_SECONDS", 7_200)
FRAME_INTERVAL_SECONDS = 30
MAX_FRAMES = _env_int("VIDEO_MAX_FRAMES", MAX_DURATION_SECONDS // FRAME_INTERVAL_SECONDS)
FFMPEG_TIMEOUT_SECONDS = _env_int("VIDEO_FFMPEG_TIMEOUT_SECONDS", 120)
MAX_CONCURRENT_JOBS = _env_int("VIDEO_MAX_CONCURRENT_JOBS", 2)
MAX_PENDING_JOBS = _env_int("VIDEO_MAX_PENDING_JOBS", 8)
JOB_RETENTION_SECONDS = _env_int("VIDEO_JOB_RETENTION_SECONDS", 3_600)
MAX_STORED_JOBS = _env_int("VIDEO_MAX_STORED_JOBS", 200)
MAX_TOPICS = 500
MAX_FILENAME_CHARS = 255
UPLOAD_CHUNK_BYTES = 1024 * 1024

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
TERMINAL_STATES = {"completed", "failed"}
ACTIVE_STATES = {"queued", "processing"}
_jobs: dict[str, dict[str, Any]] = {}
_processing_slots = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


class VideoErrorCode(str, Enum):
    """Stable, client-visible error codes."""

    UNSUPPORTED_FORMAT = "unsupported_format"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    EMPTY_UPLOAD = "empty_upload"
    TRANSCRIPT_TOO_LARGE = "transcript_too_large"
    SERVICE_BUSY = "service_busy"
    PROCESSING_TIMEOUT = "processing_timeout"
    DECODE_FAILED = "decode_failed"
    INTERNAL_ERROR = "internal_error"


ERROR_MESSAGES: dict[VideoErrorCode, str] = {
    VideoErrorCode.UNSUPPORTED_FORMAT: "Only MP4, MOV, AVI, and WebM videos are supported.",
    VideoErrorCode.PAYLOAD_TOO_LARGE: "Video must be 100MB or smaller.",
    VideoErrorCode.EMPTY_UPLOAD: "The uploaded video is empty.",
    VideoErrorCode.TRANSCRIPT_TOO_LARGE: "The transcript is longer than the allowed limit.",
    VideoErrorCode.SERVICE_BUSY: "Too many video analyses are in progress. Retry later.",
    VideoErrorCode.PROCESSING_TIMEOUT: "Media processing took too long and was stopped.",
    VideoErrorCode.DECODE_FAILED: "The video could not be decoded and no transcript was supplied.",
    VideoErrorCode.INTERNAL_ERROR: "Video analysis failed unexpectedly.",
}


class MediaTimeoutError(RuntimeError):
    """An ffmpeg call exceeded its timeout."""


class MediaDecodeError(RuntimeError):
    """ffmpeg is installed but produced neither audio nor frames."""


def _client_error(status_code: int, code: VideoErrorCode) -> HTTPException:
    detail = ERROR_MESSAGES[code]
    if code == VideoErrorCode.PAYLOAD_TOO_LARGE:
        detail = f"Video must be {MAX_VIDEO_BYTES // (1024 * 1024)}MB or smaller."
    elif code == VideoErrorCode.TRANSCRIPT_TOO_LARGE:
        detail = f"The transcript must be at most {MAX_TRANSCRIPT_CHARS} characters."
    return HTTPException(status_code=status_code, detail={"code": code.value, "detail": detail})


def extract_quran_references(text: str) -> list[str]:
    """Return Quran references in ``surah:ayah`` or ``surah ayah`` form."""
    return sorted(set(re.findall(r"\b(?:surah\s+)?(\d{1,3}[:\s]\d{1,3})\b", text, re.I)))


def extract_hadith_references(text: str) -> list[str]:
    """Return hadith citations such as ``Sahih Bukhari 123`` or ``Muslim 45``."""
    pattern = r"\b(?:sahih\s+)?(?:bukhari|muslim|abu\s+dawud|tirmidhi|nasai|ibn\s+majah)\s+\d+\b"
    return sorted(set(match.strip() for match in re.findall(pattern, text, re.I)))


def detect_language(text: str) -> str:
    """Best-effort language detection for Arabic, Urdu, and English."""
    if re.search(r"[؀-ۿ]", text):
        if re.search(r"[ٹڈڑںہے]", text):
            return "ur"
        return "ar"
    return "en"


def segment_topics(transcript: str, timestamps: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Create topic segments from transcript paragraphs with optional timestamps."""
    segments = []
    paragraphs = list(filter(None, re.split(r"\n{2,}|(?<=[.!?])\s+", transcript.strip())))
    for index, paragraph in enumerate(paragraphs):
        words = re.findall(r"[\w'-]+", paragraph)
        if not words:
            continue
        start = index * 30
        if timestamps and index < len(timestamps):
            start = int(timestamps[index].get("start", start))
        segments.append({"index": index, "title": " ".join(words[:8]), "text": paragraph, "start_seconds": start})
    return segments


def build_search_index(transcript: str, frames: list[dict[str, Any]]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for position, token in enumerate(re.findall(r"[\w'-]+", transcript.lower())):
        index.setdefault(token, []).append(position)
    for frame in frames:
        for token in set(re.findall(r"[\w'-]+", frame.get("text", "").lower())):
            index.setdefault(token, []).append(int(frame.get("timestamp_seconds", 0)))
    return index


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with a fixed argv (never a shell) under a hard timeout."""
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise MediaTimeoutError("ffmpeg timed out") from None


def extract_media(video_path: Path, workdir: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    """Extract mono audio and one frame every 30 seconds when ffmpeg exists.

    At most ``MAX_DURATION_SECONDS`` of media and ``MAX_FRAMES`` frames are
    processed. Raises :class:`MediaDecodeError` when ffmpeg is installed but
    yields neither audio nor frames (corrupt or non-video input).
    """
    if not shutil.which("ffmpeg"):
        return None, []
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path)]
    limit = ["-t", str(MAX_DURATION_SECONDS)]
    audio: Path | None = workdir / "audio.wav"
    audio_result = _run([*base, *limit, "-vn", "-ac", "1", "-ar", "16000", str(audio)])
    if audio_result.returncode != 0:
        audio = None
    frames_dir = workdir / "frames"
    frames_dir.mkdir()
    _run(
        [
            *base,
            *limit,
            "-vf",
            f"fps=1/{FRAME_INTERVAL_SECONDS}",
            "-frames:v",
            str(MAX_FRAMES),
            str(frames_dir / "frame-%04d.jpg"),
        ]
    )
    frames = []
    for number, frame in enumerate(sorted(frames_dir.glob("*.jpg"))[:MAX_FRAMES]):
        frames.append({"path": str(frame), "timestamp_seconds": number * FRAME_INTERVAL_SECONDS, "text": ""})
    if audio is None and not frames:
        raise MediaDecodeError("ffmpeg produced no audio or frames")
    return audio, frames


def transcribe_audio(audio: Path | None) -> tuple[str, list[dict[str, Any]]]:
    """Use an installed Whisper adapter if available; otherwise return empty text."""
    if audio is None:
        return "", []
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel("base", compute_type="int8")
        segments, _ = model.transcribe(str(audio), vad_filter=True)
        collected = []
        for segment in segments:
            collected.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "speaker": getattr(segment, "speaker", None) or "speaker_1",
                    "text": segment.text.strip(),
                }
            )
        return " ".join(item["text"] for item in collected).strip(), collected
    except (ImportError, OSError, RuntimeError):
        return "", []


def ocr_frame(frame: Path) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        return pytesseract.image_to_string(Image.open(frame), lang="ara+eng").strip()
    except (ImportError, OSError, RuntimeError):
        return ""


def analyze_video(video_path: Path, transcript: str = "") -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dnb-video-") as directory:
        user_supplied = bool(transcript.strip())
        try:
            audio, frames = extract_media(video_path, Path(directory))
            media_decoded = True
        except MediaDecodeError:
            if not user_supplied:
                raise
            # A supplied transcript still makes the job useful; report the media as undecoded.
            audio, frames, media_decoded = None, [], False
        timestamps: list[dict[str, Any]]
        if user_supplied:
            transcript, timestamps = transcript.strip(), []
        else:
            transcript, timestamps = transcribe_audio(audio)
        for frame in frames:
            frame["text"] = ocr_frame(Path(frame["path"]))

    sources: list[ContentSource] = []
    if transcript:
        sources.append("user_transcript" if user_supplied else "asr")
    visual_text = " ".join(frame["text"] for frame in frames if frame["text"])
    if visual_text:
        sources.append("ocr")

    transcript, transcript_cut = truncate(transcript, MAX_EXTRACTED_CHARS)
    visual_text, visual_cut = truncate(visual_text, max(0, MAX_EXTRACTED_CHARS - len(transcript)))
    combined = " ".join(part for part in (transcript, visual_text) if part)
    topics = segment_topics(combined, timestamps)[:MAX_TOPICS]
    key_points = [
        {"timestamp_seconds": item["start"], "text": item["text"]} for item in timestamps if item.get("text")
    ][:20]
    public_frames = [{k: v for k, v in frame.items() if k != "path"} for frame in frames]
    return {
        "transcript": transcript,
        "language": detect_language(combined),
        "speakers": sorted({item["speaker"] for item in timestamps}),
        "timestamps": timestamps,
        "key_points": key_points,
        "summary": " ".join(segment["text"] for segment in topics[:5]),
        "summary_method": "extractive",
        "key_takeaways": [segment["title"] for segment in topics[:5]],
        "topics": topics,
        "quran_references": extract_quran_references(combined),
        "hadith_references": extract_hadith_references(combined),
        "reference_status": "unverified_pattern_matches",
        "frames": public_frames,
        "search_index": build_search_index(combined, frames),
        "extraction_status": "content" if combined.strip() else "empty",
        "content_trust": describe_untrusted(
            [combined, *(frame["text"] for frame in public_frames)],
            sources,
            [FLAG_TRUNCATED] if (transcript_cut or visual_cut) else [],
        ).model_dump(),
        "capabilities": {
            "ffmpeg": audio is not None or bool(frames),
            "asr": bool(transcript),
            "ocr": bool(visual_text),
            "media_decoded": media_decoded,
        },
        "metrics": {
            "frames_processed": len(frames),
            "transcript_chars": len(transcript),
            "ocr_chars": len(visual_text),
            "processing_ms": int((time.perf_counter() - started) * 1000),
            "provider_cost_usd": 0.0,
        },
    }


def _purge_jobs(now: datetime | None = None) -> None:
    """Drop expired results, then the oldest finished jobs beyond the stored-job cap."""
    now = now or datetime.now(UTC)
    for job_id in [
        job_id
        for job_id, job in _jobs.items()
        if job["status"] in TERMINAL_STATES and datetime.fromisoformat(job["expires_at"]) <= now
    ]:
        del _jobs[job_id]
    finished = sorted(
        (job for job in _jobs.values() if job["status"] in TERMINAL_STATES), key=lambda job: job["completed_at"]
    )
    while len(_jobs) > MAX_STORED_JOBS and finished:
        del _jobs[finished.pop(0)["id"]]


def _finish(job_id: str, status: str, **fields: Any) -> None:
    completed = datetime.now(UTC)
    _jobs[job_id].update(
        status=status,
        completed_at=completed.isoformat(),
        expires_at=(completed + timedelta(seconds=JOB_RETENTION_SECONDS)).isoformat(),
        **fields,
    )


async def _process(job_id: str, upload_path: Path, transcript: str) -> None:
    """Run one job. The uploaded file is always deleted, whatever happens."""
    error_code: VideoErrorCode | None = None
    try:
        async with _processing_slots:
            _jobs[job_id]["status"] = "processing"
            result = await asyncio.to_thread(analyze_video, upload_path, transcript)
        _finish(job_id, "completed", result=result)
    except MediaTimeoutError:
        error_code = VideoErrorCode.PROCESSING_TIMEOUT
    except MediaDecodeError:
        error_code = VideoErrorCode.DECODE_FAILED
    except Exception as error:  # keep job failures observable without leaking internals
        error_code = VideoErrorCode.INTERNAL_ERROR
        # Type and stack frames only: an exception message may quote media content.
        logger.error(
            "Video analysis job %s failed unexpectedly: %s\n%s",
            job_id,
            type(error).__name__,
            "".join(traceback.format_tb(error.__traceback__)),
        )
    finally:
        upload_path.unlink(missing_ok=True)
    if error_code is not None:
        _finish(job_id, "failed", error_code=error_code.value, error=ERROR_MESSAGES[error_code])
    job = _jobs[job_id]
    logger.info(
        "Video analysis job %s %s error_code=%s bytes=%s metrics=%s",
        job_id,
        job["status"],
        job.get("error_code"),
        job.get("input_bytes"),
        json.dumps((job.get("result") or {}).get("metrics", {})),
    )


def _safe_filename(name: str | None) -> str:
    """Untrusted client filename: drop control characters and bound its length."""
    cleaned = re.sub(r"[\x00-\x1f\x7f‪-‮⁦-⁩]", "", name or "")
    return cleaned[:MAX_FILENAME_CHARS]


async def _spool_upload(file: UploadFile, suffix: str) -> tuple[Path, int]:
    """Stream the upload to a private temporary file, enforcing the byte cap."""
    handle = tempfile.NamedTemporaryFile(prefix="dnb-upload-", suffix=suffix, delete=False)
    path = Path(handle.name)
    size = 0
    try:
        with handle:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise _client_error(413, VideoErrorCode.PAYLOAD_TOO_LARGE)
                handle.write(chunk)
        if size == 0:
            raise _client_error(422, VideoErrorCode.EMPTY_UPLOAD)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path, size


@router.post("/analyze", status_code=202)
async def submit_video_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="MP4, MOV, AVI, or WebM video up to 100MB"),
    transcript: str = Form("", description="Optional transcript; treated as untrusted data"),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise _client_error(415, VideoErrorCode.UNSUPPORTED_FORMAT)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        raise _client_error(413, VideoErrorCode.TRANSCRIPT_TOO_LARGE)
    _purge_jobs()
    if sum(1 for job in _jobs.values() if job["status"] in ACTIVE_STATES) >= MAX_PENDING_JOBS:
        raise _client_error(503, VideoErrorCode.SERVICE_BUSY)
    upload_path, size = await _spool_upload(file, suffix)
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "filename": _safe_filename(file.filename),
        "input_bytes": size,
        "created_at": datetime.now(UTC).isoformat(),
    }
    background_tasks.add_task(_process, job_id, upload_path, transcript)
    return {"job_id": job_id, "status": "queued", "poll_url": f"/video-analysis/jobs/{job_id}"}


@router.get("/jobs/{job_id}")
async def get_video_analysis(job_id: str) -> dict[str, Any]:
    _purge_jobs()
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return json.loads(json.dumps(job, default=str))
