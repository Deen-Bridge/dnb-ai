"""Progressive screen-recording analysis for Islamic lectures (#146).

The pipeline is deliberately dependency-tolerant: ffmpeg is used when present
for audio/frame extraction, while ASR and OCR adapters are optional. A request
therefore produces useful metadata and a searchable result even on a worker
that has not installed heavyweight model packages.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/video-analysis", tags=["video-analysis"])

MAX_VIDEO_BYTES = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}
TERMINAL_STATES = {"completed", "failed"}
_jobs: dict[str, dict[str, Any]] = {}


def extract_quran_references(text: str) -> list[str]:
    """Return Quran references in ``surah:ayah`` or ``surah ayah`` form."""
    return sorted(set(re.findall(r"\b(?:surah\s+)?(\d{1,3}:\d{1,3})\b", text, re.I)))


def segment_topics(transcript: str) -> list[dict[str, Any]]:
    """Create timestamp-free topic segments from transcript paragraphs."""
    segments = []
    for index, paragraph in enumerate(filter(None, re.split(r"\n{2,}|(?<=[.!?])\s+", transcript.strip()))):
        words = re.findall(r"[\w'-]+", paragraph)
        if not words:
            continue
        segments.append({"index": index, "title": " ".join(words[:8]), "text": paragraph, "start_seconds": index * 30})
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
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)


def extract_media(video_path: Path, workdir: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    """Extract mono audio and one frame every 30 seconds when ffmpeg exists."""
    if not shutil.which("ffmpeg"):
        return None, []
    audio: Path | None = workdir / "audio.wav"
    audio_result = _run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(audio)])
    if audio_result.returncode != 0:
        audio = None
    frames_dir = workdir / "frames"
    frames_dir.mkdir()
    _run(["ffmpeg", "-y", "-i", str(video_path), "-vf", "fps=1/30", str(frames_dir / "frame-%04d.jpg")])
    frames = []
    for number, frame in enumerate(sorted(frames_dir.glob("*.jpg"))):
        frames.append({"path": str(frame), "timestamp_seconds": number * 30, "text": ""})
    return audio, frames


def transcribe_audio(audio: Path | None) -> str:
    """Use an installed Whisper adapter if available; otherwise return empty text."""
    if audio is None:
        return ""
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel("base", compute_type="int8")
        segments, _ = model.transcribe(str(audio))
        return " ".join(segment.text.strip() for segment in segments).strip()
    except (ImportError, OSError, RuntimeError):
        return ""


def ocr_frame(frame: Path) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        return pytesseract.image_to_string(Image.open(frame), lang="ara+eng").strip()
    except (ImportError, OSError, RuntimeError):
        return ""


def analyze_video(video_path: Path, transcript: str = "") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dnb-video-") as directory:
        audio, frames = extract_media(video_path, Path(directory))
        transcript = transcript.strip() or transcribe_audio(audio)
        for frame in frames:
            frame["text"] = ocr_frame(Path(frame["path"]))
        visual_text = " ".join(frame["text"] for frame in frames if frame["text"])
        combined = " ".join(part for part in (transcript, visual_text) if part)
        return {
            "transcript": transcript,
            "summary": " ".join(segment["text"] for segment in segment_topics(combined)[:5]),
            "topics": segment_topics(combined),
            "quran_references": extract_quran_references(combined),
            "frames": [{k: v for k, v in frame.items() if k != "path"} for frame in frames],
            "search_index": build_search_index(combined, frames),
            "capabilities": {
                "ffmpeg": audio is not None or bool(frames),
                "asr": bool(transcript),
                "ocr": bool(visual_text),
            },
        }


async def _process(job_id: str, payload: bytes, suffix: str, transcript: str) -> None:
    _jobs[job_id]["status"] = "processing"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
            handle.write(payload)
            handle.flush()
            result = await asyncio.to_thread(analyze_video, Path(handle.name), transcript)
        _jobs[job_id].update(status="completed", result=result, completed_at=datetime.now(UTC).isoformat())
    except Exception as error:  # keep job failures observable to clients
        _jobs[job_id].update(status="failed", error=str(error))


@router.post("/analyze", status_code=202)
async def submit_video_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="MP4, MOV, or WebM video up to 100MB"),
    transcript: str = Form(""),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only MP4, MOV, and WebM videos are supported.")
    payload = await file.read(MAX_VIDEO_BYTES + 1)
    if len(payload) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video must be 100MB or smaller.")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "filename": file.filename,
        "created_at": datetime.now(UTC).isoformat(),
    }
    background_tasks.add_task(_process, job_id, payload, suffix, transcript)
    return {"job_id": job_id, "status": "queued", "poll_url": f"/video-analysis/jobs/{job_id}"}


@router.get("/jobs/{job_id}")
async def get_video_analysis(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return json.loads(json.dumps(job, default=str))
