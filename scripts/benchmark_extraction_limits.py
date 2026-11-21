"""Cost and latency evidence for the extraction limits (#858, PROD-AUDIT AI-082).

Runs each extraction surface at a representative size and at its worst allowed
size, offline, with ffmpeg/ASR/OCR replaced by no-op fakes so only this
service's own processing is measured. Prints a Markdown table.

    python scripts/benchmark_extraction_limits.py

Timings depend on the machine; record the environment with the numbers. Every
path here is deterministic and makes no paid provider call, so the provider
cost of each request is $0. Media decoding, ASR and OCR run on the worker's CPU
and are bounded by MAX_VIDEO_DURATION_SECONDS (#853) and VIDEO_MAX_FRAMES, not
measured here.
"""

from __future__ import annotations

import platform
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_analysis as va  # noqa: E402
from audio_hadith import MAX_LECTURE_LENGTH, extract_lecture_transcript  # noqa: E402
from page_analysis import MAX_BLOCKS, MAX_PAGE_TEXT_CHARS, PageBlock, PageInput, analyze_page  # noqa: E402
from untrusted_content import inspect_text  # noqa: E402

SENTENCE = "Sheikh: Today we study patience, gratitude and Surah 2:255 with Bukhari 1. "
RUNS = 5


def _page(blocks: int, chars: int) -> PageInput:
    per_block = max(1, chars // max(1, blocks))
    return PageInput(
        width=700,
        height=blocks + 10,
        blocks=[
            PageBlock(
                x=(i % 3) * 200,
                y=i,
                width=150,
                height=1,
                text=(SENTENCE * (per_block // len(SENTENCE) + 1))[:per_block],
            )
            for i in range(blocks)
        ],
    )


def _text(chars: int) -> str:
    return (SENTENCE * (chars // len(SENTENCE) + 1))[:chars]


def _lines(chars: int) -> str:
    return "\n".join(_text(chars).split(". "))[:chars]


def _video(transcript: str) -> dict[str, Any]:
    va.extract_media = lambda video_path, workdir: (None, [])  # type: ignore[assignment]
    with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
        return va.analyze_video(Path(handle.name), transcript)


def _measure(fn: Callable[[], Any]) -> tuple[float, Any]:
    best = float("inf")
    result: Any = None
    for _ in range(RUNS):
        started = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - started)
    return best * 1000, result


def main() -> None:
    cases: list[tuple[str, str, Callable[[], Any], Callable[[Any], str]]] = [
        (
            "page-analysis",
            "60 blocks, 6k chars",
            lambda: analyze_page(_page(60, 6_000)),
            lambda r: f"{len(r.elements)} elements",
        ),
        (
            "page-analysis",
            f"worst: {MAX_BLOCKS} blocks, {MAX_PAGE_TEXT_CHARS // 1000}k chars",
            lambda: analyze_page(_page(MAX_BLOCKS, MAX_PAGE_TEXT_CHARS)),
            lambda r: f"{len(r.elements)} elements",
        ),
        (
            "lecture-transcript",
            "20k chars",
            lambda: extract_lecture_transcript(_lines(20_000)),
            lambda r: f"{len(r.segments)} segments",
        ),
        (
            "lecture-transcript",
            f"worst: {MAX_LECTURE_LENGTH // 1000}k chars",
            lambda: extract_lecture_transcript(_lines(MAX_LECTURE_LENGTH)),
            lambda r: f"{len(r.segments)} segments",
        ),
        (
            "video-analysis (no media)",
            "20k-char transcript",
            lambda: _video(_text(20_000)),
            lambda r: f"{len(r['topics'])} topics, {len(r['search_index'])} index terms",
        ),
        (
            "video-analysis (no media)",
            f"worst: {va.MAX_TRANSCRIPT_CHARS // 1000}k-char transcript",
            lambda: _video(_text(va.MAX_TRANSCRIPT_CHARS)),
            lambda r: f"{len(r['topics'])} topics, {len(r['search_index'])} index terms",
        ),
        (
            "trust inspection",
            f"worst: {va.MAX_EXTRACTED_CHARS // 1000}k chars of near-miss patterns",
            lambda: inspect_text(("ignore all previous reveal the " * 10_000)[: va.MAX_EXTRACTED_CHARS]),
            lambda r: f"{len(r)} flags",
        ),
    ]
    print(f"Environment: Python {platform.python_version()} on {platform.platform()}; best of {RUNS} runs")
    print()
    print("| Surface | Input | Best time (ms) | Output | Provider cost |")
    print("| --- | --- | --- | --- | --- |")
    for surface, label, fn, describe in cases:
        elapsed, result = _measure(fn)
        print(f"| {surface} | {label} | {elapsed:.1f} | {describe(result)} | $0 |")


if __name__ == "__main__":
    main()
