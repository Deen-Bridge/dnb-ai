"""Measure decoder throughput on synthetic frame posteriors, not model accuracy."""

from __future__ import annotations

import time

from phoneme_alignment import (
    AcousticFrames,
    AlignmentReference,
    CTCAligner,
    FrameBatch,
    Phone,
    ReferenceVerse,
    ReferenceWord,
)


def main() -> int:
    reference = AlignmentReference(
        qiraah="hafs",
        verses=[
            ReferenceVerse(
                surah=1,
                ayah=1,
                words=[
                    ReferenceWord(
                        text="synthetic", phonemes=[Phone(symbol="b" if i % 2 == 0 else "a") for i in range(32)]
                    )
                    for _ in range(8)
                ],
            )
        ],
    )
    frames = AcousticFrames(
        labels=["_", "b", "a"],
        blank_id=0,
        frame_shift_ms=20,
        model_id="synthetic throughput fixture",
        probabilities=[
            [0.02, 0.96 if (i * 256 // 3000) % 2 == 0 else 0.02, 0.02 if (i * 256 // 3000) % 2 == 0 else 0.96]
            for i in range(3000)
        ],
    )
    decoder = CTCAligner(reference, frames)
    started = time.perf_counter()
    for index in range(0, 3000, 100):
        decoder.append(FrameBatch(probabilities=frames.probabilities[index : index + 100]))
        decoder.result(final=index == 2900)
    elapsed = time.perf_counter() - started
    audio_seconds = 60
    print(f"Decoded {decoder.frames_processed} frames / 256 phonemes with 30 streaming updates in {elapsed:.3f}s")
    print(
        f"Decoder real time factor: {elapsed / audio_seconds:.4f} (synthetic posteriors; excludes acoustic inference)"
    )
    return int(elapsed >= audio_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
