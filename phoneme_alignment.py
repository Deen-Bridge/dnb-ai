"""CTC forced alignment of Quranic reference phonemes to acoustic posteriors.

References explicitly preserve the reading's phonemes and Tajweed distinctions;
we do not pretend Arabic orthography uniquely determines a Qira'ah. Model labels
and phoneme references must use the same alphabet. The decoder never stretches
characters uniformly across an audio file or invents acoustic confidence.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

MAX_PHONEMES = 512
MAX_FRAMES = 6000
MAX_CELLS = 3_000_000
MAX_AUDIO_BYTES = 2_000_000
router = APIRouter(prefix="/quran/alignment", tags=["Quran phoneme alignment"])


class Phone(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    symbol: str = Field(min_length=1, max_length=32)
    tajweed: list[Literal["madd", "ghunnah", "shadda", "ikhfa", "idgham", "iqlab", "qalqalah", "tafkhim", "tarqiq"]] = (
        Field(default_factory=list, max_length=9)
    )


class ReferenceWord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=200)
    phonemes: list[Phone] = Field(min_length=1, max_length=100)


class ReferenceVerse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    surah: int = Field(ge=1, le=114)
    ayah: int = Field(ge=1, le=286)
    words: list[ReferenceWord] = Field(min_length=1, max_length=200)


class AlignmentReference(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    qiraah: Literal["hafs", "warsh", "qalun"]
    reciter: str | None = Field(default=None, max_length=200)
    verses: list[ReferenceVerse] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def bounded_reference(self) -> AlignmentReference:
        count = sum(len(word.phonemes) for verse in self.verses for word in verse.words)
        if count > MAX_PHONEMES:
            raise ValueError(f"At most {MAX_PHONEMES} reference phonemes per alignment")
        keys = [(verse.surah, verse.ayah) for verse in self.verses]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate verse references")
        return self


class AcousticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    labels: list[str] = Field(min_length=2, max_length=512)
    blank_id: int = Field(ge=0, strict=True)
    frame_shift_ms: float = Field(gt=0, le=100)
    offset_ms: float = Field(default=0, ge=0, le=86_400_000)
    model_id: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def valid_labels(self) -> AcousticConfig:
        if self.blank_id >= len(self.labels):
            raise ValueError("blank_id must index labels")
        if any(not label or len(label) > 100 for label in self.labels) or len(set(self.labels)) != len(self.labels):
            raise ValueError("Acoustic labels must be nonempty, bounded and unique")
        return self


class FrameBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    probabilities: list[Annotated[list[float], Field(min_length=2, max_length=512)]] = Field(
        min_length=1, max_length=MAX_FRAMES
    )

    @model_validator(mode="after")
    def valid_probabilities(self) -> FrameBatch:
        width = len(self.probabilities[0])
        if not 2 <= width <= 512 or any(len(row) != width for row in self.probabilities):
            raise ValueError("Frame probabilities must be a rectangular matrix with 2 to 512 labels")
        values = np.asarray(self.probabilities, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Frame probabilities must be finite and between zero and one")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5, rtol=0):
            raise ValueError("Each frame must be a normalized probability distribution")
        return self


class AcousticFrames(AcousticConfig, FrameBatch):
    pass


class AlignmentRequest(BaseModel):
    reference: AlignmentReference
    acoustics: AcousticFrames


class AlignedPhone(BaseModel):
    symbol: str
    verse_index: int
    word_index: int
    phoneme_index: int
    tajweed: list[str]
    start_ms: float
    end_ms: float
    confidence: float


class Boundary(BaseModel):
    verse_index: int
    word_index: int | None = None
    text: str
    surah: int
    ayah: int
    start_ms: float
    end_ms: float
    confidence: float


class AlignmentResult(BaseModel):
    qiraah: str
    reciter: str | None
    model_id: str
    final: bool
    frames_processed: int
    path_log_probability: float
    phonemes: list[AlignedPhone]
    words: list[Boundary]
    verses: list[Boundary]
    confidence: float
    warnings: list[str] = Field(default_factory=list)


class AlignmentError(ValueError):
    """The reference cannot be aligned to these acoustic observations."""


class CTCAligner:
    """Incremental Viterbi decoding with CTC blank/repeated-label constraints.

    Only new frames are decoded on append. Rolling scores are O(reference),
    traceback uses one byte per frame/state, and all sessions have fixed frame
    and trellis budgets. Partial paths are provisional and may change.
    """

    def __init__(self, reference: AlignmentReference, config: AcousticConfig) -> None:
        self.reference = reference.model_copy(deep=True)
        self.config = AcousticConfig.model_validate(config.model_dump(include=set(AcousticConfig.model_fields)))
        self.targets = [
            (vi, wi, pi, phone)
            for vi, verse in enumerate(self.reference.verses)
            for wi, word in enumerate(verse.words)
            for pi, phone in enumerate(word.phonemes)
        ]
        vocabulary = {label: i for i, label in enumerate(config.labels)}
        missing = sorted({phone.symbol for _, _, _, phone in self.targets} - vocabulary.keys())
        if missing:
            raise AlignmentError(f"Reference phonemes absent from model vocabulary: {', '.join(missing)}")
        tokens = [vocabulary[phone.symbol] for _, _, _, phone in self.targets]
        if config.blank_id in tokens:
            raise AlignmentError("CTC blank cannot be a reference phoneme")
        self._labels = np.full(2 * len(tokens) + 1, config.blank_id, dtype=np.int64)
        self._labels[1::2] = tokens
        self._skip = np.zeros(len(self._labels), dtype=bool)
        self._skip[2:] = (self._labels[2:] != config.blank_id) & (self._labels[2:] != self._labels[:-2])
        self._scores = np.full(len(self._labels), -np.inf)
        self._scores[0] = 0
        self._traces: list[np.ndarray] = []
        self._emissions: list[np.ndarray] = []
        self._closed = False

    @property
    def frames_processed(self) -> int:
        return len(self._traces)

    def append(self, batch: FrameBatch) -> None:
        if self._closed:
            raise AlignmentError("Alignment already finalized")
        values = np.asarray(batch.probabilities, dtype=np.float64)
        if values.shape[1] != len(self.config.labels):
            raise AlignmentError("Frame width does not match the configured vocabulary")
        total = self.frames_processed + len(values)
        if total > MAX_FRAMES or total * len(self._scores) > MAX_CELLS:
            raise AlignmentError("Alignment exceeds the frame or trellis budget; split at verse boundaries")
        with np.errstate(divide="ignore"):
            logs = np.log(values)
        for row in logs:
            advance = np.concatenate(([-np.inf], self._scores[:-1]))
            skip = np.concatenate(([-np.inf, -np.inf], self._scores[:-2]))
            skip[~self._skip] = -np.inf
            candidates = np.stack((self._scores, advance, skip))
            steps = candidates.argmax(axis=0).astype(np.uint8)
            self._scores = candidates.max(axis=0) + row[self._labels]
            self._traces.append(steps)
            self._emissions.append(row[self._labels[1::2]])

    def result(self, *, final: bool) -> AlignmentResult:
        if not self.frames_processed:
            raise AlignmentError("No acoustic frames have been received")
        if final:
            end = len(self._scores) - 1
            if self._scores[end - 1] > self._scores[end]:
                end -= 1
        else:
            end = int(self._scores.argmax())
        if not math.isfinite(float(self._scores[end])):
            raise AlignmentError("No valid CTC path; audio may be incomplete or inconsistent with the reference")
        spans: dict[int, list[int]] = {}
        state = end
        for frame in range(self.frames_processed - 1, -1, -1):
            if state % 2:
                spans.setdefault(state // 2, []).append(frame)
            state -= int(self._traces[frame][state])
        phones = []
        for index, frames in sorted(spans.items()):
            vi, wi, pi, phone = self.targets[index]
            score = math.exp(float(np.mean([self._emissions[t][index] for t in frames])))
            phones.append(
                AlignedPhone(
                    symbol=phone.symbol,
                    verse_index=vi,
                    word_index=wi,
                    phoneme_index=pi,
                    tajweed=list(phone.tajweed),
                    start_ms=self.config.offset_ms + min(frames) * self.config.frame_shift_ms,
                    end_ms=self.config.offset_ms + (max(frames) + 1) * self.config.frame_shift_ms,
                    confidence=score,
                )
            )
        if final and len(phones) != len(self.targets):
            raise AlignmentError("Incomplete phoneme coverage")
        words = []
        verses = []
        for vi, verse in enumerate(self.reference.verses):
            for wi, word in enumerate(verse.words):
                selected = [phone for phone in phones if phone.verse_index == vi and phone.word_index == wi]
                if selected:
                    words.append(self._boundary(selected, vi, wi, word.text, verse))
            selected = [phone for phone in phones if phone.verse_index == vi]
            if selected:
                verses.append(self._boundary(selected, vi, None, " ".join(word.text for word in verse.words), verse))
        warnings = [
            "Confidence is the geometric mean acoustic posterior on the forced path, not calibrated correctness."
        ]
        if not final:
            warnings.append("Partial boundaries are provisional and can change as more audio arrives.")
        else:
            self._closed = True
        return AlignmentResult(
            qiraah=self.reference.qiraah,
            reciter=self.reference.reciter,
            model_id=self.config.model_id,
            final=final,
            frames_processed=self.frames_processed,
            path_log_probability=float(self._scores[end]),
            phonemes=phones,
            words=words,
            verses=verses,
            confidence=math.exp(float(np.mean([math.log(max(phone.confidence, 1e-300)) for phone in phones])))
            if phones
            else 0.0,
            warnings=warnings,
        )

    @staticmethod
    def _boundary(phones: list[AlignedPhone], vi: int, wi: int | None, text: str, verse: ReferenceVerse) -> Boundary:
        return Boundary(
            verse_index=vi,
            word_index=wi,
            text=text,
            surah=verse.surah,
            ayah=verse.ayah,
            start_ms=phones[0].start_ms,
            end_ms=phones[-1].end_ms,
            confidence=math.exp(sum(math.log(max(phone.confidence, 1e-300)) for phone in phones) / len(phones)),
        )


def align(reference: AlignmentReference, acoustics: AcousticFrames) -> AlignmentResult:
    decoder = CTCAligner(reference, acoustics)
    decoder.append(FrameBatch(probabilities=acoustics.probabilities))
    return decoder.result(final=True)


@router.post("/frames", response_model=AlignmentResult)
def align_frames(body: AlignmentRequest) -> AlignmentResult:
    try:
        return align(body.reference, body.acoustics)
    except AlignmentError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/audio", response_model=AlignmentResult)
async def align_audio(
    reference: Annotated[str, Form(max_length=100000)],
    audio: Annotated[UploadFile, File()],
) -> AlignmentResult:
    from phoneme_model import ModelUnavailable, get_recognizer

    try:
        target = AlignmentReference.model_validate_json(reference)
    except ValidationError as error:
        raise HTTPException(422, "Invalid phoneme reference") from error
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio exceeds the two megabyte limit")
    try:
        recognizer = get_recognizer()
        acoustics = await run_in_threadpool(recognizer.recognize, data, target.qiraah)
        return await run_in_threadpool(align, target, acoustics)
    except ModelUnavailable as error:
        raise HTTPException(503, str(error)) from error
    except (AlignmentError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


class StreamStart(BaseModel):
    reference: AlignmentReference
    acoustics: AcousticConfig


class StreamChunk(FrameBatch):
    final: bool = False


async def alignment_stream(socket: WebSocket) -> None:
    """Connection-local streaming; main.py authenticates before calling this."""
    await socket.accept()
    try:
        start = StreamStart.model_validate(await socket.receive_json())
        decoder = CTCAligner(start.reference, start.acoustics)
        await socket.send_json({"status": "ready"})
        while True:
            chunk = StreamChunk.model_validate(await socket.receive_json())
            await run_in_threadpool(decoder.append, chunk)
            result = await run_in_threadpool(decoder.result, final=chunk.final)
            await socket.send_json(result.model_dump(mode="json"))
            if chunk.final:
                await socket.close(code=1000)
                return
    except WebSocketDisconnect:
        return
    except (ValidationError, AlignmentError, ValueError) as error:
        await socket.send_json({"error": str(error)})
        await socket.close(code=1008)
