"""Optional Wav2Vec2 style phoneme recognizer for the alignment audio endpoint.

Heavy inference dependencies are lazy and optional. Model code is never trusted
from a remote repository; operators select and pin a compatible CTC checkpoint.
"""

from __future__ import annotations

import importlib
import io
import logging
import os
import re
import threading
import wave
from functools import lru_cache
from typing import Any

import numpy as np

from phoneme_alignment import AcousticFrames, AlignmentError

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """Inference is not configured or the configured model cannot be loaded."""


class HuggingFacePhonemeRecognizer:
    """Adapter for standard Transformers CTC models with phoneme vocabularies."""

    def __init__(self, model_id: str, revision: str, qiraat: set[str]) -> None:
        if not model_id or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ModelUnavailable("Set ALIGNMENT_MODEL_ID and a full commit SHA in ALIGNMENT_MODEL_REVISION")
        if not qiraat or not qiraat <= {"hafs", "warsh", "qalun"}:
            raise ModelUnavailable("ALIGNMENT_MODEL_QIRAAT must list supported readings: hafs, warsh, qalun")
        self.model_id = model_id
        self.revision = revision
        self.qiraat = frozenset(qiraat)
        self._lock = threading.Lock()
        self._model: Any = None
        self._extractor: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            options = {"revision": self.revision, "trust_remote_code": False}
            extractor = transformers.AutoFeatureExtractor.from_pretrained(self.model_id, **options)
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, do_phonemize=False, **options)
            model = transformers.AutoModelForCTC.from_pretrained(self.model_id, **options)
            model.eval()
            if extractor.sampling_rate != 16000 or model.config.pad_token_id is None:
                raise ValueError("A 16 kHz phoneme CTC model with a blank token is required")
            self._torch, self._extractor, self._tokenizer, self._model = torch, extractor, tokenizer, model
        except Exception as error:
            logger.exception("Could not load phoneme alignment model")
            raise ModelUnavailable(
                "Phoneme model could not be loaded; check configuration and optional dependencies"
            ) from error

    @staticmethod
    def read_audio(data: bytes) -> np.ndarray:
        """Validate bounded mono 16 kHz PCM16 WAV without external decoders."""
        try:
            with wave.open(io.BytesIO(data), "rb") as source:
                if (source.getnchannels(), source.getframerate(), source.getsampwidth(), source.getcomptype()) != (
                    1,
                    16000,
                    2,
                    "NONE",
                ):
                    raise AlignmentError("Audio must be mono 16 kHz PCM16 WAV")
                count = source.getnframes()
                if count < 1600 or count > 960000:
                    raise AlignmentError("Audio must contain 0.1 to 60 seconds")
                raw = source.readframes(count)
                if len(raw) != count * 2:
                    raise AlignmentError("Truncated WAV audio")
        except (wave.Error, EOFError) as error:
            raise AlignmentError("Invalid PCM WAV audio") from error
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if not np.any(samples):
            raise AlignmentError("Audio contains only digital silence")
        return samples

    def recognize(self, data: bytes, qiraah: str) -> AcousticFrames:
        if qiraah not in self.qiraat:
            raise ModelUnavailable("The configured phoneme model does not support the requested reading")
        samples = self.read_audio(data)
        # Serialize model loading/inference to bound CPU and memory use. The
        # light decoder and pre-extracted-frame API remain independently usable.
        with self._lock:
            self._load()
            try:
                inputs = self._extractor(samples, sampling_rate=16000, return_tensors="pt")
                with self._torch.inference_mode():
                    logits = self._model(**inputs).logits[0]
                    probabilities = logits.softmax(dim=-1).cpu().numpy()
                vocabulary = self._tokenizer.get_vocab()
                labels = [""] * probabilities.shape[1]
                for symbol, index in vocabulary.items():
                    if 0 <= index < len(labels):
                        labels[index] = symbol
                stride = self._model.config.inputs_to_logits_ratio
                if not isinstance(stride, (int, float)) or stride <= 0:
                    raise ValueError("Model must declare a positive inputs_to_logits_ratio")
                return AcousticFrames(
                    labels=labels,
                    blank_id=self._model.config.pad_token_id,
                    frame_shift_ms=1000 * stride / 16000,
                    model_id=f"{self.model_id}@{self.revision}",
                    probabilities=probabilities.tolist(),
                )
            except Exception as error:
                logger.exception("Phoneme inference failed")
                raise ModelUnavailable("Phoneme inference failed; check model compatibility and server logs") from error


@lru_cache
def get_recognizer() -> HuggingFacePhonemeRecognizer:
    return HuggingFacePhonemeRecognizer(
        os.getenv("ALIGNMENT_MODEL_ID", ""),
        os.getenv("ALIGNMENT_MODEL_REVISION", ""),
        {name.strip() for name in os.getenv("ALIGNMENT_MODEL_QIRAAT", "hafs").split(",")},
    )
