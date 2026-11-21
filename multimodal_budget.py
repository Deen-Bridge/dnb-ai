"""Multimodal cost and latency budgets with untrusted extraction safeguards (#859).

Enforces production latency ceilings, cost ceilings, payload limits, and retention TTLs
across audio (audio_hadith), OCR (arabic_ocr), layout analysis (page_analysis), and
lecture video (video_analysis). Treats all extracted text as untrusted data and neutralizes
prompt injection attempts before downstream model or tool consumption.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

# Prompt injection patterns in extracted multimodal content
_INJECTION_PATTERN = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions|system\s*prompt|you\s+are\s+now|override\s+instructions|<system>|tool\s*call\s*:)",
    re.IGNORECASE,
)

# Control characters except common whitespace
_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


class ModalityType(str, Enum):
    AUDIO = "audio"
    OCR = "ocr"
    PAGE_LAYOUT = "page_layout"
    VIDEO = "video"


@dataclass(frozen=True)
class ModalityBudgetConfig:
    """Pre-configured cost, latency, size, and retention budgets per modality."""

    modality: ModalityType
    max_payload_bytes: int
    max_latency_ms: int
    estimated_cost_per_unit_usd: float
    max_cost_budget_usd: float
    retention_ttl_seconds: int
    allowed_extensions: tuple[str, ...]
    allowed_mime_types: tuple[str, ...]


DEFAULT_MODALITY_BUDGETS: dict[ModalityType, ModalityBudgetConfig] = {
    ModalityType.AUDIO: ModalityBudgetConfig(
        modality=ModalityType.AUDIO,
        max_payload_bytes=25 * 1024 * 1024,  # 25 MB
        max_latency_ms=8_000,
        estimated_cost_per_unit_usd=0.006,  # per minute or segment
        max_cost_budget_usd=0.05,
        retention_ttl_seconds=3_600,  # 1 hour
        allowed_extensions=(".mp3", ".wav", ".m4a", ".aac", ".ogg"),
        allowed_mime_types=("audio/mpeg", "audio/wav", "audio/x-m4a", "audio/aac", "audio/ogg"),
    ),
    ModalityType.OCR: ModalityBudgetConfig(
        modality=ModalityType.OCR,
        max_payload_bytes=15 * 1024 * 1024,  # 15 MB
        max_latency_ms=5_000,
        estimated_cost_per_unit_usd=0.0015,  # per image page
        max_cost_budget_usd=0.03,
        retention_ttl_seconds=3_600,
        allowed_extensions=(".png", ".jpg", ".jpeg", ".webp", ".tiff"),
        allowed_mime_types=("image/png", "image/jpeg", "image/webp", "image/tiff"),
    ),
    ModalityType.PAGE_LAYOUT: ModalityBudgetConfig(
        modality=ModalityType.PAGE_LAYOUT,
        max_payload_bytes=5 * 1024 * 1024,  # 5 MB structured JSON
        max_latency_ms=3_000,
        estimated_cost_per_unit_usd=0.0005,
        max_cost_budget_usd=0.01,
        retention_ttl_seconds=1_800,  # 30 minutes
        allowed_extensions=(".json",),
        allowed_mime_types=("application/json", "text/plain"),
    ),
    ModalityType.VIDEO: ModalityBudgetConfig(
        modality=ModalityType.VIDEO,
        max_payload_bytes=100 * 1024 * 1024,  # 100 MB
        max_latency_ms=15_000,
        estimated_cost_per_unit_usd=0.02,
        max_cost_budget_usd=0.15,
        retention_ttl_seconds=7_200,  # 2 hours
        allowed_extensions=(".mp4", ".mov", ".avi", ".webm"),
        allowed_mime_types=("video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"),
    ),
}


class MultimodalBudgetError(ValueError):
    """Raised when an operation exceeds multimodal cost, latency, or size limits."""


@dataclass(frozen=True)
class SanitizedExtractedContent:
    """Wrapped extracted content guaranteeing isolation from system instructions."""

    original_length: int
    sanitized_text: str
    is_safe: bool
    injection_detected: bool
    untrusted_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_multimodal_extracted_content(raw_text: str) -> SanitizedExtractedContent:
    """Sanitize text extracted from audio transcripts, OCR glyphs, or video lectures.

    Treats all extracted text as untrusted data that must not alter system prompts or invoke tools.
    """
    if not raw_text:
        return SanitizedExtractedContent(
            original_length=0,
            sanitized_text="",
            is_safe=True,
            injection_detected=False,
            untrusted_boundary="[UNTRUSTED_MULTIMODAL_DATA]\n\n[/UNTRUSTED_MULTIMODAL_DATA]",
        )

    orig_len = len(raw_text)
    # Strip dangerous unprintable control codes
    clean = _CONTROL_CHARS_PATTERN.sub("", raw_text).strip()

    injection_found = bool(_INJECTION_PATTERN.search(clean))
    if injection_found:
        # Redact injection directives while preserving harmless context length
        clean = _INJECTION_PATTERN.sub("[REDACTED_MULTIMODAL_INJECTION]", clean)

    boundary = f"[UNTRUSTED_MULTIMODAL_DATA]\n{clean}\n[/UNTRUSTED_MULTIMODAL_DATA]"

    return SanitizedExtractedContent(
        original_length=orig_len,
        sanitized_text=clean,
        is_safe=not injection_found,
        injection_detected=injection_found,
        untrusted_boundary=boundary,
    )


@dataclass(frozen=True)
class MultimodalBudgetEvaluation:
    """Audit evidence for a multimodal request under budget."""

    modality: str
    payload_size_bytes: int
    size_within_budget: bool
    elapsed_ms: float
    latency_within_budget: bool
    estimated_cost_usd: float
    cost_within_budget: bool
    passed_all_budgets: bool
    retention_ttl_seconds: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultimodalBudgetEnforcer:
    """Observability and enforcement gatekeeper for multimodal workloads (#859)."""

    def __init__(
        self,
        modality: ModalityType,
        custom_config: ModalityBudgetConfig | None = None,
    ) -> None:
        self.config = custom_config or DEFAULT_MODALITY_BUDGETS[modality]
        self._start_time = time.monotonic()

    def validate_payload_size(self, size_bytes: int) -> None:
        """Validate payload size against modality budget."""
        if size_bytes < 0:
            raise MultimodalBudgetError("Payload size cannot be negative")
        if size_bytes > self.config.max_payload_bytes:
            raise MultimodalBudgetError(
                f"Payload size {size_bytes} exceeds {self.config.modality.value} maximum budget "
                f"of {self.config.max_payload_bytes} bytes"
            )

    def validate_extension(self, filename: str) -> None:
        """Verify file extension is explicitly permitted."""
        lowered = filename.lower()
        if not any(lowered.endswith(ext) for ext in self.config.allowed_extensions):
            raise MultimodalBudgetError(
                f"Unsupported file extension for {self.config.modality.value}. "
                f"Allowed: {', '.join(self.config.allowed_extensions)}"
            )

    def evaluate(self, payload_size_bytes: int, unit_count: float = 1.0) -> MultimodalBudgetEvaluation:
        """Evaluate latency, cost, and payload size, returning structured audit evidence."""
        elapsed_ms = (time.monotonic() - self._start_time) * 1000.0
        warnings: list[str] = []

        size_ok = payload_size_bytes <= self.config.max_payload_bytes
        if not size_ok:
            warnings.append("payload_size_budget_exceeded")

        latency_ok = elapsed_ms <= self.config.max_latency_ms
        if not latency_ok:
            warnings.append("latency_budget_exceeded")

        estimated_cost = self.config.estimated_cost_per_unit_usd * max(1.0, unit_count)
        cost_ok = estimated_cost <= self.config.max_cost_budget_usd
        if not cost_ok:
            warnings.append("cost_budget_exceeded")

        passed = size_ok and latency_ok and cost_ok

        return MultimodalBudgetEvaluation(
            modality=self.config.modality.value,
            payload_size_bytes=payload_size_bytes,
            size_within_budget=size_ok,
            elapsed_ms=round(elapsed_ms, 2),
            latency_within_budget=latency_ok,
            estimated_cost_usd=round(estimated_cost, 6),
            cost_within_budget=cost_ok,
            passed_all_budgets=passed,
            retention_ttl_seconds=self.config.retention_ttl_seconds,
            warnings=tuple(warnings),
        )
