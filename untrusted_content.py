"""Defense-in-depth sanitization for extracted multimodal text (issue #860).

Any text extracted via ASR, OCR, or third-party multimodal models is untrusted
user input. It must never alter system instructions, escape delimiters, or trigger
unintended tool calls when processed by downstream LLMs.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_INJECTION_PATTERNS = [
    re.compile(r"<\s*\|\s*im_start\s*\|.*?>", re.IGNORECASE),
    re.compile(r"<\s*\|\s*im_end\s*\|.*?>", re.IGNORECASE),
    re.compile(r"\[\s*(?:system|instruction|developer)\s*\]", re.IGNORECASE),
    re.compile(r"^\s*(?:system|assistant|human|developer)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"```+\s*(?:system|instruction)?", re.IGNORECASE),
    re.compile(r"(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior)\s+instructions?", re.IGNORECASE),
]

_ZERO_WIDTH_CHARS = re.compile(r"[\u200B-\u200D\uFEFF\u202A-\u202E]")


def sanitize_untrusted_text(text: str) -> str:
    """Sanitize extracted multimodal text to prevent prompt injection attacks.

    - Strips zero-width control and directional override characters.
    - Neutralizes known prompt injection markers and control delimiters.
    - Normalizes unicode representation.
    """
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _ZERO_WIDTH_CHARS.sub("", cleaned)

    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[REDACTED_INSTRUCTION]", cleaned)

    return cleaned.strip()


def wrap_untrusted_content(text: str, source: str = "multimodal_extraction") -> dict[str, Any]:
    """Wrap untrusted text with safe structural boundaries for downstream consumers."""
    sanitized = sanitize_untrusted_text(text)
    return {
        "text": sanitized,
        "is_untrusted": True,
        "source": source,
        "safe_wrapper": f"<<<UNTRUSTED_EXTRACTED_DATA source={source}>>>\n{sanitized}\n<<<END_UNTRUSTED_EXTRACTED_DATA>>>",
    }
