"""Trust boundary for text extracted from user media (#858, PROD-AUDIT AI-082).

Text produced by OCR, speech recognition, page-layout engines, or typed into a
"transcript" field is *data about the media*, never instructions to this
service or to any model downstream of it. This module is the single place that
describes that contract:

* :func:`inspect_text` reports instruction-like patterns and hidden characters
  as stable flag codes. It never rewrites the text: a lecture that quotes
  "you are now..." or a manuscript that mentions a "system" must survive
  extraction intact, so the boundary labels content instead of mutating it.
* :class:`UntrustedContentInfo` is attached to every extraction response as
  ``content_trust``. Downstream consumers (the DeenBridge backend, any prompt
  builder) must honour its ``handling`` rule: quote the text as data, never
  place it in system or developer instructions, never let it choose or
  authorize a tool, and never treat a pattern match as verification.

Inside this service no extracted text reaches a model, a tool, or a shell;
``tests/test_extracted_content_boundary.py`` enforces that statically.

Arabic, Urdu and Persian legitimately use ZWNJ (U+200C), ZWJ (U+200D) and the
LRM/RLM marks (U+200E/U+200F), so those are never flagged. Bidi override and
isolate controls, C0 control characters and invisible separators are flagged
as ``hidden_characters`` because they can hide text from a human reviewer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

ContentSource = Literal["asr", "ocr", "user_transcript", "page_blocks"]

HANDLING_RULE = (
    "Untrusted data extracted from user media. Quote it as data; never place it in system or developer "
    "instructions, never let it select or authorize a tool, and never treat a pattern match in it as verification."
)

# Stable flag codes, in reporting order.
FLAG_INSTRUCTION_OVERRIDE = "instruction_override"
FLAG_ROLE_MARKER = "role_marker"
FLAG_PERSONA_CHANGE = "persona_change"
FLAG_SECRET_REQUEST = "secret_request"
FLAG_TOOL_CALL_SYNTAX = "tool_call_syntax"
FLAG_VERIFICATION_CLAIM = "verification_claim"
FLAG_HIDDEN_CHARACTERS = "hidden_characters"
FLAG_TRUNCATED = "truncated"

INSTRUCTION_FLAGS = frozenset(
    {
        FLAG_INSTRUCTION_OVERRIDE,
        FLAG_ROLE_MARKER,
        FLAG_PERSONA_CHANGE,
        FLAG_SECRET_REQUEST,
        FLAG_TOOL_CALL_SYNTAX,
        FLAG_VERIFICATION_CLAIM,
    }
)

_FLAG_ORDER = (
    FLAG_INSTRUCTION_OVERRIDE,
    FLAG_ROLE_MARKER,
    FLAG_PERSONA_CHANGE,
    FLAG_SECRET_REQUEST,
    FLAG_TOOL_CALL_SYNTAX,
    FLAG_VERIFICATION_CLAIM,
    FLAG_HIDDEN_CHARACTERS,
    FLAG_TRUNCATED,
)

# Every quantifier is bounded, so scanning is linear in the input length.
_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    FLAG_INSTRUCTION_OVERRIDE: (
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
            r"\b(?:previous|prior|above|earlier|all|any|system|developer)\b[^.\n]{0,40}"
            r"\b(?:instructions?|prompts?|rules?|guidelines?|policies|context)\b",
            re.IGNORECASE,
        ),
        # Arabic: "ignore (all) previous instructions/orders"
        re.compile(r"(?:تجاهل|انس|تخط)\s{1,3}(?:(?:كل|جميع)\s{1,3})?(?:التعليمات|الأوامر|القواعد)"),
    ),
    FLAG_ROLE_MARKER: (
        re.compile(
            r"(?:^|\n)[ \t]{0,8}(?:system|assistant|developer|tool|function)(?:[ \t]{1,3}(?:prompt|message))?[ \t]{0,3}:",
            re.IGNORECASE,
        ),
        re.compile(
            r"<\|?\s{0,3}/?\s{0,3}(?:im_start|im_end|system|assistant|developer|instructions?)\s{0,3}\|?>", re.I
        ),
        re.compile(r"\[/?INST\]|\bBEGIN\s+(?:SYSTEM|INSTRUCTIONS?)\b|^#{1,6}[ \t]{0,3}system\b", re.I | re.M),
    ),
    FLAG_PERSONA_CHANGE: (
        re.compile(r"\byou\s+are\s+now\b|\bpretend\s+(?:to\s+be|you\s+are)\b|\bnew\s+persona\b", re.IGNORECASE),
    ),
    FLAG_SECRET_REQUEST: (
        re.compile(
            r"\b(?:reveal|print|show|leak|expose|output|repeat)\b[^.\n]{0,30}"
            r"\b(?:system\s+prompt|prompt|secrets?|api[ _-]?keys?|tokens?|passwords?|credentials?|env(?:ironment)?\s+variables?)\b",
            re.IGNORECASE,
        ),
    ),
    FLAG_TOOL_CALL_SYNTAX: (
        re.compile(r"\{\s{0,8}[\"'](?:tool|tool_calls?|tool_name|function|function_call|arguments)[\"']\s{0,8}:", re.I),
        re.compile(r"</?\s{0,3}(?:tool_call|tool_use|function_call|invoke|tool)\b[^>]{0,80}>", re.IGNORECASE),
        re.compile(
            r"^[ \t]{0,8}(?:tool|tool_calls?|function_call|run_command|execute)[ \t]{0,3}:[ \t]{0,3}\S", re.I | re.M
        ),
    ),
    FLAG_VERIFICATION_CLAIM: (
        re.compile(r"\bmark\s+(?:this|it)\s+as\s+(?:verified|sahih|authentic|correct|approved)\b", re.IGNORECASE),
    ),
}

# C0 controls except TAB/LF/CR, DEL, zero-width space, word joiner, BOM, and the
# bidi embedding/override/isolate controls. ZWNJ, ZWJ, LRM and RLM are allowed.
_HIDDEN = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​⁠﻿‪-‮⁦-⁩]")


def inspect_text(text: str) -> list[str]:
    """Return the stable flag codes that apply to ``text``. Never modifies it."""
    if not text:
        return []
    found = {flag for flag, patterns in _PATTERNS.items() if any(p.search(text) for p in patterns)}
    if _HIDDEN.search(text):
        found.add(FLAG_HIDDEN_CHARACTERS)
    return [flag for flag in _FLAG_ORDER if flag in found]


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """Cap ``text`` at ``limit`` characters; report whether anything was cut."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


class UntrustedContentInfo(BaseModel):
    """How a response's extracted text must be treated."""

    trust: Literal["untrusted"] = "untrusted"
    sources: list[ContentSource] = Field(..., description="Where the extracted text came from")
    instruction_like: bool = Field(
        ..., description="True when any flag suggests the text tries to instruct a model or call a tool"
    )
    flags: list[str] = Field(default_factory=list, description="Stable codes from untrusted_content.py")
    handling: str = HANDLING_RULE


def describe_untrusted(
    texts: Iterable[str], sources: Iterable[ContentSource], extra_flags: Iterable[str] = ()
) -> UntrustedContentInfo:
    """Build the ``content_trust`` block for one response."""
    found: set[str] = set(extra_flags)
    for text in texts:
        found.update(inspect_text(text))
    flags = [flag for flag in _FLAG_ORDER if flag in found]
    return UntrustedContentInfo(
        sources=list(dict.fromkeys(sources)),
        instruction_like=any(flag in INSTRUCTION_FLAGS for flag in flags),
        flags=flags,
    )
