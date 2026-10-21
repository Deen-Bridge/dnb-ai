"""Isolate untrusted retrieved documents from system instructions and tool calls (#523)."""

from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions", re.IGNORECASE),
    re.compile(r"(?:system\s+prompt|reveal\s+secret|api[_\s-]?key|seed\s+phrase)", re.IGNORECASE),
    re.compile(r"(?:call|invoke|execute)\s+tool\b", re.IGNORECASE),
    re.compile(r"تجاهل\s+جميع\s+التعليمات", re.IGNORECASE),
    re.compile(r"puuzia\s+maagizo\s+yote", re.IGNORECASE),
)


@dataclass(frozen=True)
class IsolatedEvidenceBlock:
    source_id: str
    provenance_title: str
    quoted_data: str
    injection_detected: bool
    triggered_patterns: list[str]
    allow_tool_execution: bool = False
    allow_policy_override: bool = False


def isolate_retrieved_document(
    source_id: str,
    provenance_title: str,
    raw_content: str,
) -> IsolatedEvidenceBlock:
    """Wrap retrieved text strictly as non-executable quoted evidence with provenance."""
    hits: list[str] = []
    for pat in INJECTION_PATTERNS:
        if pat.search(raw_content):
            hits.append(pat.pattern)

    sanitized_quote = raw_content.replace("</evidence>", "[/evidence]")
    envelope = (
        f'<evidence id="{source_id}" source="{provenance_title}" '
        f'executable="false" role="untrusted_data">\n'
        f"{sanitized_quote}\n"
        f"</evidence>"
    )
    return IsolatedEvidenceBlock(
        source_id=source_id,
        provenance_title=provenance_title,
        quoted_data=envelope,
        injection_detected=bool(hits),
        triggered_patterns=hits,
        allow_tool_execution=False,
        allow_policy_override=False,
    )
