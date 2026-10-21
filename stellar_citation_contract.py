"""Citation contract for protocol passages in Stellar answers (#551)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StellarSourceCitation:
    """Short source reference for a grounded Stellar protocol claim."""

    passage_id: str
    title: str
    canonical_url: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedStellarAnswer:
    """Response contract distinguishing grounded protocol claims and missing evidence."""

    status: str  # "grounded", "general_explanation", "insufficient_evidence"
    answer_text: str
    is_protocol_specific: bool
    citations: tuple[StellarSourceCitation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer_text": self.answer_text,
            "is_protocol_specific": self.is_protocol_specific,
            "citations": [c.to_dict() for c in self.citations],
        }


def build_grounded_stellar_answer(
    *,
    answer_text: str,
    is_protocol_specific: bool,
    citations: list[StellarSourceCitation],
) -> GroundedStellarAnswer:
    """Enforce citation requirements on protocol-specific Stellar claims."""
    valid_citations: list[StellarSourceCitation] = []
    seen_ids: set[str] = set()
    for cit in citations:
        pid = cit.passage_id.strip()
        title = cit.title.strip()
        url = cit.canonical_url.strip()
        if not pid or not title or not url.startswith("https://"):
            raise ValueError("Citation requires non-empty passage_id, title, and HTTPS canonical_url")
        if pid not in seen_ids:
            seen_ids.add(pid)
            valid_citations.append(StellarSourceCitation(passage_id=pid, title=title, canonical_url=url))

    cleaned_answer = answer_text.strip()
    if is_protocol_specific and not valid_citations:
        return GroundedStellarAnswer(
            status="insufficient_evidence",
            answer_text=(
                "Insufficient official Stellar documentation evidence to verify this "
                "protocol-specific claim without fabricating a citation."
            ),
            is_protocol_specific=True,
            citations=(),
        )

    if not is_protocol_specific and not valid_citations:
        return GroundedStellarAnswer(
            status="general_explanation",
            answer_text=f"[General Explanation] {cleaned_answer}",
            is_protocol_specific=False,
            citations=(),
        )

    return GroundedStellarAnswer(
        status="grounded",
        answer_text=cleaned_answer,
        is_protocol_specific=is_protocol_specific,
        citations=tuple(valid_citations),
    )
