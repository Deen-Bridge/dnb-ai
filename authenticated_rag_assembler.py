"""Retrieval-augmented generation assembler over authenticated Quran and Hadith sources (#24)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthenticatedSourcePassage:
    source_type: str  # "quran" or "hadith"
    canonical_ref: str
    arabic_text: str
    translation_text: str
    authenticity_grade: str  # e.g., "mutawatir", "sahih", "hasan"


@dataclass(frozen=True)
class GroundedRAGContext:
    query: str
    passages: list[AuthenticatedSourcePassage]
    formatted_context: str


ACCEPTED_GRADES: frozenset[str] = frozenset({"mutawatir", "sahih", "hasan"})


def assemble_authenticated_rag_context(
    query: str,
    candidates: list[AuthenticatedSourcePassage],
) -> GroundedRAGContext:
    """Filter to authenticated Quran/Hadith passages and format citation-preserving RAG context."""
    verified = [
        p
        for p in candidates
        if p.source_type in {"quran", "hadith"}
        and p.authenticity_grade.strip().lower() in ACCEPTED_GRADES
        and p.canonical_ref.strip()
    ]
    blocks = [
        f"[{p.source_type.upper()} {p.canonical_ref} | grade={p.authenticity_grade}]\n"
        f"AR: {p.arabic_text}\nEN: {p.translation_text}"
        for p in verified
    ]
    return GroundedRAGContext(
        query=query,
        passages=verified,
        formatted_context="\n\n".join(blocks),
    )
