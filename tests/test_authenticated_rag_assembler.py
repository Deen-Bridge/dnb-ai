"""Tests for authenticated Quran and Hadith RAG context assembler (#24)."""

from __future__ import annotations

from authenticated_rag_assembler import AuthenticatedSourcePassage, assemble_authenticated_rag_context


def test_filters_unauthenticated_passages_and_formats_canonical_refs() -> None:
    passages = [
        AuthenticatedSourcePassage(
            source_type="quran",
            canonical_ref="2:255",
            arabic_text="اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ",
            translation_text="Allah - there is no deity except Him",
            authenticity_grade="mutawatir",
        ),
        AuthenticatedSourcePassage(
            source_type="hadith",
            canonical_ref="Unverified 999",
            arabic_text="...",
            translation_text="...",
            authenticity_grade="mawdu",
        ),
    ]
    ctx = assemble_authenticated_rag_context("Ayat al-Kursi", passages)
    assert len(ctx.passages) == 1
    assert "[QURAN 2:255 | grade=mutawatir]" in ctx.formatted_context
