"""Tests for access-control-aware retrieval filter (#91)."""

from __future__ import annotations

from acl_retrieval_filter import CorpusChunkRecord, RetrievalAuthContext, filter_authorized_chunks


def test_prevents_cross_user_cross_tenant_and_unpublished_leakage() -> None:
    chunks = [
        CorpusChunkRecord("c-pub", "Public hadith note", published=True, visibility="public"),
        CorpusChunkRecord("c-unpub", "Draft tafsir", published=False, visibility="public"),
        CorpusChunkRecord(
            "c-u1", "User 1 private reflection", published=True, visibility="private", owner_user_id="u1"
        ),
        CorpusChunkRecord(
            "c-u2", "User 2 private reflection", published=True, visibility="private", owner_user_id="u2"
        ),
    ]
    ctx = RetrievalAuthContext(user_id="u1", tenant_id="t1", allow_unpublished=False)
    allowed = filter_authorized_chunks(chunks, ctx)
    assert [c.chunk_id for c in allowed] == ["c-pub", "c-u1"]
