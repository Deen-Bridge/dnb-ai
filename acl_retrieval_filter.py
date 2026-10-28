"""Access-control-aware retrieval preventing cross-user or unpublished document leakage (#91)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusChunkRecord:
    chunk_id: str
    text: str
    published: bool
    visibility: str  # "public", "tenant", "private"
    owner_user_id: str | None = None
    tenant_id: str | None = None


@dataclass(frozen=True)
class RetrievalAuthContext:
    user_id: str | None
    tenant_id: str | None
    allow_unpublished: bool = False


def filter_authorized_chunks(
    chunks: list[CorpusChunkRecord],
    ctx: RetrievalAuthContext,
) -> list[CorpusChunkRecord]:
    """Deny-by-default filter ensuring only published and principal-authorized chunks are returned."""
    authorized: list[CorpusChunkRecord] = []
    for chunk in chunks:
        if not chunk.published and not ctx.allow_unpublished:
            continue
        if chunk.visibility == "public":
            authorized.append(chunk)
        elif chunk.visibility == "tenant":
            if ctx.tenant_id and chunk.tenant_id == ctx.tenant_id:
                authorized.append(chunk)
        elif chunk.visibility == "private":
            if ctx.user_id and chunk.owner_user_id == ctx.user_id:
                authorized.append(chunk)
    return authorized
