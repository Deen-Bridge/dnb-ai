"""Versioned corpus for official Stellar documentation (#550)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from urllib.parse import urlparse

ALLOWED_STELLAR_DOMAINS: frozenset[str] = frozenset(
    {
        "developers.stellar.org",
        "stellar.org",
        "soroban.stellar.org",
    }
)


@dataclass(frozen=True)
class StellarCorpusDocument:
    """Immutable metadata and content for an official Stellar document."""

    doc_id: str
    title: str
    canonical_url: str
    retrieval_date: str
    content: str
    content_hash: str
    version: str


@dataclass(frozen=True)
class StellarCorpusPassage:
    """Indexed passage with full provenance metadata."""

    passage_id: str
    doc_id: str
    title: str
    canonical_url: str
    retrieval_date: str
    content_hash: str
    version: str
    text: str


def compute_content_hash(content: str) -> str:
    """Return deterministic SHA-256 hex digest of normalized document content."""
    normalized = content.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_allowed_stellar_source(url: str) -> bool:
    """Validate that a URL uses HTTPS and belongs to an allowlisted Stellar domain."""
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    host = parsed.netloc.lower()
    return host in ALLOWED_STELLAR_DOMAINS


class OfficialStellarCorpus:
    """Reproducible, offline-indexed corpus of official Stellar documentation."""

    def __init__(self, version: str = "2026.10.1") -> None:
        self.version = version
        self._documents: dict[str, StellarCorpusDocument] = {}
        self._passages: dict[str, StellarCorpusPassage] = {}

    def ingest_document(
        self,
        *,
        doc_id: str,
        title: str,
        canonical_url: str,
        retrieval_date: str,
        content: str,
    ) -> StellarCorpusDocument:
        """Ingest an allowlisted official Stellar document deterministically."""
        if not doc_id.strip() or not title.strip() or not retrieval_date.strip():
            raise ValueError("doc_id, title, and retrieval_date must be non-empty")
        if not is_allowed_stellar_source(canonical_url):
            raise ValueError(f"Disallowed documentation source URL: {canonical_url}")
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("Document content cannot be empty")
        digest = compute_content_hash(cleaned)
        doc = StellarCorpusDocument(
            doc_id=doc_id.strip(),
            title=title.strip(),
            canonical_url=canonical_url.strip(),
            retrieval_date=retrieval_date.strip(),
            content=cleaned,
            content_hash=digest,
            version=self.version,
        )
        self._documents[doc.doc_id] = doc
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        for idx, para in enumerate(paragraphs, start=1):
            pid = f"{doc.doc_id}#p{idx}"
            self._passages[pid] = StellarCorpusPassage(
                passage_id=pid,
                doc_id=doc.doc_id,
                title=doc.title,
                canonical_url=doc.canonical_url,
                retrieval_date=doc.retrieval_date,
                content_hash=doc.content_hash,
                version=doc.version,
                text=para,
            )
        return doc

    def get_passage(self, passage_id: str) -> StellarCorpusPassage | None:
        """Look up a specific passage by ID."""
        return self._passages.get(passage_id)

    def search_passages(self, query: str) -> list[StellarCorpusPassage]:
        """Deterministic token-overlap retrieval over indexed official passages."""
        tokens = {
            tok.lower()
            for tok in query.replace(",", " ").replace(".", " ").split()
            if len(tok) >= 3
        }
        if not tokens:
            return []
        scored: list[tuple[int, str, StellarCorpusPassage]] = []
        for pid, passage in sorted(self._passages.items()):
            haystack = f"{passage.title} {passage.text}".lower()
            hits = sum(1 for tok in tokens if tok in haystack)
            if hits > 0:
                scored.append((hits, pid, passage))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored]
