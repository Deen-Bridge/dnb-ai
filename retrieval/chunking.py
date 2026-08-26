"""Deterministic, content-type-aware chunking for the retrieval corpus.

A *chunk* is the smallest retrievable span the index stores. Two rules govern
how the bundled corpora become chunks:

1. **Reference records stay atomic.** A Quran ayah or a hadith grading record is
   a single indivisible unit — splitting "2:255" across two chunks would let a
   similarity search return half a verse. :func:`chunk_ayah`,
   :func:`chunk_hadith` and :func:`chunk_surah` therefore each emit exactly one
   chunk per record.
2. **Long prose is split by a token budget with overlap.** Free-form text (the
   shape #2's public-knowledge corpus will arrive in) is windowed by
   :func:`chunk_prose` so no single chunk exceeds the budget, with a configurable
   overlap so a passage that straddles a boundary is still retrievable from
   either side.

Every chunk carries stable metadata so the index can dedupe and re-sync:

* ``source`` — the corpus family (``"quran"``, ``"hadith"``, ``"surah_index"``…).
* ``source_id`` — a stable identifier for the *logical record* the chunk came
  from (e.g. ``"quran:2:255"``, ``"hadith:bukhari:1"``). Incremental reindex
  upserts and deletes by this id.
* ``content_hash`` — a SHA-256 over the embeddable text. An unchanged hash means
  the chunk never needs re-embedding (ties into backlog #13); a changed hash
  replaces the old vector.
* ``scope`` / ``published`` — the access fields #3 filters retrieval on. They are
  set here so the pipeline is ready for access-scoped retrieval without a schema
  change later.

Determinism is a hard requirement: :func:`iter_corpus_chunks` walks every corpus
in sorted order and hashing is pure, so a backfill produces byte-identical chunks
run to run — which is what makes the index idempotent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repository ``data/`` directory — the bundled corpora live here.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

QURAN_UTHMANI_PATH = DATA_DIR / "quran_uthmani.json"
SURAH_INDEX_PATH = DATA_DIR / "quran" / "surah_index.json"
HADITH_DIR = DATA_DIR / "hadith"

# Default prose windowing. Tokens are estimated by whitespace splitting (see
# :func:`estimate_tokens`) — deterministic and dependency-free, which matters
# more here than matching a specific model's tokenizer exactly.
DEFAULT_CHUNK_TOKENS = 256
DEFAULT_CHUNK_OVERLAP = 32

# Default access fields. The bundled corpora are public, canonical scripture and
# hadith, so they are published and world-readable; #3 will emit private scopes
# for per-user content through the same :class:`Chunk` shape.
DEFAULT_SCOPE = "public"


def estimate_tokens(text: str) -> int:
    """Estimate a token count by whitespace splitting.

    Deliberately not tied to any model tokenizer: the budget only needs to be a
    stable, monotonic proxy for length so chunk boundaries are reproducible.
    """
    return len(text.split())


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hex digest of *text*'s embeddable content.

    The hash is the dedup key: identical text yields an identical digest across
    processes and runs, so unchanged content is never re-embedded. Whitespace is
    normalized first so cosmetic reflowing does not force a re-embed, but casing
    and non-Latin scripts are preserved because they carry meaning.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Chunk:
    """A single retrievable span plus the metadata the index syncs on.

    ``chunk_id`` is unique within the index; ``source_id`` groups all chunks that
    came from one logical record (a prose record may yield several parts). Frozen
    so a chunk cannot drift from the ``content_hash`` computed for it.
    """

    chunk_id: str
    source: str
    source_id: str
    text: str
    content_hash: str
    scope: str = DEFAULT_SCOPE
    published: bool = True
    part: int = 0
    part_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (used by the SQLite backend and docs)."""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "source_id": self.source_id,
            "text": self.text,
            "content_hash": self.content_hash,
            "scope": self.scope,
            "published": self.published,
            "part": self.part,
            "part_count": self.part_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=data["chunk_id"],
            source=data["source"],
            source_id=data["source_id"],
            text=data["text"],
            content_hash=data["content_hash"],
            scope=data.get("scope", DEFAULT_SCOPE),
            published=bool(data.get("published", True)),
            part=int(data.get("part", 0)),
            part_count=int(data.get("part_count", 1)),
            metadata=dict(data.get("metadata", {})),
        )


def make_chunk(
    *,
    source: str,
    source_id: str,
    text: str,
    scope: str = DEFAULT_SCOPE,
    published: bool = True,
    part: int = 0,
    part_count: int = 1,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Construct a :class:`Chunk`, computing its ``chunk_id`` and ``content_hash``.

    ``chunk_id`` is the ``source_id`` for an atomic record, or ``"{source_id}#{part}"``
    when a record is split into multiple parts — stable either way, so reindexing
    the same content targets the same rows.
    """
    chunk_id = source_id if part_count == 1 else f"{source_id}#{part}"
    return Chunk(
        chunk_id=chunk_id,
        source=source,
        source_id=source_id,
        text=text,
        content_hash=content_hash(text),
        scope=scope,
        published=published,
        part=part,
        part_count=part_count,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Prose chunking (token-budget windows with overlap)
# ---------------------------------------------------------------------------


def chunk_prose(
    source: str,
    source_id: str,
    text: str,
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    scope: str = DEFAULT_SCOPE,
    published: bool = True,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split *text* into overlapping token-budget windows.

    A record short enough to fit the budget yields a single atomic chunk. Longer
    text is windowed: each window holds at most ``max_tokens`` whitespace tokens
    and starts ``max_tokens - overlap`` tokens after the previous one, so content
    on a boundary appears in two adjacent chunks and stays retrievable. The split
    is deterministic and every part shares the record's ``source_id``.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must be in [0, max_tokens)")

    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [
            make_chunk(
                source=source,
                source_id=source_id,
                text=" ".join(tokens) if tokens else text.strip(),
                scope=scope,
                published=published,
                metadata=metadata,
            )
        ]

    stride = max_tokens - overlap
    windows: list[str] = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        windows.append(" ".join(window))
        if start + max_tokens >= len(tokens):
            break
        start += stride

    part_count = len(windows)
    return [
        make_chunk(
            source=source,
            source_id=source_id,
            text=window,
            scope=scope,
            published=published,
            part=index,
            part_count=part_count,
            metadata=metadata,
        )
        for index, window in enumerate(windows)
    ]


# ---------------------------------------------------------------------------
# Atomic record chunkers
# ---------------------------------------------------------------------------


def chunk_ayah(
    surah: int,
    ayah: int,
    record: dict[str, Any],
    *,
    scope: str = DEFAULT_SCOPE,
) -> Chunk:
    """Chunk one Quran ayah — atomic, never split."""
    arabic = str(record.get("arabic", "")).strip()
    english = str(record.get("english", "")).strip()
    text = f"Quran {surah}:{ayah}\n{arabic}\n{english}".strip()
    return make_chunk(
        source="quran",
        source_id=f"quran:{surah}:{ayah}",
        text=text,
        scope=scope,
        metadata={
            "surah": surah,
            "ayah": ayah,
            "reference": f"{surah}:{ayah}",
            "arabic": arabic,
            "english": english,
        },
    )


def chunk_hadith(
    collection: str,
    record: dict[str, Any],
    *,
    scope: str = DEFAULT_SCOPE,
) -> Chunk | None:
    """Chunk one hadith grading record — atomic, never split.

    The bundled hadith corpus is grading metadata (grade, chain, book/number),
    not matn text, so the embeddable text is a rendered description of the
    record. Returns ``None`` for a record with no usable number.
    """
    number = record.get("n")
    if number is None:
        return None
    number = int(number)
    grade = str(record.get("grade", "")).strip()
    chain = str(record.get("chain", "")).strip()
    book = record.get("book")
    parts = [f"Hadith {collection} #{number}"]
    if book is not None:
        parts.append(f"book {book}")
    if grade:
        parts.append(f"grade: {grade}")
    if chain:
        parts.append(f"chain: {chain}")
    text = ", ".join(parts)
    metadata: dict[str, Any] = {"collection": collection, "number": number}
    if book is not None:
        metadata["book"] = book
    if grade:
        metadata["grade"] = grade
    if chain:
        metadata["chain"] = chain
    return make_chunk(
        source="hadith",
        source_id=f"hadith:{collection}:{number}",
        text=text,
        scope=scope,
        metadata=metadata,
    )


def chunk_surah(entry: dict[str, Any], *, scope: str = DEFAULT_SCOPE) -> Chunk:
    """Chunk one surah reference-table entry — atomic, never split."""
    number = int(entry["number"])
    name = str(entry.get("name", "")).strip()
    arabic = str(entry.get("arabic_name", "")).strip()
    place = str(entry.get("revelation_place", "")).strip()
    count = entry.get("ayah_count")
    aliases = [str(a) for a in entry.get("aliases", [])]
    text_parts = [f"Surah {number}: {name}"]
    if arabic:
        text_parts.append(arabic)
    if place:
        text_parts.append(f"revealed in {place}")
    if count is not None:
        text_parts.append(f"{count} ayat")
    if aliases:
        text_parts.append("also known as " + ", ".join(aliases))
    text = ". ".join(text_parts)
    return make_chunk(
        source="surah_index",
        source_id=f"surah:{number}",
        text=text,
        scope=scope,
        metadata={
            "number": number,
            "name": name,
            "arabic_name": arabic,
            "revelation_place": place,
            "ayah_count": count,
            "aliases": aliases,
        },
    )


# ---------------------------------------------------------------------------
# Corpus iterators (deterministic, sorted order)
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sorted_ayah_keys(keys: list[str]) -> list[str]:
    """Sort ``"surah:ayah"`` keys numerically so order is stable and natural."""

    def key_fn(ref: str) -> tuple[int, int]:
        surah_str, _, ayah_str = ref.partition(":")
        try:
            return (int(surah_str), int(ayah_str))
        except ValueError:
            return (1 << 30, 0)

    return sorted(keys, key=key_fn)


def iter_quran_chunks(path: Path = QURAN_UTHMANI_PATH) -> Iterator[Chunk]:
    """Yield one atomic chunk per ayah in ``quran_uthmani.json``."""
    if not path.exists():
        return
    content = _load_json(path)
    ayat = content.get("ayat", {}) if isinstance(content, dict) else {}
    for ref in _sorted_ayah_keys(list(ayat.keys())):
        surah_str, _, ayah_str = ref.partition(":")
        try:
            surah, ayah = int(surah_str), int(ayah_str)
        except ValueError:
            continue
        record = ayat[ref]
        if isinstance(record, dict):
            yield chunk_ayah(surah, ayah, record)


def iter_surah_index_chunks(path: Path = SURAH_INDEX_PATH) -> Iterator[Chunk]:
    """Yield one atomic chunk per surah in ``quran/surah_index.json``."""
    if not path.exists():
        return
    entries = _load_json(path)
    if not isinstance(entries, list):
        return
    for entry in sorted(entries, key=lambda e: int(e.get("number", 0))):
        if isinstance(entry, dict) and "number" in entry:
            yield chunk_surah(entry)


def iter_hadith_chunks(data_dir: Path = HADITH_DIR) -> Iterator[Chunk]:
    """Yield one atomic chunk per hadith record across every collection file."""
    if not data_dir.exists():
        return
    for path in sorted(data_dir.glob("*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        collection = str(payload.get("collection") or path.stem)
        hadiths = payload.get("hadiths", [])
        if not isinstance(hadiths, list):
            continue
        for record in sorted(hadiths, key=lambda h: int(h.get("n", 0)) if isinstance(h, dict) else 0):
            if not isinstance(record, dict):
                continue
            chunk = chunk_hadith(collection, record)
            if chunk is not None:
                yield chunk


def iter_corpus_chunks(data_dir: Path = DATA_DIR) -> Iterator[Chunk]:
    """Yield every corpus chunk in a stable order.

    The order is fixed (surah index, then Quran ayat, then hadith by collection)
    so a full backfill is reproducible run to run. ``data_dir`` is honored so
    tests can point at a fixture corpus.
    """
    yield from iter_surah_index_chunks(data_dir / "quran" / "surah_index.json")
    yield from iter_quran_chunks(data_dir / "quran_uthmani.json")
    yield from iter_hadith_chunks(data_dir / "hadith")
