"""Advanced hybrid search: vector + keyword retrieval fused with Reciprocal Rank Fusion.

Deen-Bridge/dnb-ai#226. Two retrieval channels run independently and their
rankings are merged with RRF so a passage that both channels agree on outranks
one a single channel likes:

    fused_score(d) = SUM_channels  weight_c * 1 / (rrf_k + rank_c(d))

Design seams
------------
Both channels sit behind small Protocols (:class:`VectorBackend`,
:class:`KeywordBackend`). The shipped implementations are deliberately
dependency-light and fully offline so every code path works in CI with no API
keys and no infrastructure:

* :class:`InMemoryKeywordBackend` -- Okapi-style BM25 over an in-process corpus.
* :class:`HashingVectorBackend` -- deterministic hashing-trick pseudo-embeddings
  (numpy cosine similarity). These are NOT real semantics; they exist so the
  fusion pipeline is exercised end-to-end offline. Production adapters
  (pgvector, Pinecone, Qdrant, ...) backed by a real embedding model (e.g.
  ``models/text-embedding-004`` via the seam in ``semantic_cache.embed_text``)
  implement the same Protocol and drop in without touching callers.

Explainability
--------------
Every fused result carries ``match_type`` ("semantic" | "keyword" | "both"),
its per-channel rank and raw score, and the fused score, and the response
carries the query analysis (mode, weights, fired signals) so reviewers can see
why a passage surfaced.

A/B framework
-------------
Fusion strategies live in :data:`STRATEGY_REGISTRY` (currently
``rrf_plain_v1`` and ``rrf_weighted_v1``). Queries are assigned to a variant by
a deterministic hash bucket over ``(experiment, user_id, normalized query)`` --
same input always lands in the same bucket, so a user's experience is stable
across retries. Assignments increment content-free counters exposed via
:func:`get_ab_stats` and are logged following ``telemetry.py`` conventions
(counts and labels only, never query text).

Offline safety: no network calls anywhere in this module; configuration is
read lazily so importing it never requires GEMINI_API_KEY.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

import telemetry

logger = logging.getLogger(__name__)

CHANNEL_SEMANTIC = "semantic"
CHANNEL_KEYWORD = "keyword"

DEFAULT_RRF_K = 60
DEFAULT_TOP_K = 5
CHANNEL_OVERFETCH = 2
BUCKET_COUNT = 100
BUCKET_SPLIT = BUCKET_COUNT // 2


# ---------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoredPassage:
    """One retrievable unit plus the score its channel assigned it."""

    id: str
    text: str
    source: str
    reference: str | None = None
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


def passage_from_record(record: Mapping[str, Any]) -> ScoredPassage:
    """Build a ScoredPassage from a plain dict; unknown keys land in metadata."""
    known = {f.name for f in fields(ScoredPassage)}
    kwargs = {name: record[name] for name in known if name in record}
    kwargs.setdefault("id", str(record.get("id", "")))
    kwargs.setdefault("text", str(record.get("text", "")))
    kwargs.setdefault("source", str(record.get("source", "unknown")))
    metadata = {k: v for k, v in record.items() if k not in known}
    return ScoredPassage(metadata=metadata, **kwargs)


_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


# ---------------------------------------------------------------------------
# Retrieval backends
# ---------------------------------------------------------------------------
@runtime_checkable
class VectorBackend(Protocol):
    """Semantic similarity channel. Production adapters (pgvector, Pinecone,
    Qdrant, ...) implement this Protocol backed by a real embedding model."""

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[ScoredPassage]:
        """Return up to ``k`` passages most similar to ``query``, best first."""
        ...


@runtime_checkable
class KeywordBackend(Protocol):
    """Lexical matching channel (BM25/TF-IDF family)."""

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[ScoredPassage]:
        """Return up to ``k`` passages best lexically matching ``query``."""
        ...


class InMemoryKeywordBackend:
    """Okapi BM25 over an in-process corpus. Fully offline.

    Suitable for small/medium corpora; a production deployment would swap in
    Postgres full-text search or OpenSearch behind :class:`KeywordBackend`.
    """

    def __init__(
        self,
        passages: Sequence[ScoredPassage],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._passages = list(passages)
        self._k1 = k1
        self._b = b
        self._term_freqs: list[Counter[str]] = [Counter(_tokenize(p.text)) for p in self._passages]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            self._doc_freq.update(tf.keys())

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[ScoredPassage]:
        terms = list(dict.fromkeys(_tokenize(query)))
        if not terms or not self._passages:
            return []
        scores: list[tuple[float, int]] = []
        for idx, tf in enumerate(self._term_freqs):
            length_norm = (
                self._k1 * (1 - self._b + self._b * self._lengths[idx] / self._avg_len)
                if self._avg_len
                else self._k1
            )
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                df = self._doc_freq[term]
                idf = math.log(1.0 + (len(self._passages) - df + 0.5) / (df + 0.5))
                score += idf * (freq * (self._k1 + 1.0)) / (freq + length_norm)
            if score > 0.0:
                scores.append((score, idx))
        scores.sort(key=lambda pair: (-pair[0], self._passages[pair[1]].id))
        return [replace(self._passages[idx], score=score) for score, idx in scores[:max(k, 0)]]


class HashingVectorBackend:
    """Deterministic hashing-trick pseudo-embeddings with cosine similarity.

    Tokens are hashed (sha256, stable across processes -- unlike Python's
    salted ``hash()``) into a fixed-dimension signed bag-of-vectors that is
    L2-normalized. Similarity is plain dot product. This captures lexical
    overlap only; it is a stand-in so the fusion pipeline runs offline. Real
    deployments back :class:`VectorBackend` with pgvector/Pinecone/etc. fed by
    a trained embedding model.
    """

    def __init__(self, passages: Sequence[ScoredPassage], *, dim: int = 256) -> None:
        self._passages = list(passages)
        self._dim = dim
        self._matrix = (
            np.vstack([self.embed(p.text) for p in self._passages])
            if self._passages
            else np.zeros((0, dim), dtype=np.float32)
        )

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self._dim
            sign = 1.0 if digest[8] & 1 else -1.0
            vec[index] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[ScoredPassage]:
        if not self._passages:
            return []
        similarities = self._matrix @ self.embed(query)
        order = sorted(
            range(len(self._passages)),
            key=lambda i: (-similarities[i], self._passages[i].id),
        )
        results = [
            replace(self._passages[i], score=float(similarities[i]))
            for i in order
            if similarities[i] > 0.0
        ]
        return results[:max(k, 0)]


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------
Mode = Literal["keyword", "semantic", "balanced"]
MatchType = Literal["semantic", "keyword", "both"]

_MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "keyword": {CHANNEL_SEMANTIC: 0.25, CHANNEL_KEYWORD: 0.75},
    "semantic": {CHANNEL_SEMANTIC: 0.75, CHANNEL_KEYWORD: 0.25},
    "balanced": {CHANNEL_SEMANTIC: 0.5, CHANNEL_KEYWORD: 0.5},
}

# Explicit citation/citation-shaped markers push queries toward the lexical
# channel: users asking for "2:255" or a hadith reference want exact matches,
# not paraphrase neighborhoods.
_KEYWORD_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b\d{1,3}\s*:\s*\d{1,3}\b"), "verse_reference"),
    (re.compile(r"\b(surah|sura|ayah|ayat|ayatul|juz)\b", re.IGNORECASE), "surah_or_ayah_term"),
    (
        re.compile(
            r"\b(hadith|hadeeth|hadees|sunnah|sunna)\b"
            r"|\b(sahih\s+)?(bukhari|muslim|tirmidhi|nasai|abu\s+dawud|ibn\s+majah|muwatta)\b",
            re.IGNORECASE,
        ),
        "hadith_reference",
    ),
    (re.compile("\u201c[^\u201d]+\u201d|\u00ab[^\u00bb]+\u00bb|\"[^\"]{2,}\""), "quoted_text"),
)

# Abstract/thematic phrasing benefits from semantic neighborhoods.
_THEMATIC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwhat\s+(does|do)\s+islam\s+(say|says|teach)", re.IGNORECASE), "thematic_question"),
    (re.compile(r"\bguidance\s+(on|about|for)\b", re.IGNORECASE), "guidance_request"),
    (re.compile(r"\bmeaning\s+of\b", re.IGNORECASE), "meaning_request"),
    (re.compile(r"\bhow\s+(do|can|should)\s+i\b", re.IGNORECASE), "personal_how_to"),
    (re.compile(r"\bspiritual\b|\bheart\b|\biman\b|\btaqwa\b", re.IGNORECASE), "spiritual_theme"),
)


class QueryAnalysis(BaseModel):
    """Why a query got its channel mix; surfaced on responses for review."""

    mode: Mode
    weights: dict[str, float]
    signals: list[str]


def analyze_query(query: str, override: Mode | None = None) -> QueryAnalysis:
    """Classify a query's optimal channel mix.

    Rules detect explicit citation markers (verse refs like ``2:255``, surah/
    hadith terms, quoted strings) leading to a keyword-leaning mix, and
    abstract/thematic phrasing leading to a semantic-leaning mix; anything
    else stays balanced. ``override`` forces a mode regardless of signals.
    """
    if override is not None:
        if override not in _MODE_WEIGHTS:
            raise ValueError(f"override must be one of {sorted(_MODE_WEIGHTS)}, got {override!r}")
        return QueryAnalysis(mode=override, weights=dict(_MODE_WEIGHTS[override]), signals=["override"])

    signals: list[str] = []
    for pattern, signal in _KEYWORD_MARKERS:
        if pattern.search(query):
            signals.append(signal)
    if signals:
        return QueryAnalysis(mode="keyword", weights=dict(_MODE_WEIGHTS["keyword"]), signals=signals)

    for pattern, signal in _THEMATIC_PATTERNS:
        if pattern.search(query):
            signals.append(signal)
    if signals:
        return QueryAnalysis(mode="semantic", weights=dict(_MODE_WEIGHTS["semantic"]), signals=signals)

    return QueryAnalysis(mode="balanced", weights=dict(_MODE_WEIGHTS["balanced"]), signals=["no_strong_signals"])


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FusedPassage:
    """A passage with its cross-channel fusion provenance attached."""

    passage: ScoredPassage
    fused_score: float
    match_type: MatchType
    channel_ranks: dict[str, int]
    channel_scores: dict[str, float]


def _fuse_rankings(
    rankings: Mapping[str, Sequence[ScoredPassage]],
    k: int,
    weights: Mapping[str, float] | None,
) -> list[FusedPassage]:
    """Core RRF: score(d) = SUM_c w_c / (k + rank_c(d)), ranks 1-based.

    Channels missing from ``weights`` contribute with weight 1.0. Ties break
    deterministically by passage id so identical input always yields identical
    output ordering.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    accumulated: dict[str, dict[str, Any]] = {}
    for channel, ranking in rankings.items():
        weight = 1.0 if weights is None else float(weights.get(channel, 1.0))
        for position, passage in enumerate(ranking, start=1):
            slot = accumulated.setdefault(
                passage.id,
                {"passage": passage, "score": 0.0, "ranks": {}, "scores": {}, "channels": []},
            )
            slot["score"] += weight / (k + position)
            slot["ranks"][channel] = position
            slot["scores"][channel] = passage.score
            slot["channels"].append(channel)
    fused = [
        FusedPassage(
            passage=slot["passage"],
            fused_score=slot["score"],
            match_type=cast(MatchType, "both" if len(slot["channels"]) > 1 else slot["channels"][0]),
            channel_ranks=dict(slot["ranks"]),
            channel_scores=dict(slot["scores"]),
        )
        for slot in accumulated.values()
    ]
    fused.sort(key=lambda item: (-item.fused_score, item.passage.id))
    return fused


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ScoredPassage]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedPassage]:
    """Standard RRF: every channel weighs equally."""
    return _fuse_rankings(rankings, k, None)


def weighted_reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ScoredPassage]],
    weights: Mapping[str, float],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedPassage]:
    """RRF with per-channel weight multipliers."""
    return _fuse_rankings(rankings, k, weights)


# ---------------------------------------------------------------------------
# A/B strategy framework
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchStrategy:
    """A named fusion strategy enrolled in the experiment."""

    name: str
    description: str
    uses_weights: bool


STRATEGY_REGISTRY: Mapping[str, SearchStrategy] = {
    "rrf_plain_v1": SearchStrategy(
        name="rrf_plain_v1",
        description="Standard RRF; all channels weigh equally.",
        uses_weights=False,
    ),
    "rrf_weighted_v1": SearchStrategy(
        name="rrf_weighted_v1",
        description="Weighted RRF; channel contributions scaled by query-analysis weights.",
        uses_weights=True,
    ),
}
CONTROL_VARIANT = "rrf_plain_v1"
TREATMENT_VARIANT = "rrf_weighted_v1"
DEFAULT_EXPERIMENT = "hybrid_search_fusion_v1"

_AB_LOCK = threading.Lock()
_AB_COUNTERS: Counter[str] = Counter()


def assign_variant(experiment: str, query: str, user_id: str | None = None) -> tuple[str, int]:
    """Deterministically bucket a request into an A/B variant.

    Buckets hash ``(experiment, user_id, normalized query)`` through sha256, so
    the same input always yields the same variant -- across retries, processes,
    and restarts -- while distinct queries spread uniformly across buckets.
    Lower half of the bucket space is the control, upper half the treatment.
    """
    key = f"{experiment}|{user_id or 'anonymous'}|{' '.join(query.casefold().split())}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % BUCKET_COUNT
    variant = CONTROL_VARIANT if bucket < BUCKET_SPLIT else TREATMENT_VARIANT
    return variant, bucket


def record_assignment(variant: str, bucket: int) -> None:
    """Increment the variant counter and log a content-free assignment line.

    Follows telemetry.py conventions: counts and labels only, never query text.
    """
    with _AB_LOCK:
        _AB_COUNTERS[variant] += 1
    logger.info("hybrid_ab=%s", {"variant": variant, "bucket": bucket})


def get_ab_stats() -> dict[str, int]:
    with _AB_LOCK:
        return dict(_AB_COUNTERS)


def reset_ab_stats() -> None:
    with _AB_LOCK:
        _AB_COUNTERS.clear()


# ---------------------------------------------------------------------------
# Service orchestration
# ---------------------------------------------------------------------------
class HybridSearchHit(BaseModel):
    """One fused result, carrying its per-channel provenance for explainability."""

    id: str
    text: str
    source: str
    reference: str | None = None
    fused_score: float
    match_type: MatchType
    channel_ranks: dict[str, int]
    channel_scores: dict[str, float]


class HybridSearchResponse(BaseModel):
    query: str
    results: list[HybridSearchHit]
    analysis: QueryAnalysis
    strategy: str
    ab_bucket: int
    channels: list[str]
    rrf_k: int
    latency_ms: float


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=50)
    mode_override: Mode | None = Field(
        default=None,
        description="Force the channel mix: 'keyword', 'semantic', or 'balanced'.",
    )
    user_id: str | None = Field(default=None, max_length=128)
    filters: dict[str, str | list[str]] | None = Field(
        default=None,
        description="Post-fusion metadata predicates, e.g. {'source': ['quran','hadith']}.",
    )


def _matches_filters(passage: ScoredPassage, filters: Mapping[str, Any]) -> bool:
    """True when every filter holds. Values may be scalars or lists (membership)."""
    for field_name, allowed in filters.items():
        value = getattr(passage, field_name, None)
        if value is None:
            value = passage.metadata.get(field_name)
        if value is None:
            return False
        allowed_values = allowed if isinstance(allowed, (list, tuple, set, frozenset)) else [allowed]
        if not any(str(value) == str(option) for option in allowed_values):
            return False
    return True


def _normalized_pair(sem: float, kw: float) -> dict[str, float]:
    total = sem + kw
    if total <= 0:
        return {CHANNEL_SEMANTIC: 0.5, CHANNEL_KEYWORD: 0.5}
    return {CHANNEL_SEMANTIC: sem / total, CHANNEL_KEYWORD: kw / total}


class HybridSearcher:
    """Orchestrates dual-channel retrieval, fusion, filtering, and A/B assignment."""

    def __init__(
        self,
        vector_backend: VectorBackend,
        keyword_backend: KeywordBackend,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        top_k: int = DEFAULT_TOP_K,
        enable_semantic: bool = True,
        enable_keyword: bool = True,
        balanced_weights: tuple[float, float] = (0.5, 0.5),
        experiment: str = DEFAULT_EXPERIMENT,
    ) -> None:
        self.vector_backend = vector_backend
        self.keyword_backend = keyword_backend
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.enable_semantic = enable_semantic
        self.enable_keyword = enable_keyword
        self.balanced_weights = _normalized_pair(*balanced_weights)
        self.experiment = experiment

    def effective_weights(self, analysis: QueryAnalysis) -> dict[str, float]:
        """Analysis weights, except balanced mode honors configured baselines."""
        if analysis.mode == "balanced":
            return dict(self.balanced_weights)
        return dict(analysis.weights)

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        mode_override: Mode | None = None,
        user_id: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> HybridSearchResponse:
        started = time.perf_counter()
        final_k = k if k is not None else self.top_k
        analysis = analyze_query(query, mode_override)
        strategy, bucket = assign_variant(self.experiment, query, user_id)
        record_assignment(strategy, bucket)

        depth = max(final_k * CHANNEL_OVERFETCH, final_k)
        rankings: dict[str, list[ScoredPassage]] = {}
        if self.enable_semantic:
            rankings[CHANNEL_SEMANTIC] = self.vector_backend.search(query, depth)
        if self.enable_keyword:
            rankings[CHANNEL_KEYWORD] = self.keyword_backend.search(query, depth)

        fused: list[FusedPassage] = []
        if rankings:
            weights = self.effective_weights(analysis) if STRATEGY_REGISTRY[strategy].uses_weights else None
            fused = _fuse_rankings(rankings, self.rrf_k, weights)

        if filters:
            fused = [item for item in fused if _matches_filters(item.passage, filters)]
        fused = fused[:final_k]

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        trace = telemetry.current_trace.get()
        if trace is not None:
            trace.add_span("hybrid_search", elapsed_ms)

        return HybridSearchResponse(
            query=query,
            results=[
                HybridSearchHit(
                    id=item.passage.id,
                    text=item.passage.text,
                    source=item.passage.source,
                    reference=item.passage.reference,
                    fused_score=item.fused_score,
                    match_type=item.match_type,
                    channel_ranks=item.channel_ranks,
                    channel_scores=item.channel_scores,
                )
                for item in fused
            ],
            analysis=analysis,
            strategy=strategy,
            ab_bucket=bucket,
            channels=list(rankings),
            rrf_k=self.rrf_k,
            latency_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Default service (offline, lazily built)
# ---------------------------------------------------------------------------
_DEFAULT_SERVICE: HybridSearcher | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def load_default_passages() -> list[ScoredPassage]:
    """Index whatever the bundled Quran corpus holds; empty when unavailable.

    The bundled corpus ships with the repo and loads from local JSON, so this
    never touches the network. Production deployments register richer corpora
    (and real vector stores) by constructing a HybridSearcher directly.
    """

    def _sort_key(item: tuple[str, Mapping[str, Any]]) -> tuple[Any, ...]:
        parts = item[0].split(":")
        numbers = tuple(int(part) for part in parts if part.isdigit())
        return numbers if len(numbers) == len(parts) and numbers else (item[0],)

    try:
        from corpus import corpus as quran_corpus

        passages: list[ScoredPassage] = []
        for key, payload in sorted(quran_corpus.ayat.items(), key=_sort_key):
            english = str(payload.get("english", ""))
            arabic = str(payload.get("arabic", ""))
            text = " ".join(part for part in (english, arabic) if part).strip()
            if text:
                passages.append(ScoredPassage(id=f"quran:{key}", text=text, source="quran", reference=key))
        return passages
    except Exception as exc:
        logger.warning("hybrid_search default corpus unavailable (%s); starting empty", exc)
        return []


def build_default_service() -> HybridSearcher:
    """Build a searcher from Settings knobs, falling back to offline defaults.

    Config is read lazily (and tolerantly) so importing this module never
    requires GEMINI_API_KEY or any other secret.
    """
    rrf_k, top_k = DEFAULT_RRF_K, DEFAULT_TOP_K
    enable_semantic = enable_keyword = True
    balanced = (0.5, 0.5)
    try:
        from config import get_settings

        settings = get_settings()
        rrf_k = max(1, int(settings.hybrid_rrf_k))
        top_k = min(50, max(1, int(settings.hybrid_top_k)))
        enable_semantic = bool(settings.hybrid_enable_semantic_channel)
        enable_keyword = bool(settings.hybrid_enable_keyword_channel)
        balanced = (float(settings.hybrid_semantic_weight), float(settings.hybrid_keyword_weight))
    except Exception as exc:
        logger.info("hybrid_search using built-in defaults (%s)", type(exc).__name__)
    passages = load_default_passages()
    return HybridSearcher(
        vector_backend=HashingVectorBackend(passages),
        keyword_backend=InMemoryKeywordBackend(passages),
        rrf_k=rrf_k,
        top_k=top_k,
        enable_semantic=enable_semantic,
        enable_keyword=enable_keyword,
        balanced_weights=balanced,
    )


def get_default_service() -> HybridSearcher:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        with _DEFAULT_SERVICE_LOCK:
            if _DEFAULT_SERVICE is None:
                _DEFAULT_SERVICE = build_default_service()
    return _DEFAULT_SERVICE


def handle_hybrid_search(request: HybridSearchRequest) -> HybridSearchResponse:
    """Route entry point: keep main.py's handler a thin delegation."""
    return get_default_service().search(
        request.query,
        k=request.k,
        mode_override=request.mode_override,
        user_id=request.user_id,
        filters=request.filters,
    )
