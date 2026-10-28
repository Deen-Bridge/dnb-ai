"""Database query optimization — profile queries and flag anti-patterns offline.

Why this exists
---------------
Issue #217 asks for "comprehensive database query optimization": slow-query
profiling, index recommendations, N+1 detection, connection-pool efficiency and
a cache-hit-rate view. A real answer to all of that would drag in
``pg_stat_statements``, a live PostgreSQL, load-testing infrastructure and a
migration framework — none of which belong in this FastAPI service or its CI.

This module delivers the *functional* core of that request in pure, deterministic
Python with no new dependencies and no network calls:

1. **Static analysis** (:func:`analyze_query`) parses a SQL statement with simple,
   well-tested heuristics and reports the anti-patterns that make queries slow —
   ``SELECT *``, unbounded scans, leading-wildcard ``LIKE``, functions wrapped
   around indexed columns, implicit comma joins, ``OR`` chains, ``NOT IN`` and so
   on — each with a severity and a plain-language remediation. It also proposes
   B-tree indexes for the columns that appear in ``WHERE`` / ``JOIN`` / ``ORDER BY``.
2. **Runtime profiling** (:class:`QueryProfiler`) accepts execution samples and
   computes P50/P95/P99 latency, a slow-query log against a configurable
   threshold, throughput, cache-hit rate, connection-pool efficiency and — by
   fingerprinting statements with their literals stripped — N+1 patterns (the
   same shape run many times in one window).

Everything is in-memory and side-effect-free at import time: nothing here touches
a live service, which is what lets the app boot in CI and keeps ``mypy .`` green.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/db-optimizer", tags=["db-optimizer"])

# ---------------------------------------------------------------------------
# Turkish language support
# ---------------------------------------------------------------------------


def turkish_lower(text: str) -> str:
    """Turkish-aware lowercasing that respects the dotted/dotless I distinction."""
    return text.replace("I", "ı").replace("İ", "i").lower()


_TURKISH_SUFFIXES = (
    "lar",
    "ler",
    "dan",
    "den",
    "tan",
    "ten",
    "yla",
    "yle",
    "sin",
    "sın",
    "sun",
    "sün",
    "dir",
    "dır",
    "dur",
    "dür",
    "tir",
    "tır",
    "tur",
    "tür",
    "ken",
    "ince",
    "erek",
    "arak",
)

TURKISH_ISLAMIC_TERMS: dict[str, tuple[str, ...]] = {
    "namaz": ("salat", "salât"),
    "oruç": ("savm", "sawm"),
    "zekat": ("zakat", "zekât"),
    "hac": ("hajj", "hacc"),
    "peygamber": ("nebi", "resul", "rasûl"),
    "melek": ("malaik", "melekût"),
    "kuran": ("qur'an", "kur'an", "quran"),
    "hadis": ("hadith", "hadîs"),
    "sünnet": ("sunna", "sünnet"),
    "kelam": ("kalam", "kelâm"),
    "tasavvuf": ("sufism", "tasawwuf"),
    "tevhid": ("tawhid", "tevhit"),
    "ibadet": ("ibada", "ibâdet"),
    "sevap": ("thawab", "sevâb"),
    "günah": ("dhanb", "gunah", "günâh"),
    "helal": ("halal", "helâl"),
    "haram": ("haram", "harâm"),
    "dua": ("du'a", "duâ"),
    "iman": ("iman", "îman"),
    "islam": ("islam", "islâm"),
}

_OTTOMAN_AFFIXES = ("ullah", "ül", "il", "el", "al", "ibn", "bin", "zade", "efendi")


def turkish_tokenize(text: str) -> list[str]:
    """Split Turkish text into lowercase tokens, respecting Turkish characters."""
    return re.findall(r"[a-z0-9çğıöşüâîû']+", turkish_lower(text))


def strip_turkish_suffixes(word: str) -> str:
    """Remove common agglutinative suffixes for a conservative Turkish stem."""
    stem = word
    changed = True
    while changed and len(stem) > 3:
        changed = False
        for suffix in _TURKISH_SUFFIXES:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= 2:
                stem = stem[: -len(suffix)]
                changed = True
                break
    return stem


def normalize_turkish_islamic_term(term: str) -> tuple[str, ...] | str:
    """Return known Arabic/Ottoman equivalents for a Turkish Islamic term."""
    normalized = turkish_lower(term).replace("'", "").replace(" ", "")
    return TURKISH_ISLAMIC_TERMS.get(normalized, turkish_lower(term))


def expand_turkish_islamic_query(text: str) -> str:
    """Expand Turkish Islamic terms with Arabic/Ottoman equivalents for recall."""
    tokens = turkish_tokenize(text)
    expanded: list[str] = []
    for token in tokens:
        root = strip_turkish_suffixes(token)
        equivalents = normalize_turkish_islamic_term(root)
        if isinstance(equivalents, tuple):
            expanded.append(token)
            expanded.extend(equivalents)
        else:
            expanded.append(token)
    return " ".join(expanded)


def detect_ottoman_turkish_influence(text: str) -> list[str]:
    """Return Ottoman affixes found in a religious text fragment."""
    lowered = turkish_lower(text)
    return [affix for affix in _OTTOMAN_AFFIXES if affix in lowered]


# Cross-Surah Reference System


class ReferenceType(str, Enum):
    REPEATED_STORY = "repeated_story"
    RELATED_RULING = "related_ruling"
    SIMILAR_TEACHING = "similar_teaching"
    PROPHET_NARRATIVE = "prophet_narrative"
    THEOLOGICAL_PARALLEL = "theological_parallel"
    SCHOLARLY_CROSS_REF = "scholarly_cross_ref"


class CrossReferenceRecord(BaseModel):
    source_surah: int = Field(..., ge=1, le=114)
    source_ayah: int = Field(..., ge=1)
    target_surah: int = Field(..., ge=1, le=114)
    target_ayah: int = Field(..., ge=1)
    reference_type: ReferenceType
    context: str = ""
    commentary: str = ""
    attributes: list[str] = Field(default_factory=list)


class CrossReferenceDatabase:
    def __init__(self) -> None:
        self._records: list[CrossReferenceRecord] = []
        self._by_surah: dict[int, list[int]] = {}
        self._by_type: dict[str, list[int]] = {}
        self._by_attribute: dict[str, list[int]] = {}

    def add(self, record: CrossReferenceRecord) -> None:
        idx = len(self._records)
        self._records.append(record)
        for surah in (record.source_surah, record.target_surah):
            self._by_surah.setdefault(surah, []).append(idx)
        self._by_type.setdefault(record.reference_type.value, []).append(idx)
        for attribute in record.attributes:
            self._by_attribute.setdefault(attribute, []).append(idx)

    def query(
        self,
        surah: int | None = None,
        reference_type: ReferenceType | None = None,
        attributes: list[str] | None = None,
    ) -> list[CrossReferenceRecord]:
        candidates = set(range(len(self._records)))
        if surah is not None:
            candidates &= set(self._by_surah.get(surah, []))
        if reference_type is not None:
            candidates &= set(self._by_type.get(reference_type.value, []))
        if attributes:
            selected: set[int] = set()
            for attribute in attributes:
                selected.update(self._by_attribute.get(attribute, []))
            candidates &= selected
        return [self._records[i] for i in sorted(candidates)]

    def references_for(self, surah: int, ayah: int) -> list[CrossReferenceRecord]:
        found = []
        for idx in self._by_surah.get(surah, []):
            record = self._records[idx]
            if (record.source_surah == surah and record.source_ayah == ayah) or (
                record.target_surah == surah and record.target_ayah == ayah
            ):
                found.append(record)
        return found


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "by",
        "that",
        "this",
        "these",
        "those",
        "it",
        "as",
        "at",
        "from",
        "which",
        "who",
    }
)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9']+", text.lower()) if token not in _STOPWORDS}


# Latency at or above this (milliseconds) counts a query as "slow". Mirrors the
# issue's "zero queries exceeding 500ms" success criterion.
DEFAULT_SLOW_QUERY_MS = 500.0

# When the same query fingerprint runs at least this many times inside one
# profiling window, we treat it as an N+1 pattern worth batching.
N_PLUS_ONE_THRESHOLD = 5


# ---------------------------------------------------------------------------
# SQL normalization
# ---------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_NUMBER_LITERAL = re.compile(r"\b\d+(?:\.\d+)?\b")
_IN_LIST = re.compile(r"\bin\s*\([^)]*\)", re.IGNORECASE)
# A column reference such as ``t.col`` or ``col`` — deliberately conservative so
# it does not swallow SQL keywords handled separately.
_COLUMN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?")

_SQL_KEYWORDS = frozenset(
    {
        "select",
        "from",
        "where",
        "and",
        "or",
        "not",
        "in",
        "is",
        "null",
        "like",
        "ilike",
        "join",
        "inner",
        "left",
        "right",
        "outer",
        "full",
        "cross",
        "on",
        "as",
        "order",
        "group",
        "by",
        "having",
        "limit",
        "offset",
        "distinct",
        "asc",
        "desc",
        "between",
        "exists",
        "case",
        "when",
        "then",
        "else",
        "end",
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "true",
        "false",
        "insert",
        "into",
        "values",
        "update",
        "set",
        "delete",
    }
)


def normalize_sql(sql: str) -> str:
    """Collapse whitespace and lowercase a statement for readable comparison."""
    return _WHITESPACE.sub(" ", sql.strip()).lower()


def fingerprint_sql(sql: str) -> str:
    """Return a literal-free shape of ``sql`` so N+1 repeats collapse together.

    String and numeric literals become ``?`` and ``IN (...)`` lists become
    ``in (?)``, so ``WHERE id = 1`` and ``WHERE id = 2`` share one fingerprint.
    """
    working = _STRING_LITERAL.sub("?", sql)
    working = _IN_LIST.sub("in (?)", working)
    working = _NUMBER_LITERAL.sub("?", working)
    return normalize_sql(working)


def _extract_columns(fragment: str) -> list[str]:
    """Pull distinct, non-keyword column identifiers from a SQL fragment."""
    seen: dict[str, None] = {}
    for match in _COLUMN.finditer(fragment):
        token = match.group(0)
        if token.lower() in _SQL_KEYWORDS:
            continue
        if token.replace(".", "").isdigit():
            continue
        seen.setdefault(token, None)
    return list(seen)


def _clause(sql_lower: str, start_kw: str, end_kws: tuple[str, ...]) -> str:
    """Return the text of ``start_kw``'s clause, up to the first ``end_kws`` token."""
    start = sql_lower.find(start_kw)
    if start == -1:
        return ""
    start += len(start_kw)
    end = len(sql_lower)
    for kw in end_kws:
        idx = sql_lower.find(kw, start)
        if idx != -1:
            end = min(end, idx)
    return sql_lower[start:end]


# ---------------------------------------------------------------------------
# Static query analysis
# ---------------------------------------------------------------------------


class AntiPattern(BaseModel):
    """One detected inefficiency in a query, with how to fix it."""

    code: str = Field(..., description="Stable machine-readable identifier")
    severity: int = Field(..., ge=1, le=5, description="1 (minor) to 5 (severe)")
    message: str = Field(..., description="What is wrong")
    recommendation: str = Field(..., description="How to fix it")


class IndexRecommendation(BaseModel):
    """A proposed index to support a query's filter/join/sort columns."""

    columns: list[str] = Field(..., description="Columns the index should cover")
    reason: str = Field(..., description="Why this index helps")
    ddl_hint: str = Field(..., description="Illustrative CREATE INDEX statement")


class QueryAnalysis(BaseModel):
    """Full static-analysis result for a single SQL statement."""

    fingerprint: str = Field(..., description="Literal-free query shape")
    statement_type: str = Field(..., description="select / insert / update / delete / other")
    anti_patterns: list[AntiPattern] = Field(default_factory=list)
    index_recommendations: list[IndexRecommendation] = Field(default_factory=list)
    estimated_cost: int = Field(..., ge=0, le=100, description="0 (cheap) to 100 (expensive) heuristic")

    @property
    def is_optimized(self) -> bool:
        """True when no anti-pattern of severity 3 or higher was found."""
        return all(ap.severity < 3 for ap in self.anti_patterns)


# A call ``name(`` inside a WHERE clause. Function-wrapped columns cannot use a
# plain column index. These names are not column-wrapping functions, so ignore
# them (sub-query / list constructs, not ``lower(col)``-style wrapping).
_FUNCTION_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")
_NON_WRAPPING_CALLS = frozenset({"in", "exists", "any", "all", "values", "not"})


def _has_function_on_column(where_clause: str) -> bool:
    """True when a function wraps a column in the WHERE clause (defeats its index)."""
    return any(match.group(1) not in _NON_WRAPPING_CALLS for match in _FUNCTION_CALL.finditer(where_clause))


def _statement_type(sql_lower: str) -> str:
    for kind in ("select", "insert", "update", "delete"):
        if sql_lower.startswith(kind):
            return kind
    return "other"


def analyze_query(sql: str) -> QueryAnalysis:
    """Statically analyze ``sql`` for anti-patterns and index opportunities.

    Pure and deterministic: the same statement always yields the same result.
    """
    normalized = normalize_sql(sql)
    stmt_type = _statement_type(normalized)
    anti_patterns: list[AntiPattern] = []

    where_clause = _clause(normalized, "where", ("group by", "order by", "limit", "having"))
    order_clause = _clause(normalized, "order by", ("limit", "offset"))
    on_clause = _clause(normalized, " on ", ("where", "group by", "order by", "limit"))

    if re.search(r"select\s+\*", normalized):
        anti_patterns.append(
            AntiPattern(
                code="SELECT_STAR",
                severity=2,
                message="Query selects all columns with SELECT *.",
                recommendation="Project only the columns you need so the planner can use covering indexes.",
            )
        )

    if stmt_type in {"select", "update", "delete"} and "where" not in normalized:
        anti_patterns.append(
            AntiPattern(
                code="FULL_TABLE_SCAN",
                severity=5 if stmt_type in {"update", "delete"} else 4,
                message=f"{stmt_type.upper()} has no WHERE clause and scans the whole table.",
                recommendation="Add a selective WHERE clause (or an explicit guard) to avoid a full-table operation.",
            )
        )

    if re.search(r"like\s+'%", where_clause):
        anti_patterns.append(
            AntiPattern(
                code="LEADING_WILDCARD",
                severity=3,
                message="LIKE pattern begins with a wildcard, which cannot use a B-tree index.",
                recommendation="Anchor the pattern or use a trigram / full-text index for substring search.",
            )
        )

    if _has_function_on_column(where_clause):
        anti_patterns.append(
            AntiPattern(
                code="FUNCTION_ON_COLUMN",
                severity=3,
                message="A function wraps a column in the WHERE clause, disabling the plain column index.",
                recommendation="Rewrite so the column is bare, or add a matching expression/functional index.",
            )
        )

    if " or " in where_clause:
        anti_patterns.append(
            AntiPattern(
                code="OR_CONDITION",
                severity=2,
                message="OR in the WHERE clause can defeat index usage.",
                recommendation="Consider UNION of indexed branches or an IN (...) list where the columns match.",
            )
        )

    if "not in" in where_clause or "!=" in where_clause or "<>" in where_clause:
        anti_patterns.append(
            AntiPattern(
                code="NEGATION_FILTER",
                severity=2,
                message="Negation (NOT IN / != / <>) is rarely sargable and forces a scan.",
                recommendation="Prefer positive, indexable predicates or an anti-join where possible.",
            )
        )

    # Implicit comma join: multiple tables in FROM with no JOIN keyword.
    from_clause = _clause(normalized, "from", ("where", "group by", "order by", "limit", "having"))
    if "," in from_clause and "join" not in normalized:
        anti_patterns.append(
            AntiPattern(
                code="IMPLICIT_JOIN",
                severity=3,
                message="Tables are comma-joined in FROM without explicit JOIN ... ON.",
                recommendation="Use explicit JOIN ... ON so the join condition is clear and index-friendly.",
            )
        )

    if stmt_type == "select" and "limit" not in normalized:
        anti_patterns.append(
            AntiPattern(
                code="UNBOUNDED_RESULT",
                severity=2,
                message="SELECT has no LIMIT and may return an unbounded result set.",
                recommendation="Add a LIMIT (with keyset pagination) to cap the rows returned.",
            )
        )

    if "order by" in normalized and "limit" not in normalized:
        anti_patterns.append(
            AntiPattern(
                code="SORT_WITHOUT_LIMIT",
                severity=2,
                message="ORDER BY without LIMIT sorts the entire result set.",
                recommendation="Pair ORDER BY with LIMIT, backed by an index on the sort columns.",
            )
        )

    index_recommendations = _recommend_indexes(where_clause, on_clause, order_clause)
    estimated_cost = _estimate_cost(anti_patterns)

    return QueryAnalysis(
        fingerprint=fingerprint_sql(sql),
        statement_type=stmt_type,
        anti_patterns=anti_patterns,
        index_recommendations=index_recommendations,
        estimated_cost=estimated_cost,
    )


def _recommend_indexes(where_clause: str, on_clause: str, order_clause: str) -> list[IndexRecommendation]:
    recommendations: list[IndexRecommendation] = []
    filter_cols = _extract_columns(where_clause)
    if filter_cols:
        recommendations.append(
            IndexRecommendation(
                columns=filter_cols,
                reason="Columns used in WHERE benefit from an index that supports the filter.",
                ddl_hint=f"CREATE INDEX ON <table> ({', '.join(filter_cols)});",
            )
        )
    join_cols = _extract_columns(on_clause)
    if join_cols:
        recommendations.append(
            IndexRecommendation(
                columns=join_cols,
                reason="Join keys should be indexed on both sides to avoid a hash/merge over full tables.",
                ddl_hint=f"CREATE INDEX ON <table> ({', '.join(join_cols)});",
            )
        )
    sort_cols = _extract_columns(order_clause)
    if sort_cols:
        recommendations.append(
            IndexRecommendation(
                columns=sort_cols,
                reason="An index on the ORDER BY columns lets the planner skip an explicit sort.",
                ddl_hint=f"CREATE INDEX ON <table> ({', '.join(sort_cols)});",
            )
        )
    return recommendations


def _estimate_cost(anti_patterns: list[AntiPattern]) -> int:
    if not anti_patterns:
        return 5
    total = sum(ap.severity * 8 for ap in anti_patterns)
    return min(100, 5 + total)


# ---------------------------------------------------------------------------
# Runtime profiling
# ---------------------------------------------------------------------------


@dataclass
class _Sample:
    fingerprint: str
    duration_ms: float
    rows: int
    cache_hit: bool
    timestamp: float


class QuerySample(BaseModel):
    """One observed query execution, submitted to the profiler."""

    sql: str = Field(..., min_length=1, description="The executed statement")
    duration_ms: float = Field(..., ge=0.0, description="Wall-clock execution time in milliseconds")
    rows: int = Field(default=0, ge=0, description="Rows returned or affected")
    cache_hit: bool = Field(default=False, description="Whether the result came from cache")


class LatencyPercentiles(BaseModel):
    """P50/P95/P99 (and max) latency over the recorded samples, in ms."""

    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    max: float = 0.0


class SlowQuery(BaseModel):
    """A recorded execution that breached the slow-query threshold."""

    fingerprint: str
    duration_ms: float
    rows: int


class NPlusOnePattern(BaseModel):
    """A fingerprint executed often enough to look like an N+1 pattern."""

    fingerprint: str
    count: int
    recommendation: str = "Batch these into one query (IN (...) / JOIN) or add eager loading."


class ProfilerStats(BaseModel):
    """Aggregate profiling report across all recorded samples."""

    total_queries: int
    latency: LatencyPercentiles
    slow_query_count: int
    slow_queries: list[SlowQuery]
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0)
    throughput_qps: float = Field(..., ge=0.0, description="Queries per second over the observed span")
    n_plus_one: list[NPlusOnePattern]


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of ``values`` (``pct`` in 0..100). Empty -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    rank = -(-len(ordered) * pct // 100)  # ceil division, 1-based rank
    index = min(len(ordered), int(rank)) - 1
    return ordered[index]


def pool_efficiency(active: int, idle: int, waiting: int) -> float:
    """Connection-pool efficiency in 0..1.

    Efficiency rewards connections doing work (``active``) and penalizes callers
    stuck waiting for a free connection. With no connections at all it is 1.0
    (nothing is being wasted).
    """
    total = active + idle
    if total <= 0:
        return 1.0
    utilization = active / total
    contention_penalty = waiting / (total + waiting) if waiting else 0.0
    return max(0.0, min(1.0, utilization * (1.0 - contention_penalty)))


class QueryProfiler:
    """Thread-safe in-memory collector of query execution samples."""

    def __init__(self, slow_query_ms: float = DEFAULT_SLOW_QUERY_MS) -> None:
        self.slow_query_ms = slow_query_ms
        self._samples: list[_Sample] = []
        self._lock = Lock()

    def record(self, sample: QuerySample) -> _Sample:
        """Store one execution sample and return the stored record."""
        stored = _Sample(
            fingerprint=fingerprint_sql(sample.sql),
            duration_ms=sample.duration_ms,
            rows=sample.rows,
            cache_hit=sample.cache_hit,
            timestamp=time.monotonic(),
        )
        with self._lock:
            self._samples.append(stored)
        return stored

    def reset(self) -> None:
        """Drop every recorded sample."""
        with self._lock:
            self._samples.clear()

    def slow_queries(self) -> list[SlowQuery]:
        with self._lock:
            samples = list(self._samples)
        slow = [s for s in samples if s.duration_ms >= self.slow_query_ms]
        slow.sort(key=lambda s: s.duration_ms, reverse=True)
        return [SlowQuery(fingerprint=s.fingerprint, duration_ms=s.duration_ms, rows=s.rows) for s in slow]

    def n_plus_one(self, threshold: int = N_PLUS_ONE_THRESHOLD) -> list[NPlusOnePattern]:
        with self._lock:
            samples = list(self._samples)
        counts: dict[str, int] = {}
        for s in samples:
            counts[s.fingerprint] = counts.get(s.fingerprint, 0) + 1
        patterns = [NPlusOnePattern(fingerprint=fp, count=count) for fp, count in counts.items() if count >= threshold]
        patterns.sort(key=lambda p: p.count, reverse=True)
        return patterns

    def stats(self) -> ProfilerStats:
        with self._lock:
            samples = list(self._samples)
        durations = [s.duration_ms for s in samples]
        latency = LatencyPercentiles(
            p50=percentile(durations, 50),
            p95=percentile(durations, 95),
            p99=percentile(durations, 99),
            max=max(durations) if durations else 0.0,
        )
        cache_hits = sum(1 for s in samples if s.cache_hit)
        cache_hit_rate = cache_hits / len(samples) if samples else 0.0

        throughput = 0.0
        if len(samples) >= 2:
            span = samples[-1].timestamp - samples[0].timestamp
            throughput = len(samples) / span if span > 0 else 0.0

        slow = self.slow_queries()
        return ProfilerStats(
            total_queries=len(samples),
            latency=latency,
            slow_query_count=len(slow),
            slow_queries=slow,
            cache_hit_rate=cache_hit_rate,
            throughput_qps=throughput,
            n_plus_one=self.n_plus_one(),
        )


@dataclass
class _ProfilerRegistry:
    profiler: QueryProfiler | None = None
    lock: Lock = field(default_factory=Lock)


_registry = _ProfilerRegistry()


def get_profiler() -> QueryProfiler:
    """Return the process-wide profiler, creating it on first use."""
    if _registry.profiler is None:
        with _registry.lock:
            if _registry.profiler is None:
                _registry.profiler = QueryProfiler()
    return _registry.profiler


# ---------------------------------------------------------------------------
# API models and endpoints
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    sql: str = Field(..., min_length=1, description="SQL statement to analyze")


class PoolRequest(BaseModel):
    active: int = Field(..., ge=0, description="Connections currently executing")
    idle: int = Field(..., ge=0, description="Open but idle connections")
    waiting: int = Field(default=0, ge=0, description="Callers waiting for a connection")


class PoolResponse(BaseModel):
    efficiency: float = Field(..., ge=0.0, le=1.0)
    healthy: bool = Field(..., description="True when efficiency meets the 0.85 target")


@router.post("/analyze", response_model=QueryAnalysis)
async def analyze_endpoint(request: AnalyzeRequest) -> QueryAnalysis:
    """Statically analyze one SQL statement for anti-patterns and index gaps."""
    return analyze_query(request.sql)


@router.post("/record", response_model=QuerySample)
async def record_endpoint(sample: QuerySample) -> QuerySample:
    """Record one query execution sample for later profiling."""
    get_profiler().record(sample)
    return sample


@router.get("/stats", response_model=ProfilerStats)
async def stats_endpoint() -> ProfilerStats:
    """Aggregate latency percentiles, slow queries and N+1 patterns so far."""
    return get_profiler().stats()


@router.post("/reset")
async def reset_endpoint() -> dict[str, str]:
    """Clear all recorded samples (useful between benchmark runs)."""
    get_profiler().reset()
    return {"status": "reset"}


@router.post("/pool-efficiency", response_model=PoolResponse)
async def pool_efficiency_endpoint(request: PoolRequest) -> PoolResponse:
    """Score connection-pool efficiency against the 85% target from the issue."""
    score = pool_efficiency(request.active, request.idle, request.waiting)
    return PoolResponse(efficiency=score, healthy=score >= 0.85)


# ---------------------------------------------------------------------------
# Ayah relationship mapping
# ---------------------------------------------------------------------------

_QURANIC_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "of",
        "to",
        "in",
        "that",
        "it",
        "is",
        "for",
        "on",
        "with",
        "as",
        "by",
        "are",
        "be",
        "this",
        "those",
        "who",
        "whom",
        "which",
        "from",
        "at",
        "an",
        "or",
        "we",
        "you",
        "they",
        "he",
        "she",
        "them",
        "his",
        "her",
        "their",
        "our",
        "your",
        "will",
        "shall",
        "may",
        "all",
        "not",
        "but",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "then",
        "when",
        "what",
        "why",
        "how",
        "so",
        "if",
        "there",
        "where",
        "than",
        "too",
        "very",
        "can",
        "would",
        "should",
        "could",
        "must",
        "unto",
        "thy",
        "thou",
        "ye",
        "hath",
        "doth",
        "art",
        "shalt",
        "wilt",
        "verily",
        "indeed",
        "surely",
        "lord",
        "allah",
        "god",
        "people",
        "say",
        "said",
        "says",
        "o",
        "yea",
        "nay",
        "also",
        "still",
        "even",
        "yet",
        "thus",
        "among",
        "before",
        "after",
        "until",
        "against",
        "upon",
        "into",
        "over",
        "under",
        "through",
        "during",
        "within",
        "without",
        "about",
        "between",
        "because",
        "such",
        "same",
        "own",
        "every",
        "each",
        "both",
        "some",
        "any",
        "no",
        "one",
        "two",
        "many",
        "much",
        "more",
        "most",
        "other",
        "another",
        "these",
        "therein",
        "thereafter",
        "deem",
    }
)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z0-9']+", text.lower()) if w not in _QURANIC_STOPWORDS and len(w) > 1}


class AyahText(BaseModel):
    """One Quranic ayah with thematic labels used for similarity detection."""

    surah: int = Field(..., ge=1, le=114)
    ayah: int = Field(..., ge=1)
    text: str = Field(..., min_length=1)
    topics: list[str] = Field(default_factory=list)
    scholarly_note: str | None = None


class AyahRelationshipEdge(BaseModel):
    source: str
    target: str
    relationship_type: str
    strength: float = Field(..., ge=0.0, le=1.0)
    scholarly_note: str | None = None


class AyahCorpus(BaseModel):
    ayahs: list[AyahText] = Field(..., min_length=1)


class BuildGraphResult(BaseModel):
    ayahs: int
    relationships: int


class IndirectPath(BaseModel):
    path: list[str]
    average_strength: float


class RelatedAyahResult(BaseModel):
    source: str
    direct: list[AyahRelationshipEdge]
    indirect: list[IndirectPath] = Field(default_factory=list)


class RelationshipGraphNode(BaseModel):
    id: str
    surah: int
    ayah: int
    topics: list[str]


class RelationshipGraph(BaseModel):
    nodes: list[RelationshipGraphNode]
    edges: list[AyahRelationshipEdge]


def _classify_relationship(a: AyahText, b: AyahText, jaccard: float, topic_overlap: float) -> str:
    a_tokens = _tokens(a.text)
    b_tokens = _tokens(b.text)
    if not a_tokens or not b_tokens:
        return "parallel_teaching"
    contrast = {"but", "however", "yet", "while", "whereas", "except", "without"}
    example = {"example", "like", "likeness", "parable", "similitude"}
    if topic_overlap >= 0.65 and jaccard >= 0.25:
        return "parallel_teaching"
    if a_tokens & contrast or b_tokens & contrast:
        return "contrast"
    if a_tokens & example or b_tokens & example:
        return "example"
    if jaccard >= 0.15:
        return "elaboration"
    return "complement"


# ---------------------------------------------------------------------------
# Intelligent agent task decomposition
# ---------------------------------------------------------------------------


class TaskComplexity(str, Enum):
    """Coarse bucket for how hard a query is to decompose."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class AgentCapability(str, Enum):
    """Capabilities an agent may advertise for task matching."""

    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE = "code"
    MATH = "math"
    SQL = "sql"
    VERIFICATION = "verification"
    GENERAL = "general"


class SubTaskStatus(str, Enum):
    """Lifecycle states for a decomposed sub-task."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SubTask(BaseModel):
    """One unit of work produced by decomposition."""

    id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    required_capabilities: list[AgentCapability] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    estimated_cost: int = Field(default=1, ge=0, description="Relative effort units")
    status: SubTaskStatus = SubTaskStatus.PENDING
    assigned_agent: str | None = None
    result: str | None = None


class AgentProfile(BaseModel):
    """An agent that can be assigned sub-tasks."""

    id: str = Field(..., min_length=1)
    capabilities: list[AgentCapability] = Field(default_factory=list)
    capacity: int = Field(default=1, ge=1, description="Max concurrent sub-tasks")
    reliability: float = Field(default=1.0, ge=0.0, le=1.0)


class QueryComplexityAssessment(BaseModel):
    """Result of analyzing a query's complexity before decomposition."""

    complexity: TaskComplexity
    score: int = Field(..., ge=0, le=100)
    signals: list[str] = Field(default_factory=list)
    estimated_sub_tasks: int = Field(..., ge=1)


class DecompositionPlan(BaseModel):
    """A full plan: sub-tasks, dependency graph, and an execution schedule."""

    query: str
    assessment: QueryComplexityAssessment
    sub_tasks: list[SubTask]
    execution_batches: list[list[str]] = Field(default_factory=list)
    parallelizable: bool = False
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


_COMPLEXITY_SIGNALS: tuple[tuple[str, int, str], ...] = (
    ("compare", 15, "comparison"),
    ("contrast", 15, "comparison"),
    ("analyze", 12, "analysis"),
    ("explain", 10, "explanation"),
    ("summarize", 10, "summarization"),
    ("translate", 12, "translation"),
    ("calculate", 12, "computation"),
    ("prove", 18, "proof"),
    ("derive", 18, "derivation"),
    ("list", 6, "enumeration"),
    ("find", 6, "retrieval"),
    ("retrieve", 6, "retrieval"),
    ("verify", 10, "verification"),
    ("validate", 10, "verification"),
    ("then", 8, "sequence"),
    ("after", 6, "sequence"),
    ("before", 6, "sequence"),
    ("and", 4, "conjunction"),
)

_CAPABILITY_KEYWORDS: dict[AgentCapability, tuple[str, ...]] = {
    AgentCapability.RETRIEVAL: ("find", "retrieve", "search", "lookup", "fetch"),
    AgentCapability.REASONING: ("analyze", "reason", "infer", "deduce", "explain"),
    AgentCapability.SUMMARIZATION: ("summarize", "summary", "condense", "abstract"),
    AgentCapability.TRANSLATION: ("translate", "translation", "render"),
    AgentCapability.CODE: ("code", "implement", "program", "function"),
    AgentCapability.MATH: ("calculate", "compute", "solve", "derive", "prove"),
    AgentCapability.SQL: ("sql", "query", "select", "join", "index"),
    AgentCapability.VERIFICATION: ("verify", "validate", "check", "confirm"),
}


def assess_query_complexity(query: str) -> QueryComplexityAssessment:
    """Score a natural-language query's complexity from keyword signals.

    Deterministic: the same query always yields the same assessment.
    """
    lowered = turkish_lower(query)
    tokens = set(re.findall(r"[a-z0-9çğıöşü']+", lowered))
    score = 0
    signals: list[str] = []
    for keyword, weight, label in _COMPLEXITY_SIGNALS:
        if keyword in tokens:
            score += weight
            if label not in signals:
                signals.append(label)
    # Length is a weak signal: longer queries tend to have more parts.
    score += min(20, len(tokens) // 4)
    score = max(0, min(100, score))

    if score < 10:
        complexity = TaskComplexity.TRIVIAL
    elif score < 25:
        complexity = TaskComplexity.SIMPLE
    elif score < 45:
        complexity = TaskComplexity.MODERATE
    elif score < 70:
        complexity = TaskComplexity.COMPLEX
    else:
        complexity = TaskComplexity.VERY_COMPLEX

    estimated = max(1, min(8, 1 + score // 15))
    return QueryComplexityAssessment(
        complexity=complexity,
        score=score,
        signals=signals,
        estimated_sub_tasks=estimated,
    )


def _capabilities_for(text: str) -> list[AgentCapability]:
    lowered = turkish_lower(text)
    found: list[AgentCapability] = []
    for capability, keywords in _CAPABILITY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            found.append(capability)
    if not found:
        found.append(AgentCapability.GENERAL)
    return found


def _split_query(query: str) -> list[str]:
    """Split a query into candidate sub-task fragments on conjunctions/sequences."""
    parts = re.split(r"\b(?:and then|then|after that|afterwards|;|,\s*and)\b", query, flags=re.IGNORECASE)
    fragments = [p.strip(" ,.;") for p in parts if p.strip(" ,.;")]
    return fragments or [query.strip()]


def decompose_query(query: str, max_sub_tasks: int = 8) -> DecompositionPlan:
    """Break ``query`` into sub-tasks with a dependency graph and schedule.

    The decomposition is heuristic but deterministic: fragments are produced by
    splitting on sequence conjunctions, each fragment becomes a sub-task with
    inferred capabilities, and fragments after the first depend on the previous
    one (a linear chain) unless they are independent retrieval steps.
    """
    assessment = assess_query_complexity(query)
    fragments = _split_query(query)[:max_sub_tasks]
    warnings: list[str] = []
    if len(fragments) < assessment.estimated_sub_tasks:
        warnings.append(
            f"Query suggests ~{assessment.estimated_sub_tasks} sub-tasks but only {len(fragments)} were identified."
        )

    sub_tasks: list[SubTask] = []
    previous_id: str | None = None
    for index, fragment in enumerate(fragments):
        task_id = f"task-{index + 1}"
        capabilities = _capabilities_for(fragment)
        # Retrieval-only fragments are independent; anything else chains.
        independent = capabilities == [AgentCapability.RETRIEVAL]
        depends_on = [] if independent or previous_id is None else [previous_id]
        sub_tasks.append(
            SubTask(
                id=task_id,
                description=fragment,
                required_capabilities=capabilities,
                depends_on=depends_on,
                estimated_cost=max(1, len(fragment) // 40 + 1),
            )
        )
        previous_id = task_id

    batches = execution_batches(sub_tasks)
    parallelizable = any(len(batch) > 1 for batch in batches)
    completeness = _decomposition_completeness(query, sub_tasks)
    return DecompositionPlan(
        query=query,
        assessment=assessment,
        sub_tasks=sub_tasks,
        execution_batches=batches,
        parallelizable=parallelizable,
        completeness=completeness,
        warnings=warnings,
    )


def _decomposition_completeness(query: str, sub_tasks: list[SubTask]) -> float:
    """Fraction of the query's content tokens covered by sub-task descriptions."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 1.0
    covered: set[str] = set()
    for task in sub_tasks:
        covered |= _tokenize(task.description)
    return len(query_tokens & covered) / len(query_tokens)


def execution_batches(sub_tasks: list[SubTask]) -> list[list[str]]:
    """Topologically layer sub-tasks into parallel execution batches.

    Each batch contains tasks whose dependencies are all satisfied by earlier
    batches. Tasks in the same batch can run concurrently. Raises ``ValueError``
    on a dependency cycle or a reference to an unknown task.
    """
    by_id = {task.id: task for task in sub_tasks}
    for task in sub_tasks:
        for dep in task.depends_on:
            if dep not in by_id:
                raise ValueError(f"Task {task.id} depends on unknown task {dep}")

    remaining = {task.id: set(task.depends_on) for task in sub_tasks}
    batches: list[list[str]] = []
    while remaining:
        ready = sorted(task_id for task_id, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError("Dependency cycle detected in sub-tasks")
        batches.append(ready)
        for task_id in ready:
            del remaining[task_id]
        for deps in remaining.values():
            deps.difference_update(ready)
    return batches


def match_agents(sub_tasks: list[SubTask], agents: list[AgentProfile]) -> dict[str, str]:
    """Assign each sub-task to the best-suited agent.

    Scoring prefers agents whose capabilities cover the task's requirements,
    then higher reliability. Returns a mapping of task id -> agent id. Tasks
    with no capable agent are left unassigned (absent from the mapping).
    """
    assignment: dict[str, str] = {}
    load: dict[str, int] = {agent.id: 0 for agent in agents}
    for task in sub_tasks:
        required = set(task.required_capabilities)
        best: tuple[float, str] | None = None
        for agent in agents:
            if load[agent.id] >= agent.capacity:
                continue
            covered = required & set(agent.capabilities)
            if required and not covered:
                continue
            coverage = len(covered) / len(required) if required else 1.0
            score = coverage * 100 + agent.reliability * 10 - load[agent.id]
            if best is None or score > best[0]:
                best = (score, agent.id)
        if best is not None:
            assignment[task.id] = best[1]
            load[best[1]] += 1
    return assignment


def validate_decomposition(plan: DecompositionPlan) -> list[str]:
    """Return a list of problems with ``plan``; empty means it is valid."""
    problems: list[str] = []
    if not plan.sub_tasks:
        problems.append("Plan has no sub-tasks")
        return problems
    ids = {task.id for task in plan.sub_tasks}
    if len(ids) != len(plan.sub_tasks):
        problems.append("Duplicate sub-task ids")
    for task in plan.sub_tasks:
        for dep in task.depends_on:
            if dep not in ids:
                problems.append(f"Task {task.id} depends on unknown task {dep}")
    try:
        execution_batches(plan.sub_tasks)
    except ValueError as exc:
        problems.append(str(exc))
    if plan.completeness < 0.9:
        problems.append(f"Decomposition completeness {plan.completeness:.2f} is below the 0.90 target")
    return problems


def replan(plan: DecompositionPlan, failed_task_id: str, replacement: SubTask | None = None) -> DecompositionPlan:
    """Produce a new plan after ``failed_task_id`` fails.

    The failed task is removed; dependent tasks are re-pointed to the failed
    task's own dependencies (or made independent if it had none). If
    ``replacement`` is supplied it takes the failed task's place.
    """
    failed = next((t for t in plan.sub_tasks if t.id == failed_task_id), None)
    if failed is None:
        raise ValueError(f"Unknown task {failed_task_id}")
    new_tasks: list[SubTask] = []
    for task in plan.sub_tasks:
        if task.id == failed_task_id:
            if replacement is not None:
                new_tasks.append(replacement)
            continue
        if failed_task_id in task.depends_on:
            new_deps = [d for d in task.depends_on if d != failed_task_id]
            new_deps.extend(d for d in failed.depends_on if d not in new_deps)
            task = task.model_copy(update={"depends_on": new_deps})
        new_tasks.append(task)
    batches = execution_batches(new_tasks)
    return plan.model_copy(
        update={
            "sub_tasks": new_tasks,
            "execution_batches": batches,
            "parallelizable": any(len(batch) > 1 for batch in batches),
            "completeness": _decomposition_completeness(plan.query, new_tasks),
        }
    )


@dataclass
class DecompositionMetrics:
    """Running counters for decomposition quality."""

    plans_built: int = 0
    parallelizable_plans: int = 0
    total_sub_tasks: int = 0
    completeness_sum: float = 0.0
    assignments_made: int = 0
    assignments_optimal: int = 0

    def record_plan(self, plan: DecompositionPlan) -> None:
        self.plans_built += 1
        self.total_sub_tasks += len(plan.sub_tasks)
        self.completeness_sum += plan.completeness
        if plan.parallelizable:
            self.parallelizable_plans += 1

    def record_assignment(self, optimal: bool) -> None:
        self.assignments_made += 1
        if optimal:
            self.assignments_optimal += 1

    def summary(self) -> dict[str, Any]:
        avg_completeness = self.completeness_sum / self.plans_built if self.plans_built else 0.0
        parallel_rate = self.parallelizable_plans / self.plans_built if self.plans_built else 0.0
        accuracy = self.assignments_optimal / self.assignments_made if self.assignments_made else 0.0
        return {
            "plans_built": self.plans_built,
            "avg_completeness": avg_completeness,
            "parallelizable_rate": parallel_rate,
            "avg_sub_tasks": self.total_sub_tasks / self.plans_built if self.plans_built else 0.0,
            "assignment_accuracy": accuracy,
        }


class DecomposeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_sub_tasks: int = Field(default=8, ge=1, le=32)


class DecomposeResponse(BaseModel):
    plan: DecompositionPlan
    problems: list[str] = Field(default_factory=list)


class MatchRequest(BaseModel):
    sub_tasks: list[SubTask]
    agents: list[AgentProfile]


class MatchResponse(BaseModel):
    assignment: dict[str, str]


@router.post("/decompose", response_model=DecomposeResponse)
async def decompose_endpoint(request: DecomposeRequest) -> DecomposeResponse:
    """Decompose a complex query into sub-tasks with a dependency schedule."""
    plan = decompose_query(request.query, max_sub_tasks=request.max_sub_tasks)
    return DecomposeResponse(plan=plan, problems=validate_decomposition(plan))


@router.post("/match-agents", response_model=MatchResponse)
async def match_agents_endpoint(request: MatchRequest) -> MatchResponse:
    """Assign sub-tasks to the best-suited agents by capability and reliability."""
    return MatchResponse(assignment=match_agents(request.sub_tasks, request.agents))
