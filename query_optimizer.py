"""Database query optimization — profile queries, flag anti-patterns, decompose agent tasks.

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

3. **Agent task decomposition** (:class:`TaskDecomposer`) breaks a complex query
   into sub-tasks, builds a dependency graph, matches sub-tasks to agents by
   capability, generates a parallel-aware execution plan and validates coverage.
   It is deterministic and in-memory, mirroring the offline philosophy above.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Iterable

from collections import defaultdict, deque

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
        self, surah: int | None = None, reference_type: ReferenceType | None = None, attributes: list[str] | None = None
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


class TaskKind(str, Enum):
    """Coarse category of a decomposed sub-task."""

    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"


class AgentCapability(str, Enum):
    """Capabilities an agent may advertise for task matching."""

    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    SQL = "sql"
    TURKISH = "turkish"
    QURAN = "quran"


_KIND_TO_CAPABILITY: dict[TaskKind, AgentCapability] = {
    TaskKind.RETRIEVAL: AgentCapability.RETRIEVAL,
    TaskKind.ANALYSIS: AgentCapability.ANALYSIS,
    TaskKind.SYNTHESIS: AgentCapability.SYNTHESIS,
    TaskKind.VALIDATION: AgentCapability.VALIDATION,
    TaskKind.TRANSFORMATION: AgentCapability.TRANSFORMATION,
}


class AgentProfile(BaseModel):
    """An agent that can be assigned sub-tasks."""

    agent_id: str = Field(..., min_length=1)
    capabilities: list[AgentCapability] = Field(default_factory=list)
    max_concurrency: int = Field(default=1, ge=1)
    cost_per_task: float = Field(default=1.0, ge=0.0)


class SubTask(BaseModel):
    """One node in a decomposition's dependency graph."""

    task_id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    kind: TaskKind
    depends_on: list[str] = Field(default_factory=list)
    required_capabilities: list[AgentCapability] = Field(default_factory=list)
    estimated_effort: float = Field(default=1.0, ge=0.0)


class DecompositionRequest(BaseModel):
    """Input to the decomposer: a complex query plus available agents."""

    query: str = Field(..., min_length=1)
    agents: list[AgentProfile] = Field(default_factory=list)
    max_sub_tasks: int = Field(default=8, ge=1, le=64)


class TaskAssignment(BaseModel):
    """A sub-task bound to a concrete agent."""

    task_id: str
    agent_id: str
    score: float = Field(..., ge=0.0)


class ExecutionWave(BaseModel):
    """A set of sub-tasks that can run in parallel (no inter-dependencies)."""

    wave_index: int = Field(..., ge=0)
    task_ids: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """A dependency-ordered, parallel-aware schedule."""

    waves: list[ExecutionWave] = Field(default_factory=list)
    sequential_estimate: float = Field(..., ge=0.0)
    parallel_estimate: float = Field(..., ge=0.0)
    speedup: float = Field(..., ge=0.0)


class DecompositionResult(BaseModel):
    """Full decomposition output including graph, assignments and plan."""

    query: str
    complexity: int = Field(..., ge=0, le=100)
    sub_tasks: list[SubTask] = Field(default_factory=list)
    assignments: list[TaskAssignment] = Field(default_factory=list)
    plan: ExecutionPlan
    coverage: float = Field(..., ge=0.0, le=1.0)
    is_complete: bool
    warnings: list[str] = Field(default_factory=list)


_COMPLEXITY_MARKERS = (
    " and ",
    " or ",
    " then ",
    " after ",
    " before ",
    " compare ",
    " versus ",
    " vs ",
    " join ",
    " aggregate ",
    " group ",
    " summarize ",
    " explain ",
    " why ",
    " how ",
    " list ",
    " find ",
    " count ",
    " average ",
    " top ",
    " rank ",
)


def assess_complexity(query: str) -> int:
    """Heuristic 0..100 complexity score for a natural-language query.

    Longer queries, more clauses and more distinct marker words raise the score.
    Deterministic and dependency-free.
    """
    lowered = f" {turkish_lower(query)} "
    tokens = [t for t in re.findall(r"[a-z0-9çğıöşüâîû']+", lowered) if t]
    if not tokens:
        return 0
    marker_hits = sum(1 for marker in _COMPLEXITY_MARKERS if marker in lowered)
    length_score = min(40, len(tokens) * 2)
    marker_score = min(40, marker_hits * 8)
    conjunction_score = min(20, (lowered.count(" and ") + lowered.count(" or ")) * 5)
    return min(100, length_score + marker_score + conjunction_score)


_KIND_KEYWORDS: tuple[tuple[TaskKind, tuple[str, ...]], ...] = (
    (TaskKind.RETRIEVAL, ("find", "list", "retrieve", "search", "select", "get", "fetch", "lookup")),
    (TaskKind.ANALYSIS, ("analyze", "compare", "count", "average", "aggregate", "group", "rank", "top", "why")),
    (TaskKind.TRANSFORMATION, ("convert", "transform", "normalize", "format", "translate", "map")),
    (TaskKind.VALIDATION, ("validate", "verify", "check", "ensure", "confirm")),
    (TaskKind.SYNTHESIS, ("summarize", "explain", "describe", "combine", "synthesize", "report")),
)


def _classify_kind(fragment: str) -> TaskKind:
    lowered = f" {turkish_lower(fragment)} "
    for kind, keywords in _KIND_KEYWORDS:
        if any(f" {kw} " in lowered or lowered.strip().startswith(kw) for kw in keywords):
            return kind
    return TaskKind.ANALYSIS


def _split_query(query: str, max_parts: int) -> list[str]:
    """Split a query into candidate sub-task fragments on conjunctions."""
    working = re.sub(r"\s+(and then|then|after that|afterwards)\s+", " ; ", query, flags=re.IGNORECASE)
    working = re.sub(r"\s+and\s+", " ; ", working, flags=re.IGNORECASE)
    working = re.sub(r"\s*;\s*", ";", working)
    raw_parts = [part.strip(" ;") for part in working.split(";")]
    parts = [part for part in raw_parts if part]
    if not parts:
        parts = [query.strip()]
    if len(parts) > max_parts:
        head = parts[: max_parts - 1]
        tail = " and ".join(parts[max_parts - 1 :])
        parts = head + [tail]
    return parts


def _required_capabilities(kind: TaskKind, fragment: str) -> list[AgentCapability]:
    caps: list[AgentCapability] = [_KIND_TO_CAPABILITY[kind]]
    lowered = turkish_lower(fragment)
    if any(word in lowered for word in ("sql", "query", "select", "join", "table")):
        caps.append(AgentCapability.SQL)
    if any(ch in fragment for ch in "çğıöşüÇĞİÖŞÜ"):
        caps.append(AgentCapability.TURKISH)
    if any(term in lowered for term in ("quran", "ayah", "surah", "kuran", "ayet", "sure")):
        caps.append(AgentCapability.QURAN)
    return caps


def build_sub_tasks(query: str, max_sub_tasks: int) -> list[SubTask]:
    """Decompose ``query`` into ordered sub-tasks with linear dependencies.

    Each fragment becomes a sub-task that depends on the previous one, which
    keeps the graph acyclic and reflects the natural reading order of a query.
    """
    fragments = _split_query(query, max_sub_tasks)
    tasks: list[SubTask] = []
    previous_id: str | None = None
    for index, fragment in enumerate(fragments):
        kind = _classify_kind(fragment)
        task_id = f"t{index + 1}"
        tasks.append(
            SubTask(
                task_id=task_id,
                description=fragment,
                kind=kind,
                depends_on=[previous_id] if previous_id else [],
                required_capabilities=_required_capabilities(kind, fragment),
                estimated_effort=1.0 + min(2.0, len(fragment) / 80.0),
            )
        )
        previous_id = task_id
    return tasks


def _score_agent(agent: AgentProfile, task: SubTask) -> float:
    """Score how well ``agent`` matches ``task`` (0..1)."""
    if not task.required_capabilities:
        return 0.5
    provided = set(agent.capabilities)
    required = set(task.required_capabilities)
    matched = len(required & provided)
    base = matched / len(required)
    cost_penalty = min(0.2, agent.cost_per_task * 0.05)
    return max(0.0, min(1.0, base - cost_penalty))


def assign_tasks(sub_tasks: list[SubTask], agents: list[AgentProfile]) -> tuple[list[TaskAssignment], list[str]]:
    """Greedily assign each sub-task to the best-scoring agent.

    Returns the assignments and a list of warnings for unassignable tasks.
    """
    assignments: list[TaskAssignment] = []
    warnings: list[str] = []
    if not agents:
        return assignments, ["no agents provided; tasks are unassigned"]
    for task in sub_tasks:
        best_agent: AgentProfile | None = None
        best_score = -1.0
        for agent in agents:
            score = _score_agent(agent, task)
            if score > best_score:
                best_score = score
                best_agent = agent
        if best_agent is None or best_score <= 0.0:
            warnings.append(f"no suitable agent for task {task.task_id}")
            continue
        assignments.append(TaskAssignment(task_id=task.task_id, agent_id=best_agent.agent_id, score=best_score))
    return assignments, warnings


def _topological_waves(sub_tasks: list[SubTask]) -> list[list[str]]:
    """Group sub-tasks into waves so each wave has no internal dependencies."""
    ids = {task.task_id for task in sub_tasks}
    indegree: dict[str, int] = {task.task_id: 0 for task in sub_tasks}
    children: dict[str, list[str]] = defaultdict(list)
    for task in sub_tasks:
        for dep in task.depends_on:
            if dep in ids:
                indegree[task.task_id] += 1
                children[dep].append(task.task_id)
    ready = deque(sorted(tid for tid, deg in indegree.items() if deg == 0))
    waves: list[list[str]] = []
    while ready:
        wave = list(ready)
        ready.clear()
        waves.append(wave)
        for tid in wave:
            for child in children[tid]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
    return waves


def build_execution_plan(sub_tasks: list[SubTask]) -> ExecutionPlan:
    """Build a parallel-aware execution plan from the dependency graph."""
    effort = {task.task_id: task.estimated_effort for task in sub_tasks}
    waves = _topological_waves(sub_tasks)
    sequential = sum(effort.values())
    parallel = 0.0
    execution_waves: list[ExecutionWave] = []
    for index, wave_ids in enumerate(waves):
        execution_waves.append(ExecutionWave(wave_index=index, task_ids=wave_ids))
        if wave_ids:
            parallel += max(effort.get(tid, 0.0) for tid in wave_ids)
    speedup = (sequential / parallel) if parallel > 0 else 1.0
    return ExecutionPlan(
        waves=execution_waves,
        sequential_estimate=sequential,
        parallel_estimate=parallel,
        speedup=speedup,
    )


def validate_decomposition(query: str, sub_tasks: list[SubTask]) -> tuple[float, list[str]]:
    """Estimate coverage of the original query by the sub-tasks (0..1)."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 1.0, []
    covered: set[str] = set()
    for task in sub_tasks:
        covered |= _tokenize(task.description)
    coverage = len(query_tokens & covered) / len(query_tokens)
    warnings: list[str] = []
    if coverage < 0.9:
        warnings.append(f"decomposition coverage {coverage:.2f} below 0.90 target")
    return coverage, warnings


class TaskDecomposer:
    """Deterministic, in-memory agent task decomposition engine."""

    def decompose(self, request: DecompositionRequest) -> DecompositionResult:
        """Decompose ``request.query`` into sub-tasks, assignments and a plan."""
        complexity = assess_complexity(request.query)
        sub_tasks = build_sub_tasks(request.query, request.max_sub_tasks)
        assignments, warnings = assign_tasks(sub_tasks, request.agents)
        plan = build_execution_plan(sub_tasks)
        coverage, coverage_warnings = validate_decomposition(request.query, sub_tasks)
        warnings.extend(coverage_warnings)
        return DecompositionResult(
            query=request.query,
            complexity=complexity,
            sub_tasks=sub_tasks,
            assignments=assignments,
            plan=plan,
            coverage=coverage,
            is_complete=coverage >= 0.9 and not any("unassigned" in w for w in warnings),
            warnings=warnings,
        )


_decomposer = TaskDecomposer()


def get_decomposer() -> TaskDecomposer:
    """Return the process-wide decomposer instance."""
    return _decomposer


@router.post("/decompose", response_model=DecompositionResult)
async def decompose_endpoint(request: DecompositionRequest) -> DecompositionResult:
    """Decompose a complex query into agent sub-tasks with a parallel plan."""
    return get_decomposer().decompose(request)
