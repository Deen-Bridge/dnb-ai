"""Tests for the deterministic database query optimizer.

Every test runs offline against ``query_optimizer`` alone — no main.py, no live
database, no GEMINI_API_KEY — so the suite is fast and dependency-free.
"""

from query_optimizer import (
    DEFAULT_SLOW_QUERY_MS,
    N_PLUS_ONE_THRESHOLD,
    QueryProfiler,
    QuerySample,
    analyze_query,
    fingerprint_sql,
    percentile,
    pool_efficiency,
)

# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def test_fingerprint_collapses_literals() -> None:
    assert fingerprint_sql("SELECT id FROM users WHERE id = 1") == fingerprint_sql("SELECT id FROM users WHERE id = 2")


def test_fingerprint_collapses_string_and_in_lists() -> None:
    a = fingerprint_sql("SELECT * FROM t WHERE name = 'ali' AND id IN (1, 2, 3)")
    b = fingerprint_sql("SELECT * FROM t WHERE name = 'zayd' AND id IN (9)")
    assert a == b
    assert "?" in a


def test_fingerprint_is_deterministic() -> None:
    q = "SELECT a, b FROM t WHERE x = 5"
    assert fingerprint_sql(q) == fingerprint_sql(q)


# ---------------------------------------------------------------------------
# Static analysis: anti-patterns
# ---------------------------------------------------------------------------


def _codes(sql: str) -> set[str]:
    return {ap.code for ap in analyze_query(sql).anti_patterns}


def test_select_star_flagged() -> None:
    assert "SELECT_STAR" in _codes("SELECT * FROM users WHERE id = 1 LIMIT 1")


def test_full_table_scan_flagged_for_select_without_where() -> None:
    codes = _codes("SELECT name FROM users LIMIT 10")
    assert "FULL_TABLE_SCAN" in codes


def test_delete_without_where_is_severe() -> None:
    analysis = analyze_query("DELETE FROM sessions")
    scan = next(ap for ap in analysis.anti_patterns if ap.code == "FULL_TABLE_SCAN")
    assert scan.severity == 5
    assert analysis.is_optimized is False


def test_leading_wildcard_flagged() -> None:
    assert "LEADING_WILDCARD" in _codes("SELECT id FROM users WHERE name LIKE '%ali%' LIMIT 5")


def test_function_on_column_flagged() -> None:
    assert "FUNCTION_ON_COLUMN" in _codes("SELECT id FROM users WHERE lower(name) = 'ali' LIMIT 5")


def test_or_condition_flagged() -> None:
    assert "OR_CONDITION" in _codes("SELECT id FROM users WHERE a = 1 OR b = 2 LIMIT 5")


def test_negation_filter_flagged() -> None:
    assert "NEGATION_FILTER" in _codes("SELECT id FROM users WHERE status != 'active' LIMIT 5")


def test_implicit_join_flagged() -> None:
    assert "IMPLICIT_JOIN" in _codes("SELECT u.id FROM users u, orders o WHERE u.id = o.user_id LIMIT 5")


def test_unbounded_result_flagged_without_limit() -> None:
    assert "UNBOUNDED_RESULT" in _codes("SELECT id FROM users WHERE id = 1")


def test_sort_without_limit_flagged() -> None:
    assert "SORT_WITHOUT_LIMIT" in _codes("SELECT id FROM users WHERE id = 1 ORDER BY created_at")


def test_clean_query_has_no_severe_anti_patterns() -> None:
    analysis = analyze_query("SELECT id, name FROM users WHERE tenant_id = 1 ORDER BY id LIMIT 20")
    assert analysis.is_optimized is True
    assert analysis.estimated_cost <= 20


def test_analysis_is_deterministic() -> None:
    sql = "SELECT * FROM users WHERE lower(name) LIKE '%x%'"
    assert analyze_query(sql).model_dump() == analyze_query(sql).model_dump()


def test_statement_type_detected() -> None:
    assert analyze_query("UPDATE t SET a = 1 WHERE id = 2").statement_type == "update"
    assert analyze_query("SELECT 1").statement_type == "select"


# ---------------------------------------------------------------------------
# Static analysis: index recommendations
# ---------------------------------------------------------------------------


def test_where_columns_recommended_as_index() -> None:
    analysis = analyze_query("SELECT id FROM users WHERE tenant_id = 1 AND status = 'active' LIMIT 5")
    where_rec = next(r for r in analysis.index_recommendations if "tenant_id" in r.columns)
    assert "status" in where_rec.columns


def test_join_and_order_columns_recommended() -> None:
    sql = "SELECT u.id FROM users u JOIN orders o ON u.id = o.user_id ORDER BY o.created_at LIMIT 10"
    cols = {c for rec in analyze_query(sql).index_recommendations for c in rec.columns}
    assert "o.user_id" in cols or "u.id" in cols
    assert "o.created_at" in cols


def test_no_index_recommendations_without_filters() -> None:
    analysis = analyze_query("SELECT id FROM users LIMIT 10")
    assert analysis.index_recommendations == []


# ---------------------------------------------------------------------------
# Percentiles and pool efficiency (pure helpers)
# ---------------------------------------------------------------------------


def test_percentile_empty_is_zero() -> None:
    assert percentile([], 95) == 0.0


def test_percentile_nearest_rank() -> None:
    values = [float(n) for n in range(1, 101)]  # 1..100
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile(values, 99) == 99.0
    assert percentile(values, 100) == 100.0


def test_pool_efficiency_bounds_and_contention() -> None:
    assert pool_efficiency(0, 0, 0) == 1.0  # nothing wasted
    assert pool_efficiency(10, 0, 0) == 1.0  # fully utilized, no waiting
    idle_heavy = pool_efficiency(1, 9, 0)
    assert 0.0 < idle_heavy < 0.5
    # Waiting callers lower efficiency versus the same active/idle with no wait.
    assert pool_efficiency(5, 5, 10) < pool_efficiency(5, 5, 0)


# ---------------------------------------------------------------------------
# Runtime profiling
# ---------------------------------------------------------------------------


def _profiler() -> QueryProfiler:
    return QueryProfiler()


def test_profiler_records_and_counts() -> None:
    p = _profiler()
    p.record(QuerySample(sql="SELECT 1", duration_ms=10.0))
    p.record(QuerySample(sql="SELECT 2", duration_ms=20.0))
    assert p.stats().total_queries == 2


def test_slow_queries_detected_against_threshold() -> None:
    p = _profiler()
    p.record(QuerySample(sql="SELECT fast", duration_ms=50.0))
    p.record(QuerySample(sql="SELECT slow", duration_ms=DEFAULT_SLOW_QUERY_MS + 1))
    slow = p.slow_queries()
    assert len(slow) == 1
    assert slow[0].duration_ms > DEFAULT_SLOW_QUERY_MS


def test_latency_percentiles_reported() -> None:
    p = _profiler()
    for n in range(1, 101):
        p.record(QuerySample(sql=f"SELECT {n}", duration_ms=float(n)))
    latency = p.stats().latency
    assert latency.p50 == 50.0
    assert latency.p95 == 95.0
    assert latency.max == 100.0


def test_cache_hit_rate_computed() -> None:
    p = _profiler()
    p.record(QuerySample(sql="SELECT a", duration_ms=5.0, cache_hit=True))
    p.record(QuerySample(sql="SELECT b", duration_ms=5.0, cache_hit=True))
    p.record(QuerySample(sql="SELECT c", duration_ms=5.0, cache_hit=False))
    assert p.stats().cache_hit_rate == 2 / 3


def test_n_plus_one_detected_for_repeated_shape() -> None:
    p = _profiler()
    for order_id in range(N_PLUS_ONE_THRESHOLD + 2):
        p.record(QuerySample(sql=f"SELECT * FROM items WHERE order_id = {order_id}", duration_ms=3.0))
    patterns = p.stats().n_plus_one
    assert len(patterns) == 1
    assert patterns[0].count == N_PLUS_ONE_THRESHOLD + 2


def test_reset_clears_samples() -> None:
    p = _profiler()
    p.record(QuerySample(sql="SELECT 1", duration_ms=1.0))
    p.reset()
    assert p.stats().total_queries == 0


def test_empty_profiler_stats_are_zeroed() -> None:
    stats = _profiler().stats()
    assert stats.total_queries == 0
    assert stats.cache_hit_rate == 0.0
    assert stats.throughput_qps == 0.0
    assert stats.latency.p95 == 0.0
