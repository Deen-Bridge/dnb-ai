"""Offline tests for latency profiling infrastructure (issue #211).

No live API calls or network I/O. Covers trace lifecycle, stats accuracy,
budget violations, baseline regression detection, and API endpoint contracts.
"""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import latency_profiler
from latency_profiler import (
    _LatencyRegistry,
    _TraceStore,
    add_span,
    check_budget,
    compare_to_baseline,
    end_trace,
    get_profiler_stats,
    get_recommendations,
    reset_profiler,
    router,
    start_trace,
    store_baseline,
    trace_component,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Reset all profiling state between tests."""
    reset_profiler()
    yield
    reset_profiler()


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# trace_component context manager
# ---------------------------------------------------------------------------


class TestTraceComponent:
    def test_records_elapsed_time(self):
        with trace_component("test_comp"):
            time.sleep(0.01)
        stats = get_profiler_stats("test_comp")
        assert "test_comp" in stats
        assert stats["test_comp"]["count"] == 1
        assert stats["test_comp"]["min_ms"] >= 5.0  # at least 5ms sleep

    def test_multiple_records_accumulate(self):
        for _ in range(3):
            with trace_component("multi"):
                time.sleep(0.005)
        stats = get_profiler_stats("multi")
        assert stats["multi"]["count"] == 3

    def test_exception_does_not_break_recording(self):
        with pytest.raises(ValueError):
            with trace_component("fail_comp"):
                raise ValueError("boom")
        stats = get_profiler_stats("fail_comp")
        # Should still have recorded the timing before the exception
        assert stats["fail_comp"]["count"] == 1


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


class TestSummaryStats:
    def test_empty_returns_zeros(self):
        stats = get_profiler_stats("nonexistent")
        assert stats == {}

    def test_percentiles_correct(self):
        samples = list(range(1, 101))  # 1..100 ms
        for s in samples:
            _LatencyRegistry.record(latency_profiler._registry, "pct_test", float(s))
        stats = get_profiler_stats("pct_test")
        s = stats["pct_test"]
        assert s["count"] == 100
        assert s["min_ms"] == 1.0
        assert s["max_ms"] == 100.0
        assert s["avg_ms"] == pytest.approx(50.5, abs=0.1)
        assert s["p50_ms"] == pytest.approx(50.5, abs=1.0)
        assert s["p95_ms"] >= 94.0
        assert s["p99_ms"] >= 98.0


# ---------------------------------------------------------------------------
# Request tracing lifecycle
# ---------------------------------------------------------------------------


class TestRequestTracing:
    def test_start_add_end(self):
        tid = start_trace()
        assert isinstance(tid, str) and len(tid) == 32

        add_span(tid, "db", 15.5)
        add_span(tid, "inference", 250.0)

        result = end_trace(tid)
        assert result is not None
        assert result["trace_id"] == tid
        assert len(result["spans"]) == 2
        assert result["total_ms"] == pytest.approx(265.5, abs=0.1)
        assert result["bottleneck"]["component"] == "inference"
        assert result["bottleneck"]["duration_ms"] == 250.0

    def test_end_unknown_trace_returns_none(self):
        result = end_trace("does_not_exist")
        assert result is None

    def test_bottleneck_is_slowest_component(self):
        tid = start_trace()
        add_span(tid, "fast", 1.0)
        add_span(tid, "slow", 999.0)
        add_span(tid, "mid", 50.0)
        result = end_trace(tid)
        assert result["bottleneck"]["component"] == "slow"

    def test_single_span_bottleneck(self):
        tid = start_trace()
        add_span(tid, "only", 42.0)
        result = end_trace(tid)
        assert result["bottleneck"]["component"] == "only"
        assert result["span_count"] == 1


# ---------------------------------------------------------------------------
# Latency budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    def test_within_budget(self):
        result = check_budget("database", 50.0)
        assert result["within_budget"] is True
        assert result["overage_ms"] == 0.0

    def test_exceeds_budget(self):
        result = check_budget("database", 200.0)
        assert result["within_budget"] is False
        assert result["overage_ms"] == pytest.approx(100.0, abs=0.1)

    def test_unknown_component_uses_default(self):
        result = check_budget("unknown_comp", 1500.0)
        assert result["within_budget"] is False
        assert result["budget_ms"] == 1000.0  # default

    def test_boundary_is_within(self):
        result = check_budget("database", 100.0)
        assert result["within_budget"] is True


# ---------------------------------------------------------------------------
# Baseline comparison / regression detection
# ---------------------------------------------------------------------------


class TestBaselineComparison:
    def test_regression_detected(self):
        baseline = {"inference": {"p95_ms": 100.0, "p99_ms": 120.0, "avg_ms": 80.0, "count": 50}}
        current = {"inference": {"p95_ms": 150.0, "p99_ms": 180.0, "avg_ms": 120.0, "count": 50}}
        result = compare_to_baseline(current, baseline)
        assert len(result["regressions"]) == 1
        assert result["regressions"][0]["component"] == "inference"
        assert result["regressions"][0]["change_pct"] == 50.0

    def test_improvement_detected(self):
        baseline = {"db": {"p95_ms": 200.0, "p99_ms": 250.0, "avg_ms": 150.0, "count": 30}}
        current = {"db": {"p95_ms": 100.0, "p99_ms": 130.0, "avg_ms": 80.0, "count": 30}}
        result = compare_to_baseline(current, baseline)
        assert len(result["improvements"]) == 1
        assert result["improvements"][0]["component"] == "db"
        assert result["improvements"][0]["change_pct"] == -50.0

    def test_no_change_within_threshold(self):
        baseline = {"x": {"p95_ms": 100.0}}
        current = {"x": {"p95_ms": 105.0}}
        result = compare_to_baseline(current, baseline, regression_threshold=0.20)
        assert len(result["regressions"]) == 0
        assert len(result["improvements"]) == 0

    def test_new_component_not_in_baseline(self):
        baseline = {}
        current = {"new_comp": {"p95_ms": 50.0}}
        result = compare_to_baseline(current, baseline)
        assert result["components_compared"] == 1
        assert len(result["regressions"]) == 0


# ---------------------------------------------------------------------------
# Optimization recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_inference_slow_triggers_rec(self):
        slow_stats = {
            "inference": {"p95_ms": 2500.0, "p99_ms": 3000.0, "avg_ms": 2000.0, "count": 10},
        }
        recs = get_recommendations(slow_stats)
        comp_names = [r["component"] for r in recs]
        assert "inference" in comp_names

    def test_fast_components_no_recs(self):
        fast_stats = {
            "database": {"p95_ms": 10.0, "p99_ms": 12.0, "avg_ms": 5.0, "count": 100},
            "retrieval": {"p95_ms": 20.0, "p99_ms": 25.0, "avg_ms": 15.0, "count": 100},
        }
        recs = get_recommendations(fast_stats)
        assert len(recs) == 0

    def test_tail_latency_spike_detected(self):
        spike_stats = {
            "api_call": {"p95_ms": 100.0, "p99_ms": 200.0, "avg_ms": 80.0, "count": 50},
        }
        recs = get_recommendations(spike_stats)
        tail_recs = [r for r in recs if "Tail latency" in r["recommendation"]]
        assert len(tail_recs) == 1


# ---------------------------------------------------------------------------
# Baseline store
# ---------------------------------------------------------------------------


class TestBaselineStore:
    def test_store_and_retrieve(self):
        store_baseline("v1", {"inference": {"p95_ms": 100.0}})
        from latency_profiler import get_baseline
        bl = get_baseline("v1")
        assert bl is not None
        assert bl["inference"]["p95_ms"] == 100.0

    def test_list_baselines(self):
        store_baseline("a", {})
        store_baseline("b", {})
        from latency_profiler import list_baselines
        names = list_baselines()
        assert set(names) == {"a", "b"}

    def test_reset_clears_baselines(self):
        store_baseline("x", {})
        reset_profiler()
        from latency_profiler import list_baselines
        assert list_baselines() == []


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestAPIEndpoints:
    def test_get_stats_empty(self, client):
        resp = client.get("/profiling/stats")
        assert resp.status_code == 200
        assert resp.json() == {}

    def test_trace_lifecycle(self, client):
        resp = client.post("/profiling/trace")
        assert resp.status_code == 200
        tid = resp.json()["trace_id"]
        assert len(tid) == 32

        client.post("/profiling/trace/span", json={
            "trace_id": tid,
            "timings": [{"component": "db", "duration_ms": 25.0}],
        })

        resp = client.post("/profiling/trace/end", json={"trace_id": tid})
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace_id"] == tid
        assert body["total_ms"] == 25.0
        assert body["bottleneck"]["component"] == "db"

    def test_budget_check(self, client):
        resp = client.post("/profiling/budget/check", json={"component": "database", "elapsed_ms": 50.0})
        assert resp.status_code == 200
        assert resp.json()["within_budget"] is True

        resp = client.post("/profiling/budget/check", json={"component": "database", "elapsed_ms": 999.0})
        assert resp.status_code == 200
        assert resp.json()["within_budget"] is False

    def test_baseline_crud(self, client):
        resp = client.post("/profiling/baseline", json={"name": "v1", "stats": {"db": {"p95_ms": 100}}})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        resp = client.get("/profiling/baseline")
        assert resp.status_code == 200
        assert "v1" in resp.json()["baselines"]

    def test_recommendations(self, client):
        resp = client.get("/profiling/recommendations")
        assert resp.status_code == 200
        assert "recommendations" in resp.json()

    def test_reset(self, client):
        client.post("/profiling/trace")
        resp = client.post("/profiling/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"
        assert client.get("/profiling/stats").json() == {}
