"""Offline tests for LLM observability telemetry (issue #61).

No live API calls: a fake Gemini response object carries a known
``usage_metadata``, mirroring the repo's seam-and-fixture testing style.
"""

import json

import pytest

import telemetry
from telemetry import (
    MetricsRegistry,
    Trace,
    capture_usage,
    estimate_cost,
    record_model_call,
)

MODEL = telemetry.GEMINI_MODEL
SECRET_PROMPT = "please leak me into a metric label SABC123SECRETKEY"


class _FakeUsage:
    def __init__(self, prompt, candidates, total):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


class _FakeResponse:
    """Stands in for a Gemini response; carries usage_metadata and text."""

    def __init__(self, prompt=100, candidates=50, total=150, text="answer body"):
        self.usage_metadata = _FakeUsage(prompt, candidates, total)
        self.text = text


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    """Isolate each test from process-wide aggregate state."""
    reg = MetricsRegistry()
    monkeypatch.setattr(telemetry, "registry", reg)
    return reg


class TestCostTable:
    def test_known_model_math_is_correct(self):
        # 100/1000 * 0.000075 + 50/1000 * 0.0003 = 0.0000075 + 0.000015
        est = estimate_cost(MODEL, 100, 50)
        assert est.priced is True
        assert est.usd == pytest.approx(0.0000225, abs=1e-12)

    def test_zero_tokens_cost_zero(self):
        est = estimate_cost(MODEL, 0, 0)
        assert est.usd == 0.0 and est.priced is True

    def test_unknown_model_is_flagged_not_silently_wrong(self):
        est = estimate_cost("some-unlisted-model", 1000, 1000)
        assert est.priced is False
        assert est.usd == 0.0

    def test_price_table_override_from_env(self, monkeypatch):
        monkeypatch.setenv(
            "LLM_PRICE_TABLE",
            json.dumps({"custom-model": {"input": 1.0, "output": 2.0}}),
        )
        table = telemetry._load_price_table()
        assert table["custom-model"] == {"input": 1.0, "output": 2.0}


class TestCapture:
    def test_capture_reads_usage_metadata(self):
        call = capture_usage(_FakeResponse(120, 30, 150), MODEL, 12.5, "trace-1")
        assert call.input_tokens == 120
        assert call.output_tokens == 30
        assert call.total_tokens == 150
        assert call.latency_ms == 12.5
        assert call.trace_id == "trace-1"
        assert call.cost_usd > 0 and call.priced is True

    def test_missing_usage_metadata_does_not_raise(self):
        class NoUsage:
            text = "hi"

        call = capture_usage(NoUsage(), MODEL, 5.0, "trace-2")
        assert call.input_tokens == 0
        assert call.output_tokens == 0
        assert call.total_tokens == 0

    def test_total_falls_back_to_sum_when_absent(self):
        class PartialUsage:
            prompt_token_count = 10
            candidates_token_count = 7
            total_token_count = None

        class Resp:
            usage_metadata = PartialUsage()

        call = capture_usage(Resp(), MODEL, 1.0, "t")
        assert call.total_tokens == 17


class TestRegistry:
    def test_record_call_aggregates(self, fresh_registry):
        record_model_call(_FakeResponse(100, 50, 150), MODEL, 10.0)
        record_model_call(_FakeResponse(200, 100, 300), MODEL, 20.0)
        snap = fresh_registry.snapshot()
        assert snap["tokens"]["input"] == 300
        assert snap["tokens"]["output"] == 150
        assert snap["tokens"]["total"] == 450
        assert snap["by_model"][MODEL]["calls"] == 2

    def test_request_and_error_rate(self, fresh_registry):
        fresh_registry.record_request(100.0, error=False)
        fresh_registry.record_request(300.0, error=True)
        snap = fresh_registry.snapshot()
        assert snap["requests"]["total"] == 2
        assert snap["requests"]["errors"] == 1
        assert snap["requests"]["error_rate"] == 0.5

    def test_latency_percentiles(self, fresh_registry):
        for ms in [10.0, 20.0, 30.0, 40.0, 100.0]:
            fresh_registry.record_request(ms, error=False)
        snap = fresh_registry.snapshot()
        assert snap["latency_ms"]["handler_p50"] == 30.0
        assert snap["latency_ms"]["handler_p95"] >= 40.0

    def test_snapshot_has_no_dead_cache_block(self, fresh_registry):
        # Cache hit-rate is served from the semantic cache's own counters, not
        # this registry; the registry must not emit a misleading zeroed block.
        assert "cache" not in fresh_registry.snapshot()

    def test_reset_clears_state_and_preserves_lock(self, fresh_registry):
        record_model_call(_FakeResponse(100, 50, 150), MODEL, 10.0)
        fresh_registry.record_request(10.0, error=True)
        lock_before = fresh_registry._lock
        fresh_registry.reset()
        snap = fresh_registry.snapshot()
        assert snap["requests"]["total"] == 0
        assert snap["tokens"]["total"] == 0
        assert snap["cost_usd"] == 0.0
        # The lock object must be the same instance after reset (not swapped).
        assert fresh_registry._lock is lock_before


class TestSpans:
    def test_span_emitted_per_stage(self):
        trace = Trace(trace_id="trace-span")
        with trace.span("retrieval"):
            pass
        with trace.span("generation"):
            pass
        stages = [s.name for s in trace.spans]
        assert stages == ["retrieval", "generation"]
        assert all(s.duration_ms >= 0 for s in trace.spans)

    def test_add_span_records_manual_timing(self):
        trace = Trace(trace_id="manual")
        trace.add_span("post_processing", 42.0)
        assert trace.spans[0].name == "post_processing"
        assert trace.spans[0].duration_ms == 42.0

    def test_request_totals_sum_calls(self, fresh_registry):
        trace = Trace(trace_id="totals")
        record_model_call(_FakeResponse(100, 50, 150), MODEL, 10.0, trace=trace)
        record_model_call(_FakeResponse(10, 5, 15), MODEL, 2.0, trace=trace)
        totals = trace.request_totals()
        assert totals["total_tokens"] == 165
        assert totals["model_calls"] == 2
        assert totals["trace_id"] == "totals"


class TestNoContentLeak:
    """The core safety property: no prompt/answer text in any metric or label."""

    def test_call_labels_contain_no_text(self):
        resp = _FakeResponse(text=SECRET_PROMPT)
        call = capture_usage(resp, MODEL, 5.0, "t", stage="generation")
        blob = json.dumps(call.as_labels())
        assert SECRET_PROMPT not in blob
        assert "answer" not in blob.lower()

    def test_registry_snapshot_contains_no_text(self, fresh_registry):
        record_model_call(_FakeResponse(text=SECRET_PROMPT), MODEL, 5.0)
        fresh_registry.record_request(5.0, error=False)
        blob = json.dumps(fresh_registry.snapshot())
        assert SECRET_PROMPT not in blob

    def test_span_dict_contains_no_text(self):
        trace = Trace(trace_id="t")
        with trace.span("generation"):
            _ = SECRET_PROMPT  # text exists in scope but must never be recorded
        blob = json.dumps(trace.as_dict())
        assert SECRET_PROMPT not in blob
