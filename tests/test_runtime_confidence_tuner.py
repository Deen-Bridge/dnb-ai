"""Tests for runtime confidence threshold tuning (#270)."""

from __future__ import annotations

import pytest

from runtime_confidence_tuner import RuntimeConfidenceTuner


def test_per_topic_and_per_source_thresholds_override_default() -> None:
    tuner = RuntimeConfidenceTuner(default_threshold=0.70)
    tuner.set_topic_threshold("fiqh", 0.85)
    tuner.set_source_threshold("secondary_blog", 0.92)
    assert tuner.effective_threshold(topic="fiqh") == 0.85
    assert tuner.effective_threshold(topic="fiqh", source="secondary_blog") == 0.92
    assert tuner.passes(0.88, topic="fiqh") is True
    assert tuner.passes(0.88, topic="fiqh", source="secondary_blog") is False


def test_rejects_out_of_range_thresholds() -> None:
    tuner = RuntimeConfidenceTuner()
    with pytest.raises(ValueError):
        tuner.set_topic_threshold("aqeedah", 1.2)
