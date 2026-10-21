"""Runtime confidence threshold tuning per Islamic topic and corpus source (#270)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeConfidenceTuner:
    default_threshold: float = 0.70
    topic_thresholds: dict[str, float] = field(default_factory=dict)
    source_thresholds: dict[str, float] = field(default_factory=dict)

    def set_topic_threshold(self, topic: str, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be in [0.0, 1.0].")
        self.topic_thresholds[topic.strip().lower()] = threshold

    def set_source_threshold(self, source: str, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be in [0.0, 1.0].")
        self.source_thresholds[source.strip().lower()] = threshold

    def effective_threshold(self, topic: str | None = None, source: str | None = None) -> float:
        candidates = [self.default_threshold]
        if topic and topic.strip().lower() in self.topic_thresholds:
            candidates.append(self.topic_thresholds[topic.strip().lower()])
        if source and source.strip().lower() in self.source_thresholds:
            candidates.append(self.source_thresholds[source.strip().lower()])
        return max(candidates)

    def passes(self, score: float, topic: str | None = None, source: str | None = None) -> bool:
        return score >= self.effective_threshold(topic=topic, source=source)
