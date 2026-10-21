"""Adaptive teaching based on learning progress (#251)

Responses should not stay at one complexity regardless of how the learner is
doing. This module watches the interaction signals the API already receives —
follow-up wording, feedback ratings, how quickly a learner replies, and
explicit "explain simpler"/"go deeper" requests — and adapts the next
explanation.

Features:
- Comprehension monitoring from interaction patterns (confusion, clarification,
  repetition, mastery, feedback, pacing)
- Dynamic vocabulary and explanation complexity, driven by the same
  :mod:`answer_depth` scale the depth controls use
- More examples when a learner shows confusion, fewer when they are fluent
- Greater depth for advanced learners, with progress tracked per topic
- Recognition of which teaching method works best for each learner
  (visual, analytical, example-based, balanced)
- Pacing that follows learning speed
- Persisted per-user learning profiles

The module exposes an APIRouter at ``/adaptive`` so a client can feed a turn's
observations in and receive the next recommendation. Recommendations include a
ready-to-use prompt block; the same advice is available to ``/chat`` through
``adaptive=true``.

Architecture:
- TeachingStyle / Pace: the dimensions being adapted
- ComprehensionAssessment: what one turn's signals imply
- LearningProfile: persisted progress and preferences for one learner
- AdaptiveTeachingEngine: rules that turn observations into a recommendation
- LearningProfileStore: JSON-backed persistence
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from answer_depth import (
    DEPTH_CONFIGS,
    DEPTH_SCALE,
    DepthLevel,
    build_depth_indicator,
    build_depth_prompt_block,
    normalize_depth,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/adaptive", tags=["adaptive-teaching"])


def _utc_now() -> str:
    """ISO-8601 UTC timestamp for profile updates."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Dimensions of adaptation
# ---------------------------------------------------------------------------


class TeachingStyle(str, Enum):
    """How an explanation is framed, independent of its depth."""

    BALANCED = "balanced"
    VISUAL = "visual"
    ANALYTICAL = "analytical"
    EXAMPLE_BASED = "example_based"


class Pace(str, Enum):
    """How much new material a turn introduces."""

    SLOW = "slow"
    STEADY = "steady"
    FAST = "fast"


STYLE_INSTRUCTIONS: dict[TeachingStyle, str] = {
    TeachingStyle.BALANCED: (
        "Balance the explanation: state the principle, give one example, and add one line of evidence."
    ),
    TeachingStyle.VISUAL: (
        "Structure the explanation visually: short labelled steps, a bullet hierarchy or small table, "
        "spatial metaphors (above/below, outer/inner, branches), and describe any diagram in words."
    ),
    TeachingStyle.ANALYTICAL: (
        "Structure the explanation analytically: state the principle, then the premises, then the "
        "conclusion. Make every inference explicit and define each term before using it."
    ),
    TeachingStyle.EXAMPLE_BASED: (
        "Teach primarily through concrete examples and cases: lead with one worked example, then "
        "extract the general rule from it."
    ),
}

STYLE_DESCRIPTIONS: dict[TeachingStyle, str] = {
    TeachingStyle.BALANCED: "A mix of principle, example, and evidence.",
    TeachingStyle.VISUAL: "Steps, structure, tables, and spatial metaphors.",
    TeachingStyle.ANALYTICAL: "Explicit premises and inferences, definition-first.",
    TeachingStyle.EXAMPLE_BASED: "Worked examples before general rules.",
}

PACE_INSTRUCTIONS: dict[Pace, str] = {
    Pace.SLOW: "Pace this slowly: one idea per paragraph, and recap the previous point before adding the next.",
    Pace.STEADY: "Keep a steady pace with clear transitions between ideas.",
    Pace.FAST: (
        "Move quickly: skip introductory framing and go straight to the distinctions an advanced learner needs."
    ),
}

PACE_DESCRIPTIONS: dict[Pace, str] = {
    Pace.SLOW: "One idea at a time, with recap.",
    Pace.STEADY: "A normal tutorial pace.",
    Pace.FAST: "Compressed, distinction-first delivery.",
}


# ---------------------------------------------------------------------------
# Comprehension signals
# ---------------------------------------------------------------------------

#: Phrases that suggest the previous explanation did not land.
CONFUSION_CUES: tuple[tuple[str, float], ...] = (
    ("i don't understand", 0.35),
    ("i dont understand", 0.35),
    ("do not understand", 0.35),
    ("still don't", 0.3),
    ("still dont", 0.3),
    ("still confused", 0.35),
    ("i'm confused", 0.3),
    ("im confused", 0.3),
    ("i'm lost", 0.3),
    ("im lost", 0.3),
    ("doesn't make sense", 0.3),
    ("does not make sense", 0.3),
    ("what do you mean", 0.3),
    ("too complicated", 0.3),
    ("too technical", 0.3),
    ("not clear", 0.2),
    ("unclear", 0.2),
    ("explain again", 0.25),
    ("in simple terms", 0.2),
    ("simpler", 0.25),
    ("i don't get it", 0.35),
    ("i dont get it", 0.35),
    ("no idea", 0.25),
    ("elaborate", 0.1),
    ("can you explain", 0.15),
    ("huh", 0.15),
    ("??", 0.15),
)

#: Phrases that suggest the learner has absorbed the previous answer.
MASTERY_CUES: tuple[tuple[str, float], ...] = (
    ("got it", 0.35),
    ("makes sense", 0.3),
    ("i understand", 0.3),
    ("understood", 0.25),
    ("clear now", 0.3),
    ("that's clear", 0.3),
    ("that answers", 0.3),
    ("exactly what i needed", 0.3),
    ("that's what i needed", 0.3),
    ("jazakallah", 0.25),
    ("jazakallahu", 0.25),
    ("barakallahu", 0.25),
    ("helpful", 0.2),
    ("perfect", 0.2),
    ("aha", 0.2),
    ("thank you", 0.12),
    ("thanks", 0.1),
)

#: Vocabulary that suggests a learner is ready for more depth.
ADVANCED_CUES: tuple[str, ...] = (
    "difference between",
    "evidence",
    "isnad",
    "usul",
    "madhhab",
    "tafsir",
    "implications",
    "nuance",
    "scholarly",
    "compare",
    "deeper",
    "more detail",
)

STYLE_CUES: dict[TeachingStyle, tuple[str, ...]] = {
    TeachingStyle.VISUAL: (
        "diagram",
        "picture",
        "visual",
        "chart",
        "table",
        "map",
        "step by step",
        "flow",
        "imagine",
    ),
    TeachingStyle.ANALYTICAL: (
        "why",
        "proof",
        "evidence",
        "reason",
        "logic",
        "difference between",
        "compare",
        "principle",
    ),
    TeachingStyle.EXAMPLE_BASED: (
        "example",
        "for instance",
        "practical",
        "case",
        "real life",
        "real-life",
        "scenario",
    ),
}

#: Weighted bounds for the rolling comprehension estimate.
_MIN_COMPREHENSION = 0.05
_MAX_COMPREHENSION = 0.95
_LOW_COMPREHENSION = 0.4
_HIGH_COMPREHENSION = 0.7


@dataclass
class ComprehensionAssessment:
    """What one turn's signals imply about the learner's understanding."""

    score: float  # 0.0 (struggling) to 1.0 (mastered)
    confidence: float  # How much evidence the score rests on
    signals: list[str]
    is_confused: bool
    needs_examples: bool
    style_hint: TeachingStyle | None
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "signals": self.signals,
            "is_confused": self.is_confused,
            "needs_examples": self.needs_examples,
            "style_hint": self.style_hint.value if self.style_hint else None,
            "reasons": self.reasons,
        }


def _cue_hits(text: str, cues: tuple[tuple[str, float], ...]) -> list[tuple[str, float]]:
    """Return matched (phrase, weight) pairs, in cue order."""
    return [(phrase, weight) for phrase, weight in cues if phrase in text]


def detect_style_hint(message: str) -> TeachingStyle | None:
    """Best-effort detection of the teaching style a learner is asking for."""
    lowered = message.lower()
    scores: dict[TeachingStyle, int] = {}
    for style, cues in STYLE_CUES.items():
        hits = sum(1 for cue in cues if cue in lowered)
        if hits:
            scores[style] = hits
    if not scores:
        return None
    return max(scores, key=lambda style: scores[style])


def analyze_comprehension(
    message: str = "",
    feedback: str | None = None,
    depth_request: str | None = None,
    elapsed_seconds: float | None = None,
) -> ComprehensionAssessment:
    """Score one turn's comprehension signals.

    The score starts neutral at 0.5. Confusion wording pulls it down, mastery
    wording and positive feedback pull it up, and explicit depth requests are
    treated as strong evidence in their own direction. ``confidence`` grows
    with the number of cues so a caller can weight the update.
    """
    text = message.lower()
    reasons: list[str] = []
    signals: list[str] = []
    score = 0.5

    confusion_hits = _cue_hits(text, CONFUSION_CUES)
    if confusion_hits:
        penalty = min(0.55, sum(weight for _, weight in confusion_hits))
        score -= penalty
        signals.append("confusion")
        reasons.append("confusion wording: " + ", ".join(sorted({phrase for phrase, _ in confusion_hits}))[:160])

    mastery_hits = _cue_hits(text, MASTERY_CUES)
    if mastery_hits:
        bonus = min(0.45, sum(weight for _, weight in mastery_hits))
        score += bonus
        signals.append("mastery")
        reasons.append("mastery wording: " + ", ".join(sorted({phrase for phrase, _ in mastery_hits}))[:160])

    advanced_hits = [cue for cue in ADVANCED_CUES if cue in text]
    if advanced_hits and not confusion_hits:
        score += min(0.12, 0.04 * len(advanced_hits))
        signals.append("advanced_interest")
        reasons.append("advanced vocabulary: " + ", ".join(sorted(set(advanced_hits))))

    request = (depth_request or "").strip().lower()
    if request == "simpler":
        score -= 0.2
        signals.append("depth_simpler")
        reasons.append("learner explicitly asked for a simpler explanation")
    elif request == "deeper":
        score += 0.15
        signals.append("depth_deeper")
        reasons.append("learner explicitly asked for more depth")

    if feedback == "down":
        score -= 0.3
        signals.append("negative_feedback")
        reasons.append("negative feedback on the previous answer")
    elif feedback == "up":
        score += 0.25
        signals.append("positive_feedback")
        reasons.append("positive feedback on the previous answer")

    if elapsed_seconds is not None and elapsed_seconds > 45 and not mastery_hits:
        score -= 0.05
        signals.append("long_pause")
        reasons.append("learner took a long time before replying")

    score = max(_MIN_COMPREHENSION, min(_MAX_COMPREHENSION, score))
    cue_count = len(confusion_hits) + len(mastery_hits) + len(advanced_hits) + (1 if feedback else 0)
    confidence = min(1.0, 0.15 + 0.15 * cue_count) if cue_count else 0.1

    is_confused = (bool(confusion_hits) and score < 0.5) or score < 0.35
    needs_examples = is_confused or any(cue in text for cue in STYLE_CUES[TeachingStyle.EXAMPLE_BASED])

    return ComprehensionAssessment(
        score=score,
        confidence=confidence,
        signals=signals,
        is_confused=is_confused,
        needs_examples=needs_examples,
        style_hint=detect_style_hint(message),
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Learning profile
# ---------------------------------------------------------------------------


def _default_style_scores() -> dict[str, float]:
    return {style.value: 0.5 for style in TeachingStyle}


@dataclass
class LearningProfile:
    """Persisted learning progress and preferences for one user."""

    user_id: str
    level: DepthLevel = DepthLevel.INTERMEDIATE
    comprehension: float = 0.5
    pace: Pace = Pace.STEADY
    preferred_style: TeachingStyle = TeachingStyle.BALANCED
    style_scores: dict[str, float] = field(default_factory=_default_style_scores)
    topic_levels: dict[str, DepthLevel] = field(default_factory=dict)
    topic_comprehension: dict[str, float] = field(default_factory=dict)
    topic_attempts: dict[str, int] = field(default_factory=dict)
    consecutive_confusions: int = 0
    consecutive_successes: int = 0
    examples_multiplier: float = 1.0
    interactions: int = 0
    avg_response_seconds: float | None = None
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "level": self.level.value,
            "comprehension": round(self.comprehension, 4),
            "pace": self.pace.value,
            "preferred_style": self.preferred_style.value,
            "style_scores": {key: round(value, 4) for key, value in self.style_scores.items()},
            "topic_levels": {topic: level.value for topic, level in self.topic_levels.items()},
            "topic_comprehension": {topic: round(value, 4) for topic, value in self.topic_comprehension.items()},
            "topic_attempts": dict(self.topic_attempts),
            "consecutive_confusions": self.consecutive_confusions,
            "consecutive_successes": self.consecutive_successes,
            "examples_multiplier": round(self.examples_multiplier, 4),
            "interactions": self.interactions,
            "avg_response_seconds": (
                round(self.avg_response_seconds, 2) if self.avg_response_seconds is not None else None
            ),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningProfile":
        """Rebuild a profile, tolerating stale or unknown values from storage."""
        try:
            level = normalize_depth(data.get("level")) or DepthLevel.INTERMEDIATE
        except ValueError:
            level = DepthLevel.INTERMEDIATE
        try:
            style = TeachingStyle(data.get("preferred_style", TeachingStyle.BALANCED.value))
        except ValueError:
            style = TeachingStyle.BALANCED
        try:
            pace = Pace(data.get("pace", Pace.STEADY.value))
        except ValueError:
            pace = Pace.STEADY

        topic_levels: dict[str, DepthLevel] = {}
        for topic, raw in data.get("topic_levels", {}).items():
            try:
                topic_levels[topic] = normalize_depth(raw) or level
            except ValueError:
                logger.warning("Ignoring invalid stored topic level %r for %r", raw, topic)

        scores = _default_style_scores()
        for key, value in data.get("style_scores", {}).items():
            if key in scores and isinstance(value, int | float):
                scores[key] = max(0.0, min(1.0, float(value)))

        return cls(
            user_id=data["user_id"],
            level=level,
            comprehension=max(0.0, min(1.0, float(data.get("comprehension", 0.5)))),
            pace=pace,
            preferred_style=style,
            style_scores=scores,
            topic_levels=topic_levels,
            topic_comprehension={
                topic: max(0.0, min(1.0, float(value))) for topic, value in data.get("topic_comprehension", {}).items()
            },
            topic_attempts={topic: int(value) for topic, value in data.get("topic_attempts", {}).items()},
            consecutive_confusions=int(data.get("consecutive_confusions", 0)),
            consecutive_successes=int(data.get("consecutive_successes", 0)),
            examples_multiplier=max(1.0, min(3.0, float(data.get("examples_multiplier", 1.0)))),
            interactions=int(data.get("interactions", 0)),
            avg_response_seconds=(
                float(data["avg_response_seconds"]) if data.get("avg_response_seconds") is not None else None
            ),
            updated_at=data.get("updated_at", _utc_now()),
        )


@dataclass
class Interaction:
    """One observed turn, as fed to the engine."""

    message: str = ""
    feedback: str | None = None  # "up" or "down"
    depth_request: str | None = None  # "simpler" or "deeper"
    elapsed_seconds: float | None = None
    topic: str | None = None
    style: TeachingStyle | None = None


@dataclass
class TeachingRecommendation:
    """The next explanation's depth, style, pace, and prompt block."""

    level: DepthLevel
    style: TeachingStyle
    pace: Pace
    examples: int
    include_analogies: bool
    comprehension: float
    rationale: list[str]
    prompt_block: str
    indicator: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        config = DEPTH_CONFIGS[self.level]
        return {
            "level": self.level.value,
            "depth": self.indicator,
            "style": self.style.value,
            "style_description": STYLE_DESCRIPTIONS[self.style],
            "pace": self.pace.value,
            "pace_description": PACE_DESCRIPTIONS[self.pace],
            "examples": self.examples,
            "include_analogies": self.include_analogies,
            "comprehension": round(self.comprehension, 4),
            "rationale": self.rationale,
            "prompt_block": self.prompt_block,
            "config": config.to_dict(),
        }


@dataclass
class ObservationResult:
    """The assessment, profile update, and recommendation for one turn."""

    assessment: ComprehensionAssessment
    profile: LearningProfile
    recommendation: TeachingRecommendation
    adjustments: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment": self.assessment.to_dict(),
            "profile": self.profile.to_dict(),
            "recommendation": self.recommendation.to_dict(),
            "adjustments": self.adjustments,
        }


class AdaptiveTeachingEngine:
    """Turns observations into an updated profile and a teaching recommendation."""

    comprehension_alpha = 0.35
    seconds_alpha = 0.4

    def _style_instruction(self, style: TeachingStyle) -> str:
        return STYLE_INSTRUCTIONS[style]

    def _next_pace(self, profile: LearningProfile) -> Pace:
        """Pick a pace from measured speed and comprehension, never from speed alone."""
        seconds = profile.avg_response_seconds
        if profile.comprehension < _LOW_COMPREHENSION:
            return Pace.SLOW
        if seconds is not None and seconds > 25:
            return Pace.SLOW
        if seconds is not None and seconds < 8 and profile.comprehension >= _HIGH_COMPREHENSION:
            return Pace.FAST
        return Pace.STEADY

    def _update_style_scores(self, profile: LearningProfile, style: TeachingStyle | None, feedback: str | None) -> None:
        """Blend this turn's feedback into the running effectiveness of a style."""
        if style is None or feedback not in ("up", "down"):
            return
        target = 1.0 if feedback == "up" else 0.0
        previous = profile.style_scores.get(style.value, 0.5)
        profile.style_scores[style.value] = max(0.0, min(1.0, 0.7 * previous + 0.3 * target))

        best_style, best_score = max(
            ((TeachingStyle(key), value) for key, value in profile.style_scores.items()),
            key=lambda item: item[1],
        )
        # Follow measured effectiveness once it is meaningfully above neutral;
        # back off a style that is measurably failing.
        if feedback == "up" and best_score >= 0.65:
            profile.preferred_style = best_style
        elif feedback == "down" and profile.style_scores.get(profile.preferred_style.value, 0.5) <= 0.35:
            profile.preferred_style = TeachingStyle.BALANCED

    def observe(self, profile: LearningProfile, interaction: Interaction) -> ObservationResult:
        """Apply one observation to *profile* and return the next recommendation."""
        assessment = analyze_comprehension(
            message=interaction.message,
            feedback=interaction.feedback,
            depth_request=interaction.depth_request,
            elapsed_seconds=interaction.elapsed_seconds,
        )
        adjustments: list[str] = []

        profile.interactions += 1
        profile.comprehension = (
            self.comprehension_alpha * assessment.score + (1 - self.comprehension_alpha) * profile.comprehension
        )

        if interaction.elapsed_seconds is not None:
            previous_seconds = profile.avg_response_seconds
            if previous_seconds is None:
                profile.avg_response_seconds = interaction.elapsed_seconds
            else:
                profile.avg_response_seconds = (
                    self.seconds_alpha * interaction.elapsed_seconds + (1 - self.seconds_alpha) * previous_seconds
                )

        # --- Explicit depth requests take precedence over inferred ones ------
        level_index = DEPTH_SCALE.index(profile.level)
        request = (interaction.depth_request or "").strip().lower()
        explicit_depth_request = request in ("simpler", "deeper")
        if request == "simpler":
            new_index = max(0, level_index - 1)
            profile.level = DEPTH_SCALE[new_index]
            profile.examples_multiplier = min(3.0, profile.examples_multiplier + 0.5)
            adjustments.append("learner asked for a simpler explanation: moved one level down")
        elif request == "deeper":
            new_index = min(len(DEPTH_SCALE) - 1, level_index + 1)
            profile.level = DEPTH_SCALE[new_index]
            adjustments.append("learner asked for more depth: moved one level up")

        # --- Inferred adaptation from confusion / mastery ---------------------
        if assessment.is_confused:
            profile.consecutive_confusions += 1
            profile.consecutive_successes = 0
            profile.examples_multiplier = min(3.0, profile.examples_multiplier + 0.5)
            adjustments.append("confusion detected: adding examples")
            if (
                not explicit_depth_request
                and profile.consecutive_confusions >= 1
                and profile.comprehension < _LOW_COMPREHENSION
                and DEPTH_SCALE.index(profile.level) > 0
            ):
                profile.level = DEPTH_SCALE[DEPTH_SCALE.index(profile.level) - 1]
                adjustments.append("comprehension below 0.40: reduced depth one level")
            if profile.preferred_style is TeachingStyle.ANALYTICAL:
                profile.preferred_style = TeachingStyle.EXAMPLE_BASED
                adjustments.append("analytical framing did not land: switching to examples")
        elif assessment.signals and ("mastery" in assessment.signals or "positive_feedback" in assessment.signals):
            profile.consecutive_successes += 1
            profile.consecutive_confusions = 0
            profile.examples_multiplier = max(1.0, profile.examples_multiplier - 0.25)
            if (
                not explicit_depth_request
                and profile.consecutive_successes >= 2
                and profile.comprehension >= _HIGH_COMPREHENSION
                and DEPTH_SCALE.index(profile.level) < len(DEPTH_SCALE) - 1
            ):
                profile.level = DEPTH_SCALE[DEPTH_SCALE.index(profile.level) + 1]
                adjustments.append("two consecutive successful turns: increased depth one level")

        # --- Style: explicit preference, detected hint, then measured results --
        if interaction.style is not None:
            profile.preferred_style = interaction.style
            adjustments.append(f"learner selected the {interaction.style.value} teaching style")
        elif assessment.style_hint is not None:
            profile.preferred_style = assessment.style_hint
            adjustments.append(f"question wording suggests the {assessment.style_hint.value} teaching style")
        self._update_style_scores(profile, interaction.style or assessment.style_hint, interaction.feedback)

        # --- Pacing ----------------------------------------------------------
        new_pace = self._next_pace(profile)
        if new_pace != profile.pace:
            profile.pace = new_pace
            adjustments.append(f"pace adjusted to {new_pace.value}")

        # --- Topic progress --------------------------------------------------
        if interaction.topic:
            profile.topic_levels[interaction.topic] = profile.level
            profile.topic_attempts[interaction.topic] = profile.topic_attempts.get(interaction.topic, 0) + 1
            previous_topic = profile.topic_comprehension.get(interaction.topic, profile.comprehension)
            profile.topic_comprehension[interaction.topic] = (
                self.comprehension_alpha * assessment.score + (1 - self.comprehension_alpha) * previous_topic
            )

        profile.updated_at = _utc_now()
        recommendation = self.recommend(profile, topic=interaction.topic)
        return ObservationResult(
            assessment=assessment,
            profile=profile,
            recommendation=recommendation,
            adjustments=adjustments,
        )

    def recommend(
        self,
        profile: LearningProfile,
        topic: str | None = None,
        style: TeachingStyle | None = None,
    ) -> TeachingRecommendation:
        """Build the recommendation implied by a profile, without changing it."""
        level = profile.topic_levels.get(topic, profile.level) if topic else profile.level
        config = DEPTH_CONFIGS[level]
        rationale: list[str] = []

        if topic and topic in profile.topic_levels:
            rationale.append(f"topic '{topic}' is tracked at the {level.value} level")
        rationale.append(f"rolling comprehension estimate is {profile.comprehension:.2f}")

        chosen_style = style or profile.preferred_style
        if profile.comprehension < _LOW_COMPREHENSION and chosen_style is TeachingStyle.ANALYTICAL:
            chosen_style = TeachingStyle.EXAMPLE_BASED
            rationale.append("comprehension is low, so examples take priority over analysis")

        examples = max(1, round(config.examples_per_concept * profile.examples_multiplier))
        if config.examples_per_concept == 0:
            examples = max(0, round(profile.examples_multiplier) - 1)
        if examples > config.examples_per_concept:
            rationale.append(
                f"{examples} examples instead of {config.examples_per_concept} because more scaffolding is needed"
            )

        pace = profile.pace
        seconds = profile.avg_response_seconds
        if seconds is not None:
            rationale.append(f"average reply time is {seconds:.1f}s")
        rationale.append(
            f"{chosen_style.value} style currently scores {profile.style_scores.get(chosen_style.value, 0.5):.2f}"
        )

        prompt_block = "\n".join(
            [
                build_depth_prompt_block(config).strip(),
                "",
                f"Teaching style: {chosen_style.value}",
                f"- {self._style_instruction(chosen_style)}",
                "",
                f"Pacing: {pace.value}",
                f"- {PACE_INSTRUCTIONS[pace]}",
                f"- Include {examples} concrete example(s) per concept.",
            ]
        )

        return TeachingRecommendation(
            level=level,
            style=chosen_style,
            pace=pace,
            examples=examples,
            include_analogies=config.use_analogies or profile.comprehension < _LOW_COMPREHENSION,
            comprehension=profile.comprehension,
            rationale=rationale,
            prompt_block=prompt_block,
            indicator=build_depth_indicator(level).to_dict(),
        )


class LearningProfileStore:
    """JSON-backed store for per-user learning profiles."""

    def __init__(self, data_file: str | None = None) -> None:
        self._data_file = str(data_file or os.getenv("ADAPTIVE_TEACHING_FILE") or "./data/adaptive_teaching.json")
        self._profiles: dict[str, LearningProfile] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._data_file):
            return
        try:
            with open(self._data_file) as handle:
                data = json.load(handle)
            for raw in data.get("profiles", []):
                profile = LearningProfile.from_dict(raw)
                self._profiles[profile.user_id] = profile
            logger.info("Loaded %d learning profiles", len(self._profiles))
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not stop the app
            logger.warning("Failed to load learning profiles: %s", exc)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._data_file) or ".", exist_ok=True)
        with open(self._data_file, "w") as handle:
            json.dump({"profiles": [p.to_dict() for p in self._profiles.values()]}, handle, indent=2)

    def get(self, user_id: str) -> LearningProfile:
        """Return the user's profile, creating a neutral one when absent."""
        if user_id not in self._profiles:
            self._profiles[user_id] = LearningProfile(user_id=user_id)
        return self._profiles[user_id]

    def save(self, profile: LearningProfile) -> None:
        """Persist a profile."""
        self._profiles[profile.user_id] = profile
        self._save()

    def reset(self, user_id: str) -> bool:
        """Delete a user's progress and preferences."""
        if user_id in self._profiles:
            del self._profiles[user_id]
            self._save()
            return True
        return False


_engine: AdaptiveTeachingEngine | None = None
_profile_store: LearningProfileStore | None = None


def get_teaching_engine() -> AdaptiveTeachingEngine:
    """Singleton adaptation engine."""
    global _engine
    if _engine is None:
        _engine = AdaptiveTeachingEngine()
    return _engine


def get_profile_store() -> LearningProfileStore:
    """Singleton profile store."""
    global _profile_store
    if _profile_store is None:
        _profile_store = LearningProfileStore()
    return _profile_store


def _style_or_400(value: str | None) -> TeachingStyle | None:
    if value is None:
        return None
    try:
        return TeachingStyle(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(style.value for style in TeachingStyle)
        raise HTTPException(status_code=400, detail=f"Unknown teaching style. Choose one of: {valid}.") from exc


def _level_or_400(value: str | None) -> DepthLevel | None:
    if value is None:
        return None
    try:
        return normalize_depth(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class StyleInfo(BaseModel):
    style: str
    description: str


class StylesResponse(BaseModel):
    styles: list[StyleInfo]
    paces: list[dict[str, str]]
    depths: list[str]


class RecommendationResponse(BaseModel):
    """A teaching recommendation for one user (and optional topic)."""

    recommendation: dict[str, Any]
    profile: dict[str, Any]


class ObserveRequest(BaseModel):
    """One turn's observable signals about a learner."""

    user_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(default="", max_length=8000)
    feedback: str | None = Field(default=None, description="'up' or 'down', when the client collected it")
    depth_request: str | None = Field(default=None, description="'simpler' or 'deeper'")
    elapsed_seconds: float | None = Field(default=None, ge=0, le=3600)
    topic: str | None = Field(default=None, max_length=120)
    style: str | None = Field(default=None, description="Explicit style choice (visual/analytical/example_based)")

    @field_validator("feedback")
    @classmethod
    def feedback_must_be_valid(cls, value: str | None) -> str | None:
        if value is not None and value not in ("up", "down"):
            raise ValueError("feedback must be 'up' or 'down'")
        return value

    @field_validator("depth_request")
    @classmethod
    def depth_request_must_be_valid(cls, value: str | None) -> str | None:
        if value is not None and value.strip().lower() not in ("simpler", "deeper"):
            raise ValueError("depth_request must be 'simpler' or 'deeper'")
        return value


class ObserveResponse(BaseModel):
    """Assessment, updated profile, and next recommendation for a turn."""

    assessment: dict[str, Any]
    adjustments: list[str]
    profile: dict[str, Any]
    recommendation: dict[str, Any]


class RecommendRequest(BaseModel):
    """Ask for the current recommendation without recording an observation."""

    user_id: str = Field(..., min_length=1, max_length=128)
    topic: str | None = Field(default=None, max_length=120)
    style: str | None = Field(default=None, description="Override the stored style for this answer")


class ProfileUpdateRequest(BaseModel):
    """Manual override of a learner's preferences."""

    style: str | None = None
    level: str | None = None
    pace: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/styles", response_model=StylesResponse)
async def list_styles() -> StylesResponse:
    """Return the teaching styles, paces, and depth levels the engine can use."""
    return StylesResponse(
        styles=[StyleInfo(style=style.value, description=STYLE_DESCRIPTIONS[style]) for style in TeachingStyle],
        paces=[{"pace": pace.value, "description": PACE_DESCRIPTIONS[pace]} for pace in Pace],
        depths=[level.value for level in DEPTH_SCALE],
    )


@router.get("/profile/{user_id}", response_model=RecommendationResponse)
async def get_profile(user_id: str) -> RecommendationResponse:
    """Return a learner's profile and the recommendation it currently implies."""
    store = get_profile_store()
    profile = store.get(user_id)
    recommendation = get_teaching_engine().recommend(profile)
    return RecommendationResponse(recommendation=recommendation.to_dict(), profile=profile.to_dict())


@router.put("/profile/{user_id}", response_model=RecommendationResponse)
async def update_profile(user_id: str, body: ProfileUpdateRequest) -> RecommendationResponse:
    """Set a learner's preferred style, level, or pace explicitly."""
    store = get_profile_store()
    profile = store.get(user_id)

    style = _style_or_400(body.style)
    if style is not None:
        profile.preferred_style = style
    level = _level_or_400(body.level)
    if level is not None:
        profile.level = level
    if body.pace is not None:
        try:
            profile.pace = Pace(body.pace.strip().lower())
        except ValueError as exc:
            valid = ", ".join(pace.value for pace in Pace)
            raise HTTPException(status_code=400, detail=f"Unknown pace. Choose one of: {valid}.") from exc

    store.save(profile)
    recommendation = get_teaching_engine().recommend(profile)
    return RecommendationResponse(recommendation=recommendation.to_dict(), profile=profile.to_dict())


@router.delete("/profile/{user_id}")
async def reset_profile(user_id: str) -> dict[str, Any]:
    """Delete a learner's stored progress."""
    removed = get_profile_store().reset(user_id)
    return {"user_id": user_id, "reset": removed}


@router.post("/observe", response_model=ObserveResponse)
async def observe(body: ObserveRequest) -> ObserveResponse:
    """Record a turn's signals and return the adapted teaching recommendation."""
    store = get_profile_store()
    profile = store.get(body.user_id)
    interaction = Interaction(
        message=body.message,
        feedback=body.feedback,
        depth_request=body.depth_request,
        elapsed_seconds=body.elapsed_seconds,
        topic=body.topic,
        style=_style_or_400(body.style),
    )
    result = get_teaching_engine().observe(profile, interaction)
    store.save(result.profile)
    payload = result.to_dict()
    return ObserveResponse(
        assessment=payload["assessment"],
        adjustments=payload["adjustments"],
        profile=payload["profile"],
        recommendation=payload["recommendation"],
    )


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(body: RecommendRequest) -> RecommendationResponse:
    """Return the recommendation for a learner without recording a turn."""
    store = get_profile_store()
    profile = store.get(body.user_id)
    style = _style_or_400(body.style)
    result = get_teaching_engine().recommend(profile, topic=body.topic, style=style)
    return RecommendationResponse(recommendation=result.to_dict(), profile=profile.to_dict())
