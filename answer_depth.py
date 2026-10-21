"""Answer Depth Customization (#221, #263)

Enable users to adjust answer granularity across a five-level scale, from an
ELI5 explanation to a full scholarly analysis, and to move an existing answer
up or down that scale without rephrasing their question.

Features:
- Five canonical depth levels: ELI5, beginner, intermediate, advanced,
  scholarly
- Interactive "Explain Simpler" / "Go Deeper" controls that move one step at
  a time and keep the original answer alongside every variation
- Conversation-aware adjustments: every adjustment is tied to a chat and
  returns the follow-up prompt that regenerates the answer in context
- Dynamic terminology density, analogy and example counts, citation density,
  and syntactic complexity
- Controlled scholarly disagreement, historical context, madhhab comparison,
  and optional Arabic text
- Per-user default levels, topic-specific levels, and preference overrides
- A visual depth indicator (badge, icon, colour, position on the scale) for
  every response
- Backward-compatible aliases for the original brief/standard/detailed scale

Architecture:
- DepthLevel / DEPTH_SCALE: the ordered scale
- DepthConfig: configuration for each depth level
- DepthIndicator: UI-facing description of where an answer sits
- DepthAdapter: adapts answers and prompts to a depth setting
- UserDepthPreferences / UserPreferencesStore: persisted per-user settings
- DepthSessionStore: per-chat original answers and depth variations

The module exposes an APIRouter at ``/depth`` so the frontend can render the
"Explain Simpler"/"Go Deeper" affordances, show the current depth, and store
each user's preferred level.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/depth", tags=["answer-depth"])


def _utc_now() -> str:
    """ISO-8601 UTC timestamp used on stored variations."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Depth levels
# ---------------------------------------------------------------------------


class DepthLevel(str, Enum):
    """The five supported answer depth levels, ordered simplest to deepest."""

    ELI5 = "eli5"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    SCHOLARLY = "scholarly"


#: Ordered canonical scale. Index + 1 is the 1-based position shown to users.
DEPTH_SCALE: tuple[DepthLevel, ...] = (
    DepthLevel.ELI5,
    DepthLevel.BEGINNER,
    DepthLevel.INTERMEDIATE,
    DepthLevel.ADVANCED,
    DepthLevel.SCHOLARLY,
)

#: Accepted aliases, including the pre-#263 names (brief/standard/detailed) so
#: existing clients and stored preferences keep working unchanged.
DEPTH_ALIASES: dict[str, DepthLevel] = {
    "brief": DepthLevel.BEGINNER,
    "short": DepthLevel.BEGINNER,
    "simple": DepthLevel.ELI5,
    "kid": DepthLevel.ELI5,
    "kids": DepthLevel.ELI5,
    "child": DepthLevel.ELI5,
    "beginner": DepthLevel.BEGINNER,
    "standard": DepthLevel.INTERMEDIATE,
    "normal": DepthLevel.INTERMEDIATE,
    "intermediate": DepthLevel.INTERMEDIATE,
    "detailed": DepthLevel.ADVANCED,
    "deep": DepthLevel.ADVANCED,
    "advanced": DepthLevel.ADVANCED,
    "expert": DepthLevel.SCHOLARLY,
    "scholar": DepthLevel.SCHOLARLY,
    "scholarly": DepthLevel.SCHOLARLY,
}


def normalize_depth(value: str | DepthLevel | None) -> DepthLevel | None:
    """Resolve a user-supplied depth name to a :class:`DepthLevel`.

    Accepts the canonical names, the legacy brief/standard/detailed names, and
    a few common synonyms. Returns ``None`` when *value* is ``None``. Raises
    ``ValueError`` with an actionable message for anything else.
    """
    if value is None:
        return None
    if isinstance(value, DepthLevel):
        return value
    key = str(value).strip().lower()
    if not key:
        return None
    if key in DEPTH_ALIASES:
        return DEPTH_ALIASES[key]
    try:
        return DepthLevel(key)
    except ValueError as exc:
        valid = ", ".join(level.value for level in DEPTH_SCALE)
        raise ValueError(f"Unknown depth level {value!r}. Choose one of: {valid}.") from exc


def depth_position(level: DepthLevel) -> int:
    """1-based position of *level* on the five-level scale."""
    return DEPTH_SCALE.index(level) + 1


def step_depth(level: DepthLevel, direction: int) -> DepthLevel:
    """Move *level* one step down (``-1``) or up (``+1``), clamped to the scale."""
    index = DEPTH_SCALE.index(level)
    return DEPTH_SCALE[min(max(index + direction, 0), len(DEPTH_SCALE) - 1)]


# ---------------------------------------------------------------------------
# Per-level configuration
# ---------------------------------------------------------------------------


@dataclass
class DepthConfig:
    """Configuration settings for a depth level."""

    level: DepthLevel
    label: str
    description: str
    max_length: int  # Soft maximum response length in tokens
    citation_density: float  # 0.0 to 1.0, how frequently to cite
    include_arabic: bool  # Include Arabic text
    include_transliteration: bool  # Include transliteration
    include_scholarly_disagreements: bool  # Show ikhtilaf (scholarly differences)
    include_historical_context: bool  # Add historical background
    include_madhhab_comparison: bool  # Compare school of thought positions
    terminology_complexity: float  # 0.0 (simple) to 1.0 (technical)
    evidence_detail: float  # 0.0 (conclusions only) to 1.0 (full reasoning)
    collapsible_sections: bool  # Use expandable sections for extra detail
    summary_position: str  # "start", "end", or "none"
    examples_per_concept: int  # Concrete examples to include per concept
    use_analogies: bool  # Whether to lean on everyday analogies
    reading_grade: int  # Approximate reading grade the prose should target

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "label": self.label,
            "description": self.description,
            "max_length": self.max_length,
            "citation_density": self.citation_density,
            "include_arabic": self.include_arabic,
            "include_transliteration": self.include_transliteration,
            "include_scholarly_disagreements": self.include_scholarly_disagreements,
            "include_historical_context": self.include_historical_context,
            "include_madhhab_comparison": self.include_madhhab_comparison,
            "terminology_complexity": self.terminology_complexity,
            "evidence_detail": self.evidence_detail,
            "collapsible_sections": self.collapsible_sections,
            "summary_position": self.summary_position,
            "examples_per_concept": self.examples_per_concept,
            "use_analogies": self.use_analogies,
            "reading_grade": self.reading_grade,
        }


DEPTH_CONFIGS: dict[DepthLevel, DepthConfig] = {
    DepthLevel.ELI5: DepthConfig(
        level=DepthLevel.ELI5,
        label="Explain like I'm 5",
        description="The shortest, plainest version: everyday words, a concrete analogy, no jargon.",
        max_length=150,
        citation_density=0.0,
        include_arabic=False,
        include_transliteration=False,
        include_scholarly_disagreements=False,
        include_historical_context=False,
        include_madhhab_comparison=False,
        terminology_complexity=0.05,
        evidence_detail=0.05,
        collapsible_sections=False,
        summary_position="none",
        examples_per_concept=3,
        use_analogies=True,
        reading_grade=5,
    ),
    DepthLevel.BEGINNER: DepthConfig(
        level=DepthLevel.BEGINNER,
        label="Beginner",
        description="A short, friendly answer in familiar language, with one or two examples.",
        max_length=250,
        citation_density=0.15,
        include_arabic=False,
        include_transliteration=True,
        include_scholarly_disagreements=False,
        include_historical_context=False,
        include_madhhab_comparison=False,
        terminology_complexity=0.25,
        evidence_detail=0.25,
        collapsible_sections=False,
        summary_position="start",
        examples_per_concept=2,
        use_analogies=True,
        reading_grade=8,
    ),
    DepthLevel.INTERMEDIATE: DepthConfig(
        level=DepthLevel.INTERMEDIATE,
        label="Intermediate",
        description="A clear, balanced answer with the key evidence and correct terminology explained.",
        max_length=450,
        citation_density=0.45,
        include_arabic=False,
        include_transliteration=True,
        include_scholarly_disagreements=False,
        include_historical_context=False,
        include_madhhab_comparison=False,
        terminology_complexity=0.5,
        evidence_detail=0.5,
        collapsible_sections=False,
        summary_position="start",
        examples_per_concept=1,
        use_analogies=True,
        reading_grade=11,
    ),
    DepthLevel.ADVANCED: DepthConfig(
        level=DepthLevel.ADVANCED,
        label="Advanced",
        description="A comprehensive answer with Arabic evidence, reasoning, and scholarly context.",
        max_length=900,
        citation_density=0.7,
        include_arabic=True,
        include_transliteration=True,
        include_scholarly_disagreements=True,
        include_historical_context=True,
        include_madhhab_comparison=False,
        terminology_complexity=0.8,
        evidence_detail=0.8,
        collapsible_sections=True,
        summary_position="start",
        examples_per_concept=1,
        use_analogies=False,
        reading_grade=14,
    ),
    DepthLevel.SCHOLARLY: DepthConfig(
        level=DepthLevel.SCHOLARLY,
        label="Scholarly",
        description="A rigorous analysis with full evidence chains, ikhtilaf, and madhhab comparison.",
        max_length=1600,
        citation_density=1.0,
        include_arabic=True,
        include_transliteration=True,
        include_scholarly_disagreements=True,
        include_historical_context=True,
        include_madhhab_comparison=True,
        terminology_complexity=1.0,
        evidence_detail=1.0,
        collapsible_sections=True,
        summary_position="start",
        examples_per_concept=0,
        use_analogies=False,
        reading_grade=18,
    ),
}


@dataclass(frozen=True)
class DepthIndicator:
    """UI-facing description of an answer's current depth.

    The frontend uses this to render the depth badge and to enable or disable
    the "Explain Simpler"/"Go Deeper" buttons at the ends of the scale.
    """

    level: DepthLevel
    label: str
    description: str
    position: int
    total: int
    scale: tuple[str, ...]
    can_go_simpler: bool
    can_go_deeper: bool
    visual: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "label": self.label,
            "description": self.description,
            "position": self.position,
            "total": self.total,
            "scale": list(self.scale),
            "can_go_simpler": self.can_go_simpler,
            "can_go_deeper": self.can_go_deeper,
            "visual": dict(self.visual),
        }


_DEPTH_VISUALS: dict[DepthLevel, dict[str, str]] = {
    DepthLevel.ELI5: {"icon": "\U0001f331", "badge": "ELI5", "color": "#34d399"},
    DepthLevel.BEGINNER: {"icon": "\U0001f4d8", "badge": "Beginner", "color": "#60a5fa"},
    DepthLevel.INTERMEDIATE: {"icon": "\U0001f4d7", "badge": "Intermediate", "color": "#a78bfa"},
    DepthLevel.ADVANCED: {"icon": "\U0001f393", "badge": "Advanced", "color": "#f59e0b"},
    DepthLevel.SCHOLARLY: {"icon": "\U0001f3db", "badge": "Scholarly", "color": "#ef4444"},
}


def build_depth_indicator(level: DepthLevel) -> DepthIndicator:
    """Build the visual indicator for *level*."""
    config = DEPTH_CONFIGS[level]
    position = depth_position(level)
    return DepthIndicator(
        level=level,
        label=config.label,
        description=config.description,
        position=position,
        total=len(DEPTH_SCALE),
        scale=tuple(item.value for item in DEPTH_SCALE),
        can_go_simpler=position > 1,
        can_go_deeper=position < len(DEPTH_SCALE),
        visual=dict(_DEPTH_VISUALS[level]),
    )


@dataclass
class UserDepthPreferences:
    """User's answer depth preferences."""

    user_id: str
    default_level: DepthLevel = DepthLevel.INTERMEDIATE
    custom_overrides: dict[str, Any] = field(default_factory=dict)
    per_topic_levels: dict[str, DepthLevel] = field(default_factory=dict)
    auto_expand_sections: bool = False
    always_show_arabic: bool = False
    preferred_madhhab: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "default_level": self.default_level.value,
            "custom_overrides": self.custom_overrides,
            "per_topic_levels": {k: v.value for k, v in self.per_topic_levels.items()},
            "auto_expand_sections": self.auto_expand_sections,
            "always_show_arabic": self.always_show_arabic,
            "preferred_madhhab": self.preferred_madhhab,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserDepthPreferences":
        raw_level = data.get("default_level", DepthLevel.INTERMEDIATE.value)
        default_level = normalize_depth(raw_level) or DepthLevel.INTERMEDIATE
        topic_levels: dict[str, DepthLevel] = {}
        for topic, raw in data.get("per_topic_levels", {}).items():
            try:
                topic_levels[topic] = normalize_depth(raw) or default_level
            except ValueError:
                # A stale value from an older scale should not make the whole
                # preference file unreadable.
                logger.warning("Ignoring invalid stored depth %r for topic %r", raw, topic)
        return cls(
            user_id=data["user_id"],
            default_level=default_level,
            custom_overrides=data.get("custom_overrides", {}),
            per_topic_levels=topic_levels,
            auto_expand_sections=data.get("auto_expand_sections", False),
            always_show_arabic=data.get("always_show_arabic", False),
            preferred_madhhab=data.get("preferred_madhhab"),
        )


@dataclass
class AnswerSection:
    """A section of an answer that can be expanded/collapsed."""

    id: str
    title: str
    content: str
    level: str  # "primary", "secondary", "tertiary"
    initially_expanded: bool = True
    arabic_content: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StructuredAnswer:
    """A hierarchically structured answer for progressive disclosure."""

    summary: str | None
    main_content: str
    sections: list[AnswerSection] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "main_content": self.main_content,
            "sections": [
                {
                    "id": s.id,
                    "title": s.title,
                    "content": s.content,
                    "level": s.level,
                    "initially_expanded": s.initially_expanded,
                    "arabic_content": s.arabic_content,
                    "citations": s.citations,
                }
                for s in self.sections
            ],
            "citations": self.citations,
            "metadata": self.metadata,
        }

    def to_flat_text(self, include_collapsed: bool = True) -> str:
        """Convert to plain text, optionally including collapsed sections."""
        parts = []
        if self.summary:
            parts.append(self.summary)
        parts.append(self.main_content)
        for section in self.sections:
            if include_collapsed or section.initially_expanded:
                parts.append(f"\n## {section.title}\n{section.content}")
        return "\n\n".join(parts)


class DepthAdapter:
    """Adapts answer generation based on depth settings."""

    def __init__(self) -> None:
        self._configs = DEPTH_CONFIGS.copy()

    def get_config(self, level: DepthLevel) -> DepthConfig:
        """Get the configuration for a depth level."""
        return self._configs[level]

    def get_effective_config(
        self,
        level: DepthLevel,
        preferences: UserDepthPreferences | None = None,
    ) -> DepthConfig:
        """Get effective configuration considering user preferences."""
        base_config = self._configs[level]

        if not preferences:
            return base_config

        # Apply user overrides
        overrides = preferences.custom_overrides
        if not overrides and not preferences.always_show_arabic:
            return base_config

        # Create modified config
        return DepthConfig(
            level=base_config.level,
            label=base_config.label,
            description=base_config.description,
            max_length=overrides.get("max_length", base_config.max_length),
            citation_density=overrides.get("citation_density", base_config.citation_density),
            include_arabic=preferences.always_show_arabic or base_config.include_arabic,
            include_transliteration=overrides.get("include_transliteration", base_config.include_transliteration),
            include_scholarly_disagreements=overrides.get(
                "include_scholarly_disagreements", base_config.include_scholarly_disagreements
            ),
            include_historical_context=overrides.get(
                "include_historical_context", base_config.include_historical_context
            ),
            include_madhhab_comparison=overrides.get(
                "include_madhhab_comparison", base_config.include_madhhab_comparison
            ),
            terminology_complexity=overrides.get("terminology_complexity", base_config.terminology_complexity),
            evidence_detail=overrides.get("evidence_detail", base_config.evidence_detail),
            collapsible_sections=overrides.get("collapsible_sections", base_config.collapsible_sections),
            summary_position=overrides.get("summary_position", base_config.summary_position),
            examples_per_concept=overrides.get("examples_per_concept", base_config.examples_per_concept),
            use_analogies=overrides.get("use_analogies", base_config.use_analogies),
            reading_grade=overrides.get("reading_grade", base_config.reading_grade),
        )

    def build_prompt_instructions(self, config: DepthConfig) -> str:
        """Build prompt instructions based on depth configuration."""
        instructions: list[str] = []

        # Length and vocabulary guidance
        if config.level == DepthLevel.ELI5:
            instructions.append(
                "Explain as you would to a curious five-year-old: short sentences, everyday words, "
                "no unexplained jargon, and a familiar everyday analogy."
            )
        elif config.level == DepthLevel.BEGINNER:
            instructions.append(
                "Give a short, friendly explanation for someone new to the topic. Introduce any "
                "technical term only after explaining it in plain language."
            )
        elif config.level == DepthLevel.INTERMEDIATE:
            instructions.append(
                "Give a clear, balanced answer that uses correct terminology but explains each term "
                "the first time it appears."
            )
        elif config.level == DepthLevel.ADVANCED:
            instructions.append(
                "Give a comprehensive answer with precise terminology, Arabic evidence, and explicit "
                "reasoning from premise to conclusion."
            )
        else:  # SCHOLARLY
            instructions.append(
                "Give an academically rigorous answer with full evidence chains, source criticism, "
                "and consideration of diverse scholarly perspectives."
            )

        instructions.append(
            f"Keep the answer to roughly {config.max_length} tokens unless the question genuinely requires more."
        )

        # Citation instructions
        if config.citation_density < 0.1:
            instructions.append("Do not add citations; this depth is for intuition, not evidence.")
        elif config.citation_density < 0.3:
            instructions.append("Include citations only for direct quotes.")
        elif config.citation_density < 0.6:
            instructions.append("Include citations for major claims and quotes.")
        else:
            instructions.append("Cite sources extensively for all claims and evidence.")

        # Arabic text
        if config.include_arabic:
            instructions.append("Include relevant Arabic text with transliteration and translation.")
        elif config.include_transliteration:
            instructions.append("Include transliteration of key Arabic terms.")

        # Scholarly content
        if config.include_scholarly_disagreements:
            instructions.append("Discuss scholarly differences of opinion (ikhtilaf) where relevant.")

        if config.include_historical_context:
            instructions.append("Provide historical context for the ruling or concept.")

        if config.include_madhhab_comparison:
            instructions.append(
                "Compare positions across the four major schools of thought "
                "(Hanafi, Maliki, Shafi'i, Hanbali) where applicable."
            )

        # Terminology and examples
        if config.terminology_complexity < 0.3:
            instructions.append("Use simple, accessible language for a general audience.")
        elif config.terminology_complexity > 0.7:
            instructions.append("Use precise scholarly terminology without simplifying it away.")

        if config.examples_per_concept > 0:
            plural = "example" if config.examples_per_concept == 1 else "examples"
            instructions.append(
                f"Include about {config.examples_per_concept} concrete {plural} per concept, drawn from "
                "everyday Muslim life where possible."
            )

        if config.use_analogies:
            instructions.append("Use at least one analogy to make the core idea memorable.")

        instructions.append(f"Write at a reading level of roughly grade {config.reading_grade}.")

        return "\n".join(f"- {item}" for item in instructions)

    def compress_answer(
        self,
        full_answer: StructuredAnswer,
        target_level: DepthLevel,
    ) -> StructuredAnswer:
        """Compress a detailed answer to a simpler depth level.

        This is a structural fallback for clients that cannot regenerate: it
        keeps the highest-signal material and drops the sections the target
        depth would not show. Regenerating with the returned instructions is
        preferred, because only the model can rewrite terminology and add
        analogies faithfully.
        """
        target_config = self._configs[target_level]
        compressed_from = full_answer.metadata.get("depth_level")

        if target_level in (DepthLevel.ELI5, DepthLevel.BEGINNER):
            # Summary or first paragraph, no extra sections.
            return StructuredAnswer(
                summary=None,
                main_content=full_answer.summary or full_answer.main_content[:200],
                sections=[],
                citations=full_answer.citations[:1] if target_config.citation_density > 0.1 else [],
                metadata={"compressed_from": compressed_from, "depth_level": target_level.value},
            )

        if target_level == DepthLevel.INTERMEDIATE:
            return StructuredAnswer(
                summary=full_answer.summary,
                main_content=full_answer.main_content,
                sections=[s for s in full_answer.sections[:2] if s.level == "primary"],
                citations=full_answer.citations[:5],
                metadata={"compressed_from": compressed_from, "depth_level": target_level.value},
            )

        if target_level == DepthLevel.ADVANCED:
            # Include most content, collapse tertiary sections.
            compressed_sections = []
            for s in full_answer.sections:
                if s.level == "tertiary":
                    s.initially_expanded = False
                compressed_sections.append(s)
            return StructuredAnswer(
                summary=full_answer.summary,
                main_content=full_answer.main_content,
                sections=compressed_sections,
                citations=full_answer.citations,
                metadata={**full_answer.metadata, "compressed_from": compressed_from},
            )

        # SCHOLARLY level - return as-is.
        return full_answer

    def expand_answer(
        self,
        brief_answer: str,
        target_level: DepthLevel,
        topic: str,
    ) -> dict[str, Any]:
        """Build a template for expanding a brief answer to more detail."""
        target_config = self._configs[target_level]

        required_sections: list[dict[str, str]] = []

        if target_config.include_scholarly_disagreements:
            required_sections.append(
                {
                    "title": "Scholarly Perspectives",
                    "description": "Different views among scholars on this matter",
                }
            )

        if target_config.include_historical_context:
            required_sections.append(
                {
                    "title": "Historical Context",
                    "description": "Background and historical development",
                }
            )

        if target_config.include_madhhab_comparison:
            required_sections.append(
                {
                    "title": "School of Thought Comparison",
                    "description": "Positions of the four major madhhabs",
                }
            )

        return {
            "original_answer": brief_answer,
            "topic": topic,
            "target_level": target_level.value,
            "required_sections": required_sections,
            "prompt_additions": self.build_prompt_instructions(target_config),
        }


def build_depth_prompt_block(config: DepthConfig) -> str:
    """Render the depth setting as a prompt block for a chat request."""
    indicator = build_depth_indicator(config.level)
    header = (
        f"\n\nANSWER DEPTH: {config.label} ({indicator.position}/{indicator.total})\n"
        f"{config.description}\n"
        "Adjust vocabulary, examples, citations, and structure to this depth. "
        "The ruling and its accuracy must not change with depth."
    )
    return f"{header}\n{DepthAdapter().build_prompt_instructions(config)}"


# ---------------------------------------------------------------------------
# Interactive depth controls
# ---------------------------------------------------------------------------


@dataclass
class DepthVariation:
    """One recorded answer variation at a particular depth."""

    level: DepthLevel
    direction: str  # "original", "simpler", "deeper", or "set"
    created_at: str
    answer: str | None = None
    instructions: str = ""
    follow_up_prompt: str = ""
    pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "direction": self.direction,
            "created_at": self.created_at,
            "answer": self.answer,
            "instructions": self.instructions,
            "follow_up_prompt": self.follow_up_prompt,
            "pending": self.pending,
        }


@dataclass
class DepthThread:
    """The depth history of one answer in one conversation."""

    chat_id: str
    message_id: str | None
    original_answer: str
    original_level: DepthLevel
    current_level: DepthLevel
    variations: list[DepthVariation] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "original_answer": self.original_answer,
            "original_level": self.original_level.value,
            "current_level": self.current_level.value,
            "current_depth": build_depth_indicator(self.current_level).to_dict(),
            "variations": [variation.to_dict() for variation in self.variations],
            "updated_at": self.updated_at,
        }


@dataclass
class DepthAdjustment:
    """Result of an "Explain Simpler" / "Go Deeper" request."""

    chat_id: str
    message_id: str | None
    direction: str
    previous_level: DepthLevel
    current_level: DepthLevel
    at_limit: bool
    original_answer: str
    instructions: str
    follow_up_prompt: str
    indicator: DepthIndicator
    variations: list[DepthVariation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "direction": self.direction,
            "previous_level": self.previous_level.value,
            "current_level": self.current_level.value,
            "at_limit": self.at_limit,
            "original_answer": self.original_answer,
            "instructions": self.instructions,
            "follow_up_prompt": self.follow_up_prompt,
            "depth": self.indicator.to_dict(),
            "variations": [variation.to_dict() for variation in self.variations],
            "context_preserved": True,
            "regenerate_required": True,
        }


def _thread_key(chat_id: str, message_id: str | None = None) -> str:
    return f"{chat_id}:{message_id}" if message_id else chat_id


def _build_follow_up_prompt(
    direction: str,
    config: DepthConfig,
    original_level: DepthLevel,
) -> str:
    """Prompt the client sends on the *same chat* to regenerate at the new depth."""
    if direction == "simpler":
        lead = "Explain your previous answer again, more simply and in the same conversation."
    elif direction == "deeper":
        lead = "Expand your previous answer with more depth, in the same conversation."
    else:
        lead = "Answer the previous question again at a different explanation depth."
    changed = (
        f"Move from the {DEPTH_CONFIGS[original_level].label} depth to the {config.label} depth "
        f"({depth_position(config.level)}/{len(DEPTH_SCALE)})."
    )
    keep = "Keep the same ruling, the same sources, and all established conversation context."
    extras: list[str] = []
    if config.use_analogies:
        extras.append("Use a familiar analogy.")
    if config.examples_per_concept:
        plural = "example" if config.examples_per_concept == 1 else "examples"
        extras.append(f"Add about {config.examples_per_concept} concrete {plural}.")
    if config.include_arabic:
        extras.append("Quote the Arabic evidence with translation.")
    if config.include_scholarly_disagreements:
        extras.append("Mention relevant scholarly differences (ikhtilaf).")
    if config.include_madhhab_comparison:
        extras.append("Compare the four major madhhabs where they differ.")
    instructions = " ".join(extras)
    return " ".join(part for part in (lead, changed, keep, instructions) if part)


class DepthSessionStore:
    """In-memory store of original answers and their depth variations.

    Answers live for the lifetime of the process, next to ``active_chats``.
    The store is bounded so a long-lived server cannot accumulate answers
    without limit.
    """

    def __init__(self, max_threads: int = 500) -> None:
        self._threads: dict[str, DepthThread] = {}
        self._max_threads = max(1, max_threads)

    def record_original(
        self,
        chat_id: str,
        message_id: str | None,
        answer: str,
        level: DepthLevel,
    ) -> DepthThread:
        """Store the answer as delivered, replacing any prior thread for it."""
        key = _thread_key(chat_id, message_id)
        thread = DepthThread(
            chat_id=chat_id,
            message_id=message_id,
            original_answer=answer,
            original_level=level,
            current_level=level,
            variations=[
                DepthVariation(
                    level=level,
                    direction="original",
                    answer=answer,
                    created_at=_utc_now(),
                )
            ],
        )
        self._threads[key] = thread
        self._evict()
        return thread

    def record_variation(
        self,
        chat_id: str,
        level: DepthLevel,
        answer: str,
        message_id: str | None = None,
        direction: str = "set",
    ) -> DepthThread:
        """Attach a regenerated answer at *level* to an existing thread."""
        thread = self.get(chat_id, message_id)
        if thread is None:
            thread = self.record_original(chat_id, message_id, answer, level)
            return thread

        # Fulfil the most recent pending variation at this level, if any.
        for variation in reversed(thread.variations):
            if variation.pending and variation.level == level:
                variation.answer = answer
                variation.pending = False
                variation.created_at = _utc_now()
                thread.current_level = level
                thread.updated_at = _utc_now()
                return thread

        thread.variations.append(
            DepthVariation(
                level=level,
                direction=direction,
                answer=answer,
                created_at=_utc_now(),
            )
        )
        thread.current_level = level
        thread.updated_at = _utc_now()
        return thread

    def get(self, chat_id: str, message_id: str | None = None) -> DepthThread | None:
        """Look up a thread by message, falling back to the chat's latest answer."""
        if message_id:
            return self._threads.get(_thread_key(chat_id, message_id))
        thread = self._threads.get(chat_id)
        if thread is not None:
            return thread
        # Latest thread recorded for this chat, by update time.
        candidates = [t for key, t in self._threads.items() if key == chat_id or key.startswith(f"{chat_id}:")]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.updated_at)

    def adjust(
        self,
        chat_id: str,
        direction: str,
        level: DepthLevel | None = None,
        message_id: str | None = None,
    ) -> DepthAdjustment:
        """Move a thread one step simpler/deeper, or to an explicit level.

        Raises ``KeyError`` when the chat has no recorded answer to adjust.
        """
        thread = self.get(chat_id, message_id)
        if thread is None:
            raise KeyError(chat_id)

        previous = thread.current_level
        if direction == "simpler":
            target = step_depth(previous, -1)
        elif direction == "deeper":
            target = step_depth(previous, 1)
        elif direction == "set":
            if level is None:
                raise ValueError("direction 'set' requires a level")
            target = level
        else:
            raise ValueError(f"Unknown direction {direction!r}; use simpler, deeper, or set.")

        config = DEPTH_CONFIGS[target]
        instructions = build_depth_prompt_block(config)
        follow_up = _build_follow_up_prompt(direction, config, previous)
        at_limit = direction in ("simpler", "deeper") and target == previous

        variation = DepthVariation(
            level=target,
            direction=direction,
            created_at=_utc_now(),
            instructions=instructions,
            follow_up_prompt=follow_up,
            pending=True,
        )
        thread.variations.append(variation)
        thread.current_level = target
        thread.updated_at = _utc_now()

        return DepthAdjustment(
            chat_id=thread.chat_id,
            message_id=thread.message_id,
            direction=direction,
            previous_level=previous,
            current_level=target,
            at_limit=at_limit,
            original_answer=thread.original_answer,
            instructions=instructions,
            follow_up_prompt=follow_up,
            indicator=build_depth_indicator(target),
            variations=thread.variations,
        )

    def clear(self, chat_id: str, message_id: str | None = None) -> bool:
        """Forget a thread (used by tests and when a chat is deleted)."""
        key = _thread_key(chat_id, message_id) if message_id else chat_id
        if key in self._threads:
            del self._threads[key]
            return True
        return False

    def _evict(self) -> None:
        while len(self._threads) > self._max_threads:
            oldest = min(self._threads, key=lambda key: self._threads[key].updated_at)
            del self._threads[oldest]


class UserPreferencesStore:
    """Store for user depth preferences."""

    def __init__(self, data_file: str | None = None) -> None:
        self._data_file: str = str(data_file or os.getenv("DEPTH_PREFS_FILE") or "./data/depth_preferences.json")
        self._preferences: dict[str, UserDepthPreferences] = {}
        self._explicit: set[str] = set()
        self._load_data()

    def _load_data(self) -> None:
        """Load preferences from file."""
        if not os.path.exists(self._data_file):
            return
        try:
            with open(self._data_file) as f:
                data = json.load(f)
            for user_data in data.get("preferences", []):
                prefs = UserDepthPreferences.from_dict(user_data)
                self._preferences[prefs.user_id] = prefs
                self._explicit.add(prefs.user_id)
            logger.info("Loaded %d user depth preferences", len(self._preferences))
        except Exception as e:
            logger.warning("Failed to load depth preferences: %s", e)

    def _save_data(self) -> None:
        """Save preferences to file."""
        os.makedirs(os.path.dirname(self._data_file) or ".", exist_ok=True)
        data = {"preferences": [p.to_dict() for p in self._preferences.values()]}
        with open(self._data_file, "w") as f:
            json.dump(data, f, indent=2)

    def has(self, user_id: str) -> bool:
        """True when this user has an explicitly stored preference."""
        return user_id in self._explicit

    def get(self, user_id: str) -> UserDepthPreferences:
        """Get preferences for a user, creating defaults if needed."""
        if user_id not in self._preferences:
            self._preferences[user_id] = UserDepthPreferences(user_id=user_id)
        return self._preferences[user_id]

    def save(self, preferences: UserDepthPreferences) -> None:
        """Save user preferences."""
        self._preferences[preferences.user_id] = preferences
        self._explicit.add(preferences.user_id)
        self._save_data()

    def set_default_level(self, user_id: str, level: DepthLevel) -> None:
        """Set user's default depth level."""
        prefs = self.get(user_id)
        prefs.default_level = level
        self.save(prefs)

    def set_topic_level(self, user_id: str, topic: str, level: DepthLevel) -> None:
        """Set depth level for a specific topic."""
        prefs = self.get(user_id)
        prefs.per_topic_levels[topic] = level
        self.save(prefs)

    def effective_level(
        self,
        user_id: str,
        requested: DepthLevel | None = None,
        topic: str | None = None,
    ) -> DepthLevel:
        """Resolve the level for a request without building the whole config."""
        if requested is not None:
            return requested
        prefs = self.get(user_id)
        if topic and topic in prefs.per_topic_levels:
            return prefs.per_topic_levels[topic]
        return prefs.default_level

    def reset(self, user_id: str) -> bool:
        """Remove a user's stored preferences."""
        if user_id in self._preferences:
            del self._preferences[user_id]
            self._explicit.discard(user_id)
            self._save_data()
            return True
        return False


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

_adapter: DepthAdapter | None = None
_prefs_store: UserPreferencesStore | None = None
_session_store: DepthSessionStore | None = None


def get_depth_adapter() -> DepthAdapter:
    """Get or create the singleton depth adapter."""
    global _adapter
    if _adapter is None:
        _adapter = DepthAdapter()
    return _adapter


def get_preferences_store() -> UserPreferencesStore:
    """Get or create the singleton preferences store."""
    global _prefs_store
    if _prefs_store is None:
        _prefs_store = UserPreferencesStore()
    return _prefs_store


def get_depth_session_store() -> DepthSessionStore:
    """Get or create the singleton per-chat depth history store."""
    global _session_store
    if _session_store is None:
        _session_store = DepthSessionStore()
    return _session_store


def get_answer_config(
    user_id: str | None = None,
    requested_level: DepthLevel | None = None,
    topic: str | None = None,
) -> DepthConfig:
    """Get the appropriate depth configuration for an answer.

    Priority:
    1. Explicitly requested level
    2. User's topic-specific level (if topic provided)
    3. User's default level
    4. System default (INTERMEDIATE)
    """
    adapter = get_depth_adapter()
    prefs_store = get_preferences_store()

    if user_id:
        level = prefs_store.effective_level(user_id, requested=requested_level, topic=topic)
        user_prefs: UserDepthPreferences | None = prefs_store.get(user_id)
    else:
        level = requested_level or DepthLevel.INTERMEDIATE
        user_prefs = None

    return adapter.get_effective_config(level, user_prefs)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class DepthLevelInfo(BaseModel):
    """One level on the depth scale, as offered to a client."""

    level: str
    label: str
    description: str
    position: int
    total: int
    config: dict[str, Any]


class DepthLevelsResponse(BaseModel):
    """The full ordered scale, for rendering the depth control."""

    levels: list[DepthLevelInfo]
    default: str = DepthLevel.INTERMEDIATE.value
    aliases: dict[str, str] = Field(default_factory=lambda: {k: v.value for k, v in DEPTH_ALIASES.items()})


class DepthPreferenceResponse(BaseModel):
    """A user's stored depth preferences plus the resulting effective depth."""

    preferences: dict[str, Any]
    effective: dict[str, Any]


class DepthPreferenceUpdate(BaseModel):
    """Partial update of a user's depth preferences."""

    default_level: str | None = Field(default=None, description="Canonical level or legacy alias")
    topic: str | None = Field(default=None, max_length=120, description="Topic to set a level for")
    level: str | None = Field(default=None, description="Level for the given topic")
    always_show_arabic: bool | None = None
    auto_expand_sections: bool | None = None
    preferred_madhhab: str | None = Field(default=None, max_length=32)
    custom_overrides: dict[str, Any] | None = None


class DepthAdjustRequest(BaseModel):
    """An "Explain Simpler" / "Go Deeper" (or explicit set) request."""

    chat_id: str = Field(..., min_length=1, max_length=200)
    message_id: str | None = Field(default=None, max_length=200)
    direction: str = Field(..., description="'simpler', 'deeper', or 'set'")
    level: str | None = Field(default=None, description="Required when direction is 'set'")
    user_id: str | None = Field(default=None, max_length=128)
    persist_preference: bool = Field(
        default=False,
        description="Also store this level as the user's default for future answers",
    )


class DepthAdjustResponse(BaseModel):
    """The new depth setting, prompt instructions, and preserved history."""

    chat_id: str
    message_id: str | None = None
    direction: str
    previous_level: str
    current_level: str
    at_limit: bool
    original_answer: str
    instructions: str
    follow_up_prompt: str
    depth: dict[str, Any]
    variations: list[dict[str, Any]]
    context_preserved: bool = True
    regenerate_required: bool = True


class DepthRegisterRequest(BaseModel):
    """Register an answer that was generated outside the depth-aware /chat path."""

    chat_id: str = Field(..., min_length=1, max_length=200)
    message_id: str | None = Field(default=None, max_length=200)
    answer: str = Field(..., min_length=1, max_length=30000)
    level: str | None = Field(default=None, description="Defaults to the user's effective level")
    user_id: str | None = Field(default=None, max_length=128)
    direction: str = Field(default="original", max_length=16)


class DepthRegisterResponse(BaseModel):
    """The stored thread, including the preserved original answer."""

    thread: dict[str, Any]


class DepthThreadResponse(BaseModel):
    """A chat's depth history: original answer plus every variation."""

    thread: dict[str, Any]


def _parse_level_or_400(value: str | None, *, field_name: str = "level") -> DepthLevel | None:
    """Normalize a client-supplied level, raising a typed 400 on bad input."""
    try:
        return normalize_depth(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc), headers={"X-Error-Field": field_name}) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/levels", response_model=DepthLevelsResponse)
async def list_levels() -> DepthLevelsResponse:
    """Return the ordered depth scale, its metadata, and accepted aliases."""
    levels = [
        DepthLevelInfo(
            level=level.value,
            label=DEPTH_CONFIGS[level].label,
            description=DEPTH_CONFIGS[level].description,
            position=depth_position(level),
            total=len(DEPTH_SCALE),
            config=DEPTH_CONFIGS[level].to_dict(),
        )
        for level in DEPTH_SCALE
    ]
    return DepthLevelsResponse(levels=levels)


@router.get("/preferences/{user_id}", response_model=DepthPreferenceResponse)
async def get_preferences(user_id: str) -> DepthPreferenceResponse:
    """Return a user's depth preferences and their effective default."""
    prefs = get_preferences_store().get(user_id)
    effective = get_preferences_store().effective_level(user_id)
    return DepthPreferenceResponse(
        preferences=prefs.to_dict(),
        effective=build_depth_indicator(effective).to_dict(),
    )


@router.put("/preferences/{user_id}", response_model=DepthPreferenceResponse)
async def update_preferences(user_id: str, body: DepthPreferenceUpdate) -> DepthPreferenceResponse:
    """Update a user's default or topic-specific depth."""
    store = get_preferences_store()
    prefs = store.get(user_id)

    if body.topic is not None:
        if body.level is None:
            raise HTTPException(status_code=400, detail="level is required when topic is provided")
        topic_level = _parse_level_or_400(body.level)
        if topic_level is not None:
            prefs.per_topic_levels[body.topic] = topic_level

    if body.default_level is not None:
        default_level = _parse_level_or_400(body.default_level, field_name="default_level")
        if default_level is not None:
            prefs.default_level = default_level

    if body.always_show_arabic is not None:
        prefs.always_show_arabic = body.always_show_arabic
    if body.auto_expand_sections is not None:
        prefs.auto_expand_sections = body.auto_expand_sections
    if body.preferred_madhhab is not None:
        prefs.preferred_madhhab = body.preferred_madhhab
    if body.custom_overrides is not None:
        prefs.custom_overrides = body.custom_overrides

    store.save(prefs)
    effective = store.effective_level(user_id)
    return DepthPreferenceResponse(
        preferences=prefs.to_dict(),
        effective=build_depth_indicator(effective).to_dict(),
    )


@router.post("/register", response_model=DepthRegisterResponse)
async def register_answer(body: DepthRegisterRequest) -> DepthRegisterResponse:
    """Record an answer so it can later be simplified or expanded."""
    store = get_depth_session_store()
    level = _parse_level_or_400(body.level)
    if level is None:
        level = get_preferences_store().effective_level(body.user_id) if body.user_id else DepthLevel.INTERMEDIATE

    existing = store.get(body.chat_id, body.message_id) if body.message_id else None
    if body.direction == "original" or existing is None:
        thread = store.record_original(body.chat_id, body.message_id, body.answer, level)
    else:
        thread = store.record_variation(
            body.chat_id,
            level,
            body.answer,
            message_id=body.message_id,
            direction=body.direction,
        )
    return DepthRegisterResponse(thread=thread.to_dict())


@router.post("/adjust", response_model=DepthAdjustResponse)
async def adjust_depth(body: DepthAdjustRequest) -> DepthAdjustResponse:
    """Move an answer one step simpler/deeper (or to an explicit level).

    The original answer is preserved verbatim. The response carries the prompt
    instructions for the new depth and a ready-made ``follow_up_prompt``: send
    it on the same ``chat_id`` (with ``depth`` set to ``current_level``) to
    regenerate the answer while keeping the conversation context.
    """
    direction = body.direction.strip().lower()
    if direction not in ("simpler", "deeper", "set"):
        raise HTTPException(
            status_code=400,
            detail="direction must be 'simpler', 'deeper', or 'set'",
        )

    level = _parse_level_or_400(body.level)
    if direction == "set" and level is None:
        raise HTTPException(status_code=400, detail="level is required when direction is 'set'")

    try:
        adjustment = get_depth_session_store().adjust(
            chat_id=body.chat_id,
            direction=direction,
            level=level,
            message_id=body.message_id,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                "No stored answer for this chat. Answers returned by /chat are recorded automatically; "
                "for other clients call POST /depth/register first."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if body.persist_preference and body.user_id:
        get_preferences_store().set_default_level(body.user_id, adjustment.current_level)

    payload = adjustment.to_dict()
    return DepthAdjustResponse(**payload)


@router.get("/response/{chat_id}", response_model=DepthThreadResponse)
async def get_depth_history(
    chat_id: str,
    message_id: str | None = Query(None, description="Specific answer; defaults to the chat's latest"),
) -> DepthThreadResponse:
    """Return the original answer and every depth variation recorded for it."""
    thread = get_depth_session_store().get(chat_id, message_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="No depth history recorded for this chat")
    return DepthThreadResponse(thread=thread.to_dict())
