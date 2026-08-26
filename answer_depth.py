"""Answer Depth Customization (#221)

Enable users to adjust answer granularity across a spectrum from brief summaries
to comprehensive scholarly analyses with full evidence and reasoning.

Features:
- Four depth levels: brief, standard, detailed, scholarly
- Dynamic terminology density adjustment
- Scaled citation frequency
- Modified syntactic complexity
- Controlled scholarly disagreement inclusion
- Toggleable historical context depth
- Madhhab (school of thought) comparisons
- Optional Arabic text display

Architecture:
- DepthLevel: Enum of available depth levels
- DepthConfig: Configuration for each depth level
- DepthAdapter: Adapts responses based on depth settings
- UserPreferences: Stores user's default preferences
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DepthLevel(str, Enum):
    """Available answer depth levels."""

    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"
    SCHOLARLY = "scholarly"


@dataclass
class DepthConfig:
    """Configuration settings for a depth level."""

    level: DepthLevel
    max_length: int  # Maximum response length in tokens
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
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
        }


# Default configurations for each depth level
DEPTH_CONFIGS: dict[DepthLevel, DepthConfig] = {
    DepthLevel.BRIEF: DepthConfig(
        level=DepthLevel.BRIEF,
        max_length=150,
        citation_density=0.1,
        include_arabic=False,
        include_transliteration=False,
        include_scholarly_disagreements=False,
        include_historical_context=False,
        include_madhhab_comparison=False,
        terminology_complexity=0.2,
        evidence_detail=0.1,
        collapsible_sections=False,
        summary_position="none",
    ),
    DepthLevel.STANDARD: DepthConfig(
        level=DepthLevel.STANDARD,
        max_length=400,
        citation_density=0.4,
        include_arabic=False,
        include_transliteration=True,
        include_scholarly_disagreements=False,
        include_historical_context=False,
        include_madhhab_comparison=False,
        terminology_complexity=0.4,
        evidence_detail=0.4,
        collapsible_sections=False,
        summary_position="start",
    ),
    DepthLevel.DETAILED: DepthConfig(
        level=DepthLevel.DETAILED,
        max_length=800,
        citation_density=0.7,
        include_arabic=True,
        include_transliteration=True,
        include_scholarly_disagreements=True,
        include_historical_context=True,
        include_madhhab_comparison=False,
        terminology_complexity=0.7,
        evidence_detail=0.7,
        collapsible_sections=True,
        summary_position="start",
    ),
    DepthLevel.SCHOLARLY: DepthConfig(
        level=DepthLevel.SCHOLARLY,
        max_length=1500,
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
    ),
}


@dataclass
class UserDepthPreferences:
    """User's answer depth preferences."""

    user_id: str
    default_level: DepthLevel = DepthLevel.STANDARD
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
        return cls(
            user_id=data["user_id"],
            default_level=DepthLevel(data.get("default_level", "standard")),
            custom_overrides=data.get("custom_overrides", {}),
            per_topic_levels={k: DepthLevel(v) for k, v in data.get("per_topic_levels", {}).items()},
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
        if not overrides:
            return base_config

        # Create modified config
        return DepthConfig(
            level=base_config.level,
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
        )

    def build_prompt_instructions(self, config: DepthConfig) -> str:
        """Build prompt instructions based on depth configuration."""
        instructions = []

        # Length guidance
        if config.level == DepthLevel.BRIEF:
            instructions.append("Provide a concise answer in 1-2 sentences. Focus on the core answer only.")
        elif config.level == DepthLevel.STANDARD:
            instructions.append("Provide a clear, moderate-length answer with key supporting evidence.")
        elif config.level == DepthLevel.DETAILED:
            instructions.append("Provide a comprehensive answer with detailed explanations and evidence.")
        else:  # SCHOLARLY
            instructions.append(
                "Provide an academically rigorous answer with full scholarly analysis, "
                "evidence chains, and consideration of diverse scholarly perspectives."
            )

        # Citation instructions
        if config.citation_density < 0.3:
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

        # Terminology
        if config.terminology_complexity < 0.3:
            instructions.append("Use simple, accessible language for a general audience.")
        elif config.terminology_complexity > 0.7:
            instructions.append("Use precise scholarly terminology with brief explanations.")

        return "\n".join(f"- {i}" for i in instructions)

    def compress_answer(
        self,
        full_answer: StructuredAnswer,
        target_level: DepthLevel,
    ) -> StructuredAnswer:
        """Compress a detailed answer to a simpler depth level."""
        target_config = self._configs[target_level]

        if target_level == DepthLevel.BRIEF:
            # Just return the summary or first sentence
            return StructuredAnswer(
                summary=None,
                main_content=full_answer.summary or full_answer.main_content[:200],
                sections=[],
                citations=full_answer.citations[:1] if target_config.citation_density > 0 else [],
                metadata={"compressed_from": full_answer.metadata.get("depth_level")},
            )

        elif target_level == DepthLevel.STANDARD:
            # Include summary and main content, minimal sections
            return StructuredAnswer(
                summary=full_answer.summary,
                main_content=full_answer.main_content,
                sections=[s for s in full_answer.sections[:2] if s.level == "primary"],
                citations=full_answer.citations[:5],
                metadata={"compressed_from": full_answer.metadata.get("depth_level")},
            )

        elif target_level == DepthLevel.DETAILED:
            # Include most content, collapse tertiary sections
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
                metadata=full_answer.metadata,
            )

        # SCHOLARLY level - return as-is
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

        expansion_template: dict[str, Any] = {
            "original_answer": brief_answer,
            "target_level": target_level.value,
            "required_sections": required_sections,
            "prompt_additions": self.build_prompt_instructions(target_config),
        }

        expansion_template: dict[str, Any] = {
            "original_answer": brief_answer,
            "target_level": target_level.value,
            "required_sections": required_sections,
            "prompt_additions": self.build_prompt_instructions(target_config),
        }
        return expansion_template


class UserPreferencesStore:
    """Store for user depth preferences."""

    def __init__(self, data_file: str | None = None) -> None:
        self._data_file: str = str(data_file or os.getenv("DEPTH_PREFS_FILE") or "./data/depth_preferences.json")
        self._preferences: dict[str, UserDepthPreferences] = {}
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
            logger.info("Loaded %d user depth preferences", len(self._preferences))
        except Exception as e:
            logger.warning("Failed to load depth preferences: %s", e)

    def _save_data(self) -> None:
        """Save preferences to file."""
        os.makedirs(os.path.dirname(self._data_file) or ".", exist_ok=True)
        data = {"preferences": [p.to_dict() for p in self._preferences.values()]}
        with open(self._data_file, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, user_id: str) -> UserDepthPreferences:
        """Get preferences for a user, creating defaults if needed."""
        if user_id not in self._preferences:
            self._preferences[user_id] = UserDepthPreferences(user_id=user_id)
        return self._preferences[user_id]

    def save(self, preferences: UserDepthPreferences) -> None:
        """Save user preferences."""
        self._preferences[preferences.user_id] = preferences
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


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instances
# ─────────────────────────────────────────────────────────────────────────────

_adapter: DepthAdapter | None = None
_prefs_store: UserPreferencesStore | None = None


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
    4. System default (STANDARD)
    """
    adapter = get_depth_adapter()
    prefs_store = get_preferences_store()

    # Determine effective level
    user_prefs: UserDepthPreferences | None = prefs_store.get(user_id) if user_id else None
    if requested_level:
        level = requested_level
    elif user_prefs:
        if topic and topic in user_prefs.per_topic_levels:
            level = user_prefs.per_topic_levels[topic]
        else:
            level = user_prefs.default_level
    else:
        level = DepthLevel.STANDARD

    # Get config with user preferences applied
    user_prefs: UserDepthPreferences | None = prefs_store.get(user_id) if user_id else None
    return adapter.get_effective_config(level, user_prefs)
