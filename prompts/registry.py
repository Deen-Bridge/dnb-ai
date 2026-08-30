"""Versioned prompt template registry with A/B experimentation support.

This module extracts prompts out of the application handler into a managed,
versioned store. Each template has a stable name, a semantic version, a
changelog entry, and typed variables. Rendering is deterministic and
injection-safe: variables are substituted via ``str.format`` with a strict
allow-list, so no arbitrary code execution is possible.

The A/B harness assigns each request (by ``chat_id``/``user_id`` hash) to a
prompt variant stickily, records the assignment with the response, and
supports a kill switch. Variant performance is computable by joining
assignments to an existing quality signal (feedback ratings, eval scores, or
confidence) — no new metrics store is built here.

The active prompt version and experiment configuration live in configuration
(coordinated with the pydantic-settings work in #10), so changing the live
prompt or starting an experiment needs no code change.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

# The variable contract published for #14/#39/#41 and any other consumers.
# Each key maps to a human-readable description of what the variable holds.
VARIABLE_CONTRACT: Dict[str, str] = {
    "madhhab": "The user's school of jurisprudence (hanafi, maliki, shafii, hanbali) or None.",
    "language": "The user's preferred language code (e.g. 'en', 'ar') or None.",
    "knowledge_level": "The user's knowledge level (beginner, intermediate, advanced) or None.",
    "intent": "The classified intent of the question (e.g. 'fiqh', 'aqeedah') or None.",
    "retrieved_context": "Retrieved context from the corpus, formatted as text, or an empty string.",
    "user_question": "The user's raw question text.",
}


@dataclass(frozen=True)
class PromptTemplate:
    """A single versioned prompt template."""

    name: str
    version: str  # semantic version, e.g. "1.0.0"
    changelog: str
    variables: frozenset  # allowed variable names for this template
    template: str  # the prompt text with {placeholders}

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables.

        Only variables declared in ``self.variables`` are accepted; unknown
        variables raise a ``ValueError``. This prevents injection of arbitrary
        keys and keeps rendering deterministic.
        """
        unknown = set(kwargs) - set(self.variables)
        if unknown:
            raise ValueError(
                f"Unknown variable(s) for template '{self.name}': {sorted(unknown)}"
            )
        missing = set(self.variables) - set(kwargs)
        if missing:
            raise ValueError(
                f"Missing variable(s) for template '{self.name}': {sorted(missing)}"
            )
        return self.template.format(**kwargs)


# The default system prompt, extracted from main.py and parameterized.
# This composes with #5's system_instruction change: the handler will pass
# this rendered prompt as the system instruction.
DEFAULT_SYSTEM_PROMPT = PromptTemplate(
    name="islamic_context",
    version="1.0.0",
    changelog="Initial version extracted from main.py; parameterized for madhhab, language, knowledge level, intent, and retrieved context.",
    variables=frozenset(
        {
            "madhhab",
            "language",
            "knowledge_level",
            "intent",
            "retrieved_context",
            "user_question",
        }
    ),
    template=(
        "You are an Islamic scholar assistant. Answer the user's question "
        "with accurate, well-reasoned guidance based on the Quran and authentic "
        "hadith. Be respectful, clear, and concise.\n\n"
        "User's madhhab: {madhhab}\n"
        "User's language: {language}\n"
        "User's knowledge level: {knowledge_level}\n"
        "Question intent: {intent}\n\n"
        "Retrieved context:\n{retrieved_context}\n\n"
        "User's question: {user_question}\n"
        "Provide a helpful answer."
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    """A registry of named, versioned prompt templates."""

    def __init__(self, templates: Optional[List[PromptTemplate]] = None) -> None:
        self._templates: Dict[str, PromptTemplate] = {}
        for template in templates or []:
            self.register(template)

    def register(self, template: PromptTemplate) -> None:
        """Register a template by name. Overwrites any existing template with the same name."""
        self._templates[template.name] = template
        logger.info("Registered prompt template '%s' version %s", template.name, template.version)

    def get(self, name: str) -> PromptTemplate:
        """Return the template with the given name, or raise KeyError."""
        return self._templates[name]

    def list_names(self) -> List[str]:
        """Return the names of all registered templates."""
        return sorted(self._templates)


# Shared registry instance for the application.
registry = PromptRegistry([DEFAULT_SYSTEM_PROMPT])


# ---------------------------------------------------------------------------
# A/B experiment harness
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment.

    ``control_template`` is the name of the template used as the control.
    ``variant_templates`` maps a variant name to a template name.
    ``traffic_percent`` is the percentage of requests that should be assigned
    to a variant (the rest go to control). ``enabled`` is the kill switch.
    """

    experiment_id: str
    control_template: str
    variant_templates: Dict[str, str]
    traffic_percent: float = 50.0  # 0-100, percent of requests assigned to a variant
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.traffic_percent <= 100:
            raise ValueError("traffic_percent must be between 0 and 100")


@dataclass
class Assignment:
    """The result of assigning a request to a prompt variant."""

    experiment_id: str
    variant: str  # "control" or a variant name
    template_name: str
    template_version: str


class ExperimentHarness:
    """Assigns requests to prompt variants stickily by session."""

    def __init__(self, config: ExperimentConfig, registry: PromptRegistry) -> None:
        self.config = config
        self.registry = registry

    def assign(self, session_id: str) -> Assignment:
        """Return the assignment for the given session.

        The assignment is deterministic: the same session_id always maps to the
        same variant. If the experiment is disabled (kill switch), always
        returns the control.
        """
        if not self.config.enabled:
            return self._make_assignment("control", self.config.control_template)

        # Hash the session_id to a stable bucket.
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100  # 0-99

        if bucket < self.config.traffic_percent:
            # Assign to a variant. Pick deterministically based on the bucket.
            variant_names = sorted(self.config.variant_templates.keys())
            if not variant_names:
                return self._make_assignment("control", self.config.control_template)
            variant_index = bucket % len(variant_names)
            variant_name = variant_names[variant_index]
            template_name = self.config.variant_templates[variant_name]
            return self._make_assignment(variant_name, template_name)

        return self._make_assignment("control", self.config.control_template)

    def _make_assignment(self, variant: str, template_name: str) -> Assignment:
        template = self.registry.get(template_name)
        return Assignment(
            experiment_id=self.config.experiment_id,
            variant=variant,
            template_name=template.name,
            template_version=template.version,
        )


# ---------------------------------------------------------------------------
# Configuration-driven active prompt and experiment
# ---------------------------------------------------------------------------


def get_active_template(settings: Any) -> PromptTemplate:
    """Return the active prompt template based on configuration.

    ``settings`` is expected to have ``active_prompt_name`` (a string) and
    optionally ``experiment`` (an ExperimentConfig or None). This is a thin
    adapter so the handler can stay configuration-driven.
    """
    name = getattr(settings, "active_prompt_name", DEFAULT_SYSTEM_PROMPT.name)
    return registry.get(name)


def get_experiment_harness(settings: Any) -> Optional[ExperimentHarness]:
    """Return an ExperimentHarness if an experiment is configured, else None."""
    experiment_config = getattr(settings, "experiment", None)
    if experiment_config is None:
        return None
    return ExperimentHarness(experiment_config, registry)
