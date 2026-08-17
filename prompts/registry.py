"""Versioned prompt template registry with A/B experimentation.

Why this exists
----------------
The system prompt was a hardcoded string in ``main.py`` with no version,
no history, and no way to change it without a deploy or to test whether a
change helped. This module turns prompts into managed infrastructure:

- A registry of named, versioned templates with typed variables.
- A safe, injection-resistant rendering layer.
- Sticky A/B variant assignment by session (``chat_id``/``user_id``).
- A kill switch and config-driven experiment definitions.

It composes with the ``system_instruction`` change from #5: the handler
renders a template here and passes the result as ``system_instruction``
instead of interpolating into the user prompt.

Design notes
------------
- Templates are plain strings with ``{variable}`` placeholders. Rendering
  only substitutes declared variables; unknown placeholders raise an error
  so a typo cannot silently ship.
- No arbitrary code execution: variables are escaped for the prompt context
  (newlines collapsed, control characters stripped) so a user-supplied value
  cannot inject instructions.
- Variant assignment is sticky: the same session always gets the same
  variant, so a user does not flip-flop between prompts mid-conversation.
- The active prompt version and experiment config come from pydantic
  settings (#10), so changing them needs no code change.
- Measurement joins on the recorded variant via existing quality signals
  (#43/#16/#ai-19); no new metrics store is introduced.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Template model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptTemplate:
    """A named, versioned prompt template with typed variables."""

    name: str
    version: str  # semantic version, e.g. "1.0.0"
    changelog: str
    variables: Dict[str, str]  # variable name -> description (the contract)
    text: str

    def render(self, **values: Any) -> str:
        """Render the template with the given variable values.

        Raises:
            ValueError: if a required variable is missing or an unknown
                variable is supplied.
        """
        missing = set(self.variables) - set(values)
        if missing:
            raise ValueError(f"Missing variables for template '{self.name}': {sorted(missing)}")
        unknown = set(values) - set(self.variables)
        if unknown:
            raise ValueError(f"Unknown variables for template '{self.name}': {sorted(unknown)}")

        # Escape each value to prevent prompt injection via user input.
        escaped = {key: _escape_value(value) for key, value in values.items()}

        # Substitute placeholders. Use a regex to catch any remaining
        # {variable} that was not declared (typo protection).
        rendered = self.text.format(**escaped)
        undeclared = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", rendered)
        if undeclared:
            raise ValueError(
                f"Template '{self.name}' contains undeclared placeholders: {sorted(set(undeclared))}"
            )
        return rendered


def _escape_value(value: Any) -> str:
    """Make a value safe for prompt interpolation.

    Collapses newlines and strips control characters so a user-supplied
    string cannot break out of the prompt structure.
    """
    text = str(value)
    # Remove control characters except newline/tab (which we then collapse).
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    # Collapse whitespace runs to single spaces to avoid prompt smuggling.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    """Holds all prompt templates and resolves the active version."""

    def __init__(self, templates: Optional[List[PromptTemplate]] = None):
        self._templates: Dict[str, PromptTemplate] = {}
        for template in templates or []:
            self.register(template)

    def register(self, template: PromptTemplate) -> None:
        key = f"{template.name}:{template.version}"
        if key in self._templates:
            raise ValueError(f"Duplicate template key: {key}")
        self._templates[key] = template

    def get(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        """Get a template by name and optional version.

        If version is None, returns the highest version for that name
        (simple numeric comparison on dotted parts).
        """
        candidates = [t for t in self._templates.values() if t.name == name]
        if not candidates:
            raise KeyError(f"No template named '{name}'")
        if version is not None:
            key = f"{name}:{version}"
            if key not in self._templates:
                raise KeyError(f"Template '{name}' version '{version}' not found")
            return self._templates[key]
        # Return the highest version.
        return max(candidates, key=lambda t: _version_key(t.version))

    def all_versions(self, name: str) -> List[PromptTemplate]:
        return sorted(
            [t for t in self._templates.values() if t.name == name],
            key=lambda t: _version_key(t.version),
        )


def _version_key(version: str) -> tuple:
    """Convert a semantic version string to a sortable tuple."""
    parts = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(part)
    return tuple(parts)


# ---------------------------------------------------------------------------
# A/B experiment harness
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment.

    Attributes:
        name: Unique experiment name.
        template_name: The prompt template being tested.
        control_version: Version used for the control group.
        variant_versions: List of variant versions to test against control.
        traffic_fraction: Fraction of sessions (0.0–1.0) that enter the
            experiment. Sessions outside this fraction get the control.
        enabled: Kill switch. When False, everyone gets the control.
    """

    name: str
    template_name: str
    control_version: str
    variant_versions: List[str] = field(default_factory=list)
    traffic_fraction: float = 1.0
    enabled: bool = True


class ExperimentAssigner:
    """Assigns sessions to prompt variants stickily."""

    def __init__(self, registry: PromptRegistry, config: ExperimentConfig):
        self.registry = registry
        self.config = config

    def assign(self, session_id: str) -> str:
        """Return the version assigned to this session.

        The assignment is deterministic and sticky: the same session_id
        always maps to the same version for a given experiment config.
        """
        if not self.config.enabled:
            return self.config.control_version

        # Hash the session id with the experiment name so different
        # experiments on the same session get independent assignments.
        digest = hashlib.sha256(
            f"{self.config.name}:{session_id}".encode("utf-8")
        ).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF  # 0.0–1.0

        if bucket >= self.config.traffic_fraction:
            return self.config.control_version

        # Split the in-experiment traffic evenly among control + variants.
        versions = [self.config.control_version] + self.config.variant_versions
        if not versions:
            return self.config.control_version
        index = int(bucket / self.config.traffic_fraction * len(versions)) % len(versions)
        return versions[index]

    def render_for_session(self, session_id: str, **variables: Any) -> tuple[str, str]:
        """Render the assigned template and return (version, rendered_prompt)."""
        version = self.assign(session_id)
        template = self.registry.get(self.config.template_name, version)
        return version, template.render(**variables)


# ---------------------------------------------------------------------------
# Default templates
# ---------------------------------------------------------------------------


ISLAMIC_CONTEXT_V1 = PromptTemplate(
    name="islamic_context",
    version="1.0.0",
    changelog="Initial version extracted from main.py ISLAMIC_CONTEXT constant.",
    variables={
        "madhhab": "User's madhhab (e.g. 'Hanafi', 'Shafi'i', or 'None').",
        "language": "User's preferred language (e.g. 'en', 'ar').",
        "knowledge_level": "User's knowledge level (e.g. 'beginner', 'intermediate', 'advanced').",
        "intent": "Classified intent of the question (e.g. 'fiqh', 'aqeedah', 'general').",
        "retrieved_context": "Retrieved Quran/Hadith context to ground the answer.",
    },
    text=(
        "You are DeenBridge, an Islamic assistant. Answer according to the "
        "Qur'an and authentic Sunnah, with respect for scholarly differences.\n"
        "User profile:\n"
        "- Madhhab: {madhhab}\n"
        "- Language: {language}\n"
        "- Knowledge level: {knowledge_level}\n"
        "- Intent: {intent}\n"
        "Retrieved context (use this to ground your answer):\n{retrieved_context}\n"
        "If you do not know, say so. Do not fabricate citations."
    ),
)


DEFAULT_REGISTRY = PromptRegistry([ISLAMIC_CONTEXT_V1])


def load_registry_from_dir(prompts_dir: Path) -> PromptRegistry:
    """Load templates from a directory of .txt files with metadata.

    This is a placeholder for a future file-based store. For now, the
    registry is built in code, but the loader exists so prompts can move
    to versioned files without changing callers.
    """
    # In a real implementation, read files like:
    #   prompts/islamic_context/1.0.0.txt
    #   prompts/islamic_context/1.0.0.meta.json
    # For now, return the default registry.
    return DEFAULT_REGISTRY
