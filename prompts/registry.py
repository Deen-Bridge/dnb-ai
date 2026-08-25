"""Prompt template registry — versioned store with A/B experimentation.

Provides:
- ``PromptRegistry``: a central store for prompt templates with lookup by
  name and version, plus a ``latest`` shortcut.
- ``ExperimentConfig`` / ``ExperimentHarness``: sticky session-based A/B
  variant assignment with a kill switch and metadata emission.
- ``resolve_system_prompt``: a convenience that composes the current
  active template(s) into a single system-context string.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from prompts.template import PromptTemplate

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------


class PromptRegistry:
    """A central store for versioned prompt templates."""

    def __init__(self) -> None:
        self._templates: dict[str, list[PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register a template.  Later registrations of the same name
        append a new version — the highest version wins ``latest``."""
        errors = template.validate()
        if errors:
            raise ValueError(
                f"Template '{template.name}' v{template.version} has validation errors: " + "; ".join(errors)
            )
        self._templates.setdefault(template.name, []).append(template)

    def get(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """Return a specific version, or ``None`` if not found."""
        versions = self._templates.get(name, [])
        if not versions:
            return None
        if version is None:
            return self.latest(name)
        for t in versions:
            if t.version == version:
                return t
        return None

    def latest(self, name: str) -> PromptTemplate | None:
        """Return the highest registered version of *name*."""
        versions = self._templates.get(name, [])
        if not versions:
            return None
        return versions[-1]

    def list_templates(self) -> dict[str, str]:
        """Return ``{name: latest_version}`` for every registered template."""
        return {name: versions[-1].version for name, versions in self._templates.items()}


# -----------------------------------------------------------------------
# A/B Experimentation
# -----------------------------------------------------------------------


@dataclass
class Variant:
    """A named prompt variant used inside an experiment."""

    name: str
    template_name: str
    template_version: str | None = None  # None → latest
    weight: float = 1.0


@dataclass
class ExperimentConfig:
    """Configuration for one A/B experiment.

    Variants are weighted: the probability of a session landing on a
    variant is ``weight / sum(weights)``.  When ``kill_switch`` is True
    every session gets the control variant.
    """

    experiment_id: str
    control: Variant
    variants: list[Variant] = field(default_factory=list)
    kill_switch: bool = False


class ExperimentHarness:
    """Sticky variant assignment for A/B experiments.

    A session is assigned to exactly one variant for the lifetime of the
    experiment.  The assignment is deterministic: the same
    (experiment_id, session_id) always yields the same variant.
    """

    def __init__(self, registry: PromptRegistry) -> None:
        self._registry = registry
        self._experiments: dict[str, ExperimentConfig] = {}

    def register_experiment(self, config: ExperimentConfig) -> None:
        """Register an experiment configuration."""
        self._experiments[config.experiment_id] = config

    def unregister_experiment(self, experiment_id: str) -> None:
        """Remove an experiment."""
        self._experiments.pop(experiment_id, None)

    def active_experiments(self) -> list[str]:
        """Return experiment IDs that are registered and not killed."""
        return [eid for eid, cfg in self._experiments.items() if not cfg.kill_switch]

    def assign(self, experiment_id: str, session_id: str) -> ExperimentAssignment:
        """Deterministically assign *session_id* to a variant.

        When the experiment has a kill switch active, the control variant
        is returned immediately.
        """
        config = self._experiments.get(experiment_id)
        if config is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")

        if config.kill_switch:
            return ExperimentAssignment(
                experiment_id=experiment_id,
                variant_name=config.control.name,
                kill_switch_active=True,
            )

        # Weighted sticky assignment using a deterministic hash.
        all_variants = [config.control] + config.variants
        total_weight = sum(v.weight for v in all_variants)
        raw = _stable_hash(f"{experiment_id}:{session_id}")
        pick = raw * total_weight
        cumulative = 0.0
        chosen = config.control
        for v in all_variants:
            cumulative += v.weight
            if pick < cumulative:
                chosen = v
                break

        return ExperimentAssignment(
            experiment_id=experiment_id,
            variant_name=chosen.name,
            kill_switch_active=False,
        )

    def resolve_template(
        self,
        experiment_id: str,
        session_id: str,
        **values: Any,
    ) -> tuple[str, ExperimentAssignment]:
        """Assign a session to a variant and render the template.

        Returns ``(rendered_prompt, assignment)``.
        """
        assignment = self.assign(experiment_id, session_id)
        config = self._experiments[experiment_id]
        all_variants = [config.control] + config.variants
        chosen_variant = next(
            (v for v in all_variants if v.name == assignment.variant_name),
            config.control,
        )
        template = self._registry.get(
            chosen_variant.template_name,
            chosen_template_version(chosen_variant),
        )
        if template is None:
            raise RuntimeError(f"Template '{chosen_variant.template_name}' not found in registry")
        return template.render(**values), assignment


def chosen_template_version(variant: Variant) -> str | None:
    """Return the explicit version, or ``None`` to use latest."""
    return variant.template_version


class ExperimentAssignment(BaseModel):
    """The result of a variant assignment — included in response metadata."""

    experiment_id: str
    variant_name: str
    kill_switch_active: bool = False


def _stable_hash(key: str) -> float:
    """Deterministic hash in [0, 1) for weighted variant assignment."""
    digest = hashlib.sha256(key.encode()).digest()
    # Use the first 8 bytes as a uint64, normalise to [0, 1).
    raw_int = int.from_bytes(digest[:8], "big")
    return raw_int / (2**64)


# -----------------------------------------------------------------------
# System prompt composition
# -----------------------------------------------------------------------


def resolve_system_prompt(
    registry: PromptRegistry,
    template_name: str = "islamic_context",
    *,
    version: str | None = None,
    **values: Any,
) -> str:
    """Render the named template and return it as the system prompt."""
    template = registry.get(template_name, version)
    if template is None:
        raise KeyError(f"Template '{template_name}' not registered")
    return template.render(**values)


# -----------------------------------------------------------------------
# Default registry (populated at import time by ``prompts``)
# -----------------------------------------------------------------------

_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    """Return (and lazily create) the global prompt registry."""
    global _registry  # noqa: PLW0603
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
