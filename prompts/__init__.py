"""Prompt template registry and A/B experimentation harness.

Usage::

    from prompts import get_registry, register_defaults, ExperimentHarness

    register_defaults()
    registry = get_registry()
    template = registry.get("islamic_context")
    rendered = template.render(response_language="en")
"""

from prompts.defaults import register_defaults
from prompts.registry import (
    ExperimentAssignment,
    ExperimentConfig,
    ExperimentHarness,
    PromptRegistry,
    PromptTemplate,
    Variant,
    get_registry,
    resolve_system_prompt,
)
from prompts.template import PromptTemplate as Template

__all__ = [
    "ExperimentAssignment",
    "ExperimentConfig",
    "ExperimentHarness",
    "PromptRegistry",
    "PromptTemplate",
    "Template",
    "Variant",
    "get_registry",
    "register_defaults",
    "resolve_system_prompt",
]
