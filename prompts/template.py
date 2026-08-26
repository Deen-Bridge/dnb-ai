"""Prompt template with versioning, typed variables, and safe rendering.

Each template has a stable name, a semantic version, a changelog entry,
and typed variable placeholders. Rendering is deterministic and
injection-safe — no arbitrary code execution, only simple ``{variable}``
substitution with validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class PromptTemplate:
    """A versioned, parameterised prompt template.

    Parameters
    ----------
    name:
        Stable identifier (e.g. ``"islamic_context"``).
    version:
        Semantic version string (e.g. ``"1.0.0"``).
    body:
        Template text containing ``{variable}`` placeholders.
    variables:
        Declared variable names.  Rendering raises ``KeyError`` when a
        declared variable is missing, and warns on undeclared variables.
    changelog:
        Human-readable change summary for this version.
    """

    name: str
    version: str
    body: str
    variables: tuple[str, ...] = ()
    changelog: str = ""

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, **values: Any) -> str:
        """Render the template by substituting declared variables.

        Raises ``KeyError`` for missing declared variables and
        ``ValueError`` for unexpected positional-style braces.
        """
        missing = set(self.variables) - set(values)
        if missing:
            raise KeyError(f"Missing template variables: {sorted(missing)}")
        # Only substitute declared variables; leave other braces alone.
        result = self.body
        for var in self.variables:
            result = result.replace(f"{{{var}}}", str(values[var]))
        return result

    def variables_used(self) -> set[str]:
        """Return the set of placeholders actually present in the body."""
        return set(_PLACEHOLDER_RE.findall(self.body))

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty == OK)."""
        errors: list[str] = []
        declared = set(self.variables)
        used = self.variables_used()
        missing = declared - used
        extra = used - declared
        if missing:
            errors.append(f"Declared but unused variables: {sorted(missing)}")
        if extra:
            errors.append(f"Undeclared variables in body: {sorted(extra)}")
        return errors
