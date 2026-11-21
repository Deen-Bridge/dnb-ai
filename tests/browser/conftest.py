"""Browser-suite options.

Baselines are committed images. Updating them changes the visual contract, so
it requires a deliberate flag (``--update-baselines``) and human review of the
resulting image diff.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Regenerate committed RTL screenshot baselines (requires reviewer approval).",
    )
