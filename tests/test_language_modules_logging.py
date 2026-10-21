"""Verification tests ensuring language modules use standard logging and zero print() debug (#425)."""

from __future__ import annotations

import ast
import importlib
import logging
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

LANGUAGE_MODULE_FILES = [
    "arabic_ocr.py",
    "calligraphy.py",
    "swahili_processor.py",
    "arabic_dialect/__init__.py",
    "arabic_dialect/dialects.py",
    "arabic_dialect/models.py",
    "arabic_dialect/router.py",
    "arabic_dialect/terminology.py",
    "swahili/__init__.py",
    "swahili/ router.py".replace(" ", ""),
]


@pytest.mark.parametrize("rel_path", LANGUAGE_MODULE_FILES)
def test_language_module_has_no_print_calls(rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not present")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
    print_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
    ]
    assert print_lines == [], f"Unexpected print() call(s) in {rel_path} at line(s) {print_lines}"


@pytest.mark.parametrize(
    "module_name",
    ["arabic_ocr", "calligraphy", "swahili_processor", "arabic_dialect.router"],
)
def test_language_module_exposes_module_logger(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "logger")
    assert isinstance(mod.logger, logging.Logger)
    assert mod.logger.name == module_name
