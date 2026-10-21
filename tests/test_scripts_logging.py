"""Tests for script diagnostic logging and --verbose / LOG_LEVEL control (#427)."""

from __future__ import annotations

import logging

from scripts import build_hadith_data


def test_configure_script_logging_verbose_and_env(monkeypatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert build_hadith_data.configure_script_logging(verbose=False) == logging.INFO
    assert build_hadith_data.configure_script_logging(verbose=True) == logging.DEBUG

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    assert build_hadith_data.configure_script_logging(verbose=False) == logging.WARNING
