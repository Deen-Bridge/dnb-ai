"""Tests for non-executing repository-integrity scanner (#522)."""

from __future__ import annotations

from pathlib import Path

from repo_integrity_scanner import SUPPRESSION_TOKEN, scan_repository_integrity


def test_benign_files_report_zero_issues(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def greet() -> str:\n    return 'salam'\n", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"name": "dnb-ai"}\n', encoding="utf-8")
    assert scan_repository_integrity(tmp_path) == []


def test_detects_indicators_and_honors_narrow_reviewed_suppression(tmp_path: Path) -> None:
    bad = tmp_path / "tasks.json"
    bad.write_text('{"runOn": "folderOpen"}\n', encoding="utf-8")
    issues = scan_repository_integrity(tmp_path)
    assert len(issues) == 1
    assert issues[0].rule_id == "vscode_folder_open_task"
    assert issues[0].line == 1

    suppressed = tmp_path / "reviewed.js"
    suppressed.write_text(f"eval(atob('eA==')) // {SUPPRESSION_TOKEN}\n", encoding="utf-8")
    assert len(scan_repository_integrity(tmp_path, allowed_suppressions=set())) == 2
    filtered = scan_repository_integrity(tmp_path, allowed_suppressions={"reviewed.js"})
    assert len(filtered) == 1
