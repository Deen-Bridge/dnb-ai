"""Tests for PolinRider indicator audit verifier (#521)."""

from __future__ import annotations

from pathlib import Path

from polinrider_audit import audit_directory_non_executing


def test_clean_directory_passes_audit(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("fastapi==0.115.0\n", encoding="utf-8")
    (tmp_path / "module.py").write_text("x = 1\n", encoding="utf-8")
    report = audit_directory_non_executing(tmp_path)
    assert report.clean is True
    assert "requirements.txt" in report.manifests_reviewed


def test_detects_folder_open_task_and_encoded_eval(tmp_path: Path) -> None:
    vscode = tmp_path / ".vscode"
    vscode.mkdir()
    (vscode / "tasks.json").write_text('{"runOn": "folderOpen"}\n', encoding="utf-8")
    (tmp_path / "tail.js").write_text("eval(Buffer.from('aGVsbG8=', 'base64'))\n", encoding="utf-8")
    report = audit_directory_non_executing(tmp_path)
    assert report.clean is False
    assert len(report.findings) == 2
