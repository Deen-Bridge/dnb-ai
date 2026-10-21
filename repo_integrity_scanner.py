"""Non-executing repository-integrity scanner for CI (#522)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPRESSION_TOKEN = "integrity-scan: allow-reviewed"

INDICATOR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vscode_folder_open_task", re.compile(r'"runOn"\s*:\s*"folderOpen"')),
    ("js_encoded_eval_payload", re.compile(r"\beval\s*\(\s*(?:atob|Buffer\.from)\s*\(")),
    ("python_install_hook_exec", re.compile(r"cmdclass\s*=\s*\{.*install", re.IGNORECASE)),
    ("trailing_null_or_high_entropy_tail", re.compile(r"\x00{4,}")),
)


@dataclass(frozen=True)
class IntegrityIssue:
    file: str
    line: int
    rule_id: str
    snippet: str


def scan_repository_integrity(
    root: Path,
    allowed_suppressions: set[str] | None = None,
) -> list[IntegrityIssue]:
    """Scan Python, JS, JSON, and config files without importing or executing project code."""
    issues: list[IntegrityIssue] = []
    allowed = allowed_suppressions or set()

    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(content.splitlines(), 1):
            if SUPPRESSION_TOKEN in line and rel in allowed:
                continue
            for rule_id, pattern in INDICATOR_PATTERNS:
                if pattern.search(line):
                    issues.append(
                        IntegrityIssue(
                            file=rel,
                            line=line_no,
                            rule_id=rule_id,
                            snippet=line.strip()[:120],
                        )
                    )
    return issues
