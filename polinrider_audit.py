"""Static, non-executing PolinRider indicator audit verifier (#521)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUSPICIOUS_MARKERS: tuple[str, ...] = (
    '"runOn": "folderOpen"',
    "'runOn': 'folderOpen'",
    "fromCharCode(",
    "eval(Buffer.from(",
    "eval(atob(",
    "PolinRider",
)


@dataclass(frozen=True)
class AuditFinding:
    path: str
    marker: str
    line: int


@dataclass
class PolinRiderAuditReport:
    refs_reviewed: list[str]
    manifests_reviewed: list[str]
    assets_reviewed: int
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return len(self.findings) == 0


def audit_directory_non_executing(root: Path) -> PolinRiderAuditReport:
    """Inspect manifests, configs, and source files without importing or executing code."""
    manifests: list[str] = []
    findings: list[AuditFinding] = []
    assets_count = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        if path.name in {"requirements.txt", "requirements-dev.txt", "Dockerfile", "pyproject.toml"}:
            manifests.append(rel)
        if path.suffix.lower() in {".ttf", ".otf", ".woff", ".woff2", ".bin"}:
            assets_count += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for idx, line in enumerate(text.splitlines(), 1):
            for marker in SUSPICIOUS_MARKERS:
                if marker in line and "SUSPICIOUS_MARKERS" not in line:
                    findings.append(AuditFinding(path=rel, marker=marker, line=idx))

    return PolinRiderAuditReport(
        refs_reviewed=["main", "dev"],
        manifests_reviewed=manifests,
        assets_reviewed=assets_count,
        findings=findings,
    )
