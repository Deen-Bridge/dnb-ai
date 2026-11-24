"""Versioned benchmark data and provenance (#825).

Every committed evaluation dataset is pinned in ``data/eval/benchmark_manifest.json``
with a semantic version, a content hash, its record count, a minimum record
threshold, where it came from, and an append-only version history. The check
fails when a dataset drifts from its manifest entry, drops below its threshold,
loses its provenance, contains secret-looking strings, or when a new data file
appears under a scanned directory without being versioned.

Offline and deterministic: no network, API keys or provider responses.

Usage:
    python benchmark_manifest.py            # verify; exit 1 with FAIL: lines on stderr
    python benchmark_manifest.py --update   # re-pin hashes after bumping ``version``
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "data" / "eval" / "benchmark_manifest.json"

SUPPORTED_FORMATS = ("json", "jsonl")
REQUIRED_FIELDS = ("path", "format", "version", "sha256", "source", "license", "contains_personal_data", "history")
DATA_SUFFIXES = (".json", ".jsonl")

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Matches are reported by kind only, never echoed, so a leaked credential does
# not end up copied into CI logs.
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "an email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    "a Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "an OpenAI/Anthropic-style API key": re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    "a GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "a Stellar secret seed": re.compile(r"\bS[A-Z2-7]{55}\b"),
    "a bearer token": re.compile(r"(?i)\bbearer\s+[a-z0-9._\-]{20,}"),
}


class ManifestError(Exception):
    """The manifest itself is unreadable or structurally invalid."""


def content_sha256(raw: bytes) -> str:
    """SHA-256 of ``raw`` with CRLF folded to LF.

    Windows checkouts with ``core.autocrlf`` rewrite line endings, so hashing the
    bytes on disk would give a different digest per platform for the same data.
    """
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def count_records(raw: bytes, fmt: str, records_key: str | None = None) -> int | None:
    """Number of records in a dataset, or None when the file is not a collection.

    ``jsonl`` files count non-blank lines, each of which must be a JSON object.
    ``json`` files count ``data[records_key]`` when a key is given, otherwise the
    top-level list. Raises ValueError (with a location) on malformed content.
    """
    text = raw.decode("utf-8-sig")
    if fmt == "jsonl":
        count = 0
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {lineno}: expected a JSON object")
            count += 1
        return count

    data = json.loads(text)
    if records_key:
        items = data.get(records_key) if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ValueError(f"'{records_key}' is not a list")
        return len(items)
    return len(data) if isinstance(data, list) else None


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), dict):
        raise ManifestError(f"manifest {path} must be an object with a 'datasets' mapping")
    return manifest


def write_manifest(manifest: dict[str, Any], path: Path | str = DEFAULT_MANIFEST) -> None:
    # newline="\n" keeps the file byte-identical whichever OS rewrote it.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_in_root(relative: str, root: Path) -> Path:
    """Resolve a manifest path, refusing anything that escapes the repo root."""
    root = root.resolve()
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root):
        raise ManifestError(f"path {relative!r} must be relative and inside the repository")
    return candidate


def manifest_errors(manifest: dict[str, Any]) -> list[str]:
    """Problems with the manifest-level ownership and rollback metadata."""
    problems = [f"manifest.{key} missing" for key in ("owner", "rollback") if not manifest.get(key)]
    if not isinstance(manifest.get("manifest_version"), int):
        problems.append("manifest.manifest_version must be an integer")
    if not manifest["datasets"]:
        problems.append("manifest lists no datasets")
    return problems


def _entry_errors(name: str, entry: Any) -> list[str]:
    """Structural problems with one dataset entry, before any file is read."""
    if not isinstance(entry, dict):
        return [f"{name}: manifest entry must be an object"]
    missing = [key for key in REQUIRED_FIELDS if key not in entry or entry[key] in ("", None)]
    if missing:
        return [f"{name}: manifest entry missing {', '.join(missing)}"]

    problems = []
    if entry["format"] not in SUPPORTED_FORMATS:
        problems.append(f"{name}: format must be one of {', '.join(SUPPORTED_FORMATS)}")
    if not isinstance(entry["version"], str) or not _SEMVER.match(entry["version"]):
        problems.append(f"{name}: version {entry['version']!r} is not MAJOR.MINOR.PATCH")
    if not isinstance(entry["sha256"], str) or not _SHA256.match(entry["sha256"]):
        problems.append(f"{name}: sha256 must be 64 lowercase hex characters")
    if entry["contains_personal_data"] is not False:
        problems.append(f"{name}: contains_personal_data must be false; benchmark data must be synthetic or redacted")
    min_records = entry.get("min_records", 0)
    if isinstance(min_records, bool) or not isinstance(min_records, int) or min_records < 0:
        problems.append(f"{name}: min_records must be a non-negative integer")

    history = entry["history"]
    if not isinstance(history, list) or not history or not all(isinstance(h, dict) for h in history):
        problems.append(f"{name}: history must be a non-empty list of {{version, sha256}} entries")
        return problems
    versions = [h.get("version") for h in history]
    if len(versions) != len(set(versions)):
        problems.append(f"{name}: history repeats a version; every data change needs a new version")
    latest = history[-1]
    if latest.get("version") != entry["version"] or latest.get("sha256") != entry["sha256"]:
        problems.append(f"{name}: latest history entry must match the current version and sha256")
    return problems


def verify_dataset(name: str, entry: Any, root: Path = ROOT) -> tuple[list[str], dict[str, Any]]:
    """Check one dataset against its entry. Returns (problems, observed facts)."""
    problems = _entry_errors(name, entry)
    if problems:
        return problems, {}

    try:
        path = resolve_in_root(entry["path"], root)
    except ManifestError as exc:
        return [f"{name}: {exc}"], {}
    try:
        raw = path.read_bytes()
    except OSError:
        return [f"{name}: {entry['path']} is missing; restore it or remove its manifest entry"], {}

    observed: dict[str, Any] = {"sha256": content_sha256(raw)}
    if observed["sha256"] != entry["sha256"]:
        problems.append(
            f"{name}: {entry['path']} changed without a version bump "
            f"(sha256 {observed['sha256'][:12]}... vs pinned {entry['sha256'][:12]}...). "
            "Bump `version` and run `python benchmark_manifest.py --update`, or revert the data change."
        )

    try:
        observed["records"] = count_records(raw, entry["format"], entry.get("records_key"))
    except ValueError as exc:
        problems.append(f"{name}: {entry['path']} is not valid {entry['format']} ({exc})")
        return problems, observed

    records = observed["records"]
    if "records" in entry and entry["records"] != records:
        problems.append(f"{name}: has {records} records, manifest pins {entry['records']}")
    min_records = entry.get("min_records", 0)
    if min_records and (records is None or records < min_records):
        problems.append(f"{name}: has {records} records, below the min_records threshold of {min_records}")

    text = raw.decode("utf-8-sig", errors="replace")
    for kind, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            problems.append(f"{name}: {entry['path']} contains what looks like {kind}; remove or redact it")
    return problems, observed


def unversioned_files(manifest: dict[str, Any], manifest_path: Path, root: Path = ROOT) -> list[str]:
    """Data files under ``scan_dirs`` that have no manifest entry."""
    listed = {manifest_path.resolve()}
    for entry in manifest["datasets"].values():
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            try:
                listed.add(resolve_in_root(entry["path"], root))
            except ManifestError:
                continue  # reported by verify_dataset
    ignore = [p for p in manifest.get("ignore", []) if isinstance(p, str)]

    problems = []
    for scan_dir in manifest.get("scan_dirs", []):
        try:
            base = resolve_in_root(scan_dir, root)
        except ManifestError as exc:
            problems.append(f"scan_dirs: {exc}")
            continue
        if not base.is_dir():
            problems.append(f"scan_dirs: {scan_dir} does not exist")
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in DATA_SUFFIXES or not path.is_file() or path.resolve() in listed:
                continue
            # ``base`` is already resolved inside ``root``, so this never raises.
            relative = path.relative_to(root.resolve()).as_posix()
            if any(fnmatch.fnmatch(relative, pattern) for pattern in ignore):
                continue
            problems.append(
                f"unversioned benchmark file {relative}: add it to {manifest_path.name} "
                f"with a version, owner and source, or move it out of {scan_dir}"
            )
    return problems


def check(manifest_path: Path | str = DEFAULT_MANIFEST, root: Path = ROOT) -> tuple[bool, dict[str, Any], list[str]]:
    """Verify every dataset. Returns (ok, summary, actionable_messages)."""
    manifest_path = Path(manifest_path)
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        return False, {"datasets": 0, "verified": 0, "failed": 0}, [str(exc)]

    messages = manifest_errors(manifest)
    versions: dict[str, Any] = {}
    failed = 0
    for name, entry in sorted(manifest["datasets"].items()):
        problems, _ = verify_dataset(name, entry, root)
        versions[name] = entry.get("version") if isinstance(entry, dict) else None
        failed += bool(problems)
        messages.extend(problems)
    messages.extend(unversioned_files(manifest, manifest_path, root))

    summary = {
        "manifest_version": manifest.get("manifest_version"),
        "datasets": len(versions),
        "verified": len(versions) - failed,
        "failed": failed,
        "versions": versions,
    }
    return not messages, summary, messages


def update(manifest_path: Path | str = DEFAULT_MANIFEST, root: Path = ROOT) -> tuple[list[str], list[str]]:
    """Re-pin hashes and record counts for datasets whose version was bumped.

    A dataset whose content changed but whose ``version`` already appears in its
    history is refused, so a version can never silently point at two different
    files. Nothing is written unless every changed dataset is valid.
    Returns (updated_names, errors).
    """
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    errors: list[str] = []
    updated: list[str] = []

    for name, entry in sorted(manifest["datasets"].items()):
        if not isinstance(entry, dict) or any(key not in entry for key in ("path", "format", "version", "history")):
            errors.append(f"{name}: manifest entry needs path, format, version and history before updating")
            continue
        try:
            raw = resolve_in_root(entry["path"], root).read_bytes()
            records = count_records(raw, entry["format"], entry.get("records_key"))
        except (ManifestError, OSError, ValueError) as exc:
            errors.append(f"{name}: cannot read {entry['path']} ({exc})")
            continue

        digest = content_sha256(raw)
        if digest == entry.get("sha256"):
            continue
        history = entry["history"] if isinstance(entry["history"], list) else []
        if any(isinstance(h, dict) and h.get("version") == entry["version"] for h in history):
            errors.append(
                f"{name}: content changed but version {entry['version']} is already recorded; "
                "bump `version` (major/minor/patch) first"
            )
            continue
        entry["sha256"] = digest
        if records is not None:
            entry["records"] = records
        entry["history"] = [*history, {"version": entry["version"], "sha256": digest}]
        updated.append(name)

    if updated and not errors:
        write_manifest(manifest, manifest_path)
    return (updated if not errors else []), errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify versioned benchmark data and provenance.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--update", action="store_true", help="re-pin hashes for datasets whose version was bumped")
    args = parser.parse_args(argv)

    if args.update:
        try:
            updated, errors = update(args.manifest)
        except ManifestError as exc:
            errors, updated = [str(exc)], []
        for message in errors:
            print(f"FAIL: {message}", file=sys.stderr)
        if errors:
            return 1
        print(f"Updated: {', '.join(updated)}" if updated else "Manifest already up to date.")

    ok, summary, messages = check(args.manifest)
    print(json.dumps(summary, sort_keys=True))
    for message in messages:
        print(f"FAIL: {message}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
