"""Emit a deterministic, content-addressed runtime build provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_PROVENANCE_PATH = ROOT / "model_provenance.json"
MODEL_ID_PATTERN = re.compile(r"\bgemini-\d+(?:\.\d+){1,2}-[a-z0-9]+(?:-[a-z0-9]+)*\b")


def _locked_version(lock_path: Path, package_name: str) -> str:
    """Read the exact first-party tool pin from a generated pip-tools lock."""
    prefix = f"{package_name.lower()}=="
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        normalized = line.strip().lower()
        if normalized.startswith(prefix):
            return normalized[len(prefix) :].strip().rstrip("\\").strip()
    raise ValueError(f"{package_name} is not pinned in {lock_path.name}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_baseline(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Provenance baseline does not exist: {path.relative_to(ROOT)}")

    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file() or "__pycache__" in child.parts or child.suffix == ".pyc":
            continue
        relative = child.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _installed_distributions() -> list[dict[str, str]]:
    installed = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed.append({"name": name.lower().replace("_", "-"), "version": distribution.version})
    return sorted(installed, key=lambda item: (item["name"], item["version"]))


def _runtime_model_ids() -> set[str]:
    """Find literal Gemini model IDs in runtime code, excluding test fixtures."""
    model_ids: set[str] = set()
    for source_path in ROOT.rglob("*.py"):
        excluded_directories = {
            ".git",
            ".venv",
            "__pycache__",
            "build",
            "dist",
            "site-packages",
            "test",
            "tests",
            "venv",
        }
        if excluded_directories.intersection(source_path.relative_to(ROOT).parts):
            continue
        model_ids.update(MODEL_ID_PATTERN.findall(source_path.read_text(encoding="utf-8")))
    return model_ids


def _deployment_model_configuration(model_provenance: dict[str, Any]) -> dict[str, str]:
    """Ensure Render and Docker declare the same pinned Python/model baseline."""
    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    render_config = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    service = render_config["services"][0]
    env_vars = {item["key"]: str(item.get("value", "")) for item in service.get("envVars", [])}
    model_name = model_provenance["models"]["experiment_default"]["id"]
    if env_vars.get("PYTHON_VERSION") != python_version:
        raise ValueError("Render PYTHON_VERSION does not match .python-version")
    if env_vars.get("MODEL_NAME") != model_name:
        raise ValueError("Render MODEL_NAME is not in sync with model_provenance.json")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    if f"FROM python:{python_version}-slim" not in dockerfile:
        raise ValueError("Docker base Python version does not match .python-version")
    if platform.python_version() != python_version:
        raise ValueError(f"Build Python {platform.python_version()} does not match pinned {python_version}")
    return {
        "python_version": python_version,
        "model_name": model_name,
        "primary_provider": env_vars.get("LLM_PRIMARY", "gemini"),
    }


def build_manifest(lock_file: Path, commit: str | None = None) -> dict[str, Any]:
    """Build metadata from locked installs and versioned model/eval baselines."""
    lock_path = lock_file if lock_file.is_absolute() else ROOT / lock_file
    if not lock_path.is_file():
        raise FileNotFoundError(f"Dependency lockfile does not exist: {lock_file}")

    model_provenance = json.loads(MODEL_PROVENANCE_PATH.read_text(encoding="utf-8"))
    allowed_model_statuses = {"approved", "approved_optional", "legacy_preview", "legacy_compatibility"}
    models = model_provenance["models"].values()
    if any(model.get("status") not in allowed_model_statuses for model in models):
        raise ValueError("Model provenance contains an unreviewed or invalid model status")
    approved_model_ids = {item["id"] for item in model_provenance["models"].values()}
    unregistered_models = _runtime_model_ids() - approved_model_ids
    if unregistered_models:
        raise ValueError(f"Runtime model IDs are missing from model_provenance.json: {sorted(unregistered_models)}")
    dependencies = _installed_distributions()
    installed_versions = {item["name"]: item["version"] for item in dependencies}
    sdk = model_provenance["provider_sdk"]
    installed_sdk_version = installed_versions.get(sdk["package"])
    if installed_sdk_version != sdk["version"]:
        raise ValueError(
            f"Approved provider SDK {sdk['package']}=={sdk['version']} "
            f"does not match installed version {installed_sdk_version!r}"
        )

    deployment_model_configuration = _deployment_model_configuration(model_provenance)
    baseline_hashes = {
        name: _hash_baseline(ROOT / relative_path) for name, relative_path in model_provenance["baselines"].items()
    }
    return {
        "schema_version": "1.0.0",
        "commit": commit or _git_commit(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "abi": sys.implementation.cache_tag,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "installer": {"pip_version": metadata.version("pip")},
        "build_tools": {
            "pip_tools": _locked_version(ROOT / "requirements-dev.lock", "pip-tools"),
            "pip_audit": _locked_version(ROOT / "requirements-dev.lock", "pip-audit"),
        },
        "dependency_lock": {
            "path": lock_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(lock_path),
        },
        "all_lockfiles": {
            name: _sha256_file(ROOT / filename)
            for name, filename in (
                ("installer", "requirements-installer.lock"),
                ("runtime", "requirements.lock"),
                ("development", "requirements-dev.lock"),
                ("performance", "requirements-perf.lock"),
                ("optional_alignment", "requirements-alignment.lock"),
            )
        },
        "source_manifests": {
            name: _sha256_file(ROOT / filename)
            for name, filename in (
                ("runtime", "requirements.txt"),
                ("development", "requirements-dev.txt"),
                ("performance", "requirements-perf.txt"),
                ("installer", "requirements-installer.txt"),
                ("optional_alignment", "requirements-alignment.txt"),
            )
        },
        "dependencies": dependencies,
        "model_provenance": {
            "path": MODEL_PROVENANCE_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(MODEL_PROVENANCE_PATH),
            "approved_configuration": model_provenance,
            "deployment_configuration": deployment_model_configuration,
        },
        "baseline_sha256": baseline_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default="requirements.lock", help="Hash-locked runtime requirements file")
    parser.add_argument("--output", required=True, help="Path for the deterministic JSON manifest")
    args = parser.parse_args()

    output_path = Path(args.output)
    manifest = build_manifest(Path(args.lock))
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
