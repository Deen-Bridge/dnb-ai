"""Prompt version lock and evaluation-baseline compatibility rules.

Every registered prompt template is pinned in a committed lock file
(``prompts/prompt_versions.lock.json``) by version and content fingerprint.
The lock turns two silent failure modes into CI failures:

* the template text changes but the version does not, so baselines and
  telemetry keep attributing new behaviour to the old version;
* the version changes but the lock (the reviewed record of what shipped)
  was not regenerated.

Compatibility with evaluation baselines follows semantic versioning: a
baseline recorded against prompt ``X.y.z`` stays comparable for any
``X.*.*``. A major bump declares that scores are no longer comparable, so the
baseline must be regenerated (and reviewed) in the same change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prompts.registry import PromptRegistry, parse_version

LOCK_SCHEMA_VERSION = 1
DEFAULT_LOCK_PATH = Path(__file__).resolve().parent / "prompt_versions.lock.json"
LOCK_OWNER = "AI quality maintainers"
UPDATE_HINT = "python -m scripts.check_prompt_eval --write-lock"


def is_eval_compatible(baseline_version: str, current_version: str) -> bool:
    """Return True when scores under both prompt versions may be compared."""
    return parse_version(baseline_version)[0] == parse_version(current_version)[0]


def build_lock(registry: PromptRegistry, owner: str = LOCK_OWNER) -> dict[str, Any]:
    """Build the lock document for every template in ``registry``."""
    templates = {}
    for name in registry.list_names():
        template = registry.get(name)
        templates[name] = {
            "version": template.version,
            "fingerprint": template.fingerprint,
            "changelog": template.changelog,
        }
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "owner": owner,
        "templates": templates,
    }


def load_lock(path: Path | str = DEFAULT_LOCK_PATH) -> dict[str, Any] | None:
    """Load the lock file, or return None when it does not exist."""
    lock_path = Path(path)
    if not lock_path.exists():
        return None
    with open(lock_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{lock_path} must contain a JSON object")
    return data


def write_lock(registry: PromptRegistry, path: Path | str = DEFAULT_LOCK_PATH) -> dict[str, Any]:
    """Regenerate the lock file from the registry and return its contents."""
    lock = build_lock(registry)
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(lock, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return lock


def check_lock(registry: PromptRegistry, lock: dict[str, Any] | None) -> list[str]:
    """Compare the registry with a lock document; return actionable errors.

    An empty list means every registered template is pinned at its current
    version and content.
    """
    if lock is None:
        return [f"prompt lock file is missing; run `{UPDATE_HINT}` and commit it"]
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        return [
            f"prompt lock schema_version {lock.get('schema_version')!r} is not supported (expected {LOCK_SCHEMA_VERSION})"
        ]
    if not lock.get("owner"):
        return ["prompt lock has no owner"]

    locked: dict[str, Any] = lock.get("templates") or {}
    errors: list[str] = []
    for name in registry.list_names():
        template = registry.get(name)
        entry = locked.get(name)
        if not isinstance(entry, dict):
            errors.append(f"prompt '{name}' is not in the lock; review it and run `{UPDATE_HINT}`")
            continue

        locked_version = str(entry.get("version", ""))
        same_content = entry.get("fingerprint") == template.fingerprint
        if locked_version == template.version and not same_content:
            errors.append(
                f"prompt '{name}' text changed but its version is still {template.version}; "
                "bump the version (major if evaluation baselines are no longer comparable), "
                f"add a changelog entry, then run `{UPDATE_HINT}`"
            )
        elif locked_version != template.version:
            errors.append(
                f"prompt '{name}' is version {template.version} but the lock records {locked_version or 'nothing'}; "
                f"run `{UPDATE_HINT}` and commit the lock with the change"
            )

    for name in sorted(set(locked) - set(registry.list_names())):
        errors.append(f"lock entry '{name}' has no registered template; run `{UPDATE_HINT}`")
    return errors
