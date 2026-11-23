"""Deterministic build provenance manifest tests."""

from pathlib import Path

import pytest

import scripts.build_provenance as provenance
from scripts.build_provenance import ROOT, build_manifest


def test_same_commit_and_environment_produce_identical_manifest() -> None:
    lock_file = ROOT / "requirements.lock"
    first = build_manifest(lock_file, commit="fixture-commit")
    second = build_manifest(lock_file, commit="fixture-commit")

    assert first == second
    assert first["commit"] == "fixture-commit"
    assert first["dependency_lock"]["sha256"]
    assert first["all_lockfiles"]["runtime"]
    assert first["all_lockfiles"]["development"]
    assert first["model_provenance"]["approved_configuration"]["models"]
    assert first["baseline_sha256"]


def test_lock_manifest_path_is_repository_relative() -> None:
    manifest = build_manifest(Path("requirements.lock"), commit="fixture-commit")
    assert manifest["dependency_lock"]["path"] == "requirements.lock"


def test_unregistered_runtime_model_id_fails_provenance(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "_runtime_model_ids", lambda: {"gemini-9.9-unreviewed"})
    with pytest.raises(ValueError, match="missing from model_provenance.json"):
        build_manifest(ROOT / "requirements.lock", commit="fixture-commit")
