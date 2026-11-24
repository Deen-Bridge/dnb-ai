"""Versioned benchmark data and provenance (#825).

Each test builds a tiny deterministic repository under ``tmp_path`` so the
failure modes can be exercised without touching the committed datasets; the
committed manifest itself is checked by ``test_committed_benchmarks_*``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark_manifest as bm

QA_LINES = [
    {"id": "qa-001", "question": "How many daily prayers are obligatory?", "expected_answer": "Five."},
    {"id": "qa-002", "question": "What is the first pillar of Islam?", "expected_answer": "The shahadah."},
    {"id": "qa-003", "question": "In which month is fasting obligatory?", "expected_answer": "Ramadan."},
]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    # Explicit LF so the fixture bytes are identical on every OS.
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8", newline="\n")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal repo: one jsonl benchmark, one keyed json results file, a pinned manifest."""
    eval_dir = tmp_path / "data" / "eval"
    eval_dir.mkdir(parents=True)
    _write_jsonl(eval_dir / "qa.jsonl", QA_LINES)
    (eval_dir / "results.json").write_text(
        json.dumps({"pass_rate": 1.0, "results": [{"id": r["id"], "passed": True} for r in QA_LINES]}),
        encoding="utf-8",
    )
    manifest = {
        "manifest_version": 1,
        "owner": "eval maintainers",
        "rollback": "git revert the dataset commit",
        "scan_dirs": ["data/eval"],
        "ignore": ["data/eval/live_results*.json"],
        "datasets": {
            "qa": {
                "path": "data/eval/qa.jsonl",
                "format": "jsonl",
                "version": "1.0.0",
                "sha256": "",
                "min_records": 3,
                "source": "unit-test fixture",
                "license": "MIT",
                "contains_personal_data": False,
                "history": [],
            },
            "results": {
                "path": "data/eval/results.json",
                "format": "json",
                "records_key": "results",
                "version": "1.0.0",
                "sha256": "",
                "min_records": 3,
                "source": "unit-test fixture",
                "license": "MIT",
                "contains_personal_data": False,
                "history": [],
            },
        },
    }
    bm.write_manifest(manifest, eval_dir / "manifest.json")
    updated, errors = bm.update(eval_dir / "manifest.json", root=tmp_path)
    assert errors == [] and updated == ["qa", "results"]
    return tmp_path


def _manifest_path(repo: Path) -> Path:
    return repo / "data" / "eval" / "manifest.json"


def _check(repo: Path) -> tuple[bool, dict, list[str]]:
    return bm.check(_manifest_path(repo), root=repo)


def _edit_entry(repo: Path, name: str, **changes: object) -> None:
    manifest = bm.load_manifest(_manifest_path(repo))
    manifest["datasets"][name].update(changes)
    bm.write_manifest(manifest, _manifest_path(repo))


# --- committed data -----------------------------------------------------------


def test_committed_benchmarks_match_manifest():
    ok, summary, messages = bm.check()
    assert ok, "\n".join(messages)
    assert summary["failed"] == 0
    assert summary["verified"] == summary["datasets"] >= 7


def test_committed_manifest_has_owner_rollback_and_provenance():
    manifest = bm.load_manifest()
    assert bm.manifest_errors(manifest) == []
    for name, entry in manifest["datasets"].items():
        assert entry["source"] and entry["license"], name
        assert entry["contains_personal_data"] is False, name
        assert entry["history"][-1] == {"version": entry["version"], "sha256": entry["sha256"]}, name


def test_main_reports_summary_and_exit_code(capsys):
    assert bm.main([]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["failed"] == 0


# --- hashing and counting -----------------------------------------------------


def test_hash_ignores_line_ending_style():
    assert bm.content_sha256(b'{"a": 1}\n{"b": 2}\n') == bm.content_sha256(b'{"a": 1}\r\n{"b": 2}\r\n')
    assert bm.content_sha256(b"a\n") != bm.content_sha256(b"b\n")


def test_count_records_by_format():
    assert bm.count_records(b'{"a": 1}\n\n{"b": 2}\n', "jsonl") == 2
    assert bm.count_records(b"[1, 2, 3]", "json") == 3
    assert bm.count_records(b'{"items": [1, 2]}', "json", "items") == 2
    assert bm.count_records(b'{"owner": "x"}', "json") is None


@pytest.mark.parametrize(
    ("raw", "fmt", "key", "fragment"),
    [
        (b'{"a": 1}\nnot json\n', "jsonl", None, "line 2"),
        (b'{"a": 1}\n[1]\n', "jsonl", None, "expected a JSON object"),
        (b'{"items": 3}', "json", "items", "'items' is not a list"),
    ],
)
def test_count_records_rejects_malformed_data(raw, fmt, key, fragment):
    with pytest.raises(ValueError, match=fragment):
        bm.count_records(raw, fmt, key)


# --- the deterministic fixture -------------------------------------------------


def test_fixture_passes_after_pinning(repo: Path):
    ok, summary, messages = _check(repo)
    assert ok, messages
    assert summary["versions"] == {"qa": "1.0.0", "results": "1.0.0"}
    entry = bm.load_manifest(_manifest_path(repo))["datasets"]["qa"]
    assert entry["records"] == 3
    assert entry["history"] == [{"version": "1.0.0", "sha256": entry["sha256"]}]


def test_crlf_checkout_still_passes(repo: Path):
    path = repo / "data" / "eval" / "qa.jsonl"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    ok, _, messages = _check(repo)
    assert ok, messages


def test_silent_data_change_is_reported_and_update_refuses(repo: Path):
    _write_jsonl(repo / "data" / "eval" / "qa.jsonl", [*QA_LINES, {"id": "qa-004", "question": "q", "answer": "a"}])

    ok, summary, messages = _check(repo)
    assert not ok and summary["failed"] == 1
    assert any("qa: data/eval/qa.jsonl changed without a version bump" in m for m in messages)
    assert any("Bump `version`" in m for m in messages)

    before = _manifest_path(repo).read_bytes()
    updated, errors = bm.update(_manifest_path(repo), root=repo)
    assert updated == []
    assert errors and "version 1.0.0 is already recorded" in errors[0]
    assert _manifest_path(repo).read_bytes() == before


def test_version_bump_repins_and_extends_history(repo: Path):
    _write_jsonl(repo / "data" / "eval" / "qa.jsonl", [*QA_LINES, {"id": "qa-004", "question": "q", "answer": "a"}])
    _edit_entry(repo, "qa", version="1.1.0")

    updated, errors = bm.update(_manifest_path(repo), root=repo)
    assert (updated, errors) == (["qa"], [])
    entry = bm.load_manifest(_manifest_path(repo))["datasets"]["qa"]
    assert [h["version"] for h in entry["history"]] == ["1.0.0", "1.1.0"]
    assert entry["records"] == 4
    assert _check(repo)[0]


def test_dropping_below_threshold_fails(repo: Path):
    _write_jsonl(repo / "data" / "eval" / "qa.jsonl", QA_LINES[:2])
    _edit_entry(repo, "qa", version="2.0.0")
    assert bm.update(_manifest_path(repo), root=repo)[1] == []

    ok, _, messages = _check(repo)
    assert not ok
    assert messages == ["qa: has 2 records, below the min_records threshold of 3"]


def test_hand_edited_hash_without_history_fails(repo: Path):
    _edit_entry(repo, "qa", sha256="0" * 64)
    ok, _, messages = _check(repo)
    assert not ok
    assert any("latest history entry must match" in m for m in messages)


def test_secret_like_content_is_flagged_without_echoing_it(repo: Path):
    leaked_key = "AIza" + "Q" * 35
    _write_jsonl(repo / "data" / "eval" / "qa.jsonl", [*QA_LINES, {"id": "qa-004", "answer": f"key={leaked_key}"}])
    _edit_entry(repo, "qa", version="1.0.1")
    bm.update(_manifest_path(repo), root=repo)

    ok, _, messages = _check(repo)
    assert not ok
    assert any("looks like a Google API key" in m for m in messages)
    assert all(leaked_key not in m for m in messages)


def test_email_address_is_flagged(repo: Path):
    _write_jsonl(repo / "data" / "eval" / "qa.jsonl", [*QA_LINES, {"id": "qa-004", "asked_by": "someone@example.com"}])
    _edit_entry(repo, "qa", version="1.0.1")
    bm.update(_manifest_path(repo), root=repo)
    assert any("an email address" in m for m in _check(repo)[2])


def test_personal_data_flag_must_be_false(repo: Path):
    _edit_entry(repo, "results", contains_personal_data=True)
    ok, _, messages = _check(repo)
    assert not ok
    assert any("results: contains_personal_data must be false" in m for m in messages)


def test_missing_provenance_and_owner_fail(repo: Path):
    manifest = bm.load_manifest(_manifest_path(repo))
    manifest["owner"] = ""
    manifest["datasets"]["qa"]["source"] = ""
    bm.write_manifest(manifest, _manifest_path(repo))

    ok, _, messages = _check(repo)
    assert not ok
    assert "manifest.owner missing" in messages
    assert "qa: manifest entry missing source" in messages


def test_invalid_version_string_fails(repo: Path):
    _edit_entry(repo, "qa", version="v1")
    assert any("is not MAJOR.MINOR.PATCH" in m for m in _check(repo)[2])


def test_missing_dataset_file_is_actionable(repo: Path):
    (repo / "data" / "eval" / "results.json").unlink()
    ok, _, messages = _check(repo)
    assert not ok
    assert messages == ["results: data/eval/results.json is missing; restore it or remove its manifest entry"]


def test_path_outside_repository_is_rejected(repo: Path):
    _edit_entry(repo, "qa", path="../outside.jsonl")
    ok, _, messages = _check(repo)
    assert not ok
    assert any("must be relative and inside the repository" in m for m in messages)


def test_unversioned_data_file_fails_unless_ignored(repo: Path):
    eval_dir = repo / "data" / "eval"
    (eval_dir / "baseline_results.json").write_text("{}", encoding="utf-8")
    ok, _, messages = _check(repo)
    assert not ok
    assert messages == [
        "unversioned benchmark file data/eval/baseline_results.json: add it to manifest.json "
        "with a version, owner and source, or move it out of data/eval"
    ]

    (eval_dir / "baseline_results.json").unlink()
    (eval_dir / "live_results.json").write_text("{}", encoding="utf-8")
    (eval_dir / "notes.md").write_text("not data", encoding="utf-8")
    assert _check(repo)[0]


def test_unreadable_manifest_fails_cleanly(tmp_path: Path):
    bad = tmp_path / "manifest.json"
    bad.write_text("{not json", encoding="utf-8")
    ok, summary, messages = bm.check(bad, root=tmp_path)
    assert not ok and summary["datasets"] == 0
    assert messages[0].startswith("cannot read manifest")
    assert bm.main(["--manifest", str(bad)]) == 1
    assert bm.main(["--manifest", str(bad), "--update"]) == 1
