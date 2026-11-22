"""Cross-reference graph consistency (#840).

Unit tests cover each validation rule, integration tests drive the FastAPI
routes against a temporary data file, and the evaluation test checks the
bundled graph that ships today. Everything is offline and needs no secrets.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import tafsir
from tafsir import (
    CROSS_REFERENCE_SCHEMA_VERSION,
    SEED_CROSS_REFERENCES,
    ReferenceType,
    build_cross_reference_graph,
    canonical_ayah_key,
    find_cross_references,
    get_cross_reference_graph,
    router,
    validate_cross_reference_graph,
)


def _edge(source: str = "2:282", target: str = "65:2", **overrides: Any) -> dict[str, Any]:
    edge: dict[str, Any] = {
        "source": source,
        "target": target,
        "reference_type": "related_ruling",
        "context": "Witnesses for binding agreements.",
        "commentary": "Linked by the classical mufassirun.",
        "source_work": "Tafsir al-Qurtubi",
    }
    edge.update(overrides)
    return edge


def _codes(issues: list[Any]) -> list[str]:
    return [issue.code for issue in issues]


@pytest.fixture
def data_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the served graph at a temporary file and reset the cache around the test."""
    path = tmp_path / "cross_references.json"
    monkeypatch.setattr(tafsir, "CROSS_REFERENCE_DATA_PATH", path)
    get_cross_reference_graph.cache_clear()
    yield path
    get_cross_reference_graph.cache_clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Ayah key canonicalisation (Arabic / English edge cases)
# ---------------------------------------------------------------------------


class TestCanonicalAyahKey:
    def test_plain_key(self) -> None:
        assert canonical_ayah_key("2:255") == ("2:255", None)

    @pytest.mark.parametrize(
        "raw",
        ["002:255", " 2 : 255 ", "٢:٢٥٥", "۲:۲۵۵"],  # leading zeros, spaces, Arabic-Indic, Persian digits
    )
    def test_variants_fold_to_ascii(self, raw: str) -> None:
        assert canonical_ayah_key(raw) == ("2:255", None)

    @pytest.mark.parametrize(
        "raw",
        ["", "2", "2:255:1", "1:1-7", "al-baqarah:255", "٢:", None, 2255],
    )
    def test_malformed_keys_rejected(self, raw: object) -> None:
        key, reason = canonical_ayah_key(raw)
        assert key is None and reason

    def test_out_of_range_surah_and_ayah(self) -> None:
        assert canonical_ayah_key("115:1")[0] is None
        assert canonical_ayah_key("0:1")[0] is None
        key, reason = canonical_ayah_key("2:287")
        assert key is None and reason is not None and "286" in reason
        assert canonical_ayah_key("2:0")[0] is None


# ---------------------------------------------------------------------------
# Edge validation rules
# ---------------------------------------------------------------------------


class TestValidateGraph:
    def test_clean_edge_is_accepted_without_issues(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge()])
        assert len(accepted) == 1 and issues == []

    def test_empty_graph_is_valid(self) -> None:
        assert validate_cross_reference_graph([]) == ([], [])

    def test_out_of_range_target_is_rejected(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge(target="2:300")])
        assert accepted == []
        assert _codes(issues) == ["invalid_target"]
        assert issues[0].severity == "error" and issues[0].edge == 0

    def test_range_source_is_rejected(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge(source="1:1-7")])
        assert accepted == [] and _codes(issues) == ["invalid_source"]

    def test_self_reference_is_rejected(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge(source="2:255", target="٢:٢٥٥")])
        assert accepted == []
        assert "self_reference" in _codes(issues)

    def test_unknown_reference_type_is_rejected(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge(reference_type="prophecy")])
        assert accepted == []
        assert _codes(issues) == ["invalid_record"]
        assert "reference_type" in issues[0].detail

    def test_non_object_and_missing_fields_are_rejected(self) -> None:
        accepted, issues = validate_cross_reference_graph(["2:255->3:2", {"source": "2:255"}])
        assert accepted == []
        assert _codes(issues) == ["invalid_record", "invalid_record"]

    def test_blank_context_is_rejected(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge(context="   ")])
        assert accepted == [] and _codes(issues) == ["missing_context"]

    def test_exact_duplicate_is_rejected(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge(), _edge()])
        assert len(accepted) == 1
        assert _codes(issues) == ["duplicate_edge"]
        assert issues[0].edge == 1 and "edge 0" in issues[0].detail

    def test_reverse_duplicate_is_rejected(self) -> None:
        # The index serves every edge from both endpoints, so B->A of the same
        # type would be returned twice for each ayah.
        accepted, issues = validate_cross_reference_graph([_edge(), _edge(source="65:2", target="2:282")])
        assert len(accepted) == 1 and _codes(issues) == ["duplicate_edge"]

    def test_duplicate_detected_after_canonicalisation(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge(), _edge(source="٢:٢٨٢", target="065:2")])
        assert len(accepted) == 1
        assert "duplicate_edge" in _codes(issues)

    def test_same_pair_different_type_is_kept_with_warning(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge(), _edge(reference_type="thematic_parallel")])
        assert len(accepted) == 2
        assert _codes(issues) == ["conflicting_type"]
        assert issues[0].severity == "warning"

    def test_non_canonical_keys_are_normalised_and_flagged(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge(source="٢:٢٨٢", target="065:002")])
        assert [(r.source, r.target) for r in accepted] == [("2:282", "65:2")]
        assert _codes(issues) == ["non_canonical_key", "non_canonical_key"]
        assert all(i.severity == "warning" for i in issues)

    def test_missing_provenance_and_commentary_are_warnings(self) -> None:
        accepted, issues = validate_cross_reference_graph([_edge(source_work=None, commentary="")])
        assert len(accepted) == 1
        assert sorted(_codes(issues)) == ["missing_commentary", "missing_provenance"]

    def test_arabic_text_in_context_is_preserved(self) -> None:
        context = "آية الكرسي — the Throne Verse"
        accepted, _ = validate_cross_reference_graph([_edge(context=context)])
        assert accepted[0].context == context

    def test_seed_data_is_fully_consistent(self) -> None:
        accepted, issues = validate_cross_reference_graph([r.model_dump() for r in SEED_CROSS_REFERENCES])
        assert issues == []
        assert len(accepted) == len(SEED_CROSS_REFERENCES)


# ---------------------------------------------------------------------------
# Loading, fallback and versioning
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_missing_file_uses_seed(self, tmp_path: Path) -> None:
        graph = build_cross_reference_graph(tmp_path / "absent.json")
        assert graph.report.data_source == "seed"
        assert graph.report.consistent is True
        assert graph.report.accepted_edges == len(SEED_CROSS_REFERENCES)

    def test_invalid_json_falls_back_to_seed(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        graph = build_cross_reference_graph(path)
        assert graph.report.data_source == "seed_fallback"
        assert graph.report.consistent is False
        assert _codes(graph.report.issues) == ["unreadable_file"]
        assert graph.report.accepted_edges == len(SEED_CROSS_REFERENCES)

    def test_non_list_json_falls_back_to_seed(self, tmp_path: Path) -> None:
        path = tmp_path / "object.json"
        path.write_text(json.dumps({"edges": []}), encoding="utf-8")
        graph = build_cross_reference_graph(path)
        assert graph.report.data_source == "seed_fallback"
        assert _codes(graph.report.issues) == ["not_a_list"]

    def test_one_bad_edge_does_not_discard_the_file(self, tmp_path: Path) -> None:
        # Previously a single invalid record threw away the whole file.
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps([_edge(), _edge(target="2:999")]), encoding="utf-8")
        graph = build_cross_reference_graph(path)
        assert graph.report.data_source == "file"
        assert graph.report.total_edges == 2
        assert graph.report.accepted_edges == 1
        assert graph.report.rejected_edges == 1
        assert graph.report.consistent is False

    def test_all_edges_invalid_falls_back_to_seed(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([_edge(target="2:999")]), encoding="utf-8")
        graph = build_cross_reference_graph(path)
        assert graph.report.data_source == "seed_fallback"
        assert "no_valid_edges" in _codes(graph.report.issues)
        assert len(graph.references) == len(SEED_CROSS_REFERENCES)

    def test_empty_list_serves_empty_graph(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")
        graph = build_cross_reference_graph(path)
        assert graph.report.data_source == "file"
        assert graph.references == [] and graph.report.consistent is True

    def test_data_version_is_deterministic_and_content_addressed(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps([_edge()]), encoding="utf-8")
        b.write_text(json.dumps([_edge(source="٢:٢٨٢")]), encoding="utf-8")  # same edge once canonicalised
        version_a = build_cross_reference_graph(a).report.data_version
        assert version_a == build_cross_reference_graph(a).report.data_version
        assert version_a == build_cross_reference_graph(b).report.data_version
        b.write_text(json.dumps([_edge(context="Changed context.")]), encoding="utf-8")
        assert build_cross_reference_graph(b).report.data_version != version_a

    def test_rejected_edges_are_logged_without_content(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps([_edge(), _edge(target="2:999", context="SENSITIVE-MARKER")]), encoding="utf-8")
        with caplog.at_level("WARNING", logger="tafsir"):
            build_cross_reference_graph(path)
        assert "invalid_target" in caplog.text
        assert "SENSITIVE-MARKER" not in caplog.text

    def test_lookup_uses_canonical_key(self, data_file: Path) -> None:
        data_file.write_text(json.dumps([_edge()]), encoding="utf-8")
        assert len(find_cross_references("2:282")) == 1
        assert len(find_cross_references("٢:٢٨٢")) == 1
        assert len(find_cross_references("65:2")) == 1  # served from both ends
        assert find_cross_references("not-a-key") == []


# ---------------------------------------------------------------------------
# HTTP integration
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_report_endpoint(self, client: TestClient, data_file: Path) -> None:
        data_file.write_text(json.dumps([_edge(), _edge(), _edge(target="2:999")]), encoding="utf-8")
        resp = client.get("/cross-reference-graph/report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["schema_version"] == CROSS_REFERENCE_SCHEMA_VERSION
        assert body["data_source"] == "file"
        assert body["consistent"] is False
        assert (body["total_edges"], body["accepted_edges"], body["rejected_edges"]) == (3, 1, 2)
        assert sorted(i["code"] for i in body["issues"]) == ["duplicate_edge", "invalid_target"]
        assert len(body["data_version"]) == 16

    def test_rejected_edges_are_not_served(self, client: TestClient, data_file: Path) -> None:
        data_file.write_text(json.dumps([_edge(), _edge(source="2:255", target="2:255")]), encoding="utf-8")
        assert client.get("/cross-references/2:255").json() == []
        network = client.get("/cross-reference-network").json()
        assert network["nodes"] == ["2:282", "65:2"]
        assert len(network["edges"]) == 1

    def test_arabic_indic_path_key_matches(self, client: TestClient, data_file: Path) -> None:
        data_file.write_text(json.dumps([_edge()]), encoding="utf-8")
        resp = client.get("/cross-references/٢:٢٨٢")
        assert resp.status_code == 200
        assert [r["target"] for r in resp.json()] == ["65:2"]

    def test_out_of_range_lookup_is_400(self, client: TestClient, data_file: Path) -> None:
        resp = client.get("/cross-references/2:300")
        assert resp.status_code == 400

    def test_reference_type_filter(self, client: TestClient, data_file: Path) -> None:
        data_file.write_text(
            json.dumps([_edge(), _edge(target="2:283", reference_type="thematic_parallel")]), encoding="utf-8"
        )
        resp = client.get("/cross-references/2:282", params={"reference_type": ReferenceType.RELATED_RULING.value})
        assert [r["target"] for r in resp.json()] == ["65:2"]


# ---------------------------------------------------------------------------
# Evaluation: the graph that actually ships
# ---------------------------------------------------------------------------


def test_bundled_graph_is_consistent() -> None:
    """The data deployed today must pass every rule with provenance on every edge."""
    get_cross_reference_graph.cache_clear()
    try:
        graph = get_cross_reference_graph()
        assert graph.report.consistent, graph.report.issues
        assert graph.report.issues == []
        assert graph.report.rejected_edges == 0
        assert all(ref.source_work for ref in graph.references)
        for ref in graph.references:
            assert canonical_ayah_key(ref.source) == (ref.source, None)
            assert canonical_ayah_key(ref.target) == (ref.target, None)
    finally:
        get_cross_reference_graph.cache_clear()
