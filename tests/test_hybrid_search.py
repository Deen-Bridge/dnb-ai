"""Tests for hybrid search (#226) — RRF fusion, query analysis, A/B buckets, filters.

All offline: the in-memory backends need no network and no GEMINI_API_KEY.
Fused scores are asserted against hand-computed values (e.g. 1/(60 + rank)).
"""

import math
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from hybrid_search import (
    BUCKET_COUNT,
    CHANNEL_KEYWORD,
    CHANNEL_SEMANTIC,
    DEFAULT_RRF_K,
    HashingVectorBackend,
    HybridSearchRequest,
    HybridSearcher,
    InMemoryKeywordBackend,
    ScoredPassage,
    STRATEGY_REGISTRY,
    analyze_query,
    assign_variant,
    get_ab_stats,
    passage_from_record,
    reciprocal_rank_fusion,
    reset_ab_stats,
    weighted_reciprocal_rank_fusion,
)


def mk(
    pid: str,
    text: str = "",
    source: str = "quran",
    reference: str | None = None,
    score: float = 0.0,
) -> ScoredPassage:
    return ScoredPassage(id=pid, text=text, source=source, reference=reference, score=score)


class StubBackend:
    """Fixed-ranking stand-in satisfying both backend Protocols."""

    def __init__(self, results: list[ScoredPassage]):
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5) -> list[ScoredPassage]:
        self.calls.append((query, k))
        return self.results[:k]


@pytest.fixture(autouse=True)
def _reset_ab_state():
    reset_ab_stats()
    yield
    reset_ab_stats()


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion — hand-computed scores
# ---------------------------------------------------------------------------
class TestReciprocalRankFusion:
    def test_plain_rrf_scores_and_order(self):
        """rankings semantic=[a,b], keyword=[b,c], k=60 (ranks are 1-based):

        b: 1/61 + 1/62   a: 1/61   c: 1/62  ->  order [b, a, c]
        """
        fused = reciprocal_rank_fusion(
            {CHANNEL_SEMANTIC: [mk("a"), mk("b")], CHANNEL_KEYWORD: [mk("b"), mk("c")]},
            k=60,
        )
        by_id = {item.passage.id: item for item in fused}
        assert [item.passage.id for item in fused] == ["b", "a", "c"]
        assert by_id["b"].fused_score == pytest.approx(1 / 61 + 1 / 62)
        assert by_id["a"].fused_score == pytest.approx(1 / 61)
        assert by_id["c"].fused_score == pytest.approx(1 / 62)

    def test_default_k_is_60(self):
        fused = reciprocal_rank_fusion({CHANNEL_SEMANTIC: [mk("x")]})
        assert fused[0].fused_score == pytest.approx(1 / (DEFAULT_RRF_K + 1))

    def test_custom_k(self):
        """k=1: rank 1 contributes 1/2, rank 2 contributes 1/3.

        b sits at rank 2 in semantic and rank 1 in keyword -> 1/3 + 1/2 = 5/6.
        """
        fused = reciprocal_rank_fusion(
            {CHANNEL_SEMANTIC: [mk("a"), mk("b")], CHANNEL_KEYWORD: [mk("b"), mk("c")]},
            k=1,
        )
        by_id = {item.passage.id: item for item in fused}
        assert by_id["b"].fused_score == pytest.approx(5 / 6)
        assert by_id["a"].fused_score == pytest.approx(0.5)
        assert by_id["c"].fused_score == pytest.approx(1 / 3)

    def test_single_channel_preserves_rank_order(self):
        fused = reciprocal_rank_fusion({CHANNEL_SEMANTIC: [mk("p"), mk("q"), mk("r")]})
        assert [item.passage.id for item in fused] == ["p", "q", "r"]
        assert [item.fused_score for item in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])

    def test_tie_breaks_deterministically_by_id(self):
        """x and y swap channels so both score 1/61 + 1/62; id order decides."""
        fused = reciprocal_rank_fusion(
            {CHANNEL_SEMANTIC: [mk("x"), mk("y")], CHANNEL_KEYWORD: [mk("y"), mk("x")]}
        )
        assert [item.fused_score for item in fused] == pytest.approx([1 / 61 + 1 / 62] * 2)
        assert [item.passage.id for item in fused] == ["x", "y"]

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion({CHANNEL_SEMANTIC: [mk("a")]}, k=0)

    def test_match_type_from_channel_overlap(self):
        fused = {
            item.passage.id: item
            for item in reciprocal_rank_fusion(
                {
                    CHANNEL_SEMANTIC: [mk("d1"), mk("d2")],
                    CHANNEL_KEYWORD: [mk("d2"), mk("d3")],
                }
            )
        }
        assert fused["d1"].match_type == "semantic"
        assert fused["d3"].match_type == "keyword"
        assert fused["d2"].match_type == "both"

    def test_per_channel_ranks_and_scores_recorded(self):
        sem_hits = [mk("d1", score=0.9), mk("d2", score=0.8)]
        kw_hits = [mk("d2", score=3.25)]
        fused = {
            item.passage.id: item
            for item in weighted_reciprocal_rank_fusion(
                {CHANNEL_SEMANTIC: sem_hits, CHANNEL_KEYWORD: kw_hits},
                {CHANNEL_SEMANTIC: 1.0, CHANNEL_KEYWORD: 1.0},
            )
        }
        assert fused["d1"].channel_ranks == {CHANNEL_SEMANTIC: 1}
        assert fused["d1"].channel_scores == pytest.approx({CHANNEL_SEMANTIC: 0.9})
        assert fused["d2"].channel_ranks == {CHANNEL_SEMANTIC: 2, CHANNEL_KEYWORD: 1}
        assert fused["d2"].channel_scores[CHANNEL_KEYWORD] == pytest.approx(3.25)


# ---------------------------------------------------------------------------
# Weighted fusion
# ---------------------------------------------------------------------------
class TestWeightedFusion:
    def test_weighted_scores_hand_computed(self):
        """weights keyword=2 doubles the lexical channel's contribution:

        a: 1*(1/61)          b: 1*(1/62) + 2*(1/61)   c: 2*(1/62)
        """
        fused = weighted_reciprocal_rank_fusion(
            {CHANNEL_SEMANTIC: [mk("a"), mk("b")], CHANNEL_KEYWORD: [mk("b"), mk("c")]},
            {CHANNEL_SEMANTIC: 1.0, CHANNEL_KEYWORD: 2.0},
            k=60,
        )
        by_id = {item.passage.id: item for item in fused}
        assert [item.passage.id for item in fused] == ["b", "c", "a"]
        assert by_id["b"].fused_score == pytest.approx(1 / 62 + 2 / 61)
        assert by_id["c"].fused_score == pytest.approx(2 / 62)
        assert by_id["a"].fused_score == pytest.approx(1 / 61)

    def test_zero_weight_channel_contributes_nothing(self):
        fused = weighted_reciprocal_rank_fusion(
            {CHANNEL_SEMANTIC: [mk("a"), mk("b")], CHANNEL_KEYWORD: [mk("c"), mk("d")]},
            {CHANNEL_SEMANTIC: 1.0, CHANNEL_KEYWORD: 0.0},
            k=60,
        )
        assert [item.passage.id for item in fused] == ["a", "b", "c", "d"]
        scores = {item.passage.id: item.fused_score for item in fused}
        assert scores["c"] == 0.0 and scores["d"] == 0.0

    def test_unweighted_channel_defaults_to_one(self):
        fused = weighted_reciprocal_rank_fusion(
            {CHANNEL_SEMANTIC: [mk("a")], CHANNEL_KEYWORD: [mk("b")]},
            {CHANNEL_SEMANTIC: 2.0},
            k=60,
        )
        scores = {item.passage.id: item.fused_score for item in fused}
        assert scores["a"] == pytest.approx(2 / 61)
        assert scores["b"] == pytest.approx(1 / 61)


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------
class TestAnalyzeQuery:
    @pytest.mark.parametrize(
        ("mode", "semantic", "keyword"),
        [("keyword", 0.25, 0.75), ("semantic", 0.75, 0.25), ("balanced", 0.5, 0.5)],
    )
    def test_mode_weights(self, mode, semantic, keyword):
        analysis = analyze_query("anything", override=mode)
        assert analysis.mode == mode
        assert analysis.weights[CHANNEL_SEMANTIC] == pytest.approx(semantic)
        assert analysis.weights[CHANNEL_KEYWORD] == pytest.approx(keyword)
        assert sum(analysis.weights.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "query",
        [
            "What is the meaning of 2:255?",
            "tafsir of 112:1",
            "benefits of surah al-fatiha",
            "explain ayat al-kursi",
            "hadith about cleanliness",
            "is this narration sahih bukhari",
            '"verily in the remembrance of Allah"',
        ],
    )
    def test_citation_markers_lean_keyword(self, query):
        analysis = analyze_query(query)
        assert analysis.mode == "keyword"
        assert analysis.signals
        assert "override" not in analysis.signals

    @pytest.mark.parametrize(
        "query",
        [
            "what does Islam say about patience",
            "guidance on dealing with anger",
            "what is the meaning of life in islam",
            "how do I increase my iman",
            "spiritual benefits of gratitude",
        ],
    )
    def test_thematic_phrasing_leans_semantic(self, query):
        analysis = analyze_query(query)
        assert analysis.mode == "semantic"
        assert analysis.weights[CHANNEL_SEMANTIC] > analysis.weights[CHANNEL_KEYWORD]

    def test_plain_topic_stays_balanced(self):
        analysis = analyze_query("charity")
        assert analysis.mode == "balanced"
        assert analysis.signals == ["no_strong_signals"]

    def test_keyword_markers_beat_thematic_phrasing(self):
        analysis = analyze_query("what does islam say about hadith 2:255")
        assert analysis.mode == "keyword"

    def test_override_wins_over_markers(self):
        analysis = analyze_query("hadith about patience", override="semantic")
        assert analysis.mode == "semantic"
        assert analysis.signals == ["override"]

    def test_invalid_override_raises(self):
        with pytest.raises(ValueError):
            analyze_query("patience", override="aggressive")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Backends (offline implementations)
# ---------------------------------------------------------------------------
CORPUS = [
    ScoredPassage(
        id="quran:2:153", text="seek help through patience and prayer", source="quran", reference="2:153"
    ),
    ScoredPassage(
        id="quran:2:255",
        text="Allah there is no deity except Him the Ever-Living",
        source="quran",
        reference="2:255",
    ),
    ScoredPassage(
        id="hadith:42", text="patience is at the first stroke of calamity", source="hadith", reference="Bukhari 42"
    ),
]


class TestKeywordBackend:
    def test_bm25_ranks_lexical_match_first(self):
        backend = InMemoryKeywordBackend(CORPUS)
        hits = backend.search("patience", k=3)
        assert hits[0].id == "quran:2:153"
        assert all(hit.score > 0 for hit in hits)

    def test_unknown_term_returns_empty(self):
        backend = InMemoryKeywordBackend(CORPUS)
        assert backend.search("photosynthesis") == []

    def test_respects_k(self):
        backend = InMemoryKeywordBackend(CORPUS)
        assert len(backend.search("patience", k=1)) == 1

    def test_empty_corpus_is_safe(self):
        assert InMemoryKeywordBackend([]).search("anything") == []


class TestVectorBackend:
    def test_shared_token_scores_highest(self):
        backend = HashingVectorBackend(CORPUS)
        hits = backend.search("patience and prayer", k=3)
        assert hits[0].id == "quran:2:153"
        assert all(hit.score > 0 for hit in hits)

    def test_deterministic_across_instances(self):
        first = HashingVectorBackend(CORPUS).search("patience", k=3)
        second = HashingVectorBackend(CORPUS).search("patience", k=3)
        assert [(h.id, h.score) for h in first] == [(h.id, h.score) for h in second]

    def test_embeddings_are_normalized(self):
        vec = HashingVectorBackend([]).embed("patience is bitter but its fruit is sweet")
        assert math.isclose(float((vec * vec).sum()) ** 0.5, 1.0, rel_tol=1e-5)

    def test_empty_corpus_is_safe(self):
        assert HashingVectorBackend([]).search("anything") == []

    def test_passage_from_record_routes_extra_keys_to_metadata(self):
        passage = passage_from_record({"id": "t1", "text": "hello", "source": "test", "collection": "bukhari"})
        assert passage.metadata == {"collection": "bukhari"}


# ---------------------------------------------------------------------------
# End-to-end service behavior
# ---------------------------------------------------------------------------
class TestHybridSearcherExplanations:
    def make_service(self, sem_results, kw_results):
        return HybridSearcher(
            StubBackend(sem_results),
            StubBackend(kw_results),
            rrf_k=60,
            top_k=5,
        )

    def test_match_types_end_to_end(self):
        service = self.make_service(
            [mk("d1"), mk("d2")],
            [mk("d2"), mk("d3")],
        )
        response = service.search("overlap please", user_id="u1")
        types = {hit.id: hit.match_type for hit in response.results}
        assert types == {"d1": "semantic", "d2": "both", "d3": "keyword"}

    def test_hit_carries_channel_provenance(self):
        service = self.make_service([mk("d1")], [mk("d2")])
        response = service.search("anything")
        hit = next(h for h in response.results if h.id == "d1")
        assert hit.channel_ranks == {CHANNEL_SEMANTIC: 1}
        assert set(response.channels) == {CHANNEL_SEMANTIC, CHANNEL_KEYWORD}
        assert response.rrf_k == 60

    def test_response_fields_present(self):
        service = self.make_service([mk("d1")], [])
        response = service.search("patience", user_id="u1")
        assert response.strategy in STRATEGY_REGISTRY
        assert 0 <= response.ab_bucket < BUCKET_COUNT
        assert response.analysis.mode in {"keyword", "semantic", "balanced"}
        assert response.latency_ms >= 0.0
        assert response.query == "patience"

    def test_channel_toggles(self):
        service = HybridSearcher(
            StubBackend([mk("d1")]),
            StubBackend([mk("d2")]),
            enable_semantic=False,
        )
        response = service.search("keyword only")
        assert response.channels == [CHANNEL_KEYWORD]
        assert all(hit.match_type == "keyword" for hit in response.results)

    def test_both_channels_disabled_yields_empty(self):
        service = HybridSearcher(StubBackend([]), StubBackend([]), enable_semantic=False, enable_keyword=False)
        response = service.search("anything")
        assert response.results == []
        assert response.channels == []

    def test_truncates_to_requested_k(self):
        service = self.make_service([mk(f"d{i}") for i in range(10)], [])
        assert len(service.search("query", k=3).results) == 3


class TestABAssignment:
    def test_same_input_same_bucket(self):
        first = assign_variant("exp", "patience in islam", "user-7")
        second = assign_variant("exp", "patience in islam", "user-7")
        third = assign_variant("exp", "patience in islam", "user-7")
        assert first == second == third

    def test_bucket_in_range_and_variant_registered(self):
        variant, bucket = assign_variant("exp", "some query", None)
        assert 0 <= bucket < BUCKET_COUNT
        assert variant in STRATEGY_REGISTRY

    def test_query_normalization_is_stable(self):
        first = assign_variant("exp", "Patience   IN Islam", "u")[0]
        second = assign_variant("exp", "patience in islam", "u")[0]
        assert first == second

    def test_search_records_counter(self):
        service = HybridSearcher(StubBackend([mk("d1")]), StubBackend([mk("d2")]))
        response = service.search("count me", user_id="u9")
        stats = get_ab_stats()
        assert stats.get(response.strategy) == 1

    def test_strategies_differ_in_weight_usage(self):
        assert STRATEGY_REGISTRY["rrf_weighted_v1"].uses_weights is True
        assert STRATEGY_REGISTRY["rrf_plain_v1"].uses_weights is False

    def test_plain_strategy_ignores_analysis_weights(self):
        """Control variant: a keyword-only doc scores 1/(k+rank), not weight-scaled."""
        with patch("hybrid_search.assign_variant", return_value=("rrf_plain_v1", 7)):
            service = HybridSearcher(StubBackend([]), StubBackend([mk("b")]), rrf_k=60)
            response = service.search("q")
        assert response.strategy == "rrf_plain_v1"
        assert response.results[0].fused_score == pytest.approx(1 / 61)

    def test_weighted_strategy_applies_analysis_weights(self):
        """Treatment variant with semantic-leaning analysis: 0.75 vs 0.25 scaling."""
        with patch("hybrid_search.assign_variant", return_value=("rrf_weighted_v1", 8)):
            service = HybridSearcher(StubBackend([mk("a")]), StubBackend([mk("b")]), rrf_k=60)
            response = service.search("q", mode_override="semantic")
        scores = {hit.id: hit.fused_score for hit in response.results}
        assert response.strategy == "rrf_weighted_v1"
        assert scores["a"] == pytest.approx(0.75 / 61)
        assert scores["b"] == pytest.approx(0.25 / 61)


# ---------------------------------------------------------------------------
# Filters (post-fusion metadata predicates)
# ---------------------------------------------------------------------------
class TestFilters:
    def make_mixed_service(self):
        return HybridSearcher(
            StubBackend(
                [
                    mk("v1", source="quran", reference="2:255"),
                    mk("v2", source="tafsir", reference="2:255-t"),
                ]
            ),
            StubBackend(
                [
                    mk("k1", source="hadith", reference="Bukhari 42"),
                    mk("k2", source="quran", reference="3:190"),
                ]
            ),
        )

    def test_scalar_filter(self):
        response = self.make_mixed_service().search("mixed", filters={"source": "hadith"})
        assert [hit.id for hit in response.results] == ["k1"]

    def test_list_filter_membership(self):
        response = self.make_mixed_service().search("mixed", filters={"source": ["quran", "hadith"]})
        assert [hit.id for hit in response.results] == ["k1", "v1", "k2"]

    def test_reference_field_filter(self):
        response = self.make_mixed_service().search("mixed", filters={"reference": "2:255"})
        assert [hit.id for hit in response.results] == ["v1"]

    def test_filter_removing_everything(self):
        response = self.make_mixed_service().search("mixed", filters={"source": "torah"})
        assert response.results == []

    def test_filters_apply_after_fusion_order(self):
        """Survivors keep their fused order; fusion, not the filter, ranks them.

        Fused scores: drop 1/62+1/61 > noise 1/61 == top 1/61 (tie -> id) >
        mid 1/62. Filtering to source=tafsir leaves [top, mid].
        """
        service = HybridSearcher(
            StubBackend([mk("top", source="tafsir"), mk("drop", source="hadith")]),
            StubBackend([mk("noise", source="hadith"), mk("mid", source="tafsir")]),
        )
        response = service.search("ordered", filters={"source": "tafsir"})
        assert [hit.id for hit in response.results] == ["top", "mid"]


# ---------------------------------------------------------------------------
# Request model validation
# ---------------------------------------------------------------------------
class TestRequestModel:
    def test_minimal_request(self):
        request = HybridSearchRequest(query="patience")
        assert request.k is None and request.mode_override is None and request.filters is None

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="")

    def test_bad_override_rejected(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="patience", mode_override="aggressive")

    def test_k_bounds_enforced(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="patience", k=0)
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="patience", k=51)

    def test_full_payload(self):
        request = HybridSearchRequest(
            query="2:255 tafsir",
            k=10,
            mode_override="balanced",
            user_id="u1",
            filters={"source": ["quran", "hadith"]},
        )
        assert request.k == 10 and request.filters == {"source": ["quran", "hadith"]}
