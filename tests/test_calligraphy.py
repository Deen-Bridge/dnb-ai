"""Tests for the Islamic calligraphy style estimator (#228).

Every test runs offline against supplied trait vectors — no images, no
GEMINI_API_KEY. Only the ``calligraphy`` module is imported, mounted on a bare
FastAPI app for the HTTP tests, so the suite needs none of the app's heavy
dependencies and never touches the network or genai.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from calligraphy import (
    AMBIGUITY_MARGIN,
    CALLIGRAPHY_STYLES,
    TRAIT_NAMES,
    StyleName,
    TraitVector,
    analyze,
    classify,
    router,
    score_styles,
)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# Representative trait vectors for a couple of textbook hands.
_KUFIC_TRAITS = {
    "angularity": 0.95,
    "curvature": 0.10,
    "stroke_contrast": 0.25,
    "diacritic_density": 0.15,
    "geometric_regularity": 0.92,
    "elongation": 0.30,
    "slant": 0.05,
    "letter_stacking": 0.20,
}

_NASKH_TRAITS = {
    "angularity": 0.18,
    "curvature": 0.82,
    "stroke_contrast": 0.45,
    "diacritic_density": 0.88,
    "geometric_regularity": 0.70,
    "elongation": 0.22,
    "slant": 0.10,
    "letter_stacking": 0.18,
}

_DIWANI_TRAITS = {
    "angularity": 0.15,
    "curvature": 0.90,
    "stroke_contrast": 0.60,
    "diacritic_density": 0.40,
    "geometric_regularity": 0.28,
    "elongation": 0.55,
    "slant": 0.88,
    "letter_stacking": 0.92,
}


# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------


def test_catalog_covers_named_styles() -> None:
    for name in (StyleName.KUFIC, StyleName.NASKH, StyleName.THULUTH, StyleName.DIWANI):
        assert name in CALLIGRAPHY_STYLES


def test_every_profile_has_full_ideal_vector() -> None:
    for profile in CALLIGRAPHY_STYLES.values():
        assert set(profile.ideal) == set(TRAIT_NAMES)
        for value in profile.ideal.values():
            assert 0.0 <= value <= 1.0
        assert profile.characteristics  # non-empty


# ---------------------------------------------------------------------------
# Deterministic classification of textbook hands
# ---------------------------------------------------------------------------


def test_kufic_traits_classify_as_kufic() -> None:
    result = classify(TraitVector(**_KUFIC_TRAITS))
    assert result.predicted_style == StyleName.KUFIC
    assert result.ambiguous is False


def test_naskh_traits_classify_as_naskh() -> None:
    result = classify(TraitVector(**_NASKH_TRAITS))
    assert result.predicted_style == StyleName.NASKH


def test_diwani_distinguished_from_naskh() -> None:
    # Strong slant + stacking should pick Diwani, not the rounded Naskh hand.
    diwani = classify(TraitVector(**_DIWANI_TRAITS))
    naskh = classify(TraitVector(**_NASKH_TRAITS))
    assert diwani.predicted_style == StyleName.DIWANI
    assert naskh.predicted_style == StyleName.NASKH
    assert diwani.predicted_style != naskh.predicted_style


def test_classification_is_deterministic() -> None:
    first = classify(TraitVector(**_KUFIC_TRAITS))
    second = classify(TraitVector(**_KUFIC_TRAITS))
    assert first.model_dump() == second.model_dump()


# ---------------------------------------------------------------------------
# Confidence normalization and ranking
# ---------------------------------------------------------------------------


def test_confidences_are_normalized_and_sorted() -> None:
    ranking = score_styles(TraitVector(**_KUFIC_TRAITS))
    assert len(ranking) == len(CALLIGRAPHY_STYLES)
    total = sum(s.confidence for s in ranking)
    assert abs(total - 1.0) < 1e-6
    confidences = [s.confidence for s in ranking]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_predicted_style_tops_the_ranking() -> None:
    result = classify(TraitVector(**_NASKH_TRAITS))
    assert result.ranking[0].style == result.predicted_style
    assert result.ranking[0].confidence == result.confidence


# ---------------------------------------------------------------------------
# Ambiguity flag
# ---------------------------------------------------------------------------


def test_neutral_vector_is_ambiguous() -> None:
    # An all-0.5 vector sits near no single hand, so the top two crowd together.
    result = classify(TraitVector())
    top_gap = result.ranking[0].confidence - result.ranking[1].confidence
    assert top_gap < AMBIGUITY_MARGIN
    assert result.ambiguous is True


def test_clear_specimen_is_not_ambiguous() -> None:
    result = classify(TraitVector(**_KUFIC_TRAITS))
    assert result.ambiguous is False


# ---------------------------------------------------------------------------
# Analysis: notes, bands, reconstruction hint
# ---------------------------------------------------------------------------


def test_analyze_reports_bands_and_hint() -> None:
    result = analyze(TraitVector(**_DIWANI_TRAITS))
    assert result.classification.predicted_style == StyleName.DIWANI
    assert result.legibility in {"high", "moderate", "low"}
    assert result.embellishment in {"high", "moderate", "low"}
    assert result.text_reconstruction_hint
    assert any("rule-based estimate" in note for note in result.notes)


def test_analyze_flags_high_embellishment() -> None:
    result = analyze(TraitVector(**_DIWANI_TRAITS))
    # Diwani specimen: high slant/stacking/contrast => high embellishment band.
    assert result.embellishment == "high"


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


def test_styles_endpoint_lists_catalog() -> None:
    resp = _client().get("/calligraphy/styles")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == len(CALLIGRAPHY_STYLES)
    names = {entry["name"] for entry in body}
    assert "kufic" in names and "naskh" in names


def test_style_detail_endpoint() -> None:
    resp = _client().get("/calligraphy/styles/thuluth")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "thuluth"
    assert body["characteristics"]
    assert set(body["ideal_traits"]) == set(TRAIT_NAMES)


def test_style_detail_unknown_returns_422() -> None:
    # An unknown style is rejected by the StyleName enum path constraint.
    resp = _client().get("/calligraphy/styles/nonexistent")
    assert resp.status_code == 422


def test_classify_endpoint_returns_ranking() -> None:
    resp = _client().post("/calligraphy/classify", json=_KUFIC_TRAITS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_style"] == "kufic"
    assert body["method"] == "heuristic-rule-based-estimate"
    assert len(body["ranking"]) == len(CALLIGRAPHY_STYLES)


def test_classify_endpoint_accepts_partial_traits() -> None:
    # Unspecified traits default to a neutral 0.5, so a partial body is valid.
    resp = _client().post("/calligraphy/classify", json={"angularity": 0.95, "geometric_regularity": 0.9})
    assert resp.status_code == 200


def test_classify_endpoint_rejects_out_of_range() -> None:
    resp = _client().post("/calligraphy/classify", json={"angularity": 1.5})
    assert resp.status_code == 422


def test_analyze_endpoint() -> None:
    resp = _client().post("/calligraphy/analyze", json=_NASKH_TRAITS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["classification"]["predicted_style"] == "naskh"
    assert "legibility" in body and "embellishment" in body
    assert body["text_reconstruction_hint"]
