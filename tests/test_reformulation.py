"""Tests for the question reformulation engine — no live API calls."""

import pytest
from fastapi.testclient import TestClient

from reformulation import (
    EXAMPLE_LIBRARY,
    ReformulationOption,
    assess_quality,
    detect_category,
    router,
    suggest_reformulations,
)


@pytest.fixture
def client() -> TestClient:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------


def test_vague_short_question_flagged_low_quality() -> None:
    assessment = assess_quality("wudu?")
    assert assessment.score < 0.7
    assert not assessment.is_well_formed
    codes = {issue.code for issue in assessment.issues}
    assert "too_short" in codes


def test_well_formed_question_scores_high() -> None:
    assessment = assess_quality(
        "What is the ruling on combining prayers while traveling according to the Hanafi school?"
    )
    assert assessment.score >= 0.7
    assert assessment.is_well_formed


# ---------------------------------------------------------------------------
# Reformulation strategies
# ---------------------------------------------------------------------------


def test_fiqh_question_gets_specify_madhhab_suggestion() -> None:
    options = suggest_reformulations("Is it permissible to pray while sitting?")
    strategies = {opt.strategy for opt in options}
    assert "specify_madhhab" in strategies
    madhhab_option = next(o for o in options if o.strategy == "specify_madhhab")
    assert "school" in madhhab_option.text.lower()


def test_fiqh_question_uses_provided_madhhab() -> None:
    options = suggest_reformulations("Is it permissible to pray while sitting?", madhhab="shafii")
    madhhab_option = next(o for o in options if o.strategy == "specify_madhhab")
    assert "Shafi'i" in madhhab_option.text


def test_compound_question_gets_split_into_parts_suggestion() -> None:
    options = suggest_reformulations("What is the ruling on music and how do I calculate zakat on savings?")
    strategies = {opt.strategy for opt in options}
    assert "split_compound" in strategies
    split_option = next(o for o in options if o.strategy == "split_compound")
    # The split option enumerates more than one part.
    assert "(1)" in split_option.text and "(2)" in split_option.text


def test_multiple_options_each_with_an_explanation() -> None:
    options = suggest_reformulations("it?")
    assert len(options) >= 2
    assert all(isinstance(o, ReformulationOption) for o in options)
    assert all(o.explanation.strip() for o in options)
    # Ranks are contiguous starting at 1.
    assert [o.rank for o in options] == list(range(1, len(options) + 1))


def test_category_detection() -> None:
    assert detect_category("What is the tafsir of Ayat al-Kursi in Surah Al-Baqarah?") == "tafsir"
    assert detect_category("Is the hadith in Sahih Bukhari authentic?") == "hadith"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_suggest_endpoint_returns_assessment_and_options(client: TestClient) -> None:
    resp = client.post("/reformulation/suggest", json={"question": "prayer?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "assessment" in body
    assert body["assessment"]["score"] < 1.0
    assert len(body["options"]) >= 2
    assert all(opt["explanation"] for opt in body["options"])


def test_examples_endpoint_filters_by_category(client: TestClient) -> None:
    resp = client.get("/reformulation/examples", params={"category": "fiqh"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "fiqh"
    assert body["count"] == len(EXAMPLE_LIBRARY["fiqh"])
    assert all(item["category"] == "fiqh" for item in body["examples"])


def test_examples_endpoint_returns_all_without_category(client: TestClient) -> None:
    resp = client.get("/reformulation/examples")
    assert resp.status_code == 200
    body = resp.json()
    expected_total = sum(len(v) for v in EXAMPLE_LIBRARY.values())
    assert body["count"] == expected_total
    assert body["category"] is None
