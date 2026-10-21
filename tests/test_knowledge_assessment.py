"""Tests for the Adaptive Learner Knowledge Level Assessment subsystem (#250)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from knowledge_assessment import (
    AssessmentResponseItem,
    HistoricalSnapshot,
    IslamicDiscipline,
    KnowledgeAssessmentRequest,
    ProficiencyLevel,
    evaluate_knowledge_assessment,
)
from main import app

client = TestClient(app)


def test_evaluate_knowledge_assessment_levels_and_growth() -> None:
    req = KnowledgeAssessmentRequest(
        learner_id="user-101",
        responses=[
            AssessmentResponseItem(discipline=IslamicDiscipline.aqeedah, difficulty=3, is_correct=True),
            AssessmentResponseItem(discipline=IslamicDiscipline.aqeedah, difficulty=2, is_correct=True),
            AssessmentResponseItem(discipline=IslamicDiscipline.fiqh, difficulty=2, is_correct=True),
            AssessmentResponseItem(discipline=IslamicDiscipline.fiqh, difficulty=2, is_correct=False),
            AssessmentResponseItem(discipline=IslamicDiscipline.hadith, difficulty=3, is_correct=False),
        ],
        previous_snapshots=[HistoricalSnapshot(timestamp_iso="2026-09-01T00:00:00Z", overall_score=40.0)],
    )
    res = evaluate_knowledge_assessment(req)
    assert res.overall_score == 50.0
    assert res.overall_proficiency == ProficiencyLevel.intermediate
    assert res.knowledge_growth_delta == 10.0

    by_disc = {e.discipline: e for e in res.discipline_evaluations}
    assert by_disc[IslamicDiscipline.aqeedah].proficiency == ProficiencyLevel.advanced
    assert by_disc[IslamicDiscipline.fiqh].proficiency == ProficiencyLevel.intermediate
    assert by_disc[IslamicDiscipline.hadith].proficiency == ProficiencyLevel.beginner
    assert len(res.knowledge_map["radar_axes"]) == 5


def test_questionnaire_and_evaluate_endpoints() -> None:
    q_resp = client.get("/knowledge-assessment/questionnaire", params={"discipline": "fiqh"})
    assert q_resp.status_code == 200
    items = q_resp.json()
    assert items
    assert all(i["discipline"] == "fiqh" for i in items)

    ev_resp = client.post(
        "/knowledge-assessment/evaluate",
        json={
            "learner_id": "user-101",
            "responses": [{"discipline": "aqeedah", "difficulty": 3, "is_correct": True}],
        },
    )
    assert ev_resp.status_code == 200
    assert ev_resp.json()["overall_proficiency"] == "advanced"
