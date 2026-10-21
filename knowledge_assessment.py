"""Adaptive Learner Knowledge Level Assessment Subsystem (#250).

Evaluates a learner's proficiency across core Islamic disciplines (Aqeedah,
Fiqh, Hadith, Tafsir, Seerah), provides an initial diagnostic questionnaire,
assigns domain and overall proficiency levels (beginner, intermediate,
advanced), tracks longitudinal knowledge growth, produces a visualizable
knowledge map, and recommends targeted study actions for identified gaps.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-assessment", tags=["knowledge-assessment"])


class IslamicDiscipline(str, Enum):
    aqeedah = "aqeedah"
    fiqh = "fiqh"
    hadith = "hadith"
    tafsir = "tafsir"
    seerah = "seerah"


class ProficiencyLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class DiagnosticQuestion(BaseModel):
    question_id: str
    discipline: IslamicDiscipline
    difficulty: int = Field(..., ge=1, le=3)
    prompt: str
    options: list[str]
    correct_index: int


DIAGNOSTIC_BANK: list[DiagnosticQuestion] = [
    DiagnosticQuestion(
        question_id="aq-1",
        discipline=IslamicDiscipline.aqeedah,
        difficulty=1,
        prompt="How many pillars of Iman (faith) are mentioned in the Hadith of Jibril?",
        options=["Five", "Six", "Seven", "Four"],
        correct_index=1,
    ),
    DiagnosticQuestion(
        question_id="aq-2",
        discipline=IslamicDiscipline.aqeedah,
        difficulty=3,
        prompt="Which category of Tawhid specifically concerns directing all acts of worship solely to Allah?",
        options=["Tawhid al-Rububiyyah", "Tawhid al-Uluhiyyah", "Tawhid al-Asma wa al-Sifat", "Tawhid al-Hakimiyyah"],
        correct_index=1,
    ),
    DiagnosticQuestion(
        question_id="fq-1",
        discipline=IslamicDiscipline.fiqh,
        difficulty=1,
        prompt="Which of the following invalidates wudu across all four Sunni madhhabs?",
        options=["Smiling during prayer", "Deep sleep lying down", "Drinking water", "Trimming nails"],
        correct_index=1,
    ),
    DiagnosticQuestion(
        question_id="fq-2",
        discipline=IslamicDiscipline.fiqh,
        difficulty=3,
        prompt="What is the term for legal reasoning by analogy in Usul al-Fiqh?",
        options=["Ijma", "Qiyas", "Istihsan", "Urf"],
        correct_index=1,
    ),
    DiagnosticQuestion(
        question_id="hd-1",
        discipline=IslamicDiscipline.hadith,
        difficulty=2,
        prompt="What is a hadith called when its chain of narrators is continuous with trustworthy, precise narrators free of shudhudh and 'illah?",
        options=["Hasan", "Sahih", "Mursal", "Mutawatir"],
        correct_index=1,
    ),
    DiagnosticQuestion(
        question_id="tf-1",
        discipline=IslamicDiscipline.tafsir,
        difficulty=2,
        prompt="What is the scholarly term for the historical context or occasion of revelation of a Quranic verse?",
        options=["Nasikh", "Asbab al-Nuzul", "Muhkam", "Gharib al-Quran"],
        correct_index=1,
    ),
    DiagnosticQuestion(
        question_id="sr-1",
        discipline=IslamicDiscipline.seerah,
        difficulty=1,
        prompt="In which year of Prophethood did the Hijrah from Makkah to Madinah take place?",
        options=["10th year", "13th year", "15th year", "8th year"],
        correct_index=1,
    ),
]


class AssessmentResponseItem(BaseModel):
    discipline: IslamicDiscipline
    difficulty: int = Field(default=2, ge=1, le=3)
    is_correct: bool


class HistoricalSnapshot(BaseModel):
    timestamp_iso: str
    overall_score: float = Field(..., ge=0.0, le=100.0)


class KnowledgeAssessmentRequest(BaseModel):
    learner_id: str = Field(..., min_length=1, max_length=120)
    responses: list[AssessmentResponseItem] = Field(..., min_length=1, max_length=200)
    previous_snapshots: list[HistoricalSnapshot] = Field(default_factory=list, max_length=50)


class DisciplineEvaluation(BaseModel):
    discipline: IslamicDiscipline
    weighted_score: float
    proficiency: ProficiencyLevel
    questions_answered: int
    strengths: list[str]
    gaps: list[str]


class KnowledgeAssessmentResult(BaseModel):
    learner_id: str
    overall_score: float
    overall_proficiency: ProficiencyLevel
    discipline_evaluations: list[DisciplineEvaluation]
    knowledge_growth_delta: float
    knowledge_map: dict[str, Any]
    recommendations: list[str]


def _score_to_level(score: float) -> ProficiencyLevel:
    if score >= 78.0:
        return ProficiencyLevel.advanced
    if score >= 48.0:
        return ProficiencyLevel.intermediate
    return ProficiencyLevel.beginner


def evaluate_knowledge_assessment(req: KnowledgeAssessmentRequest) -> KnowledgeAssessmentResult:
    by_disc: dict[IslamicDiscipline, list[AssessmentResponseItem]] = {}
    for item in req.responses:
        by_disc.setdefault(item.discipline, []).append(item)

    evaluations: list[DisciplineEvaluation] = []
    for disc in IslamicDiscipline:
        items = by_disc.get(disc, [])
        if not items:
            evaluations.append(
                DisciplineEvaluation(
                    discipline=disc,
                    weighted_score=0.0,
                    proficiency=ProficiencyLevel.beginner,
                    questions_answered=0,
                    strengths=[],
                    gaps=[f"No assessment items completed for {disc.value} yet."],
                )
            )
            continue
        earned = sum(i.difficulty for i in items if i.is_correct)
        possible = sum(i.difficulty for i in items)
        score = round((earned / possible) * 100.0, 2) if possible else 0.0
        level = _score_to_level(score)
        strengths = [f"Solid foundation in {disc.value} ({level.value})"] if score >= 60.0 else []
        gaps = [f"Needs foundational reinforcement in {disc.value}"] if score < 60.0 else []
        evaluations.append(
            DisciplineEvaluation(
                discipline=disc,
                weighted_score=score,
                proficiency=level,
                questions_answered=len(items),
                strengths=strengths,
                gaps=gaps,
            )
        )

    active_scores = [e.weighted_score for e in evaluations if e.questions_answered > 0]
    overall_score = round(sum(active_scores) / len(active_scores), 2) if active_scores else 0.0
    overall_level = _score_to_level(overall_score)

    if req.previous_snapshots:
        growth_delta = round(overall_score - req.previous_snapshots[-1].overall_score, 2)
    else:
        growth_delta = 0.0

    knowledge_map = {
        "radar_axes": [
            {"discipline": e.discipline.value, "score": e.weighted_score, "level": e.proficiency.value}
            for e in evaluations
        ],
        "overall_proficiency": overall_level.value,
    }

    recommendations: list[str] = []
    for e in evaluations:
        if e.questions_answered == 0:
            recommendations.append(f"Complete the initial diagnostic module for {e.discipline.value}.")
        elif e.proficiency == ProficiencyLevel.beginner:
            recommendations.append(f"Start with introductory {e.discipline.value} primers before comparative texts.")
        elif e.proficiency == ProficiencyLevel.intermediate:
            recommendations.append(f"Advance to intermediate {e.discipline.value} case studies and primary texts.")

    if not recommendations:
        recommendations.append("All disciplines at advanced proficiency; proceed to specialized scholarly tracks.")

    return KnowledgeAssessmentResult(
        learner_id=req.learner_id,
        overall_score=overall_score,
        overall_proficiency=overall_level,
        discipline_evaluations=evaluations,
        knowledge_growth_delta=growth_delta,
        knowledge_map=knowledge_map,
        recommendations=recommendations,
    )


@router.get("/questionnaire", response_model=list[DiagnosticQuestion])
async def get_questionnaire(discipline: IslamicDiscipline | None = None) -> list[DiagnosticQuestion]:
    """Return initial diagnostic assessment questions, optionally filtered by discipline."""
    if discipline is None:
        return DIAGNOSTIC_BANK
    return [q for q in DIAGNOSTIC_BANK if q.discipline == discipline]


@router.post("/evaluate", response_model=KnowledgeAssessmentResult)
async def evaluate_endpoint(body: KnowledgeAssessmentRequest) -> KnowledgeAssessmentResult:
    """Evaluate a learner's discipline-level and overall Islamic knowledge proficiency."""
    return evaluate_knowledge_assessment(body)
