"""Represent recognized fiqh differences across schools without fabricating universal consensus (#527)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MadhhabPosition:
    school: str
    ruling_summary: str
    source_reference: str


@dataclass(frozen=True)
class FiqhPluralityResponse:
    topic: str
    presents_single_universal_ruling: bool
    positions: list[MadhhabPosition]
    context_questions: list[str]
    scholar_referral_notice: str
    educational_boundary_note: str


def format_fiqh_plurality_response(
    topic: str,
    positions: list[MadhhabPosition],
    high_impact_personal_case: bool = False,
) -> FiqhPluralityResponse:
    """Present sourced multi-school views and explicit educational vs fatwa boundaries."""
    valid_positions = [p for p in positions if p.school.strip() and p.source_reference.strip()]
    context_qs = [
        "Which madhhab or local scholarly tradition do you follow?",
        "Are there specific contractual or personal circumstances involved?",
    ]
    referral = (
        "For personal, marriage, inheritance, or financial contracts, consult a qualified local scholar or mufti."
        if high_impact_personal_case or len(valid_positions) != 1
        else "Consult a qualified scholar for personal application."
    )
    boundary = (
        "This summary provides comparative educational information and is not a binding personal fatwa."
        if valid_positions
        else "Insufficient sourced evidence is available in the corpus; no consensus or ruling is asserted."
    )
    return FiqhPluralityResponse(
        topic=topic,
        presents_single_universal_ruling=False,
        positions=valid_positions,
        context_questions=context_qs,
        scholar_referral_notice=referral,
        educational_boundary_note=boundary,
    )
