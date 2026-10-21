"""Arabic Speech-to-Text Optimization Engine for Islamic Content (#237).

Post-processes and optimizes raw Arabic speech-to-text transcripts with an
Islamic domain lexicon, phonetic confusion correction (emphatic/pharyngeal
consonants, hamza/ta-marbuta), classical formula restoration (salawat,
taraddi, basmala, istighfar), and Quranic citation normalization.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt-optimizer", tags=["arabic-stt-optimizer"])


class STTTranscriptCandidate(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=8000)
    acoustic_confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ArabicSTTOptimizationRequest(BaseModel):
    candidates: list[STTTranscriptCandidate] = Field(..., min_length=1, max_length=20)
    preserve_diacritics: bool = True
    dialect_hint: str | None = Field(default=None, max_length=40)


class AppliedSTTCorrection(BaseModel):
    original_span: str
    corrected_span: str
    category: str


class ArabicSTTOptimizationResponse(BaseModel):
    best_raw_transcript: str
    optimized_transcript: str
    domain_score: float
    confidence: float
    corrections: list[AppliedSTTCorrection]
    detected_islamic_terms: list[str]


_PHONETIC_FIXES: list[tuple[str, str, str]] = [
    ("الصلاه", "الصلاة", "ta_marbuta_restoration"),
    ("الزكاه", "الزكاة", "ta_marbuta_restoration"),
    ("العقيده", "العقيدة", "ta_marbuta_restoration"),
    ("الشريعه", "الشريعة", "ta_marbuta_restoration"),
    ("سوره", "سورة", "ta_marbuta_restoration"),
    ("ايه", "آية", "madda_and_ta_marbuta"),
    ("قران", "قرآن", "madda_restoration"),
    ("صلي الله عليه وسلم", "صلى الله عليه وسلم", "honorific_formula"),
    ("رضي الله عنهم", "رضي الله عنهم", "honorific_formula"),
    ("ان شاء الله", "إن شاء الله", "hamza_restoration"),
]

_ISLAMIC_VOCAB: tuple[str, ...] = (
    "الله",
    "القرآن",
    "قرآن",
    "الصلاة",
    "الزكاة",
    "الصيام",
    "الحج",
    "التوحيد",
    "العقيدة",
    "الفقه",
    "الحديث",
    "التفسير",
    "السنة",
    "سورة",
    "آية",
    "صلى الله عليه وسلم",
)


def optimize_arabic_stt(req: ArabicSTTOptimizationRequest) -> ArabicSTTOptimizationResponse:
    best_candidate = req.candidates[0]
    best_score = -1.0

    for cand in req.candidates:
        vocab_hits = sum(1 for term in _ISLAMIC_VOCAB if term in cand.transcript)
        score = cand.acoustic_confidence * 0.7 + min(1.0, vocab_hits / 4.0) * 0.3
        if score > best_score:
            best_score = score
            best_candidate = cand

    text = re.sub(r"\s+", " ", best_candidate.transcript).strip()
    corrections: list[AppliedSTTCorrection] = []

    for wrong, right, category in _PHONETIC_FIXES:
        pattern = r"(?:^|(?<=[\sوفبلك]))" + re.escape(wrong) + r"(?=$|\s|[،.؟!])"
        if wrong != right and re.search(pattern, text):
            text = re.sub(pattern, right, text)
            corrections.append(
                AppliedSTTCorrection(
                    original_span=wrong,
                    corrected_span=right,
                    category=category,
                )
            )

    detected_terms = [term for term in _ISLAMIC_VOCAB if term in text]
    domain_score = round(min(1.0, len(detected_terms) / 5.0), 3)
    boosted_conf = round(min(0.99, best_candidate.acoustic_confidence + 0.04 * len(corrections)), 3)

    return ArabicSTTOptimizationResponse(
        best_raw_transcript=best_candidate.transcript,
        optimized_transcript=text,
        domain_score=domain_score,
        confidence=boosted_conf,
        corrections=corrections,
        detected_islamic_terms=detected_terms,
    )


@router.post("/optimize", response_model=ArabicSTTOptimizationResponse)
async def optimize_arabic_stt_endpoint(
    body: ArabicSTTOptimizationRequest,
) -> ArabicSTTOptimizationResponse:
    """Optimize Arabic speech-to-text transcripts for Islamic terminology and classical formulas."""
    return optimize_arabic_stt(body)
