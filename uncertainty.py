"""Uncertainty quantification, Islamic epistemology taxonomy, and evidence strength scoring (#199).

Why this exists
---------------
In Islamic jurisprudence (Usul al-Fiqh), knowledge is structured across rigorous epistemic
categories: definitive (Qat'i) vs probable/interpretive (Dhanni), universal consensus (Ijma)
vs valid juristic disagreement (Ikhtilaf), and established school rulings (Mu'tamad) vs
contemporary individual/institutional reasoning (Ijtihad).

An AI giving Islamic guidance must never present an interpretive or disputed opinion with
the same unconditional certainty as a foundational obligation. It must demonstrate epistemic
humility (e.g. Allahu A'lam), identify high-uncertainty and sensitive fatwa matters, quantify
evidence strength, warn when an answer rests on limited sources, provide uncertainty ranges,
and guide the user toward qualified Muftis and actionable paths to reduce uncertainty.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Islamic Epistemological Taxonomy & Enums
# ---------------------------------------------------------------------------


class EpistemicCertainty(str, Enum):
    """Categorization of ruling certainty according to Islamic legal theory (Usul al-Fiqh)."""

    QATI = "qati"  # Definitive, decisive, unambiguous textual proof (e.g. 5 pillars, clear prohibitions)
    DHANNI = "dhanni"  # Probable, interpretive, branch matters derived through ijtihad/solitary narrations
    DISPUTED = "disputed"  # Ikhtilafi: established, valid differences of opinion among Sunni madhhabs
    NOVEL_CONTEMPORARY = "novel_contemporary"  # Nawazil: modern issues requiring contemporary juristic reasoning
    GENERAL = "general"  # General Islamic history, ethics, language, or non-jurisprudential knowledge


class PositionType(str, Enum):
    """Type of legal authority or consensus backing an answer."""

    IJMA = "ijma"  # Universal scholarly consensus
    ESTABLISHED_MADHHAB = "established_madhhab"  # Relied-upon (mu'tamad) position of one or more Sunni schools
    SCHOLARLY_IKHTILAF = "scholarly_ikhtilaf"  # Recognized divergence of opinions across schools
    CONTEMPORARY_IJTIHAD = "contemporary_ijtihad"  # Modern juristic councils (e.g. OIC Fiqh Academy, AMJA)
    INDIVIDUAL_OPINION = "individual_opinion"  # Isolated, non-majority, or individual scholarly view


class EvidenceStrength(str, Enum):
    """Categorical evaluation of textual evidence backing an answer."""

    VERY_STRONG = "very_strong"  # Explicit Quranic text and/or Mutawatir / Sahih Bukhari & Muslim
    STRONG = "strong"  # Sahih / Hasan hadith with established juristic corroboration
    MODERATE = "moderate"  # Ahad / Hasan narrations with differing interpretations or solitary citation
    WEAK_OR_LIMITED = "weak_or_limited"  # Da'if narrations, unverified citations, or minimal source backing


class UncertaintyFactor(str, Enum):
    """Specific factors that contribute to or increase uncertainty in an answer."""

    DISPUTED_MATTER = "disputed_matter"
    INTERPRETIVE_EVIDENCE = "interpretive_evidence"
    LIMITED_SOURCES = "limited_sources"
    WEAK_OR_UNVERIFIED_HADITH = "weak_or_unverified_hadith"
    HEDGING_LANGUAGE_DETECTED = "hedging_language_detected"
    NOVEL_CONTEMPORARY_ISSUE = "novel_contemporary_issue"
    HIGH_STAKES_PERSONAL_RULING = "high_stakes_personal_ruling"
    ABSENCE_OF_PRIMARY_CITATIONS = "absence_of_primary_citations"


# ---------------------------------------------------------------------------
# Structured Uncertainty Quantification Model
# ---------------------------------------------------------------------------


class UncertaintyQuantification(BaseModel):
    """Complete uncertainty profile and epistemological assessment for an answer (#199)."""

    epistemic_certainty: EpistemicCertainty = Field(
        ..., description="Definitive (Qat'i), Probable (Dhanni), Disputed, or Novel"
    )
    position_type: PositionType = Field(
        ..., description="Ijma, Established Madhhab, Scholarly Ikhtilaf, or Contemporary Ijtihad"
    )
    evidence_strength: EvidenceStrength = Field(
        ..., description="Strength of textual evidence (Very Strong, Strong, Moderate, Weak/Limited)"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall calibrated confidence score (0-1)")
    uncertainty_score: float = Field(..., ge=0.0, le=1.0, description="Overall quantified uncertainty score (0-1)")
    confidence_interval: tuple[float, float] = Field(
        ..., description="Uncertainty range [lower_bound, upper_bound] for the score"
    )
    is_high_uncertainty: bool = Field(False, description="True if uncertainty exceeds the high-uncertainty threshold")
    requires_expert_consultation: bool = Field(
        False, description="True if direct consultation with a qualified Mufti/scholar is advised"
    )
    consultation_reason: str | None = Field(
        None, description="Explanation for why expert consultation is required or advised"
    )
    limited_sources_warning: bool = Field(
        False, description="True when answer is based on limited or unverified primary sources"
    )
    contributing_factors: list[str] = Field(
        default_factory=list, description="List of factors contributing to the uncertainty"
    )
    reduction_paths: list[str] = Field(
        default_factory=list, description="Actionable paths for the user to reduce uncertainty"
    )
    explanation: str = Field(..., description="User-facing summary explanation of the uncertainty profile")
    epistemic_humility_note: str = Field(
        default="والله أعلم (And Allah knows best)", description="Epistemic humility closing statement"
    )


# ---------------------------------------------------------------------------
# High-Uncertainty & Sensitive Fatwa Topic Markers
# ---------------------------------------------------------------------------

# High-stakes personal fatwas where an AI must never issue binding answers without expert consultation
_HIGH_STAKES_PERSONAL_KEYWORDS: tuple[str, ...] = (
    "divorce",
    "talaq",
    "khula",
    "annulment",
    "fasakh",
    "custody",
    "child custody",
    "hadanah",
    "inheritance",
    "inheritance distribution",
    "shares of inheritance",
    "inheritance shares",
    "estate division",
    "mirath",
    "wasiyya",
    "who inherits",
    "organ donation",
    "organ transplant",
    "euthanasia",
    "abortion",
    "terminating pregnancy",
    "assisted reproduction",
    "ivf",
    "surrogacy",
    "crypto futures",
    "forex leverage",
    "derivatives trading",
    "bankruptcy",
    "apostasy",
    "takfir",
    "excommunication",
    "swearing by allah",
    "breaking an oath",
    "kaffarah calculation",
)

# Definitive matters (Qat'i) - fundamental obligations, core creed, clear prohibitions with Ijma
_QATI_KEYWORDS: tuple[str, ...] = (
    "five pillars",
    "pillars of islam",
    "six pillars of iman",
    "oneness of allah",
    "tawhid",
    "obligation of prayer",
    "five daily prayers",
    "obligation of fasting",
    "fasting ramadan",
    "obligation of zakat",
    "obligation of hajj",
    "prohibition of murder",
    "prohibition of theft",
    "prohibition of zina",
    "prohibition of adultery",
    "prohibition of alcohol",
    "prohibition of pork",
    "prohibition of interest",
    "prohibition of riba",
    "honoring parents",
    "truthfulness",
    "prohibition of lying",
)

# Known Disputed / Ikhtilaf Topics (branch fiqh with classical divergence across 4 madhhabs)
_DISPUTED_KEYWORDS: tuple[str, ...] = (
    "bleeding breaks wudu",
    "does bleeding invalidate wudu",
    "touching opposite sex breaks wudu",
    "touching women break wudu",
    "touching skin breaks wudu",
    "eating camel meat breaks wudu",
    "wiping over socks",
    "raising hands in prayer",
    "raf al yadayn",
    "placing hands on chest",
    "sadl",
    "reciting fatiha behind imam",
    "qunut in fajr",
    "tashahhud finger movement",
    "combining prayers for rain",
    "moonsighting",
    "global vs local moonsighting",
    "astronomical calculation",
    "music",
    "musical instruments",
    "digital photography",
    "drawing living beings",
    "face veil",
    "niqab",
    "triple talaq in one sitting",
    "wiping over thin socks",
    "tarawih 8 or 20",
    "witr 1 or 3",
)

# Novel Contemporary Topics (Nawazil requiring contemporary Ijtihad)
_NOVEL_KEYWORDS: tuple[str, ...] = (
    "bitcoin",
    "cryptocurrency",
    "crypto",
    "nft",
    "nfts",
    "staking",
    "yield farming",
    "defi",
    "artificial intelligence",
    "ai generated",
    "cloning",
    "gene editing",
    "crispr",
    "brain death",
    "lab grown meat",
    "cultured meat",
    "stock options",
    "dropshipping",
    "affiliate marketing",
    "cashback",
    "credit card points",
    "mortgage in the west",
    "space travel prayer",
    "fasting in extreme latitudes",
    "midnight sun fasting",
)

# Regex compiler helpers
_HIGH_STAKES_RE = tuple(
    re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in _HIGH_STAKES_PERSONAL_KEYWORDS
)
_QATI_RE = tuple(re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in _QATI_KEYWORDS)
_DISPUTED_RE = tuple(re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in _DISPUTED_KEYWORDS)
_NOVEL_RE = tuple(re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in _NOVEL_KEYWORDS)


# ---------------------------------------------------------------------------
# Bayesian Prior Distributions
# ---------------------------------------------------------------------------

# Prior parameters (mean mu, standard deviation sigma) for different epistemic categories
_EPISTEMIC_PRIORS: dict[EpistemicCertainty, tuple[float, float]] = {
    EpistemicCertainty.QATI: (0.88, 0.06),  # High prior confidence, narrow uncertainty interval
    EpistemicCertainty.DHANNI: (0.70, 0.12),  # Solid baseline, moderate variance
    EpistemicCertainty.DISPUTED: (0.52, 0.16),  # Middling baseline, wider uncertainty interval
    EpistemicCertainty.NOVEL_CONTEMPORARY: (0.46, 0.18),  # Cautious baseline, wide uncertainty interval
    EpistemicCertainty.GENERAL: (0.62, 0.14),  # Standard general prior
}


# ---------------------------------------------------------------------------
# Epistemic & Evidence Classification Helpers
# ---------------------------------------------------------------------------


def classify_epistemic_certainty(prompt: str, answer: str, is_fiqh: bool, is_religious: bool) -> EpistemicCertainty:
    """Classify the epistemic certainty level of an Islamic topic (Qat'i vs Dhanni vs Disputed vs Novel)."""
    if not is_religious:
        return EpistemicCertainty.GENERAL

    combined = f"{prompt} {answer}".casefold()

    # 1. Check for Qat'i (foundational/definitive)
    if any(regex.search(combined) for regex in _QATI_RE):
        return EpistemicCertainty.QATI

    # 2. Check for Novel/Contemporary Nawazil
    if any(regex.search(combined) for regex in _NOVEL_RE):
        return EpistemicCertainty.NOVEL_CONTEMPORARY

    # 3. Check for Disputed / Ikhtilaf
    if any(regex.search(combined) for regex in _DISPUTED_RE):
        return EpistemicCertainty.DISPUTED

    # 4. Check for explicit ikhtilaf markers in the text
    ikhtilaf_markers = (
        "scholars differ",
        "difference of opinion",
        "there is ikhtilaf",
        "scholarly disagreement",
        "hanafi school holds",
        "shafi'i school",
        "maliki school",
        "hanbali school",
        "two views",
        "majority of scholars",
        "jumhur",
        "while other scholars",
    )
    if any(marker in combined for marker in ikhtilaf_markers):
        return EpistemicCertainty.DISPUTED

    # 5. Fiqh questions default to Dhanni (interpretive branch ruling) unless confirmed Qat'i
    if is_fiqh:
        return EpistemicCertainty.DHANNI

    return EpistemicCertainty.GENERAL


def classify_position_type(prompt: str, answer: str, epistemic: EpistemicCertainty, is_fiqh: bool) -> PositionType:
    """Identify whether an answer reflects universal consensus (Ijma), a Madhhab position, Ikhtilaf, or Ijtihad."""
    text = f"{prompt} {answer}".casefold()

    if epistemic is EpistemicCertainty.QATI or "consensus" in text or "ijma'" in text or "ijma" in text:
        return PositionType.IJMA

    if epistemic is EpistemicCertainty.NOVEL_CONTEMPORARY:
        return PositionType.CONTEMPORARY_IJTIHAD

    if epistemic is EpistemicCertainty.DISPUTED:
        return PositionType.SCHOLARLY_IKHTILAF

    madhhab_indicators = ("hanafi", "maliki", "shafi'i", "shafii", "hanbali", "mu'tamad", "school of thought")
    if is_fiqh or any(ind in text for ind in madhhab_indicators):
        return PositionType.ESTABLISHED_MADHHAB

    return PositionType.ESTABLISHED_MADHHAB if is_fiqh else PositionType.INDIVIDUAL_OPINION


def evaluate_evidence_strength(
    citations: list[Any] | None,
    hadith_refs: list[Any] | None,
    is_religious: bool,
    citation_score: float | None = None,
) -> EvidenceStrength:
    """Quantify the strength of evidence backing the answer."""
    if not is_religious:
        return EvidenceStrength.STRONG

    has_quran = False
    has_sahih_hadith = False
    has_hasan_hadith = False
    has_weak_hadith = False
    has_unverified_hadith = False

    citations_list = citations or []
    for c in citations_list:
        ctype = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else None)
        if ctype == "quran":
            has_quran = True

    hadith_list = hadith_refs or []
    for h in hadith_list:
        grade = getattr(h, "grade", "") or (h.get("grade", "") if isinstance(h, dict) else "")
        verified = getattr(h, "verified", False) or (h.get("verified", False) if isinstance(h, dict) else False)
        grade_upper = str(grade).upper()

        if "SAHIH" in grade_upper:
            has_sahih_hadith = True
        elif "HASAN" in grade_upper:
            has_hasan_hadith = True
        elif "DAIF" in grade_upper or "MAWDU" in grade_upper:
            has_weak_hadith = True
        elif not verified or "UNKNOWN" in grade_upper:
            has_unverified_hadith = True

    total_citations = len(citations_list) + len(hadith_list)

    if has_weak_hadith or (total_citations == 0 and is_religious):
        return EvidenceStrength.WEAK_OR_LIMITED

    if (has_quran and has_sahih_hadith) or (has_sahih_hadith and total_citations >= 2):
        if citation_score is None or citation_score >= 0.8:
            return EvidenceStrength.VERY_STRONG
        return EvidenceStrength.STRONG

    if has_quran or (has_sahih_hadith and total_citations >= 1) or (has_hasan_hadith and total_citations >= 2):
        return EvidenceStrength.STRONG

    if has_hasan_hadith or (total_citations >= 1 and not has_unverified_hadith):
        return EvidenceStrength.MODERATE

    return EvidenceStrength.WEAK_OR_LIMITED


def detect_high_stakes_and_consultation(
    prompt: str, answer: str, epistemic: EpistemicCertainty
) -> tuple[bool, str | None]:
    """Detect if a question deals with high-stakes personal rulings that mandate qualified Mufti consultation."""
    combined = f"{prompt} {answer}".casefold()

    for kw in _HIGH_STAKES_PERSONAL_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", combined):
            return True, f"Involves sensitive personal jurisprudence ({kw}) requiring situation-specific fatwa."

    if epistemic is EpistemicCertainty.NOVEL_CONTEMPORARY and (
        "should i" in combined or "can i" in combined or "is it halal for me" in combined
    ):
        return True, "Contemporary novel issue (nawazil) with ongoing scholarly deliberation."

    return False, None


# ---------------------------------------------------------------------------
# Bayesian Confidence & Uncertainty Range Calculation
# ---------------------------------------------------------------------------


def calculate_bayesian_confidence(
    epistemic: EpistemicCertainty,
    evidence_strength: EvidenceStrength,
    citation_score: float | None,
    expressed_certainty_score: float,
    consistency_score: float | None,
    is_high_stakes: bool,
    total_citations_count: int,
) -> tuple[float, float, tuple[float, float]]:
    """Compute calibrated confidence score, uncertainty score, and credible confidence interval.

    Returns:
        (confidence_score, uncertainty_score, (lower_bound, upper_bound))
    """
    prior_mu, prior_sigma = _EPISTEMIC_PRIORS.get(epistemic, (0.60, 0.15))

    # Evidence adjustment factors
    evidence_multipliers = {
        EvidenceStrength.VERY_STRONG: 1.15,
        EvidenceStrength.STRONG: 1.05,
        EvidenceStrength.MODERATE: 0.95,
        EvidenceStrength.WEAK_OR_LIMITED: 0.75,
    }
    evidence_mult = evidence_multipliers.get(evidence_strength, 1.0)

    # Blend prior with observed signals
    signals: list[tuple[float, float]] = []  # (value, weight)

    # 1. Prior weight
    signals.append((prior_mu, 0.35))

    # 2. Citation verification score if present
    if citation_score is not None:
        signals.append((citation_score, 0.25))

    # 3. Expressed certainty (hedging)
    signals.append((expressed_certainty_score, 0.20))

    # 4. Consistency score if present
    if consistency_score is not None:
        signals.append((consistency_score, 0.20))

    total_weight = sum(w for _, w in signals)
    raw_mean = sum(v * w for v, w in signals) / total_weight

    # Apply evidence multiplier
    calibrated_score = raw_mean * evidence_mult

    # Penalize limited sources on complex religious matters
    if total_citations_count == 0 and epistemic in (
        EpistemicCertainty.DHANNI,
        EpistemicCertainty.DISPUTED,
        EpistemicCertainty.NOVEL_CONTEMPORARY,
    ):
        calibrated_score *= 0.85

    # Apply high-stakes multiplier
    if is_high_stakes:
        calibrated_score *= 0.85

    score = round(min(1.0, max(0.0, calibrated_score)), 4)
    uncertainty = round(1.0 - score, 4)

    # Calculate dynamic variance (tighter interval for Qat'i/Very Strong, wider for Disputed/Weak)
    sigma_adj = prior_sigma
    if evidence_strength is EvidenceStrength.VERY_STRONG:
        sigma_adj *= 0.7
    elif evidence_strength is EvidenceStrength.WEAK_OR_LIMITED:
        sigma_adj *= 1.4

    margin = round(sigma_adj * 1.5, 3)
    lower = round(max(0.0, score - margin), 3)
    upper = round(min(1.0, score + margin), 3)

    return score, uncertainty, (lower, upper)


# ---------------------------------------------------------------------------
# User-Facing Explanations & Epistemic Humility Note Generation
# ---------------------------------------------------------------------------


def generate_reduction_paths(
    epistemic: EpistemicCertainty,
    requires_expert: bool,
    limited_sources: bool,
    position_type: PositionType,
) -> list[str]:
    """Generate actionable suggestions for users to clarify and reduce uncertainty."""
    paths: list[str] = []

    if requires_expert or epistemic in (EpistemicCertainty.DISPUTED, EpistemicCertainty.NOVEL_CONTEMPORARY):
        paths.append("Consult a qualified local Mufti or certified Islamic council for a binding personal verdict.")

    if position_type is PositionType.SCHOLARLY_IKHTILAF or epistemic is EpistemicCertainty.DISPUTED:
        paths.append(
            "Specify your personal Madhhab (Hanafi, Maliki, Shafi'i, or Hanbali) to receive its relied-upon (mu'tamad) position."
        )

    if limited_sources:
        paths.append("Review primary Quranic verses and authenticated Hadith collections on Sunnah.com and Quran.com.")

    if epistemic is EpistemicCertainty.NOVEL_CONTEMPORARY:
        paths.append(
            "Review resolutions from recognized contemporary bodies like the International Islamic Fiqh Academy."
        )

    if not paths:
        paths.append("Cross-reference with standard classical fiqh manuals and authenticated commentaries.")

    return paths


def generate_contributing_factors(
    epistemic: EpistemicCertainty,
    evidence_strength: EvidenceStrength,
    limited_sources: bool,
    hedging_detected: bool,
    is_high_stakes: bool,
) -> list[str]:
    """Generate human-readable list of factors contributing to the uncertainty score."""
    factors: list[str] = []

    if epistemic is EpistemicCertainty.DISPUTED:
        factors.append("Subject to recognized classical juristic disagreement (Ikhtilaf) among the four Sunni schools.")
    elif epistemic is EpistemicCertainty.NOVEL_CONTEMPORARY:
        factors.append("Contemporary emerging issue (Nawazil) requiring ongoing ijtihad and institutional review.")
    elif epistemic is EpistemicCertainty.DHANNI:
        factors.append("Interpretive branch ruling (Dhanni al-Dalalah) derived through scholarly legal deduction.")

    if evidence_strength is EvidenceStrength.WEAK_OR_LIMITED:
        factors.append("Evidence relies on limited primary texts, solitary reports, or unverified citations.")
    elif evidence_strength is EvidenceStrength.MODERATE:
        factors.append("Evidence based on solitary narrations (Ahad) with differing interpretive weight.")

    if limited_sources:
        factors.append("Answer contains minimal or single primary source citations for a complex topic.")

    if hedging_detected:
        factors.append("Language indicates semantic hesitation, uncertainty markers, or conditional phrasing.")

    if is_high_stakes:
        factors.append("Involves high-stakes personal decisions or sensitive contractual/marital rulings.")

    return factors


def generate_uncertainty_explanation(
    epistemic: EpistemicCertainty,
    position_type: PositionType,
    evidence_strength: EvidenceStrength,
    score: float,
    requires_expert: bool,
) -> str:
    """Generate a clear, respectful summary of the uncertainty profile."""
    if epistemic is EpistemicCertainty.QATI:
        return (
            "This matter is definitive and established by clear textual evidence and universal consensus (Ijma'). "
            "Confidence is high."
        )

    if epistemic is EpistemicCertainty.DISPUTED:
        return (
            "This matter contains recognized differences of opinion (Ikhtilaf) among the major Sunni schools of thought. "
            "No single view is universally binding; users should follow their school or trusted scholar."
        )

    if epistemic is EpistemicCertainty.NOVEL_CONTEMPORARY:
        return (
            "This is a contemporary matter (Nawazil) that requires modern collective Ijtihad. "
            "Rulings are derived by contemporary juristic councils and may evolve with context."
        )

    if requires_expert:
        return (
            "This question touches upon sensitive personal rulings that depend on specific individual circumstances. "
            "Personal consultation with a qualified Mufti is strongly recommended."
        )

    if score < 0.5:
        return "High uncertainty due to limited verified evidence or interpretive complexity. Verification is advised."

    return "This is an interpretive matter (Dhanni) with established scholarly foundations. Follow standard guidance."


# ---------------------------------------------------------------------------
# Main Quantification Pipeline
# ---------------------------------------------------------------------------


def quantify_uncertainty(
    prompt: str,
    answer: str,
    is_fiqh: bool,
    is_religious: bool,
    hadith_refs: list[Any] | None = None,
    citations: list[Any] | None = None,
    citation_score: float | None = None,
    consistency_score: float | None = None,
    expressed_certainty_score: float = 1.0,
) -> UncertaintyQuantification:
    """Quantify the complete uncertainty profile of an answer according to Islamic epistemology (#199)."""
    # 1. Epistemic classification
    epistemic = classify_epistemic_certainty(prompt, answer, is_fiqh=is_fiqh, is_religious=is_religious)
    position = classify_position_type(prompt, answer, epistemic=epistemic, is_fiqh=is_fiqh)

    # 2. Evidence strength evaluation
    citations_list = citations or []
    hadith_list = hadith_refs or []
    total_citations = len(citations_list) + len(hadith_list)
    evidence_strength = evaluate_evidence_strength(
        citations=citations_list,
        hadith_refs=hadith_list,
        is_religious=is_religious,
        citation_score=citation_score,
    )

    # 3. High-stakes and expert consultation evaluation
    is_high_stakes, consultation_reason = detect_high_stakes_and_consultation(
        prompt=prompt, answer=answer, epistemic=epistemic
    )

    # 4. Source limitation check
    limited_sources = is_religious and total_citations < 2 and epistemic is not EpistemicCertainty.QATI

    # 5. Bayesian confidence & interval computation
    score, uncertainty, interval = calculate_bayesian_confidence(
        epistemic=epistemic,
        evidence_strength=evidence_strength,
        citation_score=citation_score,
        expressed_certainty_score=expressed_certainty_score,
        consistency_score=consistency_score,
        is_high_stakes=is_high_stakes,
        total_citations_count=total_citations,
    )

    is_high_uncertainty = uncertainty >= 0.40 or epistemic in (
        EpistemicCertainty.DISPUTED,
        EpistemicCertainty.NOVEL_CONTEMPORARY,
    )

    # If score is very low on a religious topic, recommend expert consultation
    if score < 0.40 and is_religious and not is_high_stakes:
        is_high_stakes = True
        consultation_reason = "Low confidence in AI-generated ruling; requires verification by a scholar."

    # 6. Contributing factors & reduction paths
    hedging_detected = expressed_certainty_score < 0.8
    factors = generate_contributing_factors(
        epistemic=epistemic,
        evidence_strength=evidence_strength,
        limited_sources=limited_sources,
        hedging_detected=hedging_detected,
        is_high_stakes=is_high_stakes,
    )
    reduction_paths = generate_reduction_paths(
        epistemic=epistemic,
        requires_expert=is_high_stakes,
        limited_sources=limited_sources,
        position_type=position,
    )

    explanation = generate_uncertainty_explanation(
        epistemic=epistemic,
        position_type=position,
        evidence_strength=evidence_strength,
        score=score,
        requires_expert=is_high_stakes,
    )

    return UncertaintyQuantification(
        epistemic_certainty=epistemic,
        position_type=position,
        evidence_strength=evidence_strength,
        confidence_score=score,
        uncertainty_score=uncertainty,
        confidence_interval=interval,
        is_high_uncertainty=is_high_uncertainty,
        requires_expert_consultation=is_high_stakes,
        consultation_reason=consultation_reason,
        limited_sources_warning=limited_sources,
        contributing_factors=factors,
        reduction_paths=reduction_paths,
        explanation=explanation,
        epistemic_humility_note="والله تعالى أعلم (And Allah the Exalted knows best)",
    )
