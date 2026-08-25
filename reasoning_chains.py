"""Multi-step Islamic reasoning chains (#152).

A deterministic, dependency-free engine that turns a complex Islamic question
into an ordered chain of reasoning steps and lets a user inspect, validate and
question that chain.

Why deterministic
-----------------
The aspirational feature is chain-of-thought reasoning over fiqh, tafsir and
hadith with a live model. That belongs behind the model layer; what this module
provides is the *structure* around it — decomposition, evidence tagging,
consistency checking, weak-point detection and madhhab branching — implemented
with heuristics so it runs offline, in tests and in CI with no API key and no
network call. Nothing here reaches a live service at import time or request
time; a model can later fill richer conclusions into the same shapes.

What a chain carries
--------------------
Every step has a stable id (so a user can question one specific step), an
intermediate conclusion, evidence-source tags (Qur'an / hadith / fiqh /
tafsir), a logical connector to the next step and a confidence score. A chain
also validates itself: contradictory conclusions are flagged, the
lowest-confidence and unsupported steps are surfaced as weak points, and a
question that turns on madhhab differences fans out into parallel reasoning
branches.

No specific verse or hadith numbers are fabricated: evidence tags point at the
*category* and the authenticated sources to consult (quran.com, sunnah.com),
never an invented citation.
"""

from __future__ import annotations

import re
from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/reasoning", tags=["reasoning"])


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    """The category of evidence a step leans on."""

    QURAN = "quran"
    HADITH = "hadith"
    FIQH = "fiqh"
    TAFSIR = "tafsir"
    GENERAL = "general"


class Connector(str, Enum):
    """The logical link from one step to the next."""

    THEREFORE = "therefore"
    BECAUSE = "because"
    HOWEVER = "however"
    AND = "and"
    NONE = "none"


# Facet keywords, checked in priority order so a verse question is read as
# tafsir even when it also mentions a ruling.
_FACET_KEYWORDS: tuple[tuple[SourceType, tuple[str, ...]], ...] = (
    (
        SourceType.TAFSIR,
        ("verse", "ayah", "ayat", "surah", "surat", "qur'an", "quran", "tafsir", "meaning of", "interpret"),
    ),
    (
        SourceType.HADITH,
        ("hadith", "narration", "narrated", "sunnah", "prophet", "reported", "sahih", "authentic"),
    ),
    (
        SourceType.FIQH,
        (
            "ruling",
            "permissible",
            "permitted",
            "halal",
            "haram",
            "forbidden",
            "obligatory",
            "wajib",
            "fard",
            "makruh",
            "mustahabb",
            "prayer",
            "salah",
            "salat",
            "wudu",
            "fast",
            "fasting",
            "zakat",
            "hajj",
            "allowed",
            "valid",
            "invalid",
        ),
    ),
)

# Words that mark a genuine difference of opinion; a step resting on one is
# deliberately scored lower because the answer is contested, not settled.
_KHILAF_MARKERS: tuple[str, ...] = (
    "differ",
    "difference of opinion",
    "madhhab",
    "madhahib",
    "school",
    "schools",
    "disagree",
    "opinion",
    "opinions",
    "scholars say",
)

# Polarity markers for contradiction detection. Negatives are checked first so
# "not permissible" is read as negative rather than matching "permissible".
_NEGATIVE_MARKERS: tuple[str, ...] = (
    "not permissible",
    "not permitted",
    "not allowed",
    "impermissible",
    "haram",
    "forbidden",
    "prohibited",
    "invalid",
    "unlawful",
    "makruh",
    "disliked",
)
_POSITIVE_MARKERS: tuple[str, ...] = (
    "permissible",
    "permitted",
    "allowed",
    "halal",
    "lawful",
    "valid",
    "obligatory",
    "recommended",
    "encouraged",
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "into",
        "when",
        "what",
        "which",
        "about",
        "there",
        "their",
        "have",
        "does",
        "whether",
        "would",
        "should",
        "could",
        "muslim",
        "islamic",
        "islam",
    }
)

# The four Sunni schools, used when a question fans out into parallel paths.
MADHHABS: tuple[str, ...] = ("hanafi", "maliki", "shafii", "hanbali")

# Base confidence per facet before support and khilaf adjustments.
_FACET_BASE_CONFIDENCE: dict[SourceType, float] = {
    SourceType.QURAN: 0.78,
    SourceType.TAFSIR: 0.72,
    SourceType.HADITH: 0.68,
    SourceType.FIQH: 0.7,
    SourceType.GENERAL: 0.5,
}

# Below this a step is called out as a weak point.
WEAK_CONFIDENCE_THRESHOLD = 0.6

_UNSUPPORTED_PENALTY = 0.6
_KHILAF_PENALTY = 0.15

_SPLIT_PATTERN = re.compile(
    r"\s*(?:\?|;|\band also\b|\band\b|\bbut\b|\bhowever\b)\s*",
    re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"[a-z']+")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvidenceRef(BaseModel):
    """A pointer to a category of evidence, not an invented citation."""

    source_type: SourceType
    reference: str = Field(..., description="Category-level pointer, e.g. 'fiqh:worship' or 'Hadith (sunnah.com)'")
    note: str | None = Field(None, description="Optional clarifying note")


class ReasoningStep(BaseModel):
    """One link in the chain, individually addressable by ``id``."""

    id: str = Field(..., description="Stable step id, e.g. 'step-1', so a user can question one step")
    order: int = Field(..., ge=1, description="1-based position within its path")
    facet: str = Field(..., description="The sub-question or facet this step resolves")
    source_type: SourceType = Field(..., description="Primary evidence category for the step")
    intermediate_conclusion: str = Field(..., description="What this step concludes")
    evidence: list[EvidenceRef] = Field(default_factory=list, description="Evidence-source references")
    connector: Connector = Field(Connector.NONE, description="Logical link to the next step")
    confidence: float = Field(..., ge=0.0, le=1.0, description="How well-supported this step is")
    supported: bool = Field(..., description="Whether the step carries any evidence")
    branch: str | None = Field(None, description="Madhhab tag when the step belongs to a parallel path")


class ReasoningBranch(BaseModel):
    """A parallel reasoning path for one madhhab."""

    madhhab: str
    steps: list[ReasoningStep]


class ConsistencyIssue(BaseModel):
    """A problem found while validating a chain."""

    kind: str = Field(..., description="'contradiction', 'unsupported' or 'low_confidence'")
    step_ids: list[str]
    detail: str


class WeakPoint(BaseModel):
    """A step a user should scrutinise first."""

    step_id: str
    confidence: float
    reason: str


class ValidationReport(BaseModel):
    """The outcome of checking a chain for consistency and weak points."""

    consistent: bool = Field(..., description="True when no contradictory conclusions were found")
    issues: list[ConsistencyIssue] = Field(default_factory=list)
    weak_points: list[WeakPoint] = Field(default_factory=list)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)


class ReasoningChain(BaseModel):
    """A full decomposed answer: ordered steps, optional branches, a verdict."""

    id: str
    question: str
    madhhab: str | None = None
    steps: list[ReasoningStep]
    branches: list[ReasoningBranch] = Field(default_factory=list)
    conclusion: str
    validation: ValidationReport


class ReasoningTemplate(BaseModel):
    """A reusable step skeleton for a common Islamic question pattern."""

    pattern: str
    description: str
    steps: list[str]
    source_types: list[SourceType]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChainRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The complex question to decompose")
    madhhab: str | None = Field(None, description="Optional madhhab to tag or to force branching")


class ChainResponse(BaseModel):
    chain: ReasoningChain
    outline: str = Field(..., description="Human-friendly markdown rendering of the chain")


class StepSubmission(BaseModel):
    """A step a caller submits for validation."""

    id: str | None = None
    intermediate_conclusion: str = Field(..., min_length=1)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    steps: list[StepSubmission] = Field(..., min_length=1)


class TemplatesResponse(BaseModel):
    count: int
    templates: list[ReasoningTemplate]


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_TEMPLATES: tuple[ReasoningTemplate, ...] = (
    ReasoningTemplate(
        pattern="ruling",
        description="Establishing a fiqh ruling on an action.",
        steps=[
            "Identify the action and its relevant category of worship or dealings.",
            "Gather the primary evidence from Qur'an and Sunnah.",
            "Apply the recognised fiqh principles to the evidence.",
            "State the ruling and note any recognised difference of opinion.",
        ],
        source_types=[SourceType.QURAN, SourceType.HADITH, SourceType.FIQH],
    ),
    ReasoningTemplate(
        pattern="tafsir",
        description="Explaining the meaning of a verse.",
        steps=[
            "Locate the verse and its surah context.",
            "Summarise the occasion of revelation if it is established.",
            "Present the classical tafsir of the verse.",
            "Draw the practical or theological lesson.",
        ],
        source_types=[SourceType.QURAN, SourceType.TAFSIR],
    ),
    ReasoningTemplate(
        pattern="hadith_authenticity",
        description="Assessing a narration before acting on it.",
        steps=[
            "Identify the narration and its collection.",
            "Check the grading of its chain (sahih, hasan, da'if).",
            "Reconcile it with the Qur'an and stronger narrations.",
            "State whether it can be relied upon.",
        ],
        source_types=[SourceType.HADITH, SourceType.QURAN],
    ),
    ReasoningTemplate(
        pattern="comparative_fiqh",
        description="Comparing the positions of the madhhabs.",
        steps=[
            "State the point of difference clearly.",
            "Present each school's position and its evidence.",
            "Note where the difference is substantive versus semantic.",
            "Advise consulting a qualified scholar for one's own context.",
        ],
        source_types=[SourceType.FIQH, SourceType.QURAN, SourceType.HADITH],
    ),
)


def list_templates() -> list[ReasoningTemplate]:
    """Return the reasoning-step templates for common question patterns."""
    return list(_TEMPLATES)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    """Topic words of a phrase, lowercased and stripped of stopwords."""
    return {w for w in _WORD_PATTERN.findall(text.lower()) if len(w) >= 4 and w not in _STOPWORDS}


def split_question(question: str) -> list[str]:
    """Break a compound question into its sub-questions / facets, in order."""
    parts = [p.strip() for p in _SPLIT_PATTERN.split(question) if p and p.strip()]
    # Drop fragments too short to carry a facet (e.g. a stray "it").
    clauses = [p for p in parts if len(p) >= 3]
    return clauses or [question.strip()]


def classify_facet(text: str) -> SourceType:
    """Pick the primary evidence category for a clause by keyword priority."""
    lowered = text.lower()
    for source_type, keywords in _FACET_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return source_type
    return SourceType.GENERAL


def _fiqh_category(text: str) -> str:
    """Coarse fiqh sub-category, used only as an evidence tag."""
    lowered = text.lower()
    if any(k in lowered for k in ("prayer", "salah", "salat", "wudu", "fast", "fasting", "zakat", "hajj")):
        return "worship"
    if any(k in lowered for k in ("sale", "buy", "sell", "trade", "riba", "loan", "contract", "money")):
        return "transactions"
    if any(k in lowered for k in ("marriage", "divorce", "nikah", "talaq")):
        return "family"
    return "general"


def _evidence_for(source_type: SourceType, text: str) -> list[EvidenceRef]:
    """Category-level evidence pointers for a facet. Never an invented citation."""
    if source_type is SourceType.QURAN:
        return [EvidenceRef(source_type=SourceType.QURAN, reference="Qur'an — thematic (quran.com)")]
    if source_type is SourceType.TAFSIR:
        return [
            EvidenceRef(source_type=SourceType.QURAN, reference="Qur'an — the verse in question (quran.com)"),
            EvidenceRef(source_type=SourceType.TAFSIR, reference="Classical tafsir (e.g. Ibn Kathir, Tabari)"),
        ]
    if source_type is SourceType.HADITH:
        return [EvidenceRef(source_type=SourceType.HADITH, reference="Hadith collections (sunnah.com)")]
    if source_type is SourceType.FIQH:
        return [
            EvidenceRef(source_type=SourceType.FIQH, reference=f"fiqh:{_fiqh_category(text)}"),
            EvidenceRef(source_type=SourceType.QURAN, reference="Qur'an — thematic (quran.com)"),
        ]
    return []


def _confidence_for(source_type: SourceType, text: str, supported: bool) -> float:
    """Deterministic 0–1 confidence for a step."""
    score = _FACET_BASE_CONFIDENCE[source_type]
    if not supported:
        score *= _UNSUPPORTED_PENALTY
    if any(marker in text.lower() for marker in _KHILAF_MARKERS):
        score -= _KHILAF_PENALTY
    return round(min(1.0, max(0.0, score)), 4)


def _conclusion_for(source_type: SourceType, clause: str) -> str:
    """A structured (non-fabricated) intermediate conclusion for a facet."""
    facet = clause.rstrip("?.! ").strip()
    labels = {
        SourceType.QURAN: f"On '{facet}', the Qur'anic evidence is weighed",
        SourceType.TAFSIR: f"On '{facet}', the classical tafsir is consulted for the intended meaning",
        SourceType.HADITH: f"On '{facet}', the relevant narrations and their grading are examined",
        SourceType.FIQH: f"On '{facet}', the fiqh principles are applied to reach a ruling",
        SourceType.GENERAL: f"On '{facet}', no specific textual evidence is identified; treat with caution",
    }
    return labels[source_type]


def _assign_connectors(steps: list[ReasoningStep]) -> None:
    """Set each step's connector to the next in place."""
    for index, step in enumerate(steps):
        if index == len(steps) - 1:
            step.connector = Connector.NONE
            continue
        current = _polarity(step.intermediate_conclusion)
        nxt = _polarity(steps[index + 1].intermediate_conclusion)
        if current and nxt and current != nxt:
            step.connector = Connector.HOWEVER
        elif step.source_type == steps[index + 1].source_type:
            step.connector = Connector.THEREFORE
        else:
            step.connector = Connector.AND


def decompose(question: str, branch: str | None = None) -> list[ReasoningStep]:
    """Turn a question into ordered, evidence-tagged reasoning steps."""
    steps: list[ReasoningStep] = []
    for index, clause in enumerate(split_question(question), start=1):
        source_type = classify_facet(clause)
        evidence = _evidence_for(source_type, clause)
        supported = bool(evidence)
        steps.append(
            ReasoningStep(
                id=f"step-{index}",
                order=index,
                facet=clause,
                source_type=source_type,
                intermediate_conclusion=_conclusion_for(source_type, clause),
                evidence=evidence,
                confidence=_confidence_for(source_type, clause, supported),
                supported=supported,
                branch=branch,
            )
        )
    _assign_connectors(steps)
    return steps


def build_branches(question: str, madhhabs: tuple[str, ...] = MADHHABS) -> list[ReasoningBranch]:
    """Produce one parallel reasoning path per madhhab."""
    branches: list[ReasoningBranch] = []
    for madhhab in madhhabs:
        steps = decompose(question, branch=madhhab)
        for step in steps:
            step.id = f"{madhhab}-{step.id}"
        branches.append(ReasoningBranch(madhhab=madhhab, steps=steps))
    return branches


def _needs_branching(question: str, madhhab: str | None) -> bool:
    """A madhhab tag, or an explicit mention of schools, triggers branching."""
    if madhhab:
        return True
    return any(marker in question.lower() for marker in ("madhhab", "madhahib", "schools", "each school"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _polarity(text: str) -> int:
    """+1 if the conclusion permits, -1 if it forbids, 0 if neither."""
    lowered = text.lower()
    if any(marker in lowered for marker in _NEGATIVE_MARKERS):
        return -1
    if any(marker in lowered for marker in _POSITIVE_MARKERS):
        return 1
    return 0


def find_contradictions(steps: list[ReasoningStep]) -> list[ConsistencyIssue]:
    """Flag pairs of steps that reach opposite conclusions on a shared topic."""
    issues: list[ConsistencyIssue] = []
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            first, second = steps[i], steps[j]
            pol_a = _polarity(first.intermediate_conclusion)
            pol_b = _polarity(second.intermediate_conclusion)
            if pol_a == 0 or pol_b == 0 or pol_a == pol_b:
                continue
            shared = _tokens(first.intermediate_conclusion) & _tokens(second.intermediate_conclusion)
            if shared:
                issues.append(
                    ConsistencyIssue(
                        kind="contradiction",
                        step_ids=[first.id, second.id],
                        detail=(
                            f"Steps {first.id} and {second.id} reach opposite conclusions on "
                            f"{', '.join(sorted(shared))}."
                        ),
                    )
                )
    return issues


def _weak_reason(step: ReasoningStep, threshold: float) -> str:
    if not step.supported:
        return "No evidence source is attached to this step."
    if step.confidence < threshold:
        return f"Confidence {step.confidence} is below the {threshold} threshold."
    return "Lowest-confidence step in the chain."


def find_weak_points(steps: list[ReasoningStep], threshold: float = WEAK_CONFIDENCE_THRESHOLD) -> list[WeakPoint]:
    """Surface the lowest-confidence step plus any weak or unsupported steps."""
    if not steps:
        return []
    lowest = min(steps, key=lambda s: s.confidence)
    flagged: dict[str, ReasoningStep] = {lowest.id: lowest}
    for step in steps:
        if step.confidence < threshold or not step.supported:
            flagged[step.id] = step
    ordered = sorted(flagged.values(), key=lambda s: s.confidence)
    return [WeakPoint(step_id=s.id, confidence=s.confidence, reason=_weak_reason(s, threshold)) for s in ordered]


def validate_chain(steps: list[ReasoningStep], threshold: float = WEAK_CONFIDENCE_THRESHOLD) -> ValidationReport:
    """Check a chain for contradictions and weak points."""
    contradictions = find_contradictions(steps)
    issues: list[ConsistencyIssue] = list(contradictions)
    for step in steps:
        if not step.supported:
            issues.append(
                ConsistencyIssue(
                    kind="unsupported",
                    step_ids=[step.id],
                    detail=f"Step {step.id} carries no evidence source.",
                )
            )
        elif step.confidence < threshold:
            issues.append(
                ConsistencyIssue(
                    kind="low_confidence",
                    step_ids=[step.id],
                    detail=f"Step {step.id} has confidence {step.confidence}.",
                )
            )
    overall = round(sum(s.confidence for s in steps) / len(steps), 4) if steps else 0.0
    return ValidationReport(
        consistent=not contradictions,
        issues=issues,
        weak_points=find_weak_points(steps, threshold),
        overall_confidence=overall,
    )


# ---------------------------------------------------------------------------
# Assembly and rendering
# ---------------------------------------------------------------------------


def _summarize_conclusion(steps: list[ReasoningStep], branches: list[ReasoningBranch]) -> str:
    if branches:
        names = ", ".join(branch.madhhab for branch in branches)
        return f"The answer differs by school; parallel reasoning paths are given for: {names}."
    if not steps:
        return "No reasoning could be derived from the question."
    return f"Reasoned across {len(steps)} step(s); the final step concludes: {steps[-1].intermediate_conclusion}."


def build_chain(question: str, madhhab: str | None = None) -> ReasoningChain:
    """Decompose a question into a full, self-validated reasoning chain."""
    steps = decompose(question, branch=madhhab)
    branches = build_branches(question) if _needs_branching(question, madhhab) else []
    validation = validate_chain(steps)
    return ReasoningChain(
        id=f"chain-{abs(hash(question)) % 1_000_000:06d}",
        question=question,
        madhhab=madhhab,
        steps=steps,
        branches=branches,
        conclusion=_summarize_conclusion(steps, branches),
        validation=validation,
    )


def render_markdown(chain: ReasoningChain) -> str:
    """Render a chain as a readable markdown outline with addressable step ids."""
    lines = [f"# Reasoning for: {chain.question}", ""]
    for step in chain.steps:
        sources = ", ".join(ref.reference for ref in step.evidence) or "none"
        lines.append(f"- **[{step.id}]** ({step.source_type.value}, confidence {step.confidence})")
        lines.append(f"  - {step.intermediate_conclusion}")
        lines.append(f"  - evidence: {sources}")
        if step.connector is not Connector.NONE:
            lines.append(f"  - _{step.connector.value}_ →")
    if chain.branches:
        lines.append("")
        lines.append("## Madhhab branches")
        for branch in chain.branches:
            lines.append(f"### {branch.madhhab.capitalize()}")
            for step in branch.steps:
                lines.append(f"- **[{step.id}]** {step.intermediate_conclusion} (confidence {step.confidence})")
    lines.append("")
    lines.append(f"**Conclusion:** {chain.conclusion}")
    if chain.validation.weak_points:
        weakest = chain.validation.weak_points[0]
        lines.append(f"**Scrutinise first:** {weakest.step_id} — {weakest.reason}")
    return "\n".join(lines)


def _submission_to_step(submission: StepSubmission, index: int) -> ReasoningStep:
    supported = bool(submission.evidence)
    return ReasoningStep(
        id=submission.id or f"step-{index}",
        order=index,
        facet="",
        source_type=submission.evidence[0].source_type if submission.evidence else SourceType.GENERAL,
        intermediate_conclusion=submission.intermediate_conclusion,
        evidence=submission.evidence,
        confidence=submission.confidence,
        supported=supported,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/chain", response_model=ChainResponse)
async def create_chain(request: ChainRequest) -> ChainResponse:
    """Decompose a question into a full multi-step reasoning chain."""
    chain = build_chain(request.question, madhhab=request.madhhab)
    return ChainResponse(chain=chain, outline=render_markdown(chain))


@router.get("/templates", response_model=TemplatesResponse)
async def get_templates() -> TemplatesResponse:
    """Reasoning-step templates for common Islamic question patterns."""
    templates = list_templates()
    return TemplatesResponse(count=len(templates), templates=templates)


@router.post("/validate", response_model=ValidationReport)
async def validate_submitted_chain(request: ValidateRequest) -> ValidationReport:
    """Check a submitted chain for consistency and weak points."""
    steps = [_submission_to_step(submission, index) for index, submission in enumerate(request.steps, start=1)]
    return validate_chain(steps)
