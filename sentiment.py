"""Religious sentiment analysis — a deterministic, dependency-free read of the
emotional and spiritual dimension of a user's question.

Why this exists
---------------
A question is never just its information content. "What breaks wudu?" and
"I keep messing up my wudu and I feel like Allah is disgusted with me — what
breaks it?" ask for the same ruling, but the second needs a gentler answer and,
if the distress runs deep enough, a nudge toward a real human. This module reads
that emotional/spiritual layer so the rest of the service can answer with the
right *tone* without ever bending the underlying facts.

How it works
------------
No model, no training data, no ML dependency — just curated keyword/phrase
lexicons and a handful of weighted, negation-aware rules. That makes it
deterministic (the same text always scores the same), fast, and safe to import
in a request path with no live services behind it.

Three normalized dimensions are produced for every text:

- ``emotional``    — distress, comfort-seeking, loneliness, grief.
- ``spiritual``    — doubt, crisis of faith, feeling distant from Allah.
- ``informational`` — plain fact-finding ("what is", "how many", "ruling on").

Alongside the scores sit boolean flags (new-Muslim, urgency, seeking validation
vs guidance, cultural sensitivity, learning enthusiasm vs obligation …), a single
recommended response *tone*, and — most importantly — pastoral-care indicators
that fire when self-harm / crisis language crosses a threshold, so the caller can
surface a referral to a qualified human rather than a chatbot answer.

Nothing here diagnoses anyone. The crisis signal is intentionally conservative
and errs toward offering help; it is a routing hint, not a clinical judgement.
"""

import re

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# A dimension's raw weighted hit-sum is squashed into 0–1 by ``s / (s + SATURATION)``.
# Lower saturation → the score climbs faster with each match.
SATURATION = 3.0

# Crisis referral fires once the crisis dimension crosses this normalized score.
CRISIS_THRESHOLD = 0.20

# Distress / doubt flags fire once their dimension crosses this normalized score.
DISTRESS_THRESHOLD = 0.25
DOUBT_THRESHOLD = 0.20

# Words that flip the meaning of a following term within a small window.
NEGATIONS = ("not", "no", "never", "without", "isn't", "aren't", "don't", "doesn't", "didn't", "won't", "can't")
NEGATION_WINDOW = 3

_TOKEN_RE = re.compile(r"[a-z']+")

# ---------------------------------------------------------------------------
# Lexicons — (phrase, weight). Phrases may be multi-word; they are matched
# against the normalized text. Weights reflect how strongly a phrase signals
# its category.
# ---------------------------------------------------------------------------

CRISIS_TERMS: tuple[tuple[str, float], ...] = (
    ("want to die", 5.0),
    ("kill myself", 5.0),
    ("killing myself", 5.0),
    ("end my life", 5.0),
    ("ending my life", 5.0),
    ("suicide", 5.0),
    ("suicidal", 5.0),
    ("self harm", 4.0),
    ("self-harm", 4.0),
    ("hurt myself", 4.0),
    ("harming myself", 4.0),
    ("no reason to live", 5.0),
    ("cant go on", 4.0),
    ("can't go on", 4.0),
    ("dont want to live", 4.0),
    ("nothing to live for", 4.0),
    ("better off dead", 5.0),
)

EMOTIONAL_DISTRESS_TERMS: tuple[tuple[str, float], ...] = (
    ("depressed", 2.0),
    ("depression", 2.0),
    ("anxious", 1.5),
    ("anxiety", 1.5),
    ("panic", 1.5),
    ("scared", 1.5),
    ("afraid", 1.5),
    ("terrified", 2.0),
    ("hopeless", 2.5),
    ("worthless", 2.5),
    ("overwhelmed", 1.5),
    ("cant cope", 2.0),
    ("can't cope", 2.0),
    ("breaking down", 2.0),
    ("falling apart", 2.0),
    ("crying", 1.5),
    ("in tears", 1.5),
    ("grief", 1.5),
    ("grieving", 1.5),
    ("heartbroken", 1.5),
    ("suffering", 1.5),
    ("in pain", 1.5),
    ("hurting", 1.0),
    ("broken", 1.0),
    ("alone", 1.0),
    ("lonely", 1.5),
    ("loneliness", 1.5),
    ("empty inside", 1.5),
    ("stressed", 1.0),
    ("worried", 1.0),
    ("exhausted", 1.0),
    ("miserable", 1.5),
    ("cant sleep", 1.0),
    ("lost my", 1.5),
    ("passed away", 1.5),
    ("i feel so", 1.0),
)

COMFORT_SEEKING_TERMS: tuple[tuple[str, float], ...] = (
    ("comfort", 2.0),
    ("make dua for me", 2.5),
    ("pray for me", 2.0),
    ("please pray", 2.0),
    ("help me feel", 2.0),
    ("feel better", 2.0),
    ("how do i cope", 2.0),
    ("how to cope", 2.0),
    ("need support", 2.0),
    ("need help", 1.0),
    ("reassurance", 2.0),
    ("reassure me", 2.0),
    ("give me hope", 2.0),
    ("ease my heart", 2.0),
    ("calm my heart", 2.0),
    ("i need someone", 2.0),
)

SPIRITUAL_CRISIS_TERMS: tuple[tuple[str, float], ...] = (
    ("losing my faith", 3.0),
    ("lost my faith", 3.0),
    ("losing faith", 2.5),
    ("crisis of faith", 3.0),
    ("no longer believe", 3.0),
    ("dont believe anymore", 3.0),
    ("stopped believing", 3.0),
    ("allah abandoned me", 3.0),
    ("allah hates me", 3.0),
    ("allah is punishing me", 2.5),
    ("god abandoned me", 3.0),
    ("far from allah", 2.0),
    ("distant from allah", 2.0),
    ("feel empty in prayer", 2.0),
    ("dua never answered", 2.0),
    ("why does allah", 1.5),
    ("why would allah", 1.5),
    ("is there even a god", 3.0),
    ("does allah even care", 2.5),
    ("hate myself as a muslim", 2.5),
)

DOUBT_TERMS: tuple[tuple[str, float], ...] = (
    ("i doubt", 2.0),
    ("having doubts", 2.5),
    ("my doubts", 2.0),
    ("doubting", 2.0),
    ("unsure if", 1.5),
    ("not sure if", 1.5),
    ("confused about", 1.5),
    ("struggling to believe", 2.5),
    ("hard to believe", 1.5),
    ("questioning my faith", 2.5),
    ("questioning islam", 2.5),
    ("is islam true", 2.0),
    ("is it even true", 2.0),
    ("how do i know", 1.5),
    ("how can i be sure", 1.5),
    ("cant tell if", 1.0),
)

INFORMATIONAL_TERMS: tuple[tuple[str, float], ...] = (
    ("what is", 1.5),
    ("what are", 1.5),
    ("what breaks", 1.5),
    ("how many", 1.5),
    ("how much", 1.5),
    ("how do i perform", 1.5),
    ("how to perform", 1.5),
    ("when is", 1.5),
    ("when does", 1.5),
    ("where is", 1.0),
    ("who was", 1.5),
    ("ruling on", 2.0),
    ("is it permissible", 2.0),
    ("is it halal", 2.0),
    ("is it haram", 2.0),
    ("permissible to", 1.5),
    ("definition of", 2.0),
    ("meaning of", 2.0),
    ("explain", 1.0),
    ("difference between", 1.5),
    ("how do you", 1.0),
    ("what does the quran say", 1.5),
)

URGENCY_TERMS: tuple[tuple[str, float], ...] = (
    ("urgent", 2.0),
    ("urgently", 2.0),
    ("emergency", 2.5),
    ("immediately", 2.0),
    ("right now", 2.0),
    ("as soon as possible", 2.0),
    ("asap", 2.0),
    ("quickly", 1.0),
    ("before maghrib", 1.5),
    ("before the prayer", 1.5),
    ("need to know now", 2.0),
    ("time sensitive", 1.5),
    ("cant wait", 1.5),
)

LEARNING_ENTHUSIASM_TERMS: tuple[tuple[str, float], ...] = (
    ("i want to learn", 2.0),
    ("want to understand", 1.5),
    ("eager to", 2.0),
    ("excited to", 2.0),
    ("love learning", 2.0),
    ("keen to", 1.5),
    ("curious about", 1.5),
    ("fascinated by", 2.0),
    ("really enjoy", 1.5),
    ("passionate about", 1.5),
)

OBLIGATION_TERMS: tuple[tuple[str, float], ...] = (
    ("i have to", 1.5),
    ("i must", 1.0),
    ("required to", 1.5),
    ("obligated to", 2.0),
    ("supposed to", 1.5),
    ("forced to", 2.0),
    ("i am told i must", 2.0),
    ("do i really have to", 1.5),
)

CULTURAL_TERMS: tuple[tuple[str, float], ...] = (
    ("in my culture", 2.5),
    ("my culture", 2.0),
    ("cultural", 1.5),
    ("my family", 1.5),
    ("my parents", 1.5),
    ("my country", 1.5),
    ("back home", 1.5),
    ("our tradition", 2.0),
    ("our custom", 2.0),
    ("customary", 1.5),
    ("in my community", 1.5),
)

NEW_MUSLIM_TERMS: tuple[tuple[str, float], ...] = (
    ("new muslim", 3.0),
    ("new to islam", 3.0),
    ("recently became muslim", 3.0),
    ("just became muslim", 3.0),
    ("recently converted", 3.0),
    ("just converted", 3.0),
    ("i reverted", 3.0),
    ("i am a revert", 3.0),
    ("as a revert", 2.5),
    ("as a convert", 2.5),
    ("took my shahada", 3.0),
    ("said my shahada", 2.5),
    ("accepted islam", 2.5),
    ("still learning the basics", 2.0),
)

VALIDATION_TERMS: tuple[tuple[str, float], ...] = (
    ("am i right", 2.0),
    ("did i do the right thing", 2.5),
    ("was i wrong", 2.0),
    ("is it okay that i", 2.0),
    ("tell me its fine", 2.0),
    ("tell me it's fine", 2.0),
    ("am i a bad muslim", 2.5),
    ("am i sinful", 2.0),
    ("did i sin", 2.0),
    ("is my sin forgivable", 2.0),
    ("reassure me that", 2.0),
    ("please tell me im", 1.5),
)

GUIDANCE_TERMS: tuple[tuple[str, float], ...] = (
    ("what should i do", 2.0),
    ("how should i", 1.5),
    ("guide me", 2.0),
    ("advise me", 2.0),
    ("what do you advise", 2.0),
    ("help me decide", 2.0),
    ("what is the right thing", 2.0),
    ("how do i move forward", 2.0),
    ("what steps", 1.5),
)

# ---------------------------------------------------------------------------
# Emotion taxonomy — the human-readable map the /taxonomy endpoint returns.
# ---------------------------------------------------------------------------

EMOTION_TAXONOMY: dict[str, dict[str, str]] = {
    "emotional_distress": {
        "dimension": "emotional",
        "description": "Sadness, anxiety, fear, grief or loneliness expressed in the question.",
    },
    "comfort_seeking": {
        "dimension": "emotional",
        "description": "The person is asking for reassurance, dua or emotional support, not only facts.",
    },
    "spiritual_crisis": {
        "dimension": "spiritual",
        "description": "Feeling abandoned by Allah, loss of faith, or a collapse of religious meaning.",
    },
    "faith_doubt": {
        "dimension": "spiritual",
        "description": "Uncertainty or intellectual questioning about the truth of the faith.",
    },
    "information_seeking": {
        "dimension": "informational",
        "description": "A factual or fiqh question seeking a clear, sourced answer.",
    },
    "urgency": {
        "dimension": "informational",
        "description": "Time pressure on a personal fiqh matter that needs a prompt answer.",
    },
    "learning_enthusiasm": {
        "dimension": "emotional",
        "description": "Eagerness and positive curiosity about learning the deen.",
    },
    "obligation": {
        "dimension": "emotional",
        "description": "The question is framed around duty or compulsion rather than interest.",
    },
    "cultural_sensitivity": {
        "dimension": "informational",
        "description": "Culture, family or local custom is entangled with the religious question.",
    },
    "new_muslim": {
        "dimension": "informational",
        "description": "A revert / new Muslim who may need foundational, jargon-light answers.",
    },
    "seeking_validation": {
        "dimension": "emotional",
        "description": "Asking to be told they did the right thing or are not a bad Muslim.",
    },
    "seeking_guidance": {
        "dimension": "informational",
        "description": "Asking what to do next — actionable direction rather than reassurance.",
    },
    "crisis": {
        "dimension": "emotional",
        "description": "Self-harm or suicidal language; triggers a pastoral-care referral.",
    },
}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The user's question or message to analyze.",
        json_schema_extra={"examples": ["I feel so far from Allah lately and I don't know how to pray again."]},
    )


class DimensionScores(BaseModel):
    emotional: float = Field(..., ge=0.0, le=1.0, description="Emotional charge (distress, comfort-seeking).")
    spiritual: float = Field(..., ge=0.0, le=1.0, description="Spiritual charge (doubt, crisis of faith).")
    informational: float = Field(..., ge=0.0, le=1.0, description="Plain fact-finding intent.")


class SentimentFlags(BaseModel):
    emotional_distress: bool = Field(..., description="Distress language crossed the distress threshold.")
    spiritual_crisis: bool = Field(..., description="Crisis-of-faith language detected.")
    faith_doubt: bool = Field(..., description="Doubt / uncertainty about the faith detected.")
    comfort_seeking: bool = Field(..., description="Wants reassurance / support more than facts.")
    information_seeking: bool = Field(..., description="Primarily a factual or fiqh question.")
    urgency: bool = Field(..., description="Time pressure on a personal fiqh matter.")
    learning_enthusiasm: bool = Field(..., description="Positive eagerness to learn.")
    obligation: bool = Field(..., description="Framed around duty / compulsion.")
    cultural_sensitivity: bool = Field(..., description="Culture / family / custom is entangled with the question.")
    new_muslim: bool = Field(..., description="Revert / new-Muslim indicators present.")
    seeking_validation: bool = Field(..., description="Wants to be told they did the right thing.")
    seeking_guidance: bool = Field(..., description="Wants actionable direction on what to do.")


class CareIndicators(BaseModel):
    crisis_detected: bool = Field(..., description="Self-harm / suicidal language crossed the crisis threshold.")
    referral_recommended: bool = Field(..., description="A referral to a qualified human is recommended.")
    severity: str = Field(..., description="One of 'none', 'elevated', 'high'.")
    triggers: list[str] = Field(default_factory=list, description="The phrases that raised the care signal.")
    message: str | None = Field(default=None, description="A ready-to-surface pastoral-care / referral note.")


class ToneGuidance(BaseModel):
    tone: str = Field(..., description="The single recommended response tone.")
    guidance: str = Field(..., description="How to shape the reply for this person.")
    response_prefix: str | None = Field(default=None, description="Optional empathetic opener to prepend.")
    preserve_information_note: str = Field(
        ...,
        description="Reminder that empathy must not distort the accuracy of the religious answer.",
    )


class SentimentAnalysis(BaseModel):
    dimensions: DimensionScores
    flags: SentimentFlags
    primary_intent: str = Field(..., description="'comfort', 'information', 'spiritual_support' or 'mixed'.")
    recommended_tone: str = Field(..., description="Convenience copy of tone.tone.")
    tone: ToneGuidance
    care: CareIndicators
    matched_terms: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Phrases matched per category, for transparency / debugging.",
    )


# ---------------------------------------------------------------------------
# Matching primitives
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace so phrase matching is stable."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_negated(normalized: str, match_start: int) -> bool:
    """Return ``True`` if a negation word sits just before ``match_start``.

    Cheap, window-bounded negation handling: we only look at the few tokens
    immediately preceding the matched phrase, which catches "I do not feel
    hopeless" without the cost of real parsing.
    """
    preceding = normalized[:match_start]
    tokens = _TOKEN_RE.findall(preceding)
    window = tokens[-NEGATION_WINDOW:]
    return any(tok in NEGATIONS for tok in window)


def score_terms(normalized: str, terms: tuple[tuple[str, float], ...]) -> tuple[float, list[str]]:
    """Sum the weights of negation-aware phrase matches in ``normalized``.

    Returns ``(raw_weighted_sum, matched_phrases)``. A phrase preceded by a
    negation is skipped rather than counted.
    """
    total = 0.0
    matched: list[str] = []
    for phrase, weight in terms:
        start = normalized.find(phrase)
        if start == -1:
            continue
        if _is_negated(normalized, start):
            continue
        total += weight
        matched.append(phrase)
    return total, matched


def _saturate(raw: float) -> float:
    """Squash a raw weighted sum into 0–1 with diminishing returns."""
    if raw <= 0.0:
        return 0.0
    return round(raw / (raw + SATURATION), 4)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze(text: str) -> SentimentAnalysis:
    """Produce a full multi-dimensional sentiment analysis for ``text``.

    Deterministic and side-effect free: the same input always yields the same
    output, and nothing external is called.
    """
    normalized = _normalize(text)

    crisis_raw, crisis_hits = score_terms(normalized, CRISIS_TERMS)
    distress_raw, distress_hits = score_terms(normalized, EMOTIONAL_DISTRESS_TERMS)
    comfort_raw, comfort_hits = score_terms(normalized, COMFORT_SEEKING_TERMS)
    spiritual_raw, spiritual_hits = score_terms(normalized, SPIRITUAL_CRISIS_TERMS)
    doubt_raw, doubt_hits = score_terms(normalized, DOUBT_TERMS)
    info_raw, info_hits = score_terms(normalized, INFORMATIONAL_TERMS)
    urgency_raw, urgency_hits = score_terms(normalized, URGENCY_TERMS)
    enthusiasm_raw, enthusiasm_hits = score_terms(normalized, LEARNING_ENTHUSIASM_TERMS)
    obligation_raw, obligation_hits = score_terms(normalized, OBLIGATION_TERMS)
    cultural_raw, cultural_hits = score_terms(normalized, CULTURAL_TERMS)
    new_muslim_raw, new_muslim_hits = score_terms(normalized, NEW_MUSLIM_TERMS)
    validation_raw, validation_hits = score_terms(normalized, VALIDATION_TERMS)
    guidance_raw, guidance_hits = score_terms(normalized, GUIDANCE_TERMS)

    # Dimension roll-ups. Crisis is a hard escalator of the emotional dimension.
    emotional_score = _saturate(distress_raw + comfort_raw + validation_raw + 2.0 * crisis_raw)
    spiritual_score = _saturate(spiritual_raw + doubt_raw)
    informational_score = _saturate(info_raw + urgency_raw + guidance_raw)
    crisis_score = _saturate(crisis_raw)

    dimensions = DimensionScores(
        emotional=emotional_score,
        spiritual=spiritual_score,
        informational=informational_score,
    )

    flags = SentimentFlags(
        emotional_distress=_saturate(distress_raw + 2.0 * crisis_raw) >= DISTRESS_THRESHOLD,
        spiritual_crisis=spiritual_raw > 0.0,
        faith_doubt=_saturate(doubt_raw) >= DOUBT_THRESHOLD,
        comfort_seeking=comfort_raw > 0.0,
        information_seeking=info_raw > 0.0,
        urgency=urgency_raw > 0.0,
        learning_enthusiasm=enthusiasm_raw > 0.0,
        obligation=obligation_raw > 0.0,
        cultural_sensitivity=cultural_raw > 0.0,
        new_muslim=new_muslim_raw > 0.0,
        seeking_validation=validation_raw > 0.0,
        seeking_guidance=guidance_raw > 0.0,
    )

    matched_terms: dict[str, list[str]] = {}
    for name, hits in (
        ("crisis", crisis_hits),
        ("emotional_distress", distress_hits),
        ("comfort_seeking", comfort_hits),
        ("spiritual_crisis", spiritual_hits),
        ("faith_doubt", doubt_hits),
        ("information_seeking", info_hits),
        ("urgency", urgency_hits),
        ("learning_enthusiasm", enthusiasm_hits),
        ("obligation", obligation_hits),
        ("cultural_sensitivity", cultural_hits),
        ("new_muslim", new_muslim_hits),
        ("seeking_validation", validation_hits),
        ("seeking_guidance", guidance_hits),
    ):
        if hits:
            matched_terms[name] = hits

    care = _assess_care(crisis_score, crisis_hits, spiritual_score)
    primary_intent = _primary_intent(emotional_score, spiritual_score, informational_score, comfort_raw > 0.0)
    tone = adapt_tone_from_parts(dimensions, flags, care, primary_intent)

    return SentimentAnalysis(
        dimensions=dimensions,
        flags=flags,
        primary_intent=primary_intent,
        recommended_tone=tone.tone,
        tone=tone,
        care=care,
        matched_terms=matched_terms,
    )


def _assess_care(crisis_score: float, crisis_hits: list[str], spiritual_score: float) -> CareIndicators:
    """Decide whether the message warrants a pastoral-care / crisis referral."""
    referral_message = (
        "This message may reflect serious emotional distress. Respond with warmth and without "
        "judgement, and gently encourage the person to reach out to a trusted imam, a qualified "
        "counsellor, or a local emergency / crisis helpline. Do not attempt to handle a crisis "
        "with an automated answer alone."
    )
    if crisis_score >= CRISIS_THRESHOLD:
        return CareIndicators(
            crisis_detected=True,
            referral_recommended=True,
            severity="high",
            triggers=crisis_hits,
            message=referral_message,
        )
    if spiritual_score >= 0.5:
        return CareIndicators(
            crisis_detected=False,
            referral_recommended=True,
            severity="elevated",
            triggers=[],
            message=(
                "This message suggests a painful crisis of faith. Answer gently and consider "
                "pointing the person toward a compassionate, qualified scholar or mentor."
            ),
        )
    return CareIndicators(
        crisis_detected=False,
        referral_recommended=False,
        severity="none",
        triggers=[],
        message=None,
    )


def _primary_intent(emotional: float, spiritual: float, informational: float, comfort_seeking: bool) -> str:
    """Classify the dominant intent behind the message."""
    scores = {"comfort": emotional, "spiritual_support": spiritual, "information": informational}
    top = max(scores, key=lambda k: scores[k])
    top_value = scores[top]
    if top_value == 0.0:
        return "information"
    # If two dimensions are close, call it mixed rather than forcing one.
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) >= 2 and ordered[0] > 0.0 and (ordered[0] - ordered[1]) < 0.1:
        return "mixed"
    if top == "comfort" and not comfort_seeking and emotional < DISTRESS_THRESHOLD:
        return "information"
    return top


# ---------------------------------------------------------------------------
# Tone adaptation
# ---------------------------------------------------------------------------

_PRESERVE_NOTE = (
    "Lead with empathy, but keep the religious content accurate: do not soften, exaggerate or "
    "invent rulings to make the person feel better. Comfort and correctness are both obligations."
)


def adapt_tone_from_parts(
    dimensions: DimensionScores,
    flags: SentimentFlags,
    care: CareIndicators,
    primary_intent: str,
) -> ToneGuidance:
    """Map a completed analysis onto a single recommended tone and opener."""
    if care.crisis_detected:
        return ToneGuidance(
            tone="compassionate_urgent_referral",
            guidance=(
                "Prioritize the person's safety and dignity over information delivery. Be warm, calm "
                "and non-judgemental, affirm that they are not alone, and steer them toward immediate "
                "human help before anything else."
            ),
            response_prefix=(
                "I'm really glad you reached out, and I'm concerned about how much pain you're in. "
                "You deserve real support from people who can be there for you right now."
            ),
            preserve_information_note=_PRESERVE_NOTE,
        )
    if care.severity == "elevated" or flags.spiritual_crisis:
        return ToneGuidance(
            tone="gentle_reassuring",
            guidance=(
                "Meet the doubt or spiritual pain without alarm or rebuke. Normalize that struggling "
                "is part of faith, offer hope, and answer honestly while pointing toward a trusted scholar."
            ),
            response_prefix="Please know that wrestling with these feelings does not make you any less beloved to Allah.",
            preserve_information_note=_PRESERVE_NOTE,
        )
    if flags.emotional_distress or dimensions.emotional >= DISTRESS_THRESHOLD:
        return ToneGuidance(
            tone="empathetic_supportive",
            guidance="Acknowledge the difficulty first, then answer plainly. Keep the language soft and unhurried.",
            response_prefix="I'm sorry you're going through this — that sounds genuinely hard.",
            preserve_information_note=_PRESERVE_NOTE,
        )
    if flags.faith_doubt:
        return ToneGuidance(
            tone="patient_clarifying",
            guidance="Treat the questioning as sincere. Answer with evidence and patience, never with judgement.",
            response_prefix="That's a thoughtful question, and it's completely okay to ask it.",
            preserve_information_note=_PRESERVE_NOTE,
        )
    if flags.new_muslim:
        return ToneGuidance(
            tone="welcoming_foundational",
            guidance="Assume little prior knowledge. Avoid jargon, define terms, and be encouraging about the journey.",
            response_prefix="Welcome — it's wonderful that you're learning, and there's no such thing as a silly question.",
            preserve_information_note=_PRESERVE_NOTE,
        )
    if flags.seeking_validation:
        return ToneGuidance(
            tone="reassuring_honest",
            guidance="Offer kindness, but base any reassurance on what is actually true rather than blanket comfort.",
            response_prefix="I can hear that you want to do the right thing, which already says a lot.",
            preserve_information_note=_PRESERVE_NOTE,
        )
    if flags.learning_enthusiasm:
        return ToneGuidance(
            tone="encouraging",
            guidance="Match the person's energy. Give a clear answer and invite them to explore further.",
            response_prefix="I love the enthusiasm!",
            preserve_information_note=_PRESERVE_NOTE,
        )
    if flags.urgency and primary_intent == "information":
        return ToneGuidance(
            tone="direct_practical",
            guidance="Give the actionable answer first and concisely; save nuance and background for afterward.",
            response_prefix=None,
            preserve_information_note=_PRESERVE_NOTE,
        )
    if primary_intent == "information":
        return ToneGuidance(
            tone="clear_informative",
            guidance="A neutral, well-sourced, respectful factual answer is appropriate here.",
            response_prefix=None,
            preserve_information_note=_PRESERVE_NOTE,
        )
    return ToneGuidance(
        tone="respectful_neutral",
        guidance="No strong emotional signal detected; answer respectfully and clearly.",
        response_prefix=None,
        preserve_information_note=_PRESERVE_NOTE,
    )


def adapt_tone(analysis: SentimentAnalysis) -> ToneGuidance:
    """Public helper: return tone guidance for an already-computed analysis."""
    return analysis.tone


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=SentimentAnalysis)
async def analyze_sentiment(request: AnalyzeRequest) -> SentimentAnalysis:
    """Analyze the emotional and spiritual sentiment of a user's message.

    Returns multi-dimensional scores, detected flags, a recommended response
    tone, and pastoral-care indicators (with a referral note when self-harm or
    crisis language is present).
    """
    return analyze(request.text)


@router.get("/taxonomy")
async def get_taxonomy() -> dict[str, dict[str, dict[str, str]]]:
    """Return the emotion taxonomy this analyzer uses."""
    return {"taxonomy": EMOTION_TAXONOMY}
