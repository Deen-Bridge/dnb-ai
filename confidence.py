"""Answer confidence, abstention policy, and scholar escalation.

Why this exists
---------------
The assistant had exactly one behaviour: answer, always, with full confidence.
For a service giving Islamic guidance, "confidently wrong" is the worst possible
outcome — worse than "I don't know", and far worse than "ask a scholar". This
module gives every answer a confidence score, decides whether to answer, hedge,
or abstain, and marks low-confidence *religious* answers for human review.

The score
---------
One documented formula, one place. Confidence is a weighted mean of whatever
signals are available for a turn, so a component that has not run simply drops
out of the average rather than being guessed at:

    base  = Σ(wᵢ · sᵢ) / Σ(wᵢ)        over signals that are present
    capped = min(base, UNVERIFIED_CEILING) if no external signal ran else base
    score = capped · (1 − HIGH_STAKES_PENALTY if the question is high-stakes)

Signals (each 0–1, higher = more reliable):

- ``self_consistency`` — agreement across sampled answers. **Produced by the
  self-consistency work (#ai-18) and passed in**; this module never recomputes
  it, so the two cannot drift apart.
- ``citation_verification`` — share of the answer's citations that verified
  against a real corpus (#40), passed in the same way.
- ``expressed_certainty`` — derived here from the answer's own hedging
  language, because it is a property of the text and nothing else computes it.

``is_high_stakes`` is deliberately *not* a fourth signal. It comes from intent
classification (the ``fiqh`` classifier today, #42's richer intent hook when it
lands) and applies once, as a multiplier: the same evidence should support less
confidence when being wrong means issuing a wrong ruling. Treating it as both a
signal and a modifier would double-count it.

With no signals at all, the score is ``NO_SIGNAL_PRIOR`` — a deliberately
middling value, so an unverified answer lands in the hedge band rather than
sailing through as if it had been checked. The same reasoning gives
``UNVERIFIED_CEILING`` its job: expressed certainty is the model's opinion of
itself, so an answer nothing external has corroborated is capped below the
confident band no matter how fluently it is written.

Bands
-----
- ``score < CONFIDENCE_LOW_THRESHOLD`` → **abstain**: no answer, a pointer to a
  qualified scholar and authenticated sources.
- ``< CONFIDENCE_HIGH_THRESHOLD`` → **uncertain**: answer, with an explicit
  uncertainty note appended.
- otherwise → **confident**: answer as-is.

Thresholds are environment-configurable. Religious answers that land in the
abstain band (by default) are queued for scholar review; non-religious ones are
hedged but never sent to a scholar, whose time is for religious content.
"""

from __future__ import annotations

import logging
import os
import re
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Read a 0–1 threshold from the environment, falling back on nonsense."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if not 0.0 <= value <= 1.0:
        logger.warning("%s=%s is outside 0–1; using %s", name, value, default)
        return default
    return value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIDENCE_LOW_THRESHOLD = _env_float("CONFIDENCE_LOW_THRESHOLD", 0.40)
CONFIDENCE_HIGH_THRESHOLD = _env_float("CONFIDENCE_HIGH_THRESHOLD", 0.70)

# Religious answers at or below this score go to the scholar queue. Defaults to
# the abstain threshold: what the service would not say on its own is exactly
# what a scholar should look at.
SCHOLAR_QUEUE_THRESHOLD = _env_float(
    "SCHOLAR_QUEUE_THRESHOLD", CONFIDENCE_LOW_THRESHOLD
)

HIGH_STAKES_PENALTY = _env_float("CONFIDENCE_HIGH_STAKES_PENALTY", 0.15)
NO_SIGNAL_PRIOR = _env_float("CONFIDENCE_NO_SIGNAL_PRIOR", 0.55)

# Ceiling on an answer with no *external* corroboration. Expressed certainty is
# the model's own opinion of itself; a fluent answer that nothing has checked
# must not certify itself into the confident band, so with neither
# self-consistency nor citation verification present the score is capped below
# CONFIDENCE_HIGH_THRESHOLD and the answer is hedged.
UNVERIFIED_CEILING = _env_float("CONFIDENCE_UNVERIFIED_CEILING", 0.65)

# Signals that constitute external corroboration, as opposed to the answer's
# own account of itself.
EXTERNAL_SIGNALS = ("self_consistency", "citation_verification")

SIGNAL_WEIGHTS: Dict[str, float] = {
    "self_consistency": 0.40,
    "citation_verification": 0.30,
    "expressed_certainty": 0.30,
}

if CONFIDENCE_LOW_THRESHOLD > CONFIDENCE_HIGH_THRESHOLD:
    logger.warning(
        "CONFIDENCE_LOW_THRESHOLD (%s) exceeds CONFIDENCE_HIGH_THRESHOLD (%s); "
        "every answer will abstain",
        CONFIDENCE_LOW_THRESHOLD,
        CONFIDENCE_HIGH_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Hedging detection
# ---------------------------------------------------------------------------

# Phrases in which the model expresses its own doubt. Deliberately narrow:
# "consult a scholar" is good adab, not doubt, and does not count.
HEDGE_PATTERNS: tuple[str, ...] = (
    r"\bi(?:'m| am) not (?:sure|certain)\b",
    r"\bi (?:don't|do not) know\b",
    r"\bi (?:can(?:'t|not)) (?:verify|confirm|recall)\b",
    r"\bi (?:think|believe|suspect)\b",
    r"\bnot (?:entirely |completely )?(?:sure|certain|clear)\b",
    r"\bunclear\b",
    r"\buncertain\b",
    r"\bit (?:may|might|could) be\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bas far as i (?:know|recall)\b",
    r"\bi'?m not able to\b",
    r"\bcannot be verified\b",
    r"\bmay not be accurate\b",
)

_HEDGE_REGEXES = tuple(re.compile(p, re.IGNORECASE) for p in HEDGE_PATTERNS)


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back on nonsense."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s=%s must be at least 1; using %s", name, value, default)
        return default
    return value


# Number of hedges at which expressed certainty bottoms out at 0.
HEDGE_SATURATION = _env_int("CONFIDENCE_HEDGE_SATURATION", 3)


def count_hedges(text: str) -> int:
    """Count distinct hedging expressions in *text*."""
    if not text:
        return 0
    return sum(1 for regex in _HEDGE_REGEXES if regex.search(text))


def expressed_certainty(text: str) -> float:
    """Score how confidently the answer states itself, from 1.0 down to 0.0."""
    if not (text or "").strip():
        return 0.0
    hedges = count_hedges(text)
    if hedges == 0:
        return 1.0
    return max(0.0, 1.0 - hedges / max(1, HEDGE_SATURATION))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class ConfidenceBand(str, Enum):
    ABSTAIN = "abstain"
    UNCERTAIN = "uncertain"
    CONFIDENT = "confident"


class ConfidenceSignals(BaseModel):
    """Inputs to the score. All signals are optional; absent ones are skipped.

    ``self_consistency`` and ``citation_verification`` are produced elsewhere
    (#ai-18 and #40) and passed in — this module is their consumer, not a second
    implementation.
    """

    self_consistency: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Agreement across sampled answers (#ai-18)"
    )
    citation_verification: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Share of citations that verified (#40)"
    )
    expressed_certainty: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Inverse of the answer's own hedging"
    )
    is_religious: bool = False
    is_high_stakes: bool = False

    def present(self) -> Dict[str, float]:
        return {
            name: value
            for name, value in (
                ("self_consistency", self.self_consistency),
                ("citation_verification", self.citation_verification),
                ("expressed_certainty", self.expressed_certainty),
            )
            if value is not None
        }


class ConfidenceAssessment(BaseModel):
    """The confidence block attached to a chat answer."""

    score: float = Field(..., ge=0.0, le=1.0)
    band: ConfidenceBand
    abstained: bool
    queued: bool
    signals: Dict[str, float] = {}
    signals_used: List[str] = []
    review_id: Optional[str] = None


def compute_confidence(signals: ConfidenceSignals) -> float:
    """Fuse the available signals into a single 0–1 score.

    The one place a confidence number is produced. Callers that have their own
    signal (self-consistency agreement, citation checks) pass it in here rather
    than deriving a competing score of their own.
    """
    present = signals.present()
    if present:
        total_weight = sum(SIGNAL_WEIGHTS[name] for name in present)
        base = sum(SIGNAL_WEIGHTS[name] * value for name, value in present.items())
        score = base / total_weight
    else:
        score = NO_SIGNAL_PRIOR

    if not any(name in present for name in EXTERNAL_SIGNALS):
        score = min(score, UNVERIFIED_CEILING)

    if signals.is_high_stakes:
        score *= 1.0 - HIGH_STAKES_PENALTY

    return round(min(1.0, max(0.0, score)), 4)


def band_for(score: float) -> ConfidenceBand:
    if score < CONFIDENCE_LOW_THRESHOLD:
        return ConfidenceBand.ABSTAIN
    if score < CONFIDENCE_HIGH_THRESHOLD:
        return ConfidenceBand.UNCERTAIN
    return ConfidenceBand.CONFIDENT


def should_queue_for_scholar(score: float, signals: ConfidenceSignals) -> bool:
    """Only religious answers reach a scholar; general trivia never does.

    The comparison is strict, matching ``band_for``: with the default
    thresholds equal, "queued" then means exactly "abstained", and a score
    sitting precisely on the threshold cannot be queued while the user is shown
    an ordinary hedge.
    """
    return signals.is_religious and score < SCHOLAR_QUEUE_THRESHOLD


def assess(signals: ConfidenceSignals) -> ConfidenceAssessment:
    """Score the answer and decide what to do with it."""
    score = compute_confidence(signals)
    band = band_for(score)
    return ConfidenceAssessment(
        score=score,
        band=band,
        abstained=band is ConfidenceBand.ABSTAIN,
        queued=should_queue_for_scholar(score, signals),
        signals=signals.present(),
        signals_used=sorted(signals.present()),
    )


# ---------------------------------------------------------------------------
# Response shaping
# ---------------------------------------------------------------------------

ABSTENTION_MESSAGE = (
    "I'm not confident enough in an answer to this to give you one.\n\n"
    "Rather than risk telling you something incorrect on a matter of deen, "
    "I'd rather say so plainly. Please take this question to a qualified "
    "scholar or a trusted local imam, or check an authenticated source such as "
    "quran.com or sunnah.com directly."
)

SCHOLAR_QUEUED_NOTE = (
    "\n\nYour question has been added to our scholar-review queue so a "
    "qualified reviewer can look at it."
)

UNCERTAINTY_NOTE = (
    "⚠️ **Please verify this one.** My confidence in this answer is moderate — "
    "parts of it may be incomplete or imprecise. Confirm it against an "
    "authenticated source or with a qualified scholar before relying on it."
)


def apply_policy(answer: str, assessment: ConfidenceAssessment) -> str:
    """Return the text actually sent to the user for this assessment.

    Abstaining replaces the answer rather than prefixing it: an abstention that
    still shows the doubtful answer underneath is not an abstention.
    """
    if assessment.band is ConfidenceBand.ABSTAIN:
        message = ABSTENTION_MESSAGE
        if assessment.queued:
            message += SCHOLAR_QUEUED_NOTE
        return message
    if assessment.band is ConfidenceBand.UNCERTAIN:
        text = f"{answer.rstrip()}\n\n{UNCERTAINTY_NOTE}"
        # An operator may set SCHOLAR_QUEUE_THRESHOLD above the abstain
        # threshold to route hedged answers for review too. Nothing should ever
        # go to a scholar without the user being told it did.
        if assessment.queued:
            text += SCHOLAR_QUEUED_NOTE
        return text
    return answer


def build_signals(
    answer: str,
    is_religious: bool,
    is_high_stakes: bool = False,
    self_consistency: Optional[float] = None,
    citation_verification: Optional[float] = None,
) -> ConfidenceSignals:
    """Assemble signals for a turn, deriving only the text-based one here."""
    return ConfidenceSignals(
        self_consistency=self_consistency,
        citation_verification=citation_verification,
        expressed_certainty=expressed_certainty(answer),
        is_religious=is_religious,
        is_high_stakes=is_high_stakes,
    )


def thresholds() -> Dict[str, float]:
    """Current policy configuration, for the stats endpoint and for tests."""
    return {
        "low": CONFIDENCE_LOW_THRESHOLD,
        "high": CONFIDENCE_HIGH_THRESHOLD,
        "scholar_queue": SCHOLAR_QUEUE_THRESHOLD,
        "high_stakes_penalty": HIGH_STAKES_PENALTY,
        "no_signal_prior": NO_SIGNAL_PRIOR,
    }
