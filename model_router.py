"""Model routing intelligence — pick the right model for each query.

Why this exists
---------------
A single fixed model is a compromise: the tier that answers a delicate fiqh
question well is overkill (and overpriced, and slower) for "what time is
Maghrib?". This module chooses, per query, the cheapest available model that
still meets the accuracy the question needs and the latency the caller can
wait for — deterministically, in pure Python, with no network call and no
trained weights.

How it decides
--------------
1. **Classification** (`classify_query`) turns raw text into `QueryFeatures`:
   an estimated token count, whether it is written in Arabic script, which
   Islamic-knowledge domains it touches (fiqh, tafsir, hadith, zakat, ...) and
   a 0–1 ``complexity`` score. Pure heuristics, no model call.
2. **Scoring** (`route_query`) walks the model registry, skips anything marked
   unavailable, and scores every remaining candidate on a weighted blend of
   accuracy fit, latency fit and cost fit. The highest score wins; the rest,
   in score order, become the fallback chain.
3. **Strategies / A/B testing** bucket a query deterministically by hashing its
   text, so an experiment routes the same question the same way every time
   without any randomness.
4. **Feedback** (`record_feedback`) nudges a per-model rolling quality score
   from real outcomes, so a model that keeps disappointing is gradually
   deprioritised for future queries.

Everything is in-memory and side-effect-free at import time: constructing the
router touches no live service, which is what lets the app boot in CI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import Lock

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/routing", tags=["model-routing"])


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

# Domain keyword sets. Matched against a casefolded, whitespace-split query, so
# they catch the vocabulary that actually signals a hard religious question.
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fiqh": (
        "fiqh",
        "halal",
        "haram",
        "permissible",
        "ruling",
        "madhhab",
        "wudu",
        "salah",
        "fasting",
        "ijma",
        "qiyas",
    ),
    "tafsir": ("tafsir", "tafseer", "ayah", "verse", "surah", "quran", "mufassir"),
    "hadith": ("hadith", "sunnah", "narration", "isnad", "sahih", "bukhari", "muslim"),
    "zakat": ("zakat", "nisab", "sadaqah", "charity", "wealth"),
    "aqidah": ("aqidah", "creed", "tawhid", "shirk", "iman", "belief"),
    "seerah": ("seerah", "prophet", "sahaba", "companion", "history"),
}

# Words that mark a question as reasoning-heavy rather than a simple lookup.
COMPLEXITY_CUES: tuple[str, ...] = (
    "why",
    "compare",
    "difference",
    "explain",
    "evidence",
    "prove",
    "reconcile",
    "contradiction",
    "distinguish",
    "derive",
    "implication",
    "however",
    "whereas",
)

# An estimate of tokens-per-word for mixed English/Arabic religious text. Kept
# deliberately simple: the router only needs a monotonic size signal, not a
# tokenizer.
TOKENS_PER_WORD = 1.3


class QueryFeatures(BaseModel):
    """Heuristic features extracted from a query, used to route it."""

    length_chars: int = Field(..., description="Raw character count")
    word_count: int = Field(..., description="Whitespace-delimited word count")
    estimated_tokens: int = Field(..., description="Rough token estimate")
    is_arabic: bool = Field(..., description="Query is predominantly Arabic script")
    domains: list[str] = Field(default_factory=list, description="Detected knowledge domains")
    complexity: float = Field(..., ge=0.0, le=1.0, description="0 (trivial) to 1 (hard) reasoning load")

    @property
    def complexity_band(self) -> str:
        """Coarse bucket for the complexity score."""
        if self.complexity < 0.34:
            return "simple"
        if self.complexity < 0.67:
            return "moderate"
        return "complex"


def _arabic_ratio(text: str) -> float:
    """Fraction of the alphabetic characters that live in the Arabic block."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for ch in letters if "؀" <= ch <= "ۿ")
    return arabic / len(letters)


def _detect_domains(lowered_words: set[str]) -> list[str]:
    """Domains whose keyword set intersects the query's words, order-stable."""
    found: list[str] = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if lowered_words.intersection(keywords):
            found.append(domain)
    return found


def classify_query(text: str) -> QueryFeatures:
    """Turn raw query text into routing features. Pure, deterministic, fast.

    The ``complexity`` score blends four cheap signals — length, reasoning-cue
    words, how many knowledge domains are touched, and whether the phrasing is a
    multi-clause question — so a long comparative fiqh question scores high and a
    short factual lookup scores low, without any model call.
    """
    stripped = (text or "").strip()
    words = stripped.split()
    word_count = len(words)
    estimated_tokens = int(round(word_count * TOKENS_PER_WORD))
    lowered_words = {w.strip(".,;:!?'\"()").casefold() for w in words}

    domains = _detect_domains(lowered_words)
    cue_hits = sum(1 for cue in COMPLEXITY_CUES if cue in lowered_words)

    # Each component is clamped to 0–1, then blended with fixed weights that sum
    # to 1 so ``complexity`` itself stays in 0–1.
    length_signal = min(word_count / 40.0, 1.0)
    cue_signal = min(cue_hits / 3.0, 1.0)
    domain_signal = min(len(domains) / 2.0, 1.0)
    clause_signal = min(stripped.count("?") / 2.0, 1.0) if "?" in stripped else 0.0

    complexity = 0.4 * length_signal + 0.3 * cue_signal + 0.2 * domain_signal + 0.1 * clause_signal
    complexity = round(min(max(complexity, 0.0), 1.0), 4)

    return QueryFeatures(
        length_chars=len(stripped),
        word_count=word_count,
        estimated_tokens=estimated_tokens,
        is_arabic=_arabic_ratio(stripped) >= 0.5,
        domains=domains,
        complexity=complexity,
    )


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------


@dataclass
class ModelProfile:
    """A candidate model and its routing-relevant characteristics.

    ``accuracy``, ``cost`` and ``latency_ms`` are static profile facts;
    ``available`` gates the model out of routing entirely; ``quality_bias`` is
    the learned adjustment ``record_feedback`` moves over time.
    """

    name: str
    accuracy: float  # 0–1 intrinsic answer-quality weight
    cost: float  # relative cost per request (arbitrary units, lower is cheaper)
    latency_ms: float  # typical end-to-end latency estimate
    available: bool = True
    quality_bias: float = 0.0  # learned nudge, clamped to [-0.25, 0.25]

    @property
    def effective_accuracy(self) -> float:
        """Intrinsic accuracy adjusted by learned feedback, clamped to 0–1."""
        return min(max(self.accuracy + self.quality_bias, 0.0), 1.0)


def _default_profiles() -> dict[str, ModelProfile]:
    """The registry the router ships with — a fast tier and a pro tier.

    Returned fresh each time so a reset (and each test) starts from a known
    state rather than sharing mutated ``quality_bias`` values across the suite.
    """
    return {
        "gemini-fast": ModelProfile(
            name="gemini-fast",
            accuracy=0.72,
            cost=1.0,
            latency_ms=350.0,
        ),
        "gemini-balanced": ModelProfile(
            name="gemini-balanced",
            accuracy=0.85,
            cost=3.0,
            latency_ms=800.0,
        ),
        "gemini-pro": ModelProfile(
            name="gemini-pro",
            accuracy=0.95,
            cost=8.0,
            latency_ms=1600.0,
        ),
    }


# ---------------------------------------------------------------------------
# Routing strategies (A/B testing)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingStrategy:
    """A named set of criterion weights. Weights need not sum to 1; they are
    normalised at scoring time."""

    name: str
    accuracy_weight: float
    latency_weight: float
    cost_weight: float


STRATEGIES: dict[str, RoutingStrategy] = {
    # Sensible default: accuracy matters most, then cost, then latency.
    "balanced": RoutingStrategy("balanced", accuracy_weight=0.5, latency_weight=0.2, cost_weight=0.3),
    # A/B arm that leans on the cheap tier to measure cost savings.
    "cost_saver": RoutingStrategy("cost_saver", accuracy_weight=0.3, latency_weight=0.2, cost_weight=0.5),
    # A/B arm that always reaches for the strongest affordable model.
    "quality_first": RoutingStrategy("quality_first", accuracy_weight=0.7, latency_weight=0.1, cost_weight=0.2),
}

DEFAULT_STRATEGY = "balanced"


def bucket_strategy(query: str, experiment: list[str] | None = None) -> str:
    """Deterministically assign a query to one strategy arm of an experiment.

    Hashing the query text (not a random draw) makes the assignment stable and
    reproducible: the same question always lands in the same arm, so an A/B
    result is not muddied by the same query being routed differently on retry.
    """
    arms = experiment or [DEFAULT_STRATEGY]
    if len(arms) == 1:
        return arms[0]
    digest = hashlib.sha256(query.encode("utf-8")).digest()
    return arms[digest[0] % len(arms)]


# ---------------------------------------------------------------------------
# Routing constraints and decisions
# ---------------------------------------------------------------------------


class RoutingConstraints(BaseModel):
    """Caller-supplied service-level requirements for a routing decision."""

    min_accuracy: float = Field(0.0, ge=0.0, le=1.0, description="Reject models below this accuracy")
    max_latency_ms: float | None = Field(None, gt=0, description="Latency SLA; models slower than this are rejected")
    max_cost: float | None = Field(None, gt=0, description="Cost budget; models above this are rejected")
    strategy: str | None = Field(None, description="Force a named strategy instead of the default/experiment")


@dataclass
class RoutingDecision:
    """The outcome of one routing call, retained in the metrics store."""

    decision_id: str
    query_preview: str
    chosen_model: str
    strategy: str
    features: QueryFeatures
    scores: dict[str, float]
    fallbacks: list[str]
    rationale: str
    decision_latency_ms: float
    feedback_outcome: float | None = None


class ModelRouter:
    """The routing engine: registry + scoring + fallback + metrics + learning.

    Thread-safe around its mutable state (the registry's learned biases and the
    decision log) so concurrent requests in the ASGI app cannot corrupt it.
    """

    # How hard a single feedback signal moves a model's learned bias, and the
    # ceiling that keeps learning from overwhelming the intrinsic profile.
    FEEDBACK_LEARNING_RATE = 0.05
    MAX_QUALITY_BIAS = 0.25

    def __init__(self, profiles: dict[str, ModelProfile] | None = None) -> None:
        self._profiles: dict[str, ModelProfile] = profiles if profiles is not None else _default_profiles()
        self._decisions: dict[str, RoutingDecision] = {}
        self._decision_order: list[str] = []
        self._counter = 0
        self._lock = Lock()

    # -- registry -----------------------------------------------------------

    @property
    def profiles(self) -> dict[str, ModelProfile]:
        return self._profiles
