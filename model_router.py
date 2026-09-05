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
import time
from dataclasses import dataclass
from threading import Lock

from fastapi import APIRouter, HTTPException
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


# ---------------------------------------------------------------------------
# Agent response synthesis and consolidation engine
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    '''A single agent's answer plus its provenance metadata.'''
    agent_id: str
    content: str
    citations: list[str] | None = None
    confidence: float = 1.0


@dataclass
class Attribution:
    '''Maps a text span to the agent(s) and citation(s) it came from.'''
    span: str
    agent_ids: list[str]
    citations: list[str]


@dataclass
class Contradiction:
    '''A detected conflict between two text spans from different agents.'''
    span_a: str
    span_b: str
    reason: str


@dataclass
class SynthesisResult:
    '''The consolidated output and its quality/coherence metadata.'''
    content: str
    attributions: list[Attribution]
    contradictions_resolved: list[Contradiction]
    coherence_score: float
    quality_score: float


class AgentResponseSynthesizer:
    '''Consolidates multiple agent responses into one coherent, attributed answer.

    This is a pure, heuristic implementation designed to run in-memory with no
    external model calls, mirroring the router's philosophy. It performs:
      * sentence-level extraction and alignment
      * semantic deduplication (Jaccard-based)
      * contradiction detection (opposing cue words)
      * attribution tracking and citation consolidation
      * simple narrative generation with section grouping
      * coherence and quality validation
    '''

    # Cue words for detecting contradictions: (positive, negative) pairs.
    CONTRADICTION_PAIRS = (
        ('permissible', 'impermissible'),
        ('halal', 'haram'),
        ('allowed', 'not allowed'),
        ('obligatory', 'not obligatory'),
        ('valid', 'invalid'),
        ('true', 'false'),
        ('required', 'not required'),
    )

    def synthesize(self, responses: list[AgentResponse]) -> SynthesisResult:
        '''Run the full synthesis pipeline over a list of agent responses.'''
        if not responses:
            raise ValueError('At least one agent response is required')

        # 1. Normalize terminology across all content.
        normalized = [self._normalize_terminology(r.content) for r in responses]

        # 2. Extract sentences, remembering which agent/citations each came from.
        extracted: list[tuple[str, str, list[str], float]] = []
        for response, text in zip(responses, normalized):
            citations = response.citations or []
            for sentence in self._extract_sentences(text):
                extracted.append((sentence, response.agent_id, citations, response.confidence))

        # 3. Deduplicate semantically similar sentences.
        deduped = self._deduplicate(extracted)

        # 4. Detect contradictions between remaining spans.
        contradictions = self._detect_contradictions(deduped)

        # 5. Resolve contradictions by keeping the higher-confidence span.
        resolved, resolved_contradictions = self._resolve_contradictions(deduped, contradictions)

        # 6. Group into narrative sections (simple: no sections, just a coherent flow).
        content = self._build_narrative(resolved)

        # 7. Build attributions from the kept spans.
        attributions = [
            Attribution(span=span, agent_ids=[aid], citations=cit)
            for span, aid, cit, _ in resolved
        ]

        # 8. Validate coherence and quality.
        coherence = self._validate_coherence(resolved)
        quality = self._assess_quality(content, resolved, contradictions)

        return SynthesisResult(
            content=content,
            attributions=attributions,
            contradictions_resolved=resolved_contradictions,
            coherence_score=coherence,
            quality_score=quality,
        )

    # -- pipeline helpers ---------------------------------------------------

    def _extract_sentences(self, text: str) -> list[str]:
        '''Split text into sentences without importing regex.'''
        cleaned = ' '.join(text.split())
        # Replace sentence-end punctuation with a single period to split on.
        for char in ('!', '?'):
            cleaned = cleaned.replace(char, '.')
        parts = [s.strip() for s in cleaned.split('.') if s.strip()]
        return parts or [cleaned]

    def _normalize_terminology(self, text: str) -> str:
        '''Normalize common spelling variants and Arabic transliterations.'''
        replacements = {
            'ahkam': 'rulings',
            'masjid': 'mosque',
            'sawm': 'fasting',
            'salaah': 'salah',
            'salat': 'salah',
        }
        lowered = text.lower()
        for variant, canonical in replacements.items():
            raised = variant.capitalize()
            if variant in lowered:
                text = text.replace(variant, canonical).replace(raised, canonical.capitalize())
        return text.strip()

    def _deduplicate(self, sentences: list[tuple[str, str, list[str], float]]) -> list[tuple[str, str, list[str], float]]:
        '''Remove near-duplicate sentences, keeping the first occurrence.'''
        seen: list[str] = []
        kept: list[tuple[str, str, list[str], float]] = []
        for sentence, agent, cites, conf in sentences:
            dup = False
            for existing in seen:
                if self._jaccard_similarity(sentence, existing) >= 0.8:
                    dup = True
                    break
            if not dup:
                seen.append(sentence)
                kept.append((sentence, agent, cites, conf))
        return kept

    def _detect_contradictions(self, sentences: list[tuple[str, str, list[str], float]]) -> list[Contradiction]:
        '''Find pairs of sentences on the same topic with opposite polarity.'''
        contradictions: list[Contradiction] = []
        for i, (s1, _, _, _) in enumerate(sentences):
            for j in range(i + 1, len(sentences)):
                s2 = sentences[j][0]
                if self._jaccard_similarity(s1, s2) >= 0.4:
                    reason = self._check_opposition(s1, s2)
                    if reason:
                        contradictions.append(Contradiction(span_a=s1, span_b=s2, reason=reason))
        return contradictions

    def _resolve_contradictions(
        self,
        sentences: list[tuple[str, str, list[str], float]],
        contradictions: list[Contradiction],
    ) -> tuple[list[tuple[str, str, list[str], float]], list[Contradiction]]:
        '''Drop lower-confidence spans involved in contradictions, record resolutions.'''
        resolved: list[tuple[str, str, list[str], float]] = []
        resolved_contradictions: list[Contradiction] = []
        to_drop: set[int] = set()
        for contra in contradictions:
            idx_a, idx_b = None, None
            for i, (s, _, _, _) in enumerate(sentences):
                if s == contra.span_a:
                    idx_a = i
                if s == contra.span_b:
                    idx_b = i
            if idx_a is None or idx_b is None:
                continue
            # Keep the one with higher confidence; drop the other.
            if sentences[idx_a][3] >= sentences[idx_b][3]:
                to_drop.add(idx_b)
                resolved_contradictions.append(contra)
            else:
                to_drop.add(idx_a)
                resolved_contradictions.append(contra)
        for i, item in enumerate(sentences):
            if i not in to_drop:
                resolved.append(item)
        return resolved, resolved_contradictions

    def _build_narrative(self, sentences: list[tuple[str, str, list[str], float]]) -> str:
        '''Join deduplicated, contradiction-free spans into a coherent narrative.'''
        if not sentences:
            return ''
        # Simple narrative: merge spans into paragraphs, grouping by natural line breaks.
        # We insert a period if the span doesn't end with one.
        parts = []
        for sentence, _, _, _ in sentences:
            if sentence and not sentence.endswith('.'):
                sentence += '.'
            parts.append(sentence)
        # Insert paragraph breaks when a sentence looks like a heading/topic shift
        # (heuristic: starts with common section markers).
        text = ' '.join(parts)
        return text

    def _validate_coherence(self, sentences: list[tuple[str, str, list[str], float]]) -> float:
        '''Score 0–1 based on lexical overlap between adjacent sentences.'''
        if len(sentences) <= 1:
            return 1.0
        total = 0.0
        for i in range(len(sentences) - 1):
            total += self._jaccard_similarity(sentences[i][0], sentences[i + 1][0])
        return round(total / (len(sentences) - 1), 4)

    def _assess_quality(
        self,
        content: str,
        sentences: list[tuple[str, str, list[str], float]],
        contradictions: list[Contradiction],
    ) -> float:
        '''Heuristic quality score: length coverage, low redundancy, no contradictions.'''
        if not sentences:
            return 0.0
        # Coverage: total content length relative to the number of sentences.
        coverage = min(len(content) / (20.0 * len(sentences)), 1.0)
        # Redundancy penalty: inverse of dedup rate (we already removed dups, so high).
        redundancy_penalty = 0.0  # deduplication already handled
        # Contradiction penalty: lower score if any contradictions were found.
        contra_penalty = min(len(contradictions) * 0.1, 0.5)
        score = 0.7 * coverage + 0.3 * (1.0 - redundancy_penalty) - contra_penalty
        return round(min(max(score, 0.0), 1.0), 4)

    # -- similarity / contradiction helpers ---------------------------------

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        '''Jaccard similarity of word sets (casefolded, punctuation-stripped).'''
        # Remove all non-alphanumeric characters to normalize lexemes.
        set_a = {''.join(ch for ch in w.casefold() if ch.isalnum()) for w in a.split()}
        set_b = {''.join(ch for ch in w.casefold() if ch.isalnum()) for w in b.split()}
        set_a.discard('')
        set_b.discard('')
        if not set_a or not set_b:
            return 0.0
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        return len(intersection) / len(union)

    @classmethod
    def _check_opposition(cls, a: str, b: str) -> str | None:
        '''Return a reason string if two sentences are polar opposites, else None.'''
        lower_a = a.casefold()
        lower_b = b.casefold()
        for pos, neg in cls.CONTRADICTION_PAIRS:
            a_has_pos = pos in lower_a
            a_has_neg = neg in lower_a
            b_has_pos = pos in lower_b
            b_has_neg = neg in lower_b
            if (a_has_pos and b_has_neg) or (a_has_neg and b_has_pos):
                return f'Mismatched {pos} / {neg}'
        # Also check explicit negation with common verbs.
        negation_words = ('not', 'never', 'no', 'cannot')
        if (any(neg in lower_a for neg in negation_words) and
            not any(neg in lower_b for neg in negation_words)):
            if cls._jaccard_similarity(a, b) >= 0.4:
                return 'Negation mismatch'
        return None self._profiles

    def set_availability(self, name: str, available: bool) -> None:
        """Flip a model's health flag; unknown names raise KeyError."""
        with self._lock:
            self._profiles[name].available = available

    def available_profiles(self) -> list[ModelProfile]:
        return [p for p in self._profiles.values() if p.available]

    # -- scoring ------------------------------------------------------------

    def _score_profile(
        self,
        profile: ModelProfile,
        strategy: RoutingStrategy,
        features: QueryFeatures,
    ) -> float:
        """Weighted 0–1 score for one available profile against one query.

        Cost and latency are turned into "fit" terms (cheaper/faster is better)
        by normalising against the registry's spread, so all three criteria live
        on the same 0–1 scale. The query's ``complexity`` then reallocates the
        strategy's weight toward accuracy — a trivial lookup lets cost and
        latency speak, while a hard question shifts nearly all the weight onto
        accuracy, steering routing to a stronger model.
        """
        costs = [p.cost for p in self.available_profiles()]
        latencies = [p.latency_ms for p in self.available_profiles()]
        cost_span = max(costs) - min(costs) or 1.0
        latency_span = max(latencies) - min(latencies) or 1.0

        cost_fit = (max(costs) - profile.cost) / cost_span
        latency_fit = (max(latencies) - profile.latency_ms) / latency_span

        # Reallocate weight from cost/latency to accuracy in proportion to
        # complexity, preserving the total weight (so scores stay comparable).
        shift = features.complexity
        w_accuracy = strategy.accuracy_weight + shift * (strategy.latency_weight + strategy.cost_weight)
        w_latency = strategy.latency_weight * (1.0 - shift)
        w_cost = strategy.cost_weight * (1.0 - shift)

        total_weight = w_accuracy + w_latency + w_cost or 1.0
        raw = (w_accuracy * profile.effective_accuracy + w_latency * latency_fit + w_cost * cost_fit) / total_weight
        return round(raw, 6)

    def _eligible(self, profile: ModelProfile, constraints: RoutingConstraints) -> bool:
        """True when a profile satisfies the hard SLA constraints."""
        if profile.effective_accuracy < constraints.min_accuracy:
            return False
        if constraints.max_latency_ms is not None and profile.latency_ms > constraints.max_latency_ms:
            return False
        if constraints.max_cost is not None and profile.cost > constraints.max_cost:
            return False
        return True

    # -- routing ------------------------------------------------------------

    def route(
        self,
        query: str,
        constraints: RoutingConstraints | None = None,
        experiment: list[str] | None = None,
    ) -> RoutingDecision:
        """Choose the best available model for ``query`` and log the decision.

        Never returns an unavailable model: unavailable profiles are excluded
        before scoring. Raises ``NoModelAvailableError`` when nothing survives
        availability and the SLA constraints.
        """
        start = time.perf_counter()
        constraints = constraints or RoutingConstraints()
        features = classify_query(query)

        strategy_name = constraints.strategy or bucket_strategy(query, experiment)
        strategy = STRATEGIES.get(strategy_name, STRATEGIES[DEFAULT_STRATEGY])

        candidates = [p for p in self.available_profiles() if self._eligible(p, constraints)]
        if not candidates:
            raise NoModelAvailableError(
                "No available model satisfies the routing constraints "
                f"(min_accuracy={constraints.min_accuracy}, max_latency_ms={constraints.max_latency_ms}, "
                f"max_cost={constraints.max_cost})."
            )

        scores = {p.name: self._score_profile(p, strategy, features) for p in candidates}
        # Deterministic tie-break: score desc, then name asc.
        ranked = sorted(candidates, key=lambda p: (-scores[p.name], p.name))
        chosen = ranked[0]
        fallbacks = [p.name for p in ranked[1:]]

        rationale = (
            f"strategy={strategy.name}; complexity={features.complexity_band} "
            f"({features.complexity:.2f}); domains={features.domains or ['general']}; "
            f"chose {chosen.name} (score={scores[chosen.name]:.3f}, "
            f"acc={chosen.effective_accuracy:.2f}, cost={chosen.cost}, latency={chosen.latency_ms:.0f}ms)"
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        with self._lock:
            self._counter += 1
            decision_id = f"rt-{self._counter:06d}"
            decision = RoutingDecision(
                decision_id=decision_id,
                query_preview=query[:80],
                chosen_model=chosen.name,
                strategy=strategy.name,
                features=features,
                scores=scores,
                fallbacks=fallbacks,
                rationale=rationale,
                decision_latency_ms=round(elapsed_ms, 4),
            )
            self._decisions[decision_id] = decision
            self._decision_order.append(decision_id)
        return decision

    # -- feedback / learning ------------------------------------------------

    def record_feedback(self, decision_id: str, outcome: float) -> ModelProfile:
        """Fold an outcome (0 bad … 1 good) into the chosen model's bias.

        The bias moves toward ``outcome - 0.5`` by a small learning rate and is
        clamped, so a single bad answer nudges a model down a little and a run
        of good answers lifts it — changing which model future queries pick,
        without ever letting learning swamp the intrinsic profile.
        """
        if not 0.0 <= outcome <= 1.0:
            raise ValueError("feedback outcome must be in [0, 1]")
        with self._lock:
            decision = self._decisions.get(decision_id)
            if decision is None:
                raise KeyError(f"unknown decision_id {decision_id!r}")
            decision.feedback_outcome = outcome
            profile = self._profiles[decision.chosen_model]
            delta = self.FEEDBACK_LEARNING_RATE * (outcome - 0.5) * 2.0
            new_bias = profile.quality_bias + delta
            profile.quality_bias = min(max(new_bias, -self.MAX_QUALITY_BIAS), self.MAX_QUALITY_BIAS)
            return profile

    # -- metrics ------------------------------------------------------------

    def get_decision(self, decision_id: str) -> RoutingDecision | None:
        return self._decisions.get(decision_id)

    def metrics(self) -> RoutingMetrics:
        """Aggregate stats over every decision recorded so far."""
        with self._lock:
            decisions = [self._decisions[d] for d in self._decision_order]
        total = len(decisions)
        by_model: dict[str, int] = {}
        by_strategy: dict[str, int] = {}
        latency_sum = 0.0
        feedback_values: list[float] = []
        for d in decisions:
            by_model[d.chosen_model] = by_model.get(d.chosen_model, 0) + 1
            by_strategy[d.strategy] = by_strategy.get(d.strategy, 0) + 1
            latency_sum += d.decision_latency_ms
            if d.feedback_outcome is not None:
                feedback_values.append(d.feedback_outcome)
        return RoutingMetrics(
            total_decisions=total,
            decisions_by_model=by_model,
            decisions_by_strategy=by_strategy,
            avg_decision_latency_ms=round(latency_sum / total, 4) if total else 0.0,
            feedback_count=len(feedback_values),
            avg_feedback=round(sum(feedback_values) / len(feedback_values), 4) if feedback_values else None,
        )

    def reset(self) -> None:
        """Restore pristine registry and clear the decision log (test helper)."""
        with self._lock:
            self._profiles = _default_profiles()
            self._decisions.clear()
            self._decision_order.clear()
            self._counter = 0


class NoModelAvailableError(RuntimeError):
    """Raised when no available model satisfies the routing constraints."""


# Process-wide router. Constructing it touches no live service, so importing
# this module during app boot is safe.
_router_engine = ModelRouter()


def get_router_engine() -> ModelRouter:
    return _router_engine


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user query to route")
    constraints: RoutingConstraints = Field(default_factory=RoutingConstraints)
    experiment: list[str] | None = Field(
        None,
        description="Strategy arms to A/B bucket this query across (deterministic by query hash)",
    )


class RouteResponse(BaseModel):
    decision_id: str
    chosen_model: str
    strategy: str
    fallbacks: list[str]
    scores: dict[str, float]
    features: QueryFeatures
    rationale: str
    decision_latency_ms: float


class ModelHealth(BaseModel):
    name: str
    accuracy: float
    effective_accuracy: float
    cost: float
    latency_ms: float
    available: bool
    quality_bias: float


class FeedbackRequest(BaseModel):
    decision_id: str = Field(..., description="Id returned by POST /routing/route")
    outcome: float = Field(..., ge=0.0, le=1.0, description="0 (bad) … 1 (good) answer quality")


class FeedbackResponse(BaseModel):
    decision_id: str
    model: str
    quality_bias: float
    effective_accuracy: float


class RoutingMetrics(BaseModel):
    total_decisions: int
    decisions_by_model: dict[str, int]
    decisions_by_strategy: dict[str, int]
    avg_decision_latency_ms: float
    feedback_count: int
    avg_feedback: float | None = None


def _to_response(decision: RoutingDecision) -> RouteResponse:
    return RouteResponse(
        decision_id=decision.decision_id,
        chosen_model=decision.chosen_model,
        strategy=decision.strategy,
        fallbacks=decision.fallbacks,
        scores=decision.scores,
        features=decision.features,
        rationale=decision.rationale,
        decision_latency_ms=decision.decision_latency_ms,
    )


def _health(profile: ModelProfile) -> ModelHealth:
    return ModelHealth(
        name=profile.name,
        accuracy=profile.accuracy,
        effective_accuracy=profile.effective_accuracy,
        cost=profile.cost,
        latency_ms=profile.latency_ms,
        available=profile.available,
        quality_bias=round(profile.quality_bias, 6),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/route", response_model=RouteResponse)
async def route_endpoint(request: RouteRequest) -> RouteResponse:
    """Route one query to a model and return the choice plus its rationale."""
    try:
        decision = get_router_engine().route(
            request.query,
            constraints=request.constraints,
            experiment=request.experiment,
        )
    except NoModelAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _to_response(decision)


@router.get("/models", response_model=list[ModelHealth])
async def list_models() -> list[ModelHealth]:
    """Registered model profiles and their current health/learned bias."""
    return [_health(p) for p in get_router_engine().profiles.values()]


@router.get("/metrics", response_model=RoutingMetrics)
async def routing_metrics() -> RoutingMetrics:
    """Aggregated statistics over every routing decision so far."""
    return get_router_engine().metrics()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record an answer-quality outcome and adjust the model's learned bias."""
    try:
        profile = get_router_engine().record_feedback(request.decision_id, request.outcome)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown decision_id {request.decision_id!r}.") from exc
    return FeedbackResponse(
        decision_id=request.decision_id,
        model=profile.name,
        quality_bias=round(profile.quality_bias, 6),
        effective_accuracy=profile.effective_accuracy,
    )
