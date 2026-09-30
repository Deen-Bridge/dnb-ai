"""Source-grounded generation enforcement framework (#163).

Why this exists
---------------
An LLM that answers from memory alone will inevitably hallucinate — inventing
ayat, misquoting scholars, or blending disparate opinions into a coherent but
fabricated synthesis.  This module provides lightweight, purely local utilities
that measure *how much* a generated answer is actually grounded in the source
material it was given, and can block or flag answers that drift too far from
their sources.

Design principles
------------------
- **No external calls.**  Every function is pure Python; no LLM calls, no
  network.  This keeps latency negligible and makes the module safe to run
  on every turn as a post-processing step.
- **Composable with the existing pipeline.**  The functions here are consumed
  by ``main.py`` after the safety + generation stage, much like
  ``confidence.py`` and ``citations.py``.
- **Fail-open.**  A bad input never raises — worst case the score is 0.0
  and the blocker is a no-op, so a grounding glitch can never take down the
  chat endpoint.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (env-configurable)
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    return max(0.0, min(1.0, value))


# Minimum token-overlap ratio between a generated sentence and the best
# matching source passage below which the sentence is flagged as unsupported.
GROUNDING_OVERLAP_THRESHOLD = _env_float("GROUNDING_OVERLAP_THRESHOLD", 0.15)

# Minimum average fidelity across all sentences for the whole answer to
# pass the grounding check.
GROUNDING_FIDELITY_FLOOR = _env_float("GROUNDING_FIDELITY_FLOOR", 0.25)

# Maximum fraction of sentences that may be flagged as hallucinated before
# the answer is blocked rather than just warned.
GROUNDING_HALLUCINATION_LIMIT = _env_float("GROUNDING_HALLUCINATION_LIMIT", 0.40)


# ---------------------------------------------------------------------------
# Source passage index
# ---------------------------------------------------------------------------


@dataclass
class SourcePassage:
    """A single retrievable unit of source material."""

    text: str
    source_id: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class SourceIndex:
    """Lightweight keyword index over a list of source passages.

    This is *not* a vector store — it is a simple inverted index that maps
    normalised tokens to the passages that contain them.  Good enough for
    overlap-based grounding checks without any embedding calls.
    """

    passages: list[SourcePassage] = field(default_factory=list)
    _token_index: dict[str, list[int]] = field(default_factory=dict)

    # -- construction -------------------------------------------------------

    def add(self, passage: SourcePassage) -> None:
        idx = len(self.passages)
        self.passages.append(passage)
        for token in _tokenize(passage.text):
            self._token_index.setdefault(token, []).append(idx)

    @classmethod
    def from_texts(cls, texts: list[str], source_id: str = "") -> SourceIndex:
        index = cls()
        for i, text in enumerate(texts):
            index.add(SourcePassage(text=text, source_id=f"{source_id}:{i}" if source_id else str(i)))
        return index

    # -- query --------------------------------------------------------------

    def find_relevant(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return ``(passage_index, overlap_score)`` pairs sorted descending.

        The score is the fraction of *query* tokens present in the passage.
        """
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        candidate_counts: dict[int, int] = {}
        for token in q_tokens:
            for idx in self._token_index.get(token, []):
                candidate_counts[idx] = candidate_counts.get(idx, 0) + 1

        scored = [(idx, count / len(q_tokens)) for idx, count in candidate_counts.items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def best_overlap(self, query: str) -> float:
        """Return the highest overlap score between *query* and any passage."""
        results = self.find_relevant(query, top_k=1)
        return results[0][1] if results else 0.0

    def best_matching_passage(self, query: str) -> SourcePassage | None:
        results = self.find_relevant(query, top_k=1)
        if not results:
            return None
        return self.passages[results[0][0]]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")


def split_sentences(text: str) -> list[str]:
    """Naïve sentence splitter good enough for English prose."""
    if not text or not text.strip():
        return []
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[a-zA-Z0-9']+\b")

# Stopwords kept minimal — just enough to avoid counting "the"/"a"/"is" as
# grounding evidence.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "this",
        "that",
        "it",
        "its",
        "or",
        "and",
        "but",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "also",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens, minus stopwords."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Fidelity scoring
# ---------------------------------------------------------------------------


def token_overlap_ratio(generated: str, source: str) -> float:
    """Fraction of *generated* content tokens that appear in *source*."""
    gen_tokens = _tokenize(generated)
    if not gen_tokens:
        return 0.0
    src_tokens = set(_tokenize(source))
    if not src_tokens:
        return 0.0
    hits = sum(1 for t in gen_tokens if t in src_tokens)
    return hits / len(gen_tokens)


def sequence_similarity(generated: str, source: str) -> float:
    """Character-level SequenceMatcher ratio — catches paraphrase better than token overlap."""
    return SequenceMatcher(None, generated.lower(), source.lower()).ratio()


def sentence_fidelity(sentence: str, index: SourceIndex) -> float:
    """Score how well a single sentence is grounded in the source index.

    Returns 0.0–1.0 where 1.0 means the sentence is directly supported.
    """
    if not sentence or not sentence.strip():
        return 1.0  # empty is vacuously grounded

    # Token-overlap with best matching passage
    best_overlap = index.best_overlap(sentence)

    # Also check character-level similarity against the top passage
    best_passage = index.best_matching_passage(sentence)
    seq_sim = sequence_similarity(sentence, best_passage.text) if best_passage else 0.0

    # Blend: token overlap is the primary signal; sequence similarity adds
    # robustness for near-verbatim quotes that share few stopword-stripped
    # tokens with the source.
    return max(best_overlap, seq_sim * 0.8)


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------


@dataclass
class SentenceVerdict:
    """Grounding verdict for one sentence."""

    text: str
    fidelity: float
    supported: bool
    closest_source: str = ""


def detect_hallucinations(
    generated_text: str,
    index: SourceIndex,
    overlap_threshold: float | None = None,
) -> list[SentenceVerdict]:
    """Split *generated_text* into sentences and check each against *index*.

    Returns one ``SentenceVerdict`` per sentence.  A sentence is considered
    *supported* when its fidelity meets or exceeds *overlap_threshold*.
    """
    threshold = overlap_threshold if overlap_threshold is not None else GROUNDING_OVERLAP_THRESHOLD
    sentences = split_sentences(generated_text)
    verdicts: list[SentenceVerdict] = []
    for sent in sentences:
        fidelity = sentence_fidelity(sent, index)
        best = index.best_matching_passage(sent)
        verdicts.append(
            SentenceVerdict(
                text=sent,
                fidelity=fidelity,
                supported=fidelity >= threshold,
                closest_source=best.text[:200] if best else "",
            )
        )
    return verdicts


# ---------------------------------------------------------------------------
# Entailment check (keyword / overlap based)
# ---------------------------------------------------------------------------


@dataclass
class EntailmentResult:
    """Basic entailment verdict between a source claim and generated claim."""

    supported: bool
    score: float
    reason: str = ""


def check_entailment(source_claim: str, generated_claim: str) -> EntailmentResult:
    """Lightweight overlap-based entailment.

    This is *not* a learned NLI model — it is a fast heuristic that checks
    whether the key content tokens of the generated claim appear in the
    source.  Good enough to catch factual drift without adding LLM latency.
    """
    src_tokens = set(_tokenize(source_claim))
    gen_tokens = _tokenize(generated_claim)

    if not gen_tokens:
        return EntailmentResult(supported=True, score=1.0, reason="empty claim")

    if not src_tokens:
        return EntailmentResult(supported=False, score=0.0, reason="empty source")

    hits = sum(1 for t in gen_tokens if t in src_tokens)
    score = hits / len(gen_tokens)

    # Additional check: if the generated claim introduces *new* named entities
    # (capitalised words not in the source) that may signal fabrication.
    src_proper = {w for w in source_claim.split() if w[0:1].isupper() and len(w) > 1}
    gen_proper = {w for w in generated_claim.split() if w[0:1].isupper() and len(w) > 1}
    novel_entities = gen_proper - src_proper - _STOPWORDS  # rough; stopwords are lowercase anyway

    if novel_entities and score < 0.5:
        return EntailmentResult(
            supported=False,
            score=score,
            reason=f"low overlap ({score:.2f}) with novel entities: {', '.join(sorted(novel_entities))}",
        )

    return EntailmentResult(
        supported=score >= GROUNDING_OVERLAP_THRESHOLD,
        score=score,
    )


# ---------------------------------------------------------------------------
# Aggregate fidelity score
# ---------------------------------------------------------------------------


@dataclass
class GroundingReport:
    """Aggregated grounding assessment for a full answer."""

    fidelity_score: float
    hallucination_ratio: float
    sentence_count: int
    supported_count: int
    unsupported_count: int
    verdicts: list[SentenceVerdict] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "fidelity_score": round(self.fidelity_score, 4),
            "hallucination_ratio": round(self.hallucination_ratio, 4),
            "sentence_count": self.sentence_count,
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "passed": self.passed,
        }


def compute_grounding(
    generated_text: str,
    index: SourceIndex,
    fidelity_floor: float | None = None,
    hallucination_limit: float | None = None,
) -> GroundingReport:
    """Full grounding assessment: fidelity + hallucination detection + pass/fail.

    This is the main entry point for callers in ``main.py``.
    """
    floor = fidelity_floor if fidelity_floor is not None else GROUNDING_FIDELITY_FLOOR
    limit = hallucination_limit if hallucination_limit is not None else GROUNDING_HALLUCINATION_LIMIT

    verdicts = detect_hallucinations(generated_text, index)
    n = len(verdicts)
    if n == 0:
        return GroundingReport(
            fidelity_score=1.0,
            hallucination_ratio=0.0,
            sentence_count=0,
            supported_count=0,
            unsupported_count=0,
            passed=True,
        )

    supported = sum(1 for v in verdicts if v.supported)
    unsupported = n - supported
    avg_fidelity = sum(v.fidelity for v in verdicts) / n
    hallucination_ratio = unsupported / n

    passed = avg_fidelity >= floor and hallucination_ratio <= limit

    return GroundingReport(
        fidelity_score=avg_fidelity,
        hallucination_ratio=hallucination_ratio,
        sentence_count=n,
        supported_count=supported,
        unsupported_count=unsupported,
        verdicts=verdicts,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Grounding violation blocking rules
# ---------------------------------------------------------------------------


class GroundingAction(str, Enum):
    """Possible dispositions for a grounding check."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class GroundingDecision:
    """Final decision after applying blocking rules to a ``GroundingReport``."""

    action: GroundingAction
    report: GroundingReport
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "fidelity_score": round(self.report.fidelity_score, 4),
            "hallucination_ratio": round(self.report.hallucination_ratio, 4),
            "passed": self.report.passed,
            "reason": self.reason,
        }


def apply_grounding_rules(report: GroundingReport) -> GroundingDecision:
    """Map a ``GroundingReport`` to an actionable decision.

    Rules (evaluated in order):
    1. If the report *passed* all thresholds → PASS.
    2. If hallucination ratio is above the limit → BLOCK.
    3. Otherwise → WARN (fidelity below floor but not catastrophic).
    """
    if report.passed:
        return GroundingDecision(action=GroundingAction.PASS, report=report)

    if report.hallucination_ratio > GROUNDING_HALLUCINATION_LIMIT:
        return GroundingDecision(
            action=GroundingAction.BLOCK,
            report=report,
            reason=(
                f"Hallucination ratio {report.hallucination_ratio:.0%} exceeds "
                f"limit {GROUNDING_HALLUCINATION_LIMIT:.0%}. "
                f"{report.unsupported_count}/{report.sentence_count} sentences unsupported."
            ),
        )

    return GroundingDecision(
        action=GroundingAction.WARN,
        report=report,
        reason=(
            f"Fidelity score {report.fidelity_score:.2f} below floor "
            f"{GROUNDING_FIDELITY_FLOOR:.2f}. Answer may contain unsupported claims."
        ),
    )


# ---------------------------------------------------------------------------
# Convenience: build a SourceIndex from the types already in the codebase
# ---------------------------------------------------------------------------


def index_from_tafsir_context(tafsir_context: object) -> SourceIndex:
    """Build a ``SourceIndex`` from a ``TafsirContext`` (if available)."""
    index = SourceIndex()
    if tafsir_context is None:
        return index
    # TafsirContext has a .passages attribute (list of strings)
    passages = getattr(tafsir_context, "passages", None)
    if isinstance(passages, list):
        for p in passages:
            if isinstance(p, str) and p.strip():
                index.add(SourcePassage(text=p.strip(), source_id="tafsir"))
    return index


def index_from_strings(texts: list[str], source_id: str = "") -> SourceIndex:
    """Build a ``SourceIndex`` from a plain list of source strings."""
    return SourceIndex.from_texts(texts, source_id=source_id)


# Block message shown to the user when an answer is grounded-violated.
GROUNDING_BLOCK_MESSAGE = (
    "I was unable to generate a reliable answer grounded in authentic sources "
    "for this question. Please consult a qualified scholar or check "
    "authenticated sources directly."
)
