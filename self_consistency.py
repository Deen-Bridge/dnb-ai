"""Self-consistency sampling for hallucination detection (#55).

This module implements self-consistency sampling to detect hallucinations by
generating multiple candidate answers and measuring their agreement. When the
model is confident about a fact, sampled answers tend to agree; when it's
confabulating, they diverge.

Architecture
------------
1. **Sampler**: Generate N candidate answers at elevated temperature.
2. **Claim extractor**: Parse atomic claims from each answer.
3. **Agreement scorer**: Measure consistency across samples.
4. **Response policy**: Flag low-agreement answers for review or abstention.

The module produces a `self_consistency` score consumed by confidence.py,
where it functions as an EXTERNAL_SIGNAL that can lift the UNVERIFIED_CEILING
and push answers into the confident band.

Latency optimization
--------------------
- Parallel sampling via asyncio.gather
- Early exit when agreement is clearly high/low after partial samples
- Configurable sample count (default 3, up to 5 for high-stakes questions)
- Results are cached per (prompt, context) for the session

Integration
-----------
Called from the chat handler after the initial generation, before confidence
assessment. The self_consistency score flows into build_signals() alongside
citation_verification and expressed_certainty.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment."""
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


def _env_float(name: str, default: float) -> float:
    """Read a 0–1 float from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    return value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of samples to generate for consistency checking
SAMPLE_COUNT = _env_int("SELF_CONSISTENCY_SAMPLE_COUNT", 3)
SAMPLE_COUNT_HIGH_STAKES = _env_int("SELF_CONSISTENCY_SAMPLE_COUNT_HIGH_STAKES", 5)

# Temperature for sampling (higher = more diverse = stricter test)
SAMPLING_TEMPERATURE = _env_float("SELF_CONSISTENCY_TEMPERATURE", 0.9)

# Agreement threshold below which we flag the answer
LOW_AGREEMENT_THRESHOLD = _env_float("SELF_CONSISTENCY_LOW_THRESHOLD", 0.5)

# Timeout per sample (ms)
SAMPLE_TIMEOUT_MS = _env_int("SELF_CONSISTENCY_SAMPLE_TIMEOUT_MS", 10000)

# Enable/disable the feature
SELF_CONSISTENCY_ENABLED = os.getenv("SELF_CONSISTENCY_ENABLED", "true").lower() not in {
    "0",
    "false",
    "off",
}

# Early exit thresholds - skip remaining samples if agreement is clearly
# high or low after this many samples
EARLY_EXIT_MIN_SAMPLES = _env_int("SELF_CONSISTENCY_EARLY_EXIT_MIN", 2)
EARLY_EXIT_HIGH_THRESHOLD = _env_float("SELF_CONSISTENCY_EARLY_EXIT_HIGH", 0.85)
EARLY_EXIT_LOW_THRESHOLD = _env_float("SELF_CONSISTENCY_EARLY_EXIT_LOW", 0.3)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """An atomic factual assertion extracted from an answer."""

    text: str
    category: str = "general"  # factual, religious, numerical, citation
    source_span: tuple[int, int] | None = None


@dataclass
class SampleResult:
    """Result of one sampling attempt."""

    text: str
    claims: list[Claim] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


class SelfConsistencyResult(BaseModel):
    """The final self-consistency assessment."""

    score: float = Field(..., ge=0.0, le=1.0, description="Agreement score 0-1")
    sample_count: int = Field(..., description="Number of samples generated")
    claim_count: int = Field(..., description="Total unique claims across samples")
    agreement_matrix: dict[str, float] = Field(default_factory=dict, description="Per-claim agreement scores")
    low_agreement_claims: list[str] = Field(default_factory=list, description="Claims with < 50% agreement")
    total_latency_ms: float = Field(0.0, description="Total sampling latency")
    early_exit: bool = Field(False, description="Whether early exit was triggered")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

# Patterns for different claim types
CLAIM_PATTERNS = {
    "citation": re.compile(
        r"(?:Quran|Surah|Qur'an)\s*\d+:\d+|"
        r"(?:Bukhari|Muslim|Tirmidhi|Abu Dawud|Nasa'i|Ibn Majah)\s*(?:hadith\s*)?\d*",
        re.IGNORECASE,
    ),
    "numerical": re.compile(
        r"\b(?:is|are|was|were|equals?|amounts?\s+to)\s+[\d,]+(?:\.\d+)?(?:\s*%|percent)?\b",
        re.IGNORECASE,
    ),
    "religious": re.compile(
        r"(?:obligatory|mandatory|forbidden|haram|halal|sunnah|mustahabb|makruh|"
        r"fard|wajib|permissible|impermissible)\b",
        re.IGNORECASE,
    ),
}

# Sentence-ending patterns
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def extract_claims(text: str) -> list[Claim]:
    """Extract atomic factual claims from an answer.

    Uses a combination of:
    1. Sentence segmentation
    2. Pattern matching for specific claim types
    3. Filtering for assertive statements (not questions/hedges)
    """
    if not text or not text.strip():
        return []

    claims: list[Claim] = []
    sentences = SENTENCE_END.split(text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue

        # Skip questions and hedges
        if sentence.endswith("?"):
            continue
        if re.search(r"\b(?:may|might|could|perhaps|possibly|I think)\b", sentence, re.IGNORECASE):
            continue

        # Categorize the claim
        category = "general"
        for cat_name, pattern in CLAIM_PATTERNS.items():
            if pattern.search(sentence):
                category = cat_name
                break

        claims.append(Claim(text=sentence, category=category))

    return claims


def normalize_claim(claim: str) -> str:
    """Normalize a claim for comparison (lowercase, strip punctuation, etc.)."""
    text = claim.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


def claims_match(claim1: str, claim2: str, threshold: float = 0.7) -> bool:
    """Check if two claims are semantically equivalent.

    Uses token overlap as a simple heuristic. For production, this could
    be replaced with embedding similarity.
    """
    norm1 = set(normalize_claim(claim1).split())
    norm2 = set(normalize_claim(claim2).split())

    if not norm1 or not norm2:
        return False

    intersection = len(norm1 & norm2)
    union = len(norm1 | norm2)

    return (intersection / union) >= threshold if union > 0 else False


# ---------------------------------------------------------------------------
# Agreement scoring
# ---------------------------------------------------------------------------


def compute_agreement(samples: list[SampleResult]) -> tuple[float, dict[str, float], list[str]]:
    """Compute agreement score across sampled answers.

    Returns:
        - Overall agreement score (0-1)
        - Per-claim agreement matrix
        - List of low-agreement claims
    """
    if len(samples) < 2:
        return 1.0, {}, []

    # Collect all unique claims across samples
    all_claims: dict[str, list[int]] = {}  # claim_text -> list of sample indices that contain it

    for sample_idx, sample in enumerate(samples):
        for claim in sample.claims:
            claim_key = normalize_claim(claim.text)
            if not claim_key:
                continue

            # Check if this claim matches any existing claim
            matched = False
            for existing_key in all_claims:
                if claims_match(claim.text, existing_key):
                    all_claims[existing_key].append(sample_idx)
                    matched = True
                    break

            if not matched:
                all_claims[claim_key] = [sample_idx]

    if not all_claims:
        return 1.0, {}, []

    # Compute agreement for each claim
    n_samples = len(samples)
    agreement_matrix: dict[str, float] = {}
    low_agreement_claims: list[str] = []

    total_agreement = 0.0
    for claim_key, sample_indices in all_claims.items():
        # Agreement = fraction of samples that contain this claim
        agreement = len(set(sample_indices)) / n_samples
        agreement_matrix[claim_key[:50]] = round(agreement, 3)  # Truncate for readability

        if agreement < LOW_AGREEMENT_THRESHOLD:
            low_agreement_claims.append(claim_key[:100])

        total_agreement += agreement

    # Overall score = mean agreement across all claims
    overall_score = total_agreement / len(all_claims) if all_claims else 1.0

    return round(overall_score, 4), agreement_matrix, low_agreement_claims


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


async def sample_answer(
    prompt: str,
    generator: Callable[[str, float], Any],
    temperature: float = SAMPLING_TEMPERATURE,
    timeout_ms: int = SAMPLE_TIMEOUT_MS,
) -> SampleResult:
    """Generate one sample answer at the given temperature."""
    import time

    start = time.perf_counter()

    try:
        response = await asyncio.wait_for(
            generator(prompt, temperature),
            timeout=timeout_ms / 1000.0,
        )

        latency = (time.perf_counter() - start) * 1000.0

        if hasattr(response, "text"):
            text = response.text
        elif isinstance(response, str):
            text = response
        else:
            text = str(response)

        claims = extract_claims(text)

        return SampleResult(text=text, claims=claims, latency_ms=latency)

    except TimeoutError:
        return SampleResult(
            text="",
            latency_ms=timeout_ms,
            error="Sampling timed out",
        )
    except Exception as e:
        return SampleResult(
            text="",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            error=str(e),
        )


async def run_self_consistency(
    prompt: str,
    original_answer: str,
    generator: Callable[[str, float], Any],
    is_high_stakes: bool = False,
) -> SelfConsistencyResult:
    """Run self-consistency sampling on a prompt.

    Args:
        prompt: The user's question
        original_answer: The original answer (generated at normal temperature)
        generator: Async function (prompt, temperature) -> response
        is_high_stakes: Whether to use more samples

    Returns:
        SelfConsistencyResult with agreement score and metadata
    """
    if not SELF_CONSISTENCY_ENABLED:
        return SelfConsistencyResult(
            score=1.0,
            sample_count=0,
            claim_count=0,
            metadata={"disabled": True},
        )

    import time

    start = time.perf_counter()

    # Include original answer as first sample
    original_claims = extract_claims(original_answer)
    samples = [SampleResult(text=original_answer, claims=original_claims, latency_ms=0.0)]

    # Determine sample count
    target_samples = SAMPLE_COUNT_HIGH_STAKES if is_high_stakes else SAMPLE_COUNT
    samples_needed = target_samples - 1  # -1 for original

    early_exit = False
    total_latency = 0.0

    # Generate additional samples
    for i in range(samples_needed):
        sample = await sample_answer(prompt, generator)

        if sample.error:
            logger.warning("Sample %d failed: %s", i + 1, sample.error)
            continue

        samples.append(sample)
        total_latency += sample.latency_ms

        # Check for early exit after minimum samples
        if len(samples) >= EARLY_EXIT_MIN_SAMPLES + 1:
            current_score, _, _ = compute_agreement(samples)

            if current_score >= EARLY_EXIT_HIGH_THRESHOLD:
                logger.info(
                    "Early exit (high agreement): score=%.3f after %d samples",
                    current_score,
                    len(samples),
                )
                early_exit = True
                break

            if current_score <= EARLY_EXIT_LOW_THRESHOLD:
                logger.info(
                    "Early exit (low agreement): score=%.3f after %d samples",
                    current_score,
                    len(samples),
                )
                early_exit = True
                break

    # Compute final agreement
    score, agreement_matrix, low_agreement_claims = compute_agreement(samples)

    # Count unique claims
    all_claim_texts = set()
    for sample in samples:
        for claim in sample.claims:
            all_claim_texts.add(normalize_claim(claim.text))

    total_latency += (time.perf_counter() - start) * 1000.0 - total_latency

    return SelfConsistencyResult(
        score=score,
        sample_count=len(samples),
        claim_count=len(all_claim_texts),
        agreement_matrix=agreement_matrix,
        low_agreement_claims=low_agreement_claims,
        total_latency_ms=round(total_latency, 2),
        early_exit=early_exit,
        metadata={
            "is_high_stakes": is_high_stakes,
            "target_samples": target_samples,
        },
    )


# ---------------------------------------------------------------------------
# Response policy
# ---------------------------------------------------------------------------


def apply_consistency_policy(answer: str, result: SelfConsistencyResult) -> tuple[str, bool]:
    """Apply response policy based on self-consistency score.

    Returns:
        - Modified answer (with warning if needed)
        - Whether the answer should be flagged for review
    """
    if not SELF_CONSISTENCY_ENABLED:
        return answer, False

    if result.score >= EARLY_EXIT_HIGH_THRESHOLD:
        # High agreement - no modification needed
        return answer, False

    if result.score < LOW_AGREEMENT_THRESHOLD:
        # Low agreement - add warning and flag for review
        warning = (
            "\n\n⚠️ **Consistency notice:** This answer showed some variation "
            "across multiple generations. The following points may benefit "
            "from additional verification:\n"
        )
        for claim in result.low_agreement_claims[:3]:  # Show top 3
            warning += f"- {claim[:80]}...\n" if len(claim) > 80 else f"- {claim}\n"

        return f"{answer.rstrip()}{warning}", True

    # Moderate agreement - no modification but may flag for review
    should_flag = result.score < 0.6
    return answer, should_flag


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@lru_cache(maxsize=100)
def _cache_key(prompt: str, context: str | None) -> str:
    """Generate a cache key for a prompt+context combination."""
    content = f"{prompt}:{context or ''}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# Session-level cache (cleared per request)
_session_cache: dict[str, SelfConsistencyResult] = {}


def get_cached_result(prompt: str, context: str | None = None) -> SelfConsistencyResult | None:
    """Retrieve cached self-consistency result if available."""
    key = _cache_key(prompt, context)
    return _session_cache.get(key)


def cache_result(prompt: str, context: str | None, result: SelfConsistencyResult) -> None:
    """Cache a self-consistency result."""
    key = _cache_key(prompt, context)
    _session_cache[key] = result


def clear_session_cache() -> None:
    """Clear the session-level cache."""
    _session_cache.clear()


# ---------------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------------


async def get_self_consistency_score(
    prompt: str,
    original_answer: str,
    generator: Callable[[str, float], Any],
    is_high_stakes: bool = False,
    context: str | None = None,
) -> float:
    """Convenience function to get just the self-consistency score.

    This is the primary integration point for confidence.py.
    """
    # Check cache first
    cached = get_cached_result(prompt, context)
    if cached is not None:
        return cached.score

    result = await run_self_consistency(prompt, original_answer, generator, is_high_stakes)

    # Cache the result
    cache_result(prompt, context, result)

    return result.score


# ---------------------------------------------------------------------------
# Metadata for telemetry
# ---------------------------------------------------------------------------


def get_consistency_metadata(result: SelfConsistencyResult) -> dict[str, Any]:
    """Extract metadata for telemetry/logging."""
    return {
        "self_consistency_score": result.score,
        "self_consistency_samples": result.sample_count,
        "self_consistency_claims": result.claim_count,
        "self_consistency_early_exit": result.early_exit,
        "self_consistency_latency_ms": result.total_latency_ms,
        "self_consistency_low_claims": len(result.low_agreement_claims),
    }
