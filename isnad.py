"""Isnad (chain of transmission) analysis and visualization.

Why this exists
----------------
Hadith authenticity depends not just on the text (matn) but on the chain of
narrators (isnad) that transmitted it. This module parses isnad strings,
normalizes narrator names, detects gaps in the chain, scores narrator
credibility, and produces a text-based tree visualization for debugging and
display.

Design notes
-------------
This is a lightweight, heuristic-based implementation. A production system
would integrate biographical (rijal) databases, but the parsing, data
structures, and visualization are designed so a future RAG-backed narrator
lookup can be dropped in behind the same interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NarratorEra(str, Enum):
    """Rough generation classification for narrators."""

    SAHABI = "sahabi"  # Companion of the Prophet (peace be upon him)
    TABI = "tabi"  # Successor (student of a Companion)
    TABI_AL_TABIIN = "tabi_al_tabiin"  # Student of a Successor
    LATER = "later"
    UNKNOWN = "unknown"


class ChainStrength(str, Enum):
    """Assessment of an isnad chain, mirroring classical hadith grading."""

    SAHIH = "sahih"
    HASAN = "hasan"
    DAIF = "da'if"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Known narrators (lightweight lookup table)
# ---------------------------------------------------------------------------

# Common Companions and their approximate eras.
_KNOWN_SAHABA: set[str] = {
    "abu hurairah",
    "abdullah ibn umar",
    "aisha",
    "abu bakr",
    "umar ibn al-khattab",
    "uthman ibn affan",
    "ali ibn abi talib",
    "anas ibn malik",
    "ibn abbas",
    "jabir ibn abdullah",
    "abdullah ibn masud",
    "abu musa al-ashari",
    "muadh ibn jabal",
    "zaid ibn thabit",
    "abu dardaa",
    "ubada ibn al-samit",
    "muawiyah ibn abi sufyan",
    "amaar ibn yasir",
    "salman al-farisi",
    "bilal ibn rabah",
}

_KNOWN_TABIN: set[str] = {
    "sa'id ibn al-musayyib",
    "al-qasim ibn muhammad",
    "urwah ibn al-zubayr",
    "ibn shihab al-zuhri",
    "atabah ibn abd al-aziz",
    "abd al-malik ibn shuayb",
    "makhul",
    "habib ibn abi thabit",
    "ata ibn yasar",
    "ikrimah",
}


def _normalize_name(name: str) -> str:
    """Strip honorifics and normalize a narrator name."""
    cleaned = name.strip().lower()
    # Remove common honorifics / kunyah prefixes that clutter parsing.
    for prefix in ("imam", "sheikh", "shaikh", "al-", "sheykh"):
        cleaned = cleaned.replace(prefix, " ")
    cleaned = re.sub(r"\s*\(.*?\)\s*", " ", cleaned)  # strip parenthetical
    cleaned = re.sub(r"[^a-z\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Narrator:
    """A single narrator in an isnad chain."""

    name: str
    normalized_name: str = ""
    era: NarratorEra = NarratorEra.UNKNOWN
    credibility_score: float = 0.5  # 0.0 (very weak) to 1.0 (very strong)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.normalized_name:
            self.normalized_name = _normalize_name(self.name)


@dataclass(frozen=True)
class Gap:
    """A detected gap between two consecutive narrators."""

    between: tuple[str, str]  # (narrator_before, narrator_after)
    description: str


@dataclass
class IsnadChain:
    """A parsed isnad chain with analysis metadata."""

    narrators: list[Narrator] = field(default_factory=list)
    strength: ChainStrength = ChainStrength.UNKNOWN
    gaps: list[Gap] = field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Heuristic separator patterns found in isnad strings.
_SEPARATOR_RE = re.compile(
    r"\s*(?:->|→|-{2,}|—|\|)\s*|(?:\s+from\s+)|(?:\s+via\s+)|(?:\s+narrated\s+by\s+)",
    re.IGNORECASE,
)

# Patterns that hint at a narrator name.
_IBN_PATTERN = re.compile(r"\bibn\b", re.IGNORECASE)
_ABU_PATTERN = re.compile(r"\babu\b", re.IGNORECASE)


def _looks_like_name(text: str) -> bool:
    """Heuristic: does this chunk look like it contains a narrator name?"""
    t = text.strip()
    if not t:
        return False
    # Must have at least two words or contain a known patronymic marker.
    words = t.split()
    return len(words) >= 2 or _IBN_PATTERN.search(t) or _ABU_PATTERN.search(t)


def parse_isnad(text: str) -> IsnadChain:
    """Extract narrator names and build a chain from free-text isnad string.

    Handles common formats:
    - "Narrated A from B from C from D"
    - "A -> B -> C -> D"
    - "A | B | C | D"
    """
    if not text or not text.strip():
        return IsnadChain(raw_text=text)

    # Normalize whitespace around separators and split.
    cleaned = re.sub(r"\s+", " ", text.strip())

    # Try multiple separator strategies and pick the one yielding the most
    # narrator-looking segments.
    best_chunks: list[str] = []
    for strategy in (_SEPARATOR_RE, re.compile(r"\s+from\s+", re.IGNORECASE), re.compile(r"\s*\|\s*")):
        parts = strategy.split(cleaned)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > len(best_chunks):
            best_chunks = parts

    narrators: list[Narrator] = []
    for chunk in best_chunks:
        # Strip leading/trailing prepositions.
        chunk = re.sub(r"^(?:Narrated|Reported|On the authority of|He said|She said)\s*", "", chunk, flags=re.IGNORECASE).strip()
        chunk = re.sub(r"\s*(?:said|says|narrated|reported).*$", "", chunk, flags=re.IGNORECASE).strip()
        if _looks_like_name(chunk):
            narrators.append(Narrator(name=chunk))

    # Classify eras and scores.
    for narrator in narrators:
        _classify_narrator(narrator)

    chain = IsnadChain(narrators=narrators, raw_text=text)
    chain.strength = assess_chain(chain)
    chain.gaps = detect_gaps(chain)
    return chain


def _classify_narrator(narrator: Narrator) -> None:
    """Set era and credibility_score based on known lists and heuristics."""
    norm = narrator.normalized_name
    if norm in _KNOWN_SAHABA:
        narrator.era = NarratorEra.SAHABI
        narrator.credibility_score = 0.9
        narrator.notes.append("Companion of the Prophet (peace be upon him)")
        return
    if norm in _KNOWN_TABIN:
        narrator.era = NarratorEra.TABI
        narrator.credibility_score = 0.75
        narrator.notes.append("Known Successor (Tabi')")
        return
    # Heuristic: names with "ibn" are usually identifiable people.
    if _IBN_PATTERN.search(narrator.name):
        narrator.era = NarratorEra.TABI_AL_TABIIN
        narrator.credibility_score = 0.55
    else:
        narrator.credibility_score = 0.4


# ---------------------------------------------------------------------------
# Chain analysis
# ---------------------------------------------------------------------------

# Maximum generations expected in a sound chain from Sahabi to collector.
_MAX_GENERATIONS = 4


def detect_gaps(chain: IsnadChain) -> list[Gap]:
    """Detect discontinuities in the chain: missing links or large generational jumps."""
    gaps: list[Gap] = []
    narrators = chain.narrators
    if len(narrators) < 2:
        return gaps

    era_order = {
        NarratorEra.SAHABI: 0,
        NarratorEra.TABI: 1,
        NarratorEra.TABI_AL_TABIIN: 2,
        NarratorEra.LATER: 3,
        NarratorEra.UNKNOWN: 4,
    }

    for i in range(len(narrators) - 1):
        a, b = narrators[i], narrators[i + 1]
        era_a = era_order.get(a.era, 4)
        era_b = era_order.get(b.era, 4)
        # Backward or same-era jump suggests a problem.
        if era_b < era_a:
            gaps.append(Gap(
                between=(a.name, b.name),
                description=f"{b.normalized_name} appears earlier in generation than {a.normalized_name}",
            ))
        elif era_b - era_a > 1:
            gaps.append(Gap(
                between=(a.name, b.name),
                description=f"Possible missing link between {a.normalized_name} and {b.normalized_name} (era gap of {era_b - era_a})",
            ))

    # Chain is too long for a sound isnad.
    known_eras = [n for n in narrators if n.era != NarratorEra.UNKNOWN]
    if len(known_eras) > _MAX_GENERATIONS:
        gaps.append(Gap(
            between=(narrators[0].name, narrators[-1].name),
            description=f"Chain has {len(known_eras)} classified narrators, exceeding expected {_MAX_GENERATIONS}",
        ))

    return gaps


def assess_chain(chain: IsnadChain) -> ChainStrength:
    """Rate overall chain strength based on narrator credibility and continuity."""
    narrators = chain.narrators
    if not narrators:
        return ChainStrength.UNKNOWN

    scores = [n.credibility_score for n in narrators]
    avg = sum(scores) / len(scores)
    gaps = detect_gaps(chain)
    has_gap = bool(gaps)

    # Very few narrators with high confidence → sahih.
    if avg >= 0.75 and not has_gap and len(narrators) >= 2:
        return ChainStrength.SAHIH
    # Decent average, minor issues → hasan.
    if avg >= 0.55 and not has_gap:
        return ChainStrength.HASAN
    # Low average or gaps → da'if.
    if avg < 0.55 or has_gap:
        return ChainStrength.DAIF
    return ChainStrength.UNKNOWN


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_chain(chain: IsnadChain) -> str:
    """Return a text-based tree visualization of the chain."""
    lines: list[str] = []
    lines.append(f"Isnad Chain — strength: {chain.strength.value}")
    lines.append(f"  Narrators: {len(chain.narrators)}")
    lines.append("")

    if not chain.narrators:
        lines.append("  (empty chain)")
        return "\n".join(lines)

    # Build a vertical tree: Prophet → Sahabi → ... → Collector
    narrator_names = [n.name for n in chain.narrators]
    gap_pairs = {gap.between for gap in chain.gaps}

    for i, name in enumerate(narrator_names):
        prefix = "  " if i == 0 else "  │  " * (i - 1) + "  ├─ "
        connector = "" if i == 0 else " ──▶"
        era_tag = f" [{chain.narrators[i].era.value}]" if chain.narrators[i].era != NarratorEra.UNKNOWN else ""
        score_tag = f" ({chain.narrators[i].credibility_score:.2f})"

        lines.append(f"{prefix}{connector} {name}{era_tag}{score_tag}")

        # Indicate a gap if detected.
        if i < len(narrator_names) - 1:
            next_pair = (chain.narrators[i].name, chain.narrators[i + 1].name)
            if next_pair in gap_pairs:
                gap_prefix = "  " + "  │  " * i + "  │  "
                lines.append(f"{gap_prefix}  ⚠ GAP")

    lines.append("")
    if chain.gaps:
        lines.append("Gaps detected:")
        for gap in chain.gaps:
            lines.append(f"  - {gap.description}")

    return "\n".join(lines)
