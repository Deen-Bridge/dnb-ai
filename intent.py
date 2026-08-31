"""Hierarchical intent classification for Islamic questions (#203).

Why this exists
---------------
The chat path's only notion of question type was the binary ``classify_fiqh``
keyword pre-filter in ``fiqh.py``: a prompt either was a fiqh question or it
was not. That is too coarse to shape retrieval and answer style — a tafsir
explanation request, a comparative fiqh question, an urgent personal ruling,
and a historical query about the early caliphate all looked identical.

This module classifies each prompt along several orthogonal axes, offline and
deterministically (cue lists + word-boundary regexes — zero model calls, zero
network latency), following the same conventions as ``tafsir.py``'s cue
detection:

- **Knowledge domains** — a hierarchical taxonomy whose top-level labels match
  the ``domain`` values in ``data/eval/islamic_qa_benchmark.jsonl`` 1:1, so
  classifier output can be scored directly against benchmark labels;
- **Response format** — fatwa/ruling, tafsir explanation, comparison,
  factual lookup, practical guidance;
- **Urgency** of personal fiqh questions;
- **Orientation** — learning (understanding) vs action (doing);
- **Comparative / historical / meta-methodology** flags;
- **Multi-intent** detection across sentence-like segments;
- **Answer depth** — brief vs standard vs detailed;
- **Academic vs practical** orientation.

Every classification is recorded on ``accuracy_tracker`` so predicted-vs-actual
counts per field accumulate once ground-truth labels arrive from feedback or
the scholar-review queue.
"""

from __future__ import annotations

import logging
import re
import threading
from itertools import combinations

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Knowledge-domain taxonomy
# ---------------------------------------------------------------------------
# Top-level keys intentionally mirror data/eval/islamic_qa_benchmark.jsonl's
# ``domain`` vocabulary so benchmark scoring needs no label mapping. Cues are
# matched case-insensitively as substrings, like FIQH_KEYWORDS in fiqh.py.

DOMAIN_CUES: dict[str, tuple[str, ...]] = {
    "aqeedah": (
        "aqeedah",
        "aqida",
        "tawhid",
        "tawheed",
        "tauheed",
        "oneness of allah",
        "shirk",
        "iman",
        "pillars of faith",
        "articles of faith",
        "divine decree",
        "al-qadar",
        "qadar",
        "day of judgment",
        "day of resurrection",
        "qiyamah",
        "barzakh",
        "angels",
        "attributes of allah",
        "asma wa sifat",
        "intercession",
        "shafa'ah",
        "destiny",
        "predestination",
    ),
    "fiqh_ibadat": (
        "wudu",
        "wudhu",
        "ghusl",
        "tayammum",
        "salah",
        "salat",
        "namaz",
        "rak'ah",
        "rakah",
        "sujud",
        "fajr",
        "dhuhr",
        "zuhr",
        "asr",
        "maghrib",
        "isha",
        "tarawih",
        "jumu'ah",
        "qiblah",
        "adhan",
        "fasting",
        "siyam",
        "sawm",
        "ramadan",
        "itikaf",
        "zakat",
        "zakah",
        "hajj",
        "umrah",
        "umra",
        "tawaf",
        "sacrifice",
        "udhiyah",
        "qurbani",
        "halal",
        "haram",
        "makruh",
        "purification",
        "taharah",
        "menstruation",
        "hayd",
        "nifas",
    ),
    "fiqh_muamalat": (
        "riba",
        "usury",
        "interest",
        "mortgage",
        "loan",
        "business",
        "trade",
        "contract",
        "insurance",
        "crypto",
        "cryptocurrency",
        "stocks",
        "invest",
        "investment",
        "mortgage",
        "rent",
        "hire purchase",
        "islamic finance",
        "sukuk",
        "mudarabah",
        "musharakah",
        "employment",
        "salary",
        "copyright",
    ),
    "contemporary_issues": (
        "crypto",
        "cryptocurrency",
        "bitcoin",
        "nft",
        "organ donation",
        "vaccine",
        "artificial intelligence",
        "social media",
        "surrogacy",
        "euthanasia",
        "climate change",
        "modern issue",
        "contemporary",
        "in vitro",
        "ivf",
        "genetic engineering",
    ),
    "fiqh_munakahat_mirath": (
        "marriage",
        "nikah",
        "divorce",
        "talaq",
        "khul'",
        "khula",
        "iddah",
        "dowry",
        "mahr",
        "inheritance",
        "mirath",
        "wirathat",
        "wasyya",
        "wasiyya",
        "will",
        "estate division",
        "custody",
        "hadanah",
        "guardianship",
        "polygamy",
        "mahram",
    ),
    "ulum_al_quran": (
        "tafsir",
        "tafseer",
        "exegesis",
        "asbab al-nuzul",
        "asbab al nuzul",
        "revelation of",
        "revealed",
        "makki",
        "madani",
        "abrogation",
        "naskh",
        "qira'at",
        "recitation",
        "preservation of the quran",
        "compilation of the quran",
        "mufassir",
        "occasions of revelation",
        "context of the verse",
        "context of surah",
    ),
    "mustalah_al_hadith": (
        "hadith",
        "hadeeth",
        "sunnah of the prophet",
        "isnad",
        "chain of narration",
        "matn",
        "sahih",
        "hasan",
        "da'if",
        "daeef",
        "weak narration",
        "authenticity of",
        "graded",
        "bukhari",
        "muslim",
        "tirmidhi",
        "abu dawud",
        "nasai",
        "ibn majah",
        "muwatta",
        "riwayah",
    ),
    "seerah": (
        "seerah",
        "sirah",
        "life of the prophet",
        "prophet muhammad",
        "prophet's biography",
        "migration",
        "hijrah",
        "hegira",
        "battle of badr",
        "battle of uhud",
        "conquest of makkah",
        "meccan period",
        "medinan period",
        "companions of the prophet",
        "year of the elephant",
    ),
    "tarikh_islami": (
        "history of islam",
        "islamic history",
        "historical",
        "caliphate",
        "caliph",
        "khalifa",
        "abbasid",
        "umayyad",
        "ottoman",
        "golden age",
        "battle of",
        "siege of",
        "treaty of hudaybiyyah",
        "era of",
        "century of islam",
        "spread of islam",
    ),
    "tasawwuf_adab_akhlaq": (
        "akhlaq",
        "adab",
        "character",
        "manners",
        "etiquette",
        "sincerity",
        "ikhlas",
        "riya",
        "humility",
        "patience",
        "sabr",
        "gratitude",
        "shukr",
        "tazkiyah",
        "purification of the heart",
        "spirituality",
        "dhikr",
        "ihsan",
        "good deeds",
        "lying",
        "backbiting",
        "gheebah",
        "anger management",
        "parents",
        "neighbours",
    ),
    "general_islam": (
        "islam",
        "muslim",
        "allah",
        "quran",
        "five pillars",
        "shahada",
        "convert",
        "revert",
        "new muslim",
    ),
}

# Deterministic priority when picking a primary domain: the more specific
# jurisprudence and science domains outrank the catch-all.
_PRIMARY_DOMAIN_PRIORITY = (
    "fiqh_munakahat_mirath",
    "fiqh_muamalat",
    "fiqh_ibadat",
    "ulum_al_quran",
    "mustalah_al_hadith",
    "aqeedah",
    "seerah",
    "tarikh_islami",
    "tasawwuf_adab_akhlaq",
    "contemporary_issues",
    "general_islam",
)

# ---------------------------------------------------------------------------
# Response-format cues
# ---------------------------------------------------------------------------

RESPONSE_FORMATS = frozenset({"fatwa", "tafsir_explanation", "comparison", "factual_lookup", "practical_guidance"})

# Evaluated in order; the first matching format wins. Comparison outranks
# fatwa because "compare the rulings on X" wants an analysis, not a verdict.
_RESPONSE_FORMAT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "comparison",
        (
            "difference between",
            "differences between",
            "compare",
            "comparison",
            "versus",
            " vs ",
            "vs.",
            "similarities between",
            "which is better",
            "which is more",
            "contrasting",
        ),
    ),
    (
        "fatwa",
        (
            "fatwa",
            "ruling on",
            "the ruling",
            "is it permissible",
            "is it allowed",
            "is it haram",
            "is it halal",
            "is it sinful",
            "am i allowed",
            "what should i do",
            "does it invalidate",
            "does it break",
        ),
    ),
    (
        "tafsir_explanation",
        (
            "tafsir",
            "tafseer",
            "explain",
            "explanation",
            "interpret",
            "commentary",
            "meaning of",
            "what does",
            "what do",
            "significance of",
            "why was",
            "mufassir",
        ),
    ),
    (
        "practical_guidance",
        (
            "how do i",
            "how to",
            "step by step",
            "guide me",
            "teach me how",
            "show me how",
            "beginner",
        ),
    ),
    (
        "factual_lookup",
        (
            "who was",
            "who is",
            "when was",
            "when did",
            "where is",
            "where did",
            "how many",
            "what year",
            "list of",
            "name the",
            "define",
            "definition of",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Facet cues: urgency, orientation, meta, depth, academic register
# ---------------------------------------------------------------------------

URGENCY_CUES: tuple[str, ...] = (
    "urgent",
    "urgently",
    "emergency",
    "asap",
    "right away",
    "immediately",
    "time-sensitive",
    "time sensitive",
    "deadline",
    "tonight",
    "before sunrise",
    "before sunset",
    "in a few hours",
    "by tomorrow",
)

LEARNING_CUES: tuple[str, ...] = (
    "learn about",
    "learn more",
    "understand",
    "study ",
    "i want to know",
    "help me understand",
    "what is the concept",
    "explain the concept",
    "deeper understanding",
    "for my knowledge",
)

ACTION_CUES: tuple[str, ...] = (
    "can i",
    "could i",
    "should i",
    "shall i",
    "may i",
    "am i allowed",
    "do i have to",
    "must i",
    "how do i",
    "what should i do",
    "i want to ",
    "i plan to ",
    "i am about to",
    "i'm about to",
    "is my ",
    "does my ",
)

HISTORICAL_CUES: tuple[str, ...] = (
    "history of",
    "historical",
    "historically",
    "when was",
    "when did",
    "who was",
    "in the time of",
    "during the era",
    "century",
    "caliphate",
    "caliph",
    "battle of",
    "origins of",
    "originated",
    "background of",
)

META_CUES: tuple[str, ...] = (
    "how do you determine",
    "how do you decide",
    "how does this assistant",
    "how are answers",
    "what methodology",
    "your methodology",
    "which school do you follow",
    "are your sources",
    "where do you get your",
    "how reliable are your",
    "usul al-fiqh",
    "usul al fiqh",
    "principles of jurisprudence",
    "how is this classified",
    "what criteria",
)

BRIEF_DEPTH_CUES: tuple[str, ...] = (
    "briefly",
    "in short",
    "short answer",
    "quick question",
    "quick answer",
    "one sentence",
    "just tell me",
    "summarize",
    "summarise",
    "tldr",
    "no details",
)

DETAILED_DEPTH_CUES: tuple[str, ...] = (
    "in detail",
    "detailed",
    "elaborate",
    "in depth",
    "in-depth",
    "comprehensive",
    "thoroughly",
    "everything about",
    "long answer",
    "full explanation",
    "with examples",
    "deep dive",
)

COMPARATIVE_CUES: tuple[str, ...] = (
    "difference between",
    "differences between",
    "compare",
    "comparison",
    "versus",
    " vs ",
    "vs.",
    "similarities between",
    "which is better",
    "which is more",
    "contrasting",
)

ACADEMIC_CUES: tuple[str, ...] = (
    "academic",
    "academically",
    "for my research",
    "my thesis",
    "my dissertation",
    "research paper",
    "term paper",
    "scholarly",
    "with citations",
    "with references",
    "cite sources",
    "for my exam",
    "for my studies",
    "university assignment",
    "footnotes",
)

PRACTICAL_CUES: tuple[str, ...] = ACTION_CUES

# Segmentation for multi-intent detection: sentence ends, semicolons, and the
# "and also"/"and what about" style joins that signal a second ask.
_SEGMENT_SPLIT = re.compile(
    r"[.?;!\n]+"
    r"|\band also\b"
    r"|\band what about\b"
    r"|\balso,?\s+(?:what|how|why|is|are|can|do)\b"
    r"|\b(?:and|but)\s+(?:how|why|when|where)\s+(?:do|does|did|is|are|can|should|i)\b",
    re.IGNORECASE,
)


_WORD_ONLY = re.compile(r"^[a-z0-9'\-]+$")
_WORD_CACHE: dict[str, re.Pattern[str]] = {}


def _cue_matches(text: str, cue: str) -> bool:
    """Word-boundary match for single-word cues so 'iman' never fires on
    'imagine'; plain substring for phrases, like fiqh.py's keyword matcher."""
    if " " not in cue and _WORD_ONLY.fullmatch(cue):
        pattern = _WORD_CACHE.get(cue)
        if pattern is None:
            pattern = re.compile(rf"\b{re.escape(cue)}\b")
            _WORD_CACHE[cue] = pattern
        return pattern.search(text) is not None
    return cue in text


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(_cue_matches(text, cue) for cue in cues)


def detect_domains(text: str) -> list[str]:
    """Return every knowledge domain whose cues appear, in taxonomy order."""
    return [domain for domain, cues in DOMAIN_CUES.items() if _has_any(text, cues)]


def pick_primary_domain(domains: list[str]) -> str | None:
    """Deterministically choose the most specific domain, if any."""
    for candidate in _PRIMARY_DOMAIN_PRIORITY:
        if candidate in domains:
            return candidate
    return None


def detect_response_format(text: str) -> str | None:
    """First matching response format wins; rules are ordered by specificity."""
    for fmt, cues in _RESPONSE_FORMAT_RULES:
        if _has_any(text, cues):
            return fmt
    return None


def detect_fiqh_urgency(text: str, domains: list[str], response_format: str | None) -> str:
    """High only for time-sensitive *personal* fiqh questions."""
    is_fiqh = any(d.startswith("fiqh_") for d in domains) or response_format == "fatwa"
    if is_fiqh and _has_any(text, URGENCY_CUES):
        return "high"
    return "none"


def detect_orientation(text: str) -> str | None:
    """Learning (understanding) vs action (doing). Learning checked first so
    'I want to understand whether I can...' reads as learning."""
    if _has_any(text, LEARNING_CUES):
        return "learning"
    if _has_any(text, ACTION_CUES):
        return "action"
    return None


def detect_answer_depth(text: str) -> str:
    if _has_any(text, BRIEF_DEPTH_CUES):
        return "brief"
    if _has_any(text, DETAILED_DEPTH_CUES):
        return "detailed"
    return "standard"


def detect_academic_orientation(text: str) -> str | None:
    if _has_any(text, ACADEMIC_CUES):
        return "academic"
    if _has_any(text, PRACTICAL_CUES):
        return "practical"
    return None


def is_multi_intent(prompt: str, domains: list[str]) -> bool:
    """True when distinct sentence-like segments carry different domain cues.

    A single question spanning two domains ("What is riba?") stays single-
    intent; two asks ("What is riba and how do I pray?") is multi-intent.
    """
    if len(domains) < 2:
        return False
    segments = [s.casefold() for s in _SEGMENT_SPLIT.split(prompt) if s and s.strip()]
    if len(segments) < 2:
        return False
    segment_domains = [
        frozenset(d for d, cues in DOMAIN_CUES.items() if _has_any(segment, cues)) for segment in segments
    ]
    distinct = [s for s in segment_domains if s]
    if len(distinct) < 2:
        return False
    return any(a != b for a, b in combinations(distinct, 2))


class IntentClassification(BaseModel):
    """Structured result of one prompt's classification."""

    domains: list[str] = Field(default_factory=list)
    primary_domain: str | None = None
    response_format: str | None = None
    fiqh_urgency: str = "none"  # "none" | "high"
    orientation: str | None = None  # "learning" | "action" | None
    is_comparative: bool = False
    is_historical: bool = False
    is_meta: bool = False
    is_multi_intent: bool = False
    answer_depth: str = "standard"  # "brief" | "standard" | "detailed"
    academic_register: str | None = None  # "academic" | "practical" | None


def classify_intent(prompt: str) -> IntentClassification:
    """Classify one prompt along every axis. Never raises on ordinary input."""
    text = (prompt or "").casefold()
    if not text.strip():
        return IntentClassification()

    domains = detect_domains(text)
    fmt = detect_response_format(text)

    return IntentClassification(
        domains=domains,
        primary_domain=pick_primary_domain(domains),
        response_format=fmt,
        fiqh_urgency=detect_fiqh_urgency(text, domains, fmt),
        orientation=detect_orientation(text),
        is_comparative=_has_any(text, COMPARATIVE_CUES),
        is_historical=_has_any(text, HISTORICAL_CUES),
        is_meta=_has_any(text, META_CUES),
        is_multi_intent=is_multi_intent(prompt, domains),
        answer_depth=detect_answer_depth(text),
        academic_register=detect_academic_orientation(text),
    )


# ---------------------------------------------------------------------------
# Accuracy tracking
# ---------------------------------------------------------------------------

TRACKED_FIELDS = (
    "primary_domain",
    "response_format",
    "fiqh_urgency",
    "orientation",
    "answer_depth",
    "register",
)


class IntentAccuracyTracker:
    """Thread-safe accumulator of predicted-vs-actual counts per field.

    ``record_prediction`` runs on every classification and tracks volume per
    primary domain. ``record`` adds a labelled example once a ground truth is
    known (feedback, review-queue verdict, benchmark label); accuracy per
    field is then simply correct/total from ``snapshot``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._predictions_by_domain: dict[str, int] = {}
        self._labelled: dict[str, dict[str, int]] = {}

    def record_prediction(self, classification: IntentClassification) -> None:
        key = classification.primary_domain or "unclassified"
        with self._lock:
            self._predictions_by_domain[key] = self._predictions_by_domain.get(key, 0) + 1

    def record(self, field: str, predicted: object, actual: object) -> None:
        if field not in TRACKED_FIELDS:
            raise ValueError(f"Unknown tracked field '{field}'; expected one of {TRACKED_FIELDS}")
        with self._lock:
            bucket = self._labelled.setdefault(field, {"total": 0, "correct": 0})
            bucket["total"] += 1
            if predicted == actual:
                bucket["correct"] += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            result: dict[str, object] = {
                "predictions_by_domain": dict(sorted(self._predictions_by_domain.items())),
                "total_predictions": sum(self._predictions_by_domain.values()),
            }
            accuracy: dict[str, dict[str, float]] = {}
            for field, bucket in self._labelled.items():
                total = bucket["total"]
                correct = bucket["correct"]
                accuracy[field] = {
                    "total": float(total),
                    "correct": float(correct),
                    "accuracy": round(correct / total, 4) if total else 0.0,
                }
            result["accuracy"] = accuracy
            return result

    def reset(self) -> None:
        with self._lock:
            self._predictions_by_domain.clear()
            self._labelled.clear()


# Process-wide tracker the chat path records into and telemetry reads from.
accuracy_tracker = IntentAccuracyTracker()
