"""Question reformulation suggestions — a deterministic, dependency-free engine.

Users often ask vague, underspecified, or compound questions that are hard to
answer well. This module assesses the *quality* of an incoming question and
proposes better-formed reformulations, entirely offline: no model call, no
network, no import-time side effects.

The engine covers the enhancement criteria in issue #218:

* **Quality assessment** — flags questions that are too short, missing a clear
  subject, built around ambiguous pronouns, or lacking scope, and scores the
  question on a 0–1 scale with human-readable reasons.
* **Reformulation options** — offers multiple ranked rewrites, each with an
  explanation of *why* it is an improvement: more precise phrasing, adding
  context, splitting a compound question into parts, correcting to standard
  Islamic terminology, specifying a madhhab for fiqh questions, and adding
  missing scope or constraints.
* **Example library** — a small catalogue of well-formed questions by category
  (aqidah / fiqh / tafsir / hadith / history) the caller can learn from.

The router mirrors the other feature modules: ``router = APIRouter(...)`` with
pydantic request/response models and no authentication of its own.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/reformulation", tags=["reformulation"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_QUESTION_WORDS = 4
WELL_FORMED_THRESHOLD = 0.7
MAX_QUESTION_LENGTH = 2000

# Interrogative openers that signal a genuine, scoped question.
QUESTION_WORDS = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "is",
        "are",
        "do",
        "does",
        "did",
        "can",
        "should",
        "could",
        "would",
        "may",
        "must",
    }
)

# Pronouns that, when a question leans on them without an antecedent, leave the
# reader guessing what the question is actually about.
AMBIGUOUS_PRONOUNS = frozenset({"it", "this", "that", "these", "those", "they", "them", "he", "she"})

# Connectives that usually join two distinct questions into one compound ask.
COMPOUND_CONNECTORS = (
    " and also ",
    " and how ",
    " and what ",
    " and when ",
    " and why ",
    " and is ",
    " and are ",
    " and does ",
    " and can ",
    " and should ",
    "; ",
)

# Fiqh (jurisprudence) signals — a ruling question benefits from naming a
# madhhab, since the four Sunni schools can differ.
FIQH_KEYWORDS = frozenset(
    {
        "halal",
        "haram",
        "permissible",
        "impermissible",
        "prohibited",
        "allowed",
        "ruling",
        "wudu",
        "ghusl",
        "tayammum",
        "salah",
        "salat",
        "prayer",
        "pray",
        "fasting",
        "sawm",
        "zakat",
        "hajj",
        "umrah",
        "nikah",
        "marriage",
        "divorce",
        "talaq",
        "inheritance",
        "riba",
        "interest",
        "makruh",
        "wajib",
        "fard",
        "sunnah",
        "fatwa",
    }
)

MADHHABS = ("Hanafi", "Maliki", "Shafi'i", "Hanbali")

# Colloquial phrasings mapped to the standard Islamic term to prefer. Each entry
# is (regex, replacement-term, short label used in the explanation).
TERMINOLOGY_HINTS: tuple[tuple[str, str, str], ...] = (
    (r"\bwashing (up )?before pray\w*", "wudu (ablution)", "wudu"),
    (r"\bfasting month\b", "Ramadan", "Ramadan"),
    (r"\bislamic pilgrimage\b", "Hajj", "Hajj"),
    (r"\bcharity tax\b", "zakat", "zakat"),
    (r"\bcall to prayer\b", "adhan", "adhan"),
    (r"\bislamic law\b", "sharia (or fiqh, for the applied rulings)", "sharia/fiqh"),
    (r"\bprophet'?s sayings?\b", "hadith", "hadith"),
    (r"\bquran verse commentary\b", "tafsir", "tafsir"),
)

# Topical category detection for routing to the example library and for tone.
CATEGORY_KEYWORDS: dict[str, frozenset[str]] = {
    "aqidah": frozenset(
        {"aqidah", "belief", "tawhid", "iman", "shirk", "creed", "attributes", "qadr", "predestination", "angels"}
    ),
    "fiqh": frozenset(
        {"halal", "haram", "permissible", "ruling", "wudu", "salah", "prayer", "zakat", "fasting", "nikah", "riba"}
    ),
    "tafsir": frozenset(
        {"tafsir", "verse", "ayah", "surah", "quran", "interpretation", "revealed", "meaning of the verse"}
    ),
    "hadith": frozenset(
        {"hadith", "narration", "bukhari", "muslim", "isnad", "sanad", "authentic", "sahih", "daif", "narrator"}
    ),
    "history": frozenset(
        {"history", "caliph", "sahaba", "companion", "battle", "conquest", "umayyad", "abbasid", "seerah", "dynasty"}
    ),
}

# A small catalogue of well-formed reference questions, one lever a user can
# pull to see what "good" looks like in each area.
EXAMPLE_LIBRARY: dict[str, list[str]] = {
    "aqidah": [
        "What is the difference between tawhid al-uluhiyyah and tawhid al-rububiyyah in Sunni creed?",
        "How do Ash'ari and Athari scholars differ in interpreting the divine attributes (sifat)?",
        "What is the ruling on a Muslim who denies a matter known necessarily from the religion?",
    ],
    "fiqh": [
        "What is the ruling on combining Dhuhr and Asr while traveling according to the Hanafi school?",
        "Does touching one's spouse invalidate wudu in the Shafi'i madhhab, and what is the evidence?",
        "How is zakat calculated on gold held for personal use versus gold held as an investment?",
    ],
    "tafsir": [
        "What is the classical tafsir of Surah Al-Baqarah 2:255 (Ayat al-Kursi) on the divine attributes?",
        "What is the asbab al-nuzul (occasion of revelation) of Surah An-Nur 24:11-20?",
        "How do Ibn Kathir and al-Tabari differ in interpreting Surah Al-Kahf 18:65-82?",
    ],
    "hadith": [
        "Is the hadith 'seek knowledge even unto China' authentic, and what is its grading and chain?",
        "What is the difference between a sahih and a hasan hadith in the terminology of al-Nawawi?",
        "In which collections is the hadith of Jibril (Umar ibn al-Khattab) recorded, and what is its status?",
    ],
    "history": [
        "What were the main causes of the First Fitna during the caliphate of Ali ibn Abi Talib?",
        "How did the Abbasid Revolution of 750 CE change the administration of the Islamic state?",
        "What was the role of the Ansar in the events immediately following the Prophet's death?",
    ],
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QualityIssue(BaseModel):
    """A single defect detected in a question, with the penalty it carries."""

    code: str = Field(..., description="Stable machine-readable identifier for the issue")
    message: str = Field(..., description="Human-readable explanation of the problem")
    penalty: float = Field(..., ge=0.0, le=1.0, description="Weight subtracted from the quality score")


class QualityAssessment(BaseModel):
    """The overall verdict on how well-formed a question is."""

    score: float = Field(..., ge=0.0, le=1.0, description="Quality score from 0 (poor) to 1 (well-formed)")
    is_well_formed: bool = Field(..., description="True when the question needs no material improvement")
    word_count: int = Field(..., ge=0)
    issues: list[QualityIssue] = Field(default_factory=list)


class ReformulationOption(BaseModel):
    """One suggested rewrite, ranked by how much it should help."""

    text: str = Field(..., description="The suggested reformulated question")
    strategy: str = Field(..., description="Which improvement technique this option applies")
    explanation: str = Field(..., description="Why this reformulation is an improvement")
    rank: int = Field(..., ge=1, description="1 is the most strongly recommended option")


class SuggestRequest(BaseModel):
    """A question to assess and reformulate."""

    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    madhhab: str | None = Field(
        default=None,
        description="Optional preferred school of fiqh (hanafi, maliki, shafii, hanbali)",
    )


class SuggestResponse(BaseModel):
    """Quality assessment plus ranked reformulation options for a question."""

    question: str
    category: str | None = Field(None, description="Detected topical category, when one is clear")
    assessment: QualityAssessment
    options: list[ReformulationOption]


class ExampleQuestion(BaseModel):
    category: str
    question: str


class ExamplesResponse(BaseModel):
    count: int
    category: str | None = None
    examples: list[ExampleQuestion]


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    """Lowercase word tokens, punctuation stripped."""
    return re.findall(r"[a-z']+", text.lower())


def detect_category(question: str) -> str | None:
    """Best-effort topical category from keyword overlap, or None when unclear."""
    tokens = set(_words(question))
    best: str | None = None
    best_score = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = len(tokens & keywords)
        if score > best_score:
            best = category
            best_score = score
    return best


def is_fiqh_question(question: str) -> bool:
    """True when the question reads like a jurisprudence/ruling question."""
    tokens = set(_words(question))
    return bool(tokens & FIQH_KEYWORDS)


def _split_compound(question: str) -> list[str]:
    """Split a compound question into its constituent asks."""
    lowered = question.lower()
    pattern = "|".join(re.escape(c) for c in COMPOUND_CONNECTORS)
    parts = re.split(pattern, lowered)
    cleaned = [p.strip(" ?.,;") for p in parts if p.strip(" ?.,;")]
    return cleaned if len(cleaned) > 1 else []


def _is_compound(question: str) -> bool:
    """A question is compound if it joins two asks or contains multiple ``?``."""
    if question.count("?") > 1:
        return True
    return bool(_split_compound(question))


def _terminology_hint(question: str) -> tuple[str, str] | None:
    """Return (rewritten_question, preferred_term) when a colloquial phrasing is found."""
    for pattern, replacement, label in TERMINOLOGY_HINTS:
        if re.search(pattern, question, flags=re.IGNORECASE):
            rewritten = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
            return rewritten, label
    return None


def assess_quality(question: str) -> QualityAssessment:
    """Score how well-formed a question is and enumerate its defects."""
    stripped = question.strip()
    tokens = _words(stripped)
    word_count = len(tokens)
    issues: list[QualityIssue] = []

    if word_count < MIN_QUESTION_WORDS:
        issues.append(
            QualityIssue(
                code="too_short",
                message=(
                    f"The question is only {word_count} word(s); it is likely too short to convey "
                    "what you actually want to know. Add the specific subject and scope."
                ),
                penalty=0.4,
            )
        )

    if tokens and tokens[0] not in QUESTION_WORDS and "?" not in stripped:
        issues.append(
            QualityIssue(
                code="not_a_question",
                message=(
                    "This does not read as a clear question. Start with an interrogative "
                    "(what/how/is/does…) or end with a question mark so the ask is explicit."
                ),
                penalty=0.2,
            )
        )

    if tokens and tokens[0] in AMBIGUOUS_PRONOUNS:
        issues.append(
            QualityIssue(
                code="ambiguous_pronoun",
                message=(
                    f"The question opens with the pronoun '{tokens[0]}' but never names the subject "
                    "it refers to. State the concrete topic instead of a pronoun."
                ),
                penalty=0.25,
            )
        )

    if _is_compound(stripped):
        issues.append(
            QualityIssue(
                code="compound",
                message=(
                    "This bundles more than one question together. Compound questions get shallow, "
                    "partial answers — ask each part separately."
                ),
                penalty=0.25,
            )
        )

    if is_fiqh_question(stripped) and not _mentions_madhhab(stripped):
        issues.append(
            QualityIssue(
                code="missing_madhhab",
                message=(
                    "This is a fiqh (ruling) question but no school of thought is specified. The four "
                    "Sunni madhhabs can differ; naming one gives you a precise, actionable answer."
                ),
                penalty=0.15,
            )
        )

    if word_count >= MIN_QUESTION_WORDS and not _has_scope(stripped):
        issues.append(
            QualityIssue(
                code="missing_scope",
                message=(
                    "The question lacks scope or constraints (time, place, condition, source). Adding "
                    "context narrows the answer to your actual situation."
                ),
                penalty=0.1,
            )
        )

    score = max(0.0, min(1.0, 1.0 - sum(issue.penalty for issue in issues)))
    high_severity = any(issue.penalty >= 0.25 for issue in issues)
    is_well_formed = score >= WELL_FORMED_THRESHOLD and not high_severity

    return QualityAssessment(
        score=round(score, 3),
        is_well_formed=is_well_formed,
        word_count=word_count,
        issues=issues,
    )


def _mentions_madhhab(question: str) -> bool:
    lowered = question.lower()
    return any(m.lower().replace("'", "") in lowered.replace("'", "") for m in MADHHABS) or "madhhab" in lowered


def _has_scope(question: str) -> bool:
    """Heuristic: a scoped question names a condition, source, or qualifier."""
    lowered = question.lower()
    scope_markers = (
        "according to",
        "when",
        "while",
        "during",
        "if",
        "in the",
        "for a",
        "for the",
        "based on",
        "evidence",
        "school",
        "madhhab",
        "surah",
        "hadith",
    )
    return any(marker in lowered for marker in scope_markers)


# ---------------------------------------------------------------------------
# Reformulation generation
# ---------------------------------------------------------------------------


def _normalize_madhhab(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip().casefold().replace("'", "")
    for m in MADHHABS:
        if m.casefold().replace("'", "") == cleaned:
            return m
    return None


def _clean_stub(text: str) -> str:
    """Trim trailing punctuation so a fragment can be recomposed into a question."""
    return text.strip().rstrip("?.! ").strip()


def suggest_reformulations(question: str, madhhab: str | None = None) -> list[ReformulationOption]:
    """Produce ranked, explained reformulation options for a question."""
    stripped = question.strip()
    options: list[ReformulationOption] = []
    assessment = assess_quality(stripped)
    codes = {issue.code for issue in assessment.issues}
    stub = _clean_stub(stripped)

    # 1. Split a compound question — the highest-value fix when it applies.
    if "compound" in codes:
        parts = _split_compound(stripped)
        if parts:
            numbered = "  ".join(f"({i + 1}) {p.capitalize()}?" for i, p in enumerate(parts))
        else:
            numbered = "Ask each part as its own question."
        options.append(
            ReformulationOption(
                text=numbered,
                strategy="split_compound",
                explanation=(
                    "Breaking the compound question into separate parts lets each one receive a full, "
                    "well-sourced answer instead of a shallow blended response."
                ),
                rank=1,
            )
        )

    # 2. Specify a madhhab for fiqh questions lacking one.
    if is_fiqh_question(stripped) and not _mentions_madhhab(stripped):
        chosen = _normalize_madhhab(madhhab)
        if chosen:
            text = f"{stub} according to the {chosen} school?"
            why = (
                f"You indicated the {chosen} school, so scoping the question to it yields the specific "
                "relied-upon (mu'tamad) ruling rather than a survey of all four madhhabs."
            )
        else:
            text = f"{stub} according to the {MADHHABS[0]} school (or specify your own madhhab)?"
            why = (
                "Fiqh rulings can differ between the Hanafi, Maliki, Shafi'i, and Hanbali schools. Naming "
                "the madhhab you follow turns a broad comparison into a precise, actionable answer."
            )
        options.append(ReformulationOption(text=text, strategy="specify_madhhab", explanation=why, rank=1))

    # 3. Correct colloquial phrasing to standard Islamic terminology.
    hint = _terminology_hint(stripped)
    if hint is not None:
        rewritten, label = hint
        options.append(
            ReformulationOption(
                text=rewritten if rewritten.endswith("?") else f"{_clean_stub(rewritten)}?",
                strategy="correct_terminology",
                explanation=(
                    f"Using the precise term '{label}' matches how scholars index the topic, so you "
                    "reach authentic sources faster and avoid ambiguity."
                ),
                rank=1,
            )
        )

    # 4. Resolve an ambiguous opening pronoun.
    if "ambiguous_pronoun" in codes:
        options.append(
            ReformulationOption(
                text=f"What is the Islamic ruling on [name the specific subject]? ({stub})",
                strategy="clarify_subject",
                explanation=(
                    "Replacing the opening pronoun with the concrete subject removes the guesswork about "
                    "what the question actually concerns."
                ),
                rank=1,
            )
        )

    # 5. Expand a too-short or subject-less question with a precise template.
    if "too_short" in codes or "not_a_question" in codes:
        category = detect_category(stripped) or "the relevant topic"
        options.append(
            ReformulationOption(
                text=(
                    f"What is the authentic Islamic position on {stub or category}, with evidence from "
                    "the Qur'an and Sunnah?"
                ),
                strategy="add_specificity",
                explanation=(
                    "Framing the question as a complete, evidence-seeking sentence tells the answerer the "
                    "exact subject and the level of detail you want."
                ),
                rank=1,
            )
        )

    # 6. Add missing scope / constraints.
    if "missing_scope" in codes:
        options.append(
            ReformulationOption(
                text=f"{stub} — specifically in the context of [time / place / condition]?",
                strategy="add_context",
                explanation=(
                    "Stating the situation (travel, illness, a particular era, a named source) narrows a "
                    "general question to your circumstances and prevents an over-broad answer."
                ),
                rank=1,
            )
        )

    # 7. Always offer at least one clearer alternative phrasing, so a caller
    #    never receives an empty list even for an already-decent question.
    alt = stub[:1].upper() + stub[1:] if stub else stripped
    options.append(
        ReformulationOption(
            text=f"{alt}?" if alt and not alt.endswith("?") else alt,
            strategy="alternative_phrasing",
            explanation=(
                "A clean, capitalized, single-sentence phrasing ending in a question mark reads "
                "unambiguously and is the minimum every question should meet."
            ),
            rank=1,
        )
    )

    # Rank: keep the generation order (most impactful first) and stamp 1-based ranks.
    ranked: list[ReformulationOption] = []
    for index, option in enumerate(options, start=1):
        ranked.append(option.model_copy(update={"rank": index}))
    return ranked


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/suggest", response_model=SuggestResponse)
async def suggest(request: SuggestRequest) -> SuggestResponse:
    """Assess a question and return ranked, explained reformulation options."""
    assessment = assess_quality(request.question)
    options = suggest_reformulations(request.question, request.madhhab)
    category = detect_category(request.question)
    return SuggestResponse(
        question=request.question,
        category=category,
        assessment=assessment,
        options=options,
    )


@router.get("/examples", response_model=ExamplesResponse)
async def examples(
    category: str | None = Query(None, description="Filter by aqidah, fiqh, tafsir, hadith, or history"),
) -> ExamplesResponse:
    """Return the library of well-formed example questions, optionally filtered."""
    if category is not None:
        key = category.strip().lower()
        selected = EXAMPLE_LIBRARY.get(key, [])
        items = [ExampleQuestion(category=key, question=q) for q in selected]
        return ExamplesResponse(count=len(items), category=key, examples=items)

    items = [ExampleQuestion(category=cat, question=q) for cat, questions in EXAMPLE_LIBRARY.items() for q in questions]
    return ExamplesResponse(count=len(items), category=None, examples=items)
