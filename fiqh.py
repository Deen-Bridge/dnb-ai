"""Madhhab-aware fiqh question detection and handling.

Provides:
- Madhhab normalization (common spellings → canonical form)
- Fiqh-question classification (keyword pre-filter + Gemini classifier)
- Fiqh-specific system prompt instructions

The classifier uses a two-stage approach: a deterministic keyword pre-filter
(~0ms) for obvious cases, falling back to a lightweight structured-output
Gemini call (~1-2s) for ambiguous prompts. This minimizes latency for
obvious non-fiqh questions (greetings, seerah, etc.) while maintaining
accuracy for edge cases.

Trade-off: The Gemini fallback adds latency for ambiguous prompts. In
production this could be replaced with a smaller distilled model or a
tighter keyword list if the latency budget is constrained.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MADHHABS = frozenset({"hanafi", "maliki", "shafii", "hanbali"})

MADHHAB_NORMALIZATION: dict[str, str] = {
    "hanafi": "hanafi",
    "hanafee": "hanafi",
    "hanafiy": "hanafi",
    "hanafiyah": "hanafi",
    "maliki": "maliki",
    "malikee": "maliki",
    "malikiy": "maliki",
    "malikiyah": "maliki",
    "shafii": "shafii",
    "shafie": "shafii",
    "shafiy": "shafii",
    "hanbali": "hanbali",
    "hanbalee": "hanbali",
    "hanbaliy": "hanbali",
    "hambali": "hanbali",
    "hanbaliyah": "hanbali",
}

FIQH_KEYWORDS: list[str] = [
    "halal",
    "haram",
    "makruh",
    "mustahab",
    "mubah",
    "wajib",
    "fard",
    "sunna",
    "sunnah",
    "bidah",
    "wudu",
    "wudhu",
    "ghusl",
    "tayammum",
    "salah",
    "salat",
    "prayer",
    "zakat",
    "sawm",
    "fasting",
    "hajj",
    "umrah",
    "talaq",
    "divorce",
    "nikah",
    "marriage",
    "mahr",
    "dowry",
    "riba",
    "interest",
    "gharar",
    "maysir",
    "fiqh",
    "ruling",
    "fatwa",
    "madhab",
    "madhhab",
    "raises.*hands.*prayer",
    "wiping.*socks",
    "masah",
    "break.*wudu",
    "nullif",
    "invalidat",
    "qiyam",
    "ruku",
    "sujood",
    "tashahhud",
    "dhabiha",
    "slaughter",
    "awrah",
    "hijab",
    "niqab",
    "dress code",
]

# Patterns that override FIQH_KEYWORDS — basic factual questions about
# prayer/worship that are NOT fiqh ruling questions.
NON_FIQH_OVERRIDES: list[str] = [
    r"how many rak",
    r"how many.*rak",
    r"what is the time of",
    r"what time is",
]

NON_FIQH_PATTERNS: list[str] = [
    r"^(hello|hi|hey|assalamu|salam)\b",
    r"how (are|r) you",
    r"what can you do",
    r"who (are|r) you",
    r"what is your function",
    r"tell me about yourself",
    r"what is your name",
]

CLASSIFIER_INSTRUCTION = (
    "Classify whether the following user query is a fiqh (Islamic jurisprudence) "
    "question asking about a religious ruling.\n\n"
    "A fiqh question asks about:\n"
    "- What is halal, haram, makruh, mustahab, wajib\n"
    "- Ritual purity (wudu, ghusl, tayammum)\n"
    "- Prayer rulings (how to pray, what breaks prayer)\n"
    "- Fasting, zakat, hajj/umrah rulings\n"
    "- Marriage, divorce, inheritance\n"
    "- Food, slaughter, dietary laws\n"
    "- Business transactions, riba/interest\n"
    "- Clothing, dress code, awrah\n\n"
    "A non-fiqh question includes:\n"
    "- General greetings, introductions\n"
    "- Questions about aqeedah/belief (who is Allah, what is iman)\n"
    "- Seerah/stories of prophets\n"
    "- Platform questions ('what can you do')\n"
    "- Duas and adhkar\n"
    "- General Islamic knowledge that is not about rulings\n\n"
    "Return ONLY valid JSON with key 'is_fiqh' (boolean)."
)


def normalize_madhhab(value: Optional[str]) -> Optional[str]:
    """Normalize a madhhab value to canonical form.

    Returns *None* for unknown/empty values (logs a warning).
    """
    if not value:
        return None
    cleaned = re.sub(r"[^a-zA-Z]", "", value).strip().lower()
    normalized = MADHHAB_NORMALIZATION.get(cleaned)
    if normalized is not None:
        return normalized
    logger.warning("Unknown madhhab value: %r — degrading to None", value)
    return None


def _quick_is_fiqh(prompt: str) -> Optional[bool]:
    """Deterministic pre-filter for fiqh questions.

    Returns:
        ``True`` if the prompt looks like a fiqh question.
        ``False`` for obvious non-fiqh (greetings, etc.).
        ``None`` if uncertain (needs classifier call).
    """
    lowered = prompt.lower().strip()

    for pattern in NON_FIQH_PATTERNS:
        if re.match(pattern, lowered):
            return False

    has_keyword = False
    for kw in FIQH_KEYWORDS:
        if re.search(kw, lowered, re.IGNORECASE):
            has_keyword = True
            break

    if not has_keyword:
        return None

    for pattern in NON_FIQH_OVERRIDES:
        if re.search(pattern, lowered, re.IGNORECASE):
            return None

    return True


class FiqhClassifier:
    """Lightweight two-stage classifier for fiqh questions.

    Stage 1 — deterministic keyword pre-filter (zero latency).
    Stage 2 — structured-output Gemini call for uncertain cases.
    """

    def __init__(self, genai_model=None):
        self._model = genai_model

    def is_fiqh_question(self, prompt: str) -> bool:
        quick = _quick_is_fiqh(prompt)
        if quick is not None:
            return quick
        if self._model is not None:
            return _classify_fiqh(prompt, self._model)
        return False


def _classify_fiqh(prompt: str, genai_model) -> bool:
    """Classify via Gemini structured output."""
    response = genai_model.generate_content(
        f"{CLASSIFIER_INSTRUCTION}\n\nUser query: {prompt}",
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
        request_options={"timeout": 15},
    )
    result = json.loads(response.text)
    return bool(result.get("is_fiqh", False))


# ---------------------------------------------------------------------------
# Fiqh-specific system prompt extension
# ---------------------------------------------------------------------------

FIQH_INSTRUCTIONS = """FIQH ANSWER GUIDELINES — APPLY WHEN THE USER ASKS ABOUT ISLAMIC RULINGS:

1. STATE AGREEMENT FIRST: Begin with points on which all four major Sunni schools (Hanafi, Maliki, Shafi'i, Hanbali) agree.

2. ATTRIBUTE EACH POSITION: For points of difference, present each school's position with clear attribution:
   - "The Hanafi school holds that..., based on..."
   - "According to the Maliki school..."
   - "The Shafi'i position is..."
   - "The Hanbali school rules..."

3. DO NOT RANK: Never describe any school's position as "correct", "stronger", "more authentic", or "preferred". Do not rank or imply superiority.

4. DISTINGUISH RELIED-UPON VIEWS: When known, distinguish the relied-upon (rajih/mufta bihi) position within a school from minority or weak views in that same school.

5. MADHHAB PREFERENCE: {MADHHAB_LEAD}

6. PERSONAL APPLICATION: If the question asks "what should I do" or is a personal ruling question, include a recommendation to consult a qualified local scholar.

7. CITE EVIDENCE: Mention the primary evidence (Quran verses, hadith) that each school relies on.

Remember: Legitimate scholarly difference (ikhtilaf) between the schools is a core feature of fiqh methodology, not a controversy to be avoided or resolved.
"""
