import logging
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

MADHHAB_MAP = {
    "hanafi": "hanafi",
    "hanafee": "hanafi",
    "hanifite": "hanafi",
    "maliki": "maliki",
    "malikee": "maliki",
    "malikite": "maliki",
    "shafii": "shafii",
    "shafi'i": "shafii",
    "shafie": "shafii",
    "shafi": "shafii",
    "shafiy": "shafii",
    "shafite": "shafii",
    "hanbali": "hanbali",
    "hanbalee": "hanbali",
    "hanbalite": "hanbali",
}

VALID_MADHHABS = frozenset({"hanafi", "maliki", "shafii", "hanbali"})


def normalize_madhhab(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    cleaned = raw.strip().casefold()
    normalized = MADHHAB_MAP.get(cleaned)
    if normalized is not None:
        return normalized
    if cleaned in VALID_MADHHABS:
        return cleaned
    logger.warning("Unknown madhhab '%s' — degrading to None", raw)
    return None


FIQH_KEYWORDS = frozenset({
    "wudu", "wudhu", "ghusl", "tayammum", "salah", "salat", "namaz",
    "prayer", "pray", "rak'ah", "raka", "sajdah", "sujud", "ruku",
    "fajr", "dhuhr", "zuhr", "asr", "maghrib", "isha",
    "qiblah", "qibla", "adhan", "azan",
    "fasting", "siyam", "sawm", "zakat", "zakah",
    "hajj", "umrah", "umra",
    "halal", "haram", "makruh", "mustahabb", "mubah", "wajib", "fard",
    "sunna", "sunnah", "bid'ah", "bida",
    "riba", "usury", "interest", "mortgage", "loan",
    "contract", "marriage", "nikah", "divorce", "talaq",
    "inheritance", "mirath", "wasiyya",
    "impurity", "najis", "najasa", "taharah",
    "menstruation", "hayd", "nifas", "istihada",
    "fatwa", "ruling", "permissible", "prohibited",
    "ijtihad", "taqlid", "madhhab",
    "is it allowed", "is it permissible", "can i",
    "does it break", "does it invalidate",
    "what does islam say about",
    "what is the ruling on",
})


def keyword_match(text: str) -> bool:
    lower = text.casefold()
    return any(kw in lower for kw in FIQH_KEYWORDS)


def classify_fiqh(prompt: str, classify_fn=None) -> bool:
    kw = keyword_match(prompt)
    if not kw:
        return False
    if classify_fn is None:
        return True
    return classify_fn(prompt)


FIQH_IKHTILAF_CONTEXT = """

FIQH / MADHHAB METHODOLOGY:
When the user asks a fiqh (jurisprudence) question, follow these rules:

1. State points of agreement (ijma') among the schools first.
2. Then present the position of each major Sunni school with attribution:
   - Hanafi school
   - Maliki school
   - Shafi'i school
   - Hanbali school
3. Never rank one school above another or call any school "correct" or "stronger".
4. Distinguish the relied-upon position (mu'tamad) within a school from minority views when known.
5. For personal-ruling questions ("what should I do?"), close with a recommendation to consult a qualified local scholar.
"""

MADHHAB_LEAD_INSTRUCTION = """
6. The user follows the {madhhab} school. Lead with that school's position first, then summarize the other schools' views.
"""


class FiqhInfo(BaseModel):
    is_fiqh_question: bool
    madhhab_requested: Optional[str] = None
