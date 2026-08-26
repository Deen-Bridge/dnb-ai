"""Historical context injection — asbab al-nuzul, hadith circumstances,
scholar biographies, and the historical development of fiqh rulings.

Why this exists
---------------
A bare answer to "why is wine forbidden?" or "what does 9:5 mean?" gives the
contemporary ruling with none of the ground it grew from: the *asbab al-nuzul*
(occasion of revelation), the staged way the prohibition of intoxicants was
revealed, the century a cited scholar actually lived in, or the fact that a
ruling was time-bound to a specific circumstance rather than universal. That
missing context is where flattened or anachronistic readings do the most harm.

This module is a small, curated, **offline** knowledge base of that context. It
does not generate history with a language model — every entry is a fixed record
with its own attribution, so nothing here is fabricated at request time. A
relevance detector scans a question (or a drafted answer) for verse references,
scholar names, and topic keywords, and returns only the historical records that
actually match, plus a compact system-prompt block the chat layer can inject.

Design constraints
------------------
No network, no live services, no import-time side effects: the data lives in
module-level constants and every public function is pure. That keeps ``mypy .``
and the Docker ``/ping`` boot green and makes the whole surface unit-testable
without mocking anything.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])

MAX_TEXT_LENGTH = 8_000

# ---------------------------------------------------------------------------
# Curated knowledge base
#
# Every record is a fixed, attributed datum. The point is not exhaustiveness
# but a faithful, checkable core that the relevance detector can draw from.
# ---------------------------------------------------------------------------

# Occasions of revelation, keyed by "surah:ayah". ``attribution`` names the
# classical source the report is transmitted through.
ASBAB_AL_NUZUL: dict[str, dict[str, str]] = {
    "2:219": {
        "summary": (
            "Revealed as an intermediate step in the staged prohibition of "
            "intoxicants: it names both benefit and greater sin in khamr, "
            "prompting some Companions to begin abstaining before the final "
            "prohibition came down."
        ),
        "attribution": "al-Wahidi, Asbab al-Nuzul; reports via Ibn Abbas",
    },
    "4:43": {
        "summary": (
            "The second stage on intoxicants — forbidding approaching prayer "
            "while intoxicated — revealed after a Companion led prayer and "
            "confused the words of a recitation."
        ),
        "attribution": "al-Tirmidhi; al-Wahidi, Asbab al-Nuzul",
    },
    "5:90": {
        "summary": (
            "The final, decisive prohibition of khamr and gambling. Its "
            "staged revelation over years is the classic example of gradual "
            "legislation (tadarruj) accommodating an established custom."
        ),
        "attribution": "al-Bukhari; al-Wahidi, Asbab al-Nuzul",
    },
    "2:256": {
        "summary": (
            "'There is no compulsion in religion' — reported to have been "
            "revealed concerning the Ansar, some of whose children had been "
            "raised in other faiths and whom the parents wished to compel."
        ),
        "attribution": "Abu Dawud; al-Nasa'i; via Ibn Abbas",
    },
    "9:5": {
        "summary": (
            "The 'verse of the sword', revealed in the specific setting of "
            "expired treaties with hostile tribes after Tabuk; classical "
            "exegetes read it against the treaty-keeping clauses immediately "
            "following (9:4, 9:7), not as an open-ended command."
        ),
        "attribution": "al-Tabari, Jami' al-Bayan",
    },
    "80:1": {
        "summary": (
            "'He frowned and turned away' — revealed when the Prophet turned "
            "from the blind Companion Ibn Umm Maktum while addressing Quraysh "
            "notables; a rebuke recorded against the Prophet himself."
        ),
        "attribution": "al-Tirmidhi; al-Wahidi, Asbab al-Nuzul",
    },
}

# Circumstances surrounding notable hadith, keyed by a stable topic slug.
HADITH_CONTEXTS: dict[str, dict[str, str]] = {
    "actions-by-intentions": {
        "text": "Actions are but by intentions (innama al-a'mal bi-l-niyyat).",
        "circumstance": (
            "Traditionally linked to a man who emigrated to Madinah to marry "
            "rather than for faith — 'the emigrant to what he emigrated for'. "
            "Placed first in al-Bukhari's Sahih as a foundational principle."
        ),
        "attribution": "al-Bukhari 1; Muslim 1907; via Umar ibn al-Khattab",
    },
    "standing-for-funeral": {
        "text": "The Prophet stood as a funeral passed.",
        "circumstance": (
            "When told the deceased was a Jew, he replied 'Was he not a "
            "soul?' — narrated to show the ruling's rationale ('illa) and how "
            "later jurists debated whether the standing remained recommended."
        ),
        "attribution": "al-Bukhari 1312; Muslim 961",
    },
}

# Classical scholars with their time period, so a cited name can be placed in
# its century rather than read as a contemporary voice.
SCHOLARS: dict[str, dict[str, Any]] = {
    "abu-hanifa": {
        "name": "Abu Hanifa al-Nu'man",
        "hijri": "80–150 AH",
        "gregorian": "699–767 CE",
        "century_ce": 8,
        "madhhab": "Hanafi (eponym)",
        "region": "Kufa, Iraq",
        "note": "Earliest of the four eponymous imams; emphasised reasoned analogy (qiyas) and istihsan.",
    },
    "malik": {
        "name": "Malik ibn Anas",
        "hijri": "93–179 AH",
        "gregorian": "711–795 CE",
        "century_ce": 8,
        "madhhab": "Maliki (eponym)",
        "region": "Madinah",
        "note": "Author of al-Muwatta'; weighted the practice of the people of Madinah ('amal ahl al-Madinah).",
    },
    "al-shafii": {
        "name": "Muhammad ibn Idris al-Shafi'i",
        "hijri": "150–204 AH",
        "gregorian": "767–820 CE",
        "century_ce": 9,
        "madhhab": "Shafi'i (eponym)",
        "region": "Studied in Madinah, Iraq, Egypt",
        "note": "Systematised usul al-fiqh in al-Risala, formalising the hierarchy of sources.",
    },
    "ahmad": {
        "name": "Ahmad ibn Hanbal",
        "hijri": "164–241 AH",
        "gregorian": "780–855 CE",
        "century_ce": 9,
        "madhhab": "Hanbali (eponym)",
        "region": "Baghdad",
        "note": "Traditionist and muhaddith; compiler of the Musnad; foregrounded narrated evidence.",
    },
    "al-tabari": {
        "name": "Muhammad ibn Jarir al-Tabari",
        "hijri": "224–310 AH",
        "gregorian": "839–923 CE",
        "century_ce": 10,
        "madhhab": "Independent (his Jariri school did not survive)",
        "region": "Baghdad",
        "note": "Author of the landmark tafsir Jami' al-Bayan and a universal history.",
    },
    "al-ghazali": {
        "name": "Abu Hamid al-Ghazali",
        "hijri": "450–505 AH",
        "gregorian": "1058–1111 CE",
        "century_ce": 11,
        "madhhab": "Shafi'i",
        "region": "Tus, Baghdad, Nishapur",
        "note": "Author of Ihya' 'Ulum al-Din; reconciled fiqh, kalam and tasawwuf.",
    },
    "ibn-taymiyya": {
        "name": "Taqi al-Din Ibn Taymiyya",
        "hijri": "661–728 AH",
        "gregorian": "1263–1328 CE",
        "century_ce": 14,
        "madhhab": "Hanbali",
        "region": "Damascus",
        "note": "Reformer stressing return to Qur'an, Sunnah and the salaf; prolific in fiqh and creed.",
    },
    "al-nawawi": {
        "name": "Yahya ibn Sharaf al-Nawawi",
        "hijri": "631–676 AH",
        "gregorian": "1233–1277 CE",
        "century_ce": 13,
        "madhhab": "Shafi'i",
        "region": "Damascus",
        "note": "Author of Riyad al-Salihin and the Forty Hadith; major Shafi'i codifier.",
    },
    "ibn-kathir": {
        "name": "Isma'il ibn Kathir",
        "hijri": "701–774 AH",
        "gregorian": "1301–1373 CE",
        "century_ce": 14,
        "madhhab": "Shafi'i",
        "region": "Damascus",
        "note": "Student of Ibn Taymiyya; author of a widely used tafsir grounded in narration.",
    },
}

# The historical development of selected fiqh rulings: how understanding and
# application changed over centuries, and whether a ruling is read as
# time-bound to its circumstance or universal.
FIQH_TIMELINE: dict[str, dict[str, Any]] = {
    "intoxicants": {
        "title": "Prohibition of intoxicants (khamr)",
        "scope": "universal",
        "stages": [
            {"period": "Early Makkan/Madinan", "development": "No prohibition; wine an accepted custom (ref. 16:67)."},
            {"period": "2:219", "development": "Named as containing great sin outweighing benefit — a moral warning."},
            {
                "period": "4:43",
                "development": "Approaching prayer while intoxicated forbidden — a partial, timed restriction.",
            },
            {
                "period": "5:90–91",
                "development": "Decisive, total prohibition. Later jurists extended the 'illa (intoxication) to all intoxicants by analogy.",
            },
        ],
        "note": "A textbook case of gradual legislation (tadarruj); the final ruling is universal, the earlier stages time-bound.",
    },
    "riba": {
        "title": "Riba (usury) rulings",
        "scope": "universal",
        "stages": [
            {"period": "Revelation", "development": "Progressive condemnation culminating in 2:275–279."},
            {
                "period": "Classical fiqh",
                "development": "Distinguished riba al-fadl and riba al-nasi'a; debated the 'six commodities' hadith.",
            },
            {
                "period": "Modern",
                "development": "Extended by most contemporary councils to interest-based banking; a minority reads institutional interest differently.",
            },
        ],
        "note": "The prohibition is universal; its application to new financial instruments is an ongoing ijtihad.",
    },
    "tobacco": {
        "title": "Ruling on tobacco",
        "scope": "time-bound-to-knowledge",
        "stages": [
            {"period": "Pre-17th c.", "development": "Unknown in the Muslim world; no ruling existed."},
            {
                "period": "17th–18th c.",
                "development": "On introduction, scholars split between permissible, disliked, and forbidden for lack of clear harm.",
            },
            {
                "period": "Modern",
                "development": "With established medical harm, most contemporary scholars rule it forbidden or strongly disliked.",
            },
        ],
        "note": "A clear case where the ruling evolved as socio-historical knowledge (of harm) changed.",
    },
    "coffee": {
        "title": "Ruling on coffee",
        "scope": "resolved",
        "stages": [
            {
                "period": "15th–16th c.",
                "development": "Controversial on introduction in Yemen and later Makkah; briefly banned by some authorities.",
            },
            {
                "period": "Later",
                "development": "Consensus settled on permissibility once it was clear it is not an intoxicant.",
            },
        ],
        "note": "Illustrates how an unfamiliar substance moved from dispute to settled permissibility.",
    },
}

# ---------------------------------------------------------------------------
# Relevance detection
# ---------------------------------------------------------------------------

_AYAH_REF = re.compile(r"\b(\d{1,3})\s*[:.]\s*(\d{1,3})\b")

# Topic keyword -> timeline key. Matched case-insensitively as whole words.
_TOPIC_KEYWORDS: dict[str, str] = {
    "intoxicant": "intoxicants",
    "khamr": "intoxicants",
    "wine": "intoxicants",
    "alcohol": "intoxicants",
    "riba": "riba",
    "usury": "riba",
    "interest": "riba",
    "tobacco": "tobacco",
    "smoking": "tobacco",
    "cigarette": "tobacco",
    "coffee": "coffee",
    "qahwa": "coffee",
}

_HADITH_KEYWORDS: dict[str, str] = {
    "intention": "actions-by-intentions",
    "niyya": "actions-by-intentions",
    "niyyah": "actions-by-intentions",
    "funeral": "standing-for-funeral",
    "janaza": "standing-for-funeral",
}


def _strip_apostrophes(value: str) -> str:
    """Drop ASCII and curly apostrophes so 'al-Shafi'i' matches 'al-shafii'."""
    return value.replace("'", "").replace("’", "")


def _match_scholars(text: str) -> list[str]:
    """Return the keys of scholars whose name is mentioned in ``text``."""
    lowered = _strip_apostrophes(text.lower())
    hits: list[str] = []
    for key, record in SCHOLARS.items():
        full = _strip_apostrophes(str(record["name"]).lower())
        # Match on the key, the full name, or the distinctive last element.
        distinctive = full.split()[-1]
        needles = {key.replace("-", " "), key.replace("-", ""), full, distinctive}
        if any(n and n in lowered for n in needles):
            hits.append(key)
    return hits


def _match_keywords(text: str, table: dict[str, str]) -> list[str]:
    """Return distinct values from ``table`` whose keyword appears as a word."""
    lowered = text.lower()
    seen: list[str] = []
    for word, value in table.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered) and value not in seen:
            seen.append(value)
    return seen


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AsbabEntry(BaseModel):
    reference: str
    summary: str
    attribution: str


class HadithContextEntry(BaseModel):
    slug: str
    text: str
    circumstance: str
    attribution: str


class ScholarBio(BaseModel):
    key: str
    name: str
    hijri: str
    gregorian: str
    century_ce: int
    madhhab: str
    region: str
    note: str


class TimelineStage(BaseModel):
    period: str
    development: str


class FiqhTimeline(BaseModel):
    key: str
    title: str
    scope: str = Field(..., description="'universal', 'time-bound-to-knowledge', 'resolved', etc.")
    stages: list[TimelineStage]
    note: str


class HistoricalContext(BaseModel):
    """Everything the detector matched for a piece of text, plus a ready-to-inject block."""

    asbab: list[AsbabEntry] = Field(default_factory=list)
    hadith_contexts: list[HadithContextEntry] = Field(default_factory=list)
    scholars: list[ScholarBio] = Field(default_factory=list)
    timelines: list[FiqhTimeline] = Field(default_factory=list)
    has_context: bool = False
    context_block: str = ""


class HistoryContextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _asbab_entries(text: str) -> list[AsbabEntry]:
    out: list[AsbabEntry] = []
    for surah, ayah in _AYAH_REF.findall(text):
        ref = f"{int(surah)}:{int(ayah)}"
        record = ASBAB_AL_NUZUL.get(ref)
        if record and all(e.reference != ref for e in out):
            out.append(AsbabEntry(reference=ref, **record))
    return out


def _hadith_entries(text: str) -> list[HadithContextEntry]:
    out: list[HadithContextEntry] = []
    for slug in _match_keywords(text, _HADITH_KEYWORDS):
        record = HADITH_CONTEXTS[slug]
        out.append(HadithContextEntry(slug=slug, **record))
    return out


def _scholar_entries(text: str) -> list[ScholarBio]:
    return [ScholarBio(key=key, **SCHOLARS[key]) for key in _match_scholars(text)]


def _timeline_entries(text: str) -> list[FiqhTimeline]:
    out: list[FiqhTimeline] = []
    for key in _match_keywords(text, _TOPIC_KEYWORDS):
        record = FIQH_TIMELINE[key]
        out.append(FiqhTimeline(key=key, **record))
    return out


def _render_block(ctx: HistoricalContext) -> str:
    """Render matched context as a compact system-prompt block for chat injection."""
    lines: list[str] = []
    if ctx.asbab:
        lines.append("Occasion of revelation (asbab al-nuzul):")
        lines += [f"- {e.reference}: {e.summary} [{e.attribution}]" for e in ctx.asbab]
    if ctx.hadith_contexts:
        lines.append("Hadith circumstances:")
        lines += [f"- {e.text} — {e.circumstance} [{e.attribution}]" for e in ctx.hadith_contexts]
    if ctx.timelines:
        lines.append("Historical development of the ruling:")
        for t in ctx.timelines:
            lines.append(f"- {t.title} (scope: {t.scope}):")
            lines += [f"    {s.period}: {s.development}" for s in t.stages]
            lines.append(f"    Note: {t.note}")
    if ctx.scholars:
        lines.append("Scholars cited, placed in their period:")
        lines += [f"- {s.name} ({s.hijri} / {s.gregorian}), {s.madhhab}, {s.region}." for s in ctx.scholars]
    if not lines:
        return ""
    header = (
        "Relevant historical context for this answer. Weave it in where it aids "
        "understanding, distinguish time-bound circumstances from universal rulings, "
        "and keep the attributions:"
    )
    return header + "\n" + "\n".join(lines)


def build_historical_context(text: str) -> HistoricalContext:
    """Detect and assemble all historical context relevant to ``text``.

    Pure and offline: scans for verse references, scholar names, and topic
    keywords, returning only records that actually match.
    """
    ctx = HistoricalContext(
        asbab=_asbab_entries(text),
        hadith_contexts=_hadith_entries(text),
        scholars=_scholar_entries(text),
        timelines=_timeline_entries(text),
    )
    ctx.context_block = _render_block(ctx)
    ctx.has_context = bool(ctx.context_block)
    return ctx


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/context", response_model=HistoricalContext)
def historical_context(request: HistoryContextRequest) -> HistoricalContext:
    """Return the historical context relevant to a question or drafted answer."""
    return build_historical_context(request.text)


@router.get("/asbab/{surah}/{ayah}", response_model=AsbabEntry)
def asbab_al_nuzul(surah: int, ayah: int) -> AsbabEntry:
    """Return the occasion of revelation for a verse, if one is recorded."""
    ref = f"{surah}:{ayah}"
    record = ASBAB_AL_NUZUL.get(ref)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No recorded asbab al-nuzul for {ref}")
    return AsbabEntry(reference=ref, **record)


@router.get("/scholar/{name}", response_model=ScholarBio)
def scholar_biography(name: str) -> ScholarBio:
    """Return the biography and time period of a classical scholar."""
    matched = _match_scholars(name)
    if not matched:
        raise HTTPException(status_code=404, detail=f"No biographical record for '{name}'")
    key = matched[0]
    return ScholarBio(key=key, **SCHOLARS[key])


@router.get("/timeline/{topic}", response_model=FiqhTimeline)
def fiqh_timeline(topic: str) -> FiqhTimeline:
    """Return the historical development of a ruling by topic key or keyword."""
    key = topic.lower()
    if key not in FIQH_TIMELINE:
        matched = _match_keywords(topic, _TOPIC_KEYWORDS)
        if not matched:
            raise HTTPException(status_code=404, detail=f"No timeline for topic '{topic}'")
        key = matched[0]
    return FiqhTimeline(key=key, **FIQH_TIMELINE[key])


# ---------------------------------------------------------------------------
# Conversation history budget management (#49)
# ---------------------------------------------------------------------------

MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "16000"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "50"))


def estimate_tokens(text: str) -> int:
    """Cheap local token estimate — ~4 chars per token on average."""
    return max(1, len(text) // 4)


def trim_history(chat_session: Any) -> bool:
    """Enforce token budget and turn cap on chat history.

    Drops oldest turn-pairs (user + model) until both budgets are satisfied.
    Returns True if any turns were dropped.
    """
    if not chat_session or not getattr(chat_session, "history", None):
        return False

    original_len = len(chat_session.history)
    truncated = False

    while len(chat_session.history) > MAX_HISTORY_TURNS * 2:
        chat_session.history.pop(0)
        chat_session.history.pop(0)
        truncated = True

    while len(chat_session.history) >= 2:
        total = sum(
            estimate_tokens(m.parts[0].text) if hasattr(m, "parts") and m.parts and hasattr(m.parts[0], "text") else 0
            for m in chat_session.history
        )
        if total <= MAX_HISTORY_TOKENS:
            break
        chat_session.history.pop(0)
        chat_session.history.pop(0)
        truncated = True

    if truncated:
        turns_dropped = (original_len - len(chat_session.history)) // 2
        logger.info("Trimmed chat history: dropped %d turn pair(s)", turns_dropped)

    return truncated
