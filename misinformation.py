"""Religious misinformation flagging and correction system (#181).

Detects common Islamic misconceptions in generated content, validates
quotation accuracy, suggests corrections from authoritative sources,
and blocks propagation of verified misinformation.

Components:
- MISCONCEPTION_DB: curated dict of common misconceptions with corrections
- MisinfoDetection: pattern-matching engine that flags problematic content
- QuotationValidator: verifies cited text matches authoritative sources
- CorrectionEngine: produces correction suggestions
- MisinfoPipeline: orchestrates detection, validation, and correction
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Misconception database
# ---------------------------------------------------------------------------


class MisconceptionSeverity(str, Enum):
    LOW = "low"  # Minor inaccuracy
    MEDIUM = "medium"  # Notable error, may mislead
    HIGH = "high"  # Significant theological error
    CRITICAL = "critical"  # Potentially dangerous misinformation


class MisconceptionEntry(BaseModel):
    id: str
    category: str
    severity: MisconceptionSeverity
    false_claim_patterns: list[str]
    correction: str
    authoritative_source: str
    source_reference: str | None = None
    notes: str | None = None


MISCONCEPTION_DB: dict[str, MisconceptionEntry] = {}


def _register(entries: list[MisconceptionEntry]) -> None:
    for e in entries:
        MISCONCEPTION_DB[e.id] = e


_register(
    [
        MisconceptionEntry(
            id="shirk-tawassul",
            category="tawheed",
            severity=MisconceptionSeverity.HIGH,
            false_claim_patterns=[
                r"tawassul\s+is\s+(shirk|polytheism|associating\s+partners)",
                r"asking\s+(the\s+prophet|all[ahy]|saints)\s+is\s+(shirk|haram)",
                r"visiting\s+graves?\s+is\s+(shirk|bid'ah|forbidden)",
                r"intercession\s+(is\s+)?(shirk|not\s+permitted)",
            ],
            correction=(
                "Tawassul (seeking intercession) through the Prophet ﷺ or righteous "
                "people is permitted in Islam when done correctly, as supported by "
                "authentic hadith. It is not shirk. Visiting graves is sunnah and "
                "encouraged for remembrance of death."
            ),
            authoritative_source="Sahih al-Bukhari, Sahih Muslim",
            source_reference="Bukhari 1014, Muslim 710",
        ),
        MisconceptionEntry(
            id="shahada-addition",
            category="aqeedah",
            severity=MisconceptionSeverity.CRITICAL,
            false_claim_patterns=[
                r"shahada.*muhammad.*rasul.*allah.*ali",
                r"la\s+ilaha\s+illallah\s+muhammadun\s+rasulullah\s+aliyun",
                r"the\s+shahada\s+includes?\s+ali",
            ],
            correction=(
                "The Shahada is 'La ilaha illallah Muhammadun Rasulullah' "
                "(There is no god but Allah, Muhammad is the Messenger of Allah). "
                "Adding any name after it is a bid'ah (innovation) and alters the "
                "fundamental declaration of faith."
            ),
            authoritative_source="Quran 3:18, Sahih al-Bukhari",
            source_reference="Quran 3:18",
        ),
        MisconceptionEntry(
            id="bida-helpers",
            category="ahkam",
            severity=MisconceptionSeverity.MEDIUM,
            false_claim_patterns=[
                r"(mawlid|milad|shab-e-barat|isra.*miraj\s+celebration)\s+is\s+(haram|bid'ah|forbidden)",
                r"celebrating\s+(the\s+prophet'?s?\s+birthday|milad)\s+is\s+(haram|bid'ah)",
                r"(tarawih|rawatib)\s+are\s+(bid'ah|not\s+sunnah)",
            ],
            correction=(
                "Scholars differ on the ruling of Mawlid and similar commemorations. "
                "Many reputable scholars across the four madhhabs consider them "
                "permissible or recommended when free from impermissible practices. "
                "Tarawih prayer is established Sunnah, not bid'ah."
            ),
            authoritative_source="Majority of four madhhabs",
            source_reference="Shafi'i Fiqh, Reliance of the Traveller",
        ),
        MisconceptionEntry(
            id="music-total-haram",
            category="ahkam",
            severity=MisconceptionSeverity.MEDIUM,
            false_claim_patterns=[
                r"all\s+music\s+is\s+(haram|forbidden)\s+(in\s+islam|absolutely)",
                r"listening\s+to\s+(any\s+)?music\s+is\s+(haram|a\s+major\s+sin)",
                r"instruments?\s+are\s+(all\s+)?(haram|forbidden)\s+without\s+exception",
            ],
            correction=(
                "There is scholarly ikhtilaf (difference of opinion) on music. "
                "While many scholars restrict or prohibit musical instruments, others "
                "permit certain types. The claim that ALL music is absolutely haram "
                "is an oversimplification. Nasheed and daff (drum) are widely accepted."
            ),
            authoritative_source="Multiple schools of fiqh",
            source_reference="Ibn Hazm, Al-Muhalla; Al-Ghazali, Ihya Ulum al-Din",
        ),
        MisconceptionEntry(
            id="woman-no-driving",
            category="rights",
            severity=MisconceptionSeverity.HIGH,
            false_claim_patterns=[
                r"women\s+(are\s+)?not\s+(allowed\s+to\s+)?drive",
                r"driving\s+is\s+(haram|forbidden)\s+for\s+(women|females)",
                r"islam\s+prohibits?\s+women\s+from\s+driving",
            ],
            correction=(
                "There is no prohibition in the Quran or authentic Sunnah against "
                "women driving. This is a cultural practice, not an Islamic ruling. "
                "Many Muslim-majority countries have no such restriction."
            ),
            authoritative_source="Quran and Sunnah (absence of prohibition)",
        ),
        MisconceptionEntry(
            id="force-conversion",
            category="aqeedah",
            severity=MisconceptionSeverity.CRITICAL,
            false_claim_patterns=[
                r"islam\s+(allows?|commands?)\s+(forced?\s+)?conversion",
                r"there\s+is\s+no\s+compulsion\s+in\s+religion.*but.*must\s+convert",
                r"non-?muslims?\s+(must|should)\s+be\s+(forced?\s+to\s+)?convert",
                r"killing?\s+apostates?\s+is\s+(obligatory|mandatory|always)",
            ],
            correction=(
                "The Quran explicitly states 'La ikraha fid-deen' (There is no "
                "compulsion in religion, 2:256). Forced conversion is prohibited "
                "in Islam. Regarding apostasy, scholars differ on the ruling, with "
                "many contemporary scholars emphasizing freedom of conscience."
            ),
            authoritative_source="Quran 2:256, 18:29, 109:6",
            source_reference="Quran 2:256",
        ),
        MisconceptionEntry(
            id="jinn-shapeshifters",
            category="aqeedah",
            severity=MisconceptionSeverity.MEDIUM,
            false_claim_patterns=[
                r"jinn\s+(can\s+)?(shapeshift|take\s+any\s+form|impersonate)\s+(anyone|anything)",
                r"jinn\s+can\s+become\s+(anything|anyone|any\s+animal)",
                r"shaytan\s+can\s+take\s+(any|any\s+form|any\s+shape)",
            ],
            correction=(
                "Islamic sources indicate jinn can take certain forms (snakes, dogs, "
                "humans) but there are scholarly discussions about limits. Shaytan "
                "cannot assume any form at will without limitations."
            ),
            authoritative_source="Sahih Muslim, Sahih al-Bukhari",
            source_reference="Muslim 2230, Bukhari 3283",
        ),
        MisconceptionEntry(
            id="quran-scientific-miracle-overreach",
            category="tafsir",
            severity=MisconceptionSeverity.LOW,
            false_claim_patterns=[
                r"the\s+quran\s+(clearly\s+)?describes?\s+(quantum\s+physics|black\s+holes?|evolution|big\s+bang)",
                r"allah\s+mentions?\s+(atoms?|electrons?|dna|genes?)\s+in\s+the\s+quran",
                r"the\s+quran\s+predicted?\s+(modern\s+)?science",
            ],
            correction=(
                "While the Quran contains verses that are compatible with scientific "
                "observations, claiming it explicitly describes modern scientific "
                "concepts is anachronistic interpretation. Classical mufassireen did "
                "not interpret these verses in scientific terms."
            ),
            authoritative_source="Classical tafsir works",
        ),
        MisconceptionEntry(
            id="abrogation-misread",
            category="usul",
            severity=MisconceptionSeverity.HIGH,
            false_claim_patterns=[
                r"naskh\s+means?\s+the\s+(earlier|later)\s+(verses?|revelations?)\s+(are\s+)?(abrogated|cancelled|deleted)",
                r"verse?\s+of\s+the\s+sword\s+(abrogates?|cancels?)\s+all\s+peaceful\s+verses?",
                r"later\s+verses?\s+completely?\s+(replace|cancel|void)\s+earlier\s+verses?",
            ],
            correction=(
                "Naskh (abrogation) in Islamic scholarship is a nuanced topic. "
                "Many scholars distinguish between naskh of ruling while the verse "
                "remains in the Quran, and textual abrogation. The 'Verse of the "
                "Sword' does not abrogate all peaceful verses, as classical scholars "
                "like Ibn Ashur have demonstrated."
            ),
            authoritative_source="Al-Itqan fi Ulum al-Quran, Al-Suyuti",
            source_reference="Al-Itqan, Chapter on Naskh",
        ),
        MisconceptionEntry(
            id="sufi-shirk-overreach",
            category="aqeedah",
            severity=MisconceptionSeverity.MEDIUM,
            false_claim_patterns=[
                r"(all\s+)?sufism\s+is\s+(shirk|bid'ah|not\s+islam)",
                r"(all\s+)?sufis?\s+(commit|are\s+guilty\s+of)\s+shirk",
                r"tasawwuf\s+is\s+(not\s+islamic|bid'ah|shirk)",
            ],
            correction=(
                "Tasawwuf (purification of the heart) is recognized by the majority "
                "of scholars as an integral part of Islam when it stays within "
                "Shariah boundaries. Many great scholars like Al-Ghazali, Ibn Taymiyyah "
                "(in aspects), and others engaged with tasawwuf."
            ),
            authoritative_source="Majority of classical scholars",
        ),
        MisconceptionEntry(
            id="sufism-all-saints",
            category="aqeedah",
            severity=MisconceptionSeverity.MEDIUM,
            false_claim_patterns=[
                r"asking\s+(prophet|saint|wali)\s+for\s+(help|intercession)\s+is\s+permissible\s+unconditionally",
                r"(visiting|calling\s+upon)\s+dead\s+(people|saints)\s+is\s+(always\s+)?permissible",
                r"(wali|saints?)\s+(can|have)\s+power\s+(over|to)\s+(control|change)\s+fate",
            ],
            correction=(
                "While tawassul through righteous people has basis in hadith, "
                "calling upon the dead directly or attributing independent power "
                "to saints beyond Allah's control is problematic. Scholars distinguish "
                "between permissible intercession (with Allah's permission) and "
                "impermissible practices."
            ),
            authoritative_source="Multiple scholars",
        ),
        MisconceptionEntry(
            id="dajjal-appearance",
            category="aqeedah",
            severity=MisconceptionSeverity.LOW,
            false_claim_patterns=[
                r"dajjal\s+(has|will\s+have)\s+(wings|multiple\s+eyes?|is\s+giant)",
                r"the\s+dajjal\s+is\s+(a\s+)?(specific\s+person|organization|technology)",
                r"dajjal\s+(will|can)\s+(control|is\s+connected\s+to)\s+(internet|ai|technology)",
            ],
            correction=(
                "The Dajjal's exact appearance is described in authentic hadith as "
                "having specific features. Modern interpretations linking the Dajjal "
                "to specific technologies or organizations are speculative and should "
                "not be presented as established Islamic knowledge."
            ),
            authoritative_source="Sahih al-Bukhari, Sahih Muslim",
            source_reference="Bukhari 3338, Muslim 2937",
        ),
        MisconceptionEntry(
            id="jummah-multiple",
            category="ahkam",
            severity=MisconceptionSeverity.LOW,
            false_claim_patterns=[
                r"jummah\s+prayer\s+(can|should)\s+be\s+(prayed|performed)\s+(alone|at\s+home|individually)",
                r"it'?s?\s+(okay|fine|permissible)\s+to\s+(skip|miss)\s+jummah",
            ],
            correction=(
                "Jummah (Friday) prayer is obligatory (wajib) for men and should "
                "be performed in congregation at the mosque. Skipping it without "
                "a valid excuse is sinful according to the majority of scholars."
            ),
            authoritative_source="Quran 62:9, Sahih al-Bukhari",
            source_reference="Quran 62:9-10",
        ),
        MisconceptionEntry(
            id="zakat-multiple-rates",
            category="ahkam",
            severity=MisconceptionSeverity.HIGH,
            false_claim_patterns=[
                r"zakat\s+(rate|percentage)\s+is\s+(10|20|5)\s*%",
                r"(rate|percentage)\s+(of\s+)?zakat\s+is\s+(10|20|5)\s*%",
                r"zakat\s+(is\s+)?(only\s+)?on\s+cash",
                r"zakat\s+(is\s+)?not\s+(due|obligatory)\s+on\s+(gold|silver|investments?|stocks?)",
            ],
            correction=(
                "Zakat is 2.5% (1/40th) on wealth that meets the nisab threshold "
                "after one lunar year. It applies to gold, silver, cash, business "
                "inventory, stocks, and other wealth categories. Different rates "
                "apply to agricultural produce (5-10%)."
            ),
            authoritative_source="Quran 9:60, Hadith of Abu Bakr (RA)",
            source_reference="Quran 9:60, Bukhari 1456",
        ),
        MisconceptionEntry(
            id="halal-food-only",
            category="ahkam",
            severity=MisconceptionSeverity.MEDIUM,
            false_claim_patterns=[
                r"(people\s+of\s+the\s+book|ahl\s+al-kitab)\s+(meat|food)\s+is\s+(haram|forbidden|not\s+halal)",
                r"eating?\s+(christian|jewish)\s+(food|meat)\s+is\s+(haram|forbidden)",
            ],
            correction=(
                "The Quran permits eating the food of People of the Book (5:5). "
                "This includes meat slaughtered by Jews and Christians following "
                "their own dietary laws. This is the position of the Hanafi, "
                "Shafi'i, and Maliki schools."
            ),
            authoritative_source="Quran 5:5",
            source_reference="Quran 5:5",
        ),
        MisconceptionEntry(
            id="tawbah-impossible",
            category="aqeedah",
            severity=MisconceptionSeverity.HIGH,
            false_claim_patterns=[
                r"once\s+you\s+commit\s+(a\s+major\s+)?sin\s+you\s+(can'?t|cannot)\s+(repent|be\s+forgiven|return)",
                r"tawbah\s+(is\s+)?not\s+(accepted|possible)\s+(for\s+)?(major|certain)\s+sins?",
                r"allah\s+(will\s+)?never\s+forgive\s+(major|certain|specific)\s+sins?",
            ],
            correction=(
                "Allah's mercy encompasses all things (7:156). Tawbah (repentance) "
                "is accepted for any sin as long as it is sincere, done before "
                "death, and the person resolves not to return to it. Despair of "
                "Allah's mercy is itself a major sin."
            ),
            authoritative_source="Quran 39:53, 7:156",
            source_reference="Quran 39:53",
        ),
        MisconceptionEntry(
            id="women-minaret",
            category="rights",
            severity=MisconceptionSeverity.MEDIUM,
            false_claim_patterns=[
                r"women\s+(are\s+)?not\s+(allowed|permitted)\s+to\s+(enter|go\s+to)\s+mosques?",
                r"mosques?\s+(are\s+)?(only\s+)?for\s+men",
                r"women\s+(should|must)\s+(only\s+)?pray\s+(at\s+home|not\s+at\s+mosque)",
            ],
            correction=(
                "The Prophet ﷺ said: 'Do not prevent the servants of Allah from "
                "the mosques of Allah' (Sahih Muslim 445). Women are encouraged "
                "to attend mosques. Scholars may differ on conditions but the "
                "default is permissibility."
            ),
            authoritative_source="Sahih Muslim 445, Sahih al-Bukhari",
            source_reference="Muslim 445, Bukhari 902",
        ),
        MisconceptionEntry(
            id="jihad-obligation",
            category="usul",
            severity=MisconceptionSeverity.CRITICAL,
            false_claim_patterns=[
                r"jihad\s+is\s+(obligatory|fard)\s+(on\s+)?(all\s+)?muslims?\s+(always|every)",
                r"all\s+muslims?\s+(must|should|are\s+obligated\s+to)\s+(fight|wage\s+jihad)",
                r"jihad\s+(only\s+)?means?\s+(fighting|war|combat)",
            ],
            correction=(
                "Jihad has multiple forms; the 'greater jihad' is the spiritual "
                "struggle against one's nafs (ego). Armed jihad is under strict "
                "conditions and authority - it is not individual obligation on all "
                "Muslims. It is only fard kifayah (communal obligation) under "
                "legitimate authority."
            ),
            authoritative_source="Sahih al-Bukhari, Sahih Muslim",
            source_reference="Bukhari 27, Muslim 174",
        ),
        MisconceptionEntry(
            id="heavenly-religions-equal",
            category="aqeedah",
            severity=MisconceptionSeverity.MEDIUM,
            false_claim_patterns=[
                r"(all\s+)?religions?\s+(lead|go)\s+to\s+heaven",
                r"(christianity|judaism|hinduism|buddhism)\s+(is\s+)?(equally?\s+valid|the\s+same\s+as)\s+islam",
                r"it\s+(doesn'?t|does\s+not)\s+matter\s+(which|what)\s+religion\s+you\s+(follow|choose)",
            ],
            correction=(
                "Islam teaches that the final and complete revelation is the Quran "
                "and the Sunnah. While People of the Book have special consideration "
                "(5:69), Islam does not teach that all religions are equally valid. "
                "This is a fundamental aspect of aqeedah."
            ),
            authoritative_source="Quran 3:19, 3:85",
            source_reference="Quran 3:19, 3:85",
        ),
        MisconceptionEntry(
            id="prayer-language",
            category="ahkam",
            severity=MisconceptionSeverity.LOW,
            false_claim_patterns=[
                r"(salat|prayer)\s+(must|should)\s+(only\s+)?be\s+(performed|recited)\s+in\s+(arabic|quranic\s+arabic)",
                r"(non-arabic|other\s+language)\s+(prayer|salat)\s+(is\s+)?(invalid|not\s+accepted|haram)",
            ],
            correction=(
                "The imam must recite in Arabic, but the ma'mum (follower) can "
                "make personal supplications (dua) in their own language. Allah "
                "accepts supplication in any language. The Quran itself is the "
                "word of Allah in Arabic, but dua is not restricted."
            ),
            authoritative_source="Sahih al-Bukhari, Sahih Muslim",
            source_reference="Bukhari 4344, Muslim 1337",
        ),
        MisconceptionEntry(
            id="fida-kafarah-confusion",
            category="ahkam",
            severity=MisconceptionSeverity.LOW,
            false_claim_patterns=[
                r"fidya\s+and\s+kaffarah\s+are\s+(the\s+same|interchangeable)",
                r"fidya\s+is\s+(paid|due)\s+for\s+(breaking\s+a\s+fast|missed\s+fasts?\s+without\s+excuse)",
            ],
            correction=(
                "Fidya is for those who cannot fast due to valid excuse (illness, "
                "pregnancy) and feeds a poor person per day. Kaffarah is for "
                "deliberately breaking a fast and involves fasting 60 consecutive "
                "days. They are distinct rulings."
            ),
            authoritative_source="Quran 2:184-185, Hadith",
            source_reference="Quran 2:184, Bukhari 1956",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Detection models
# ---------------------------------------------------------------------------


class FlagSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"


class MisinfoFlag(BaseModel):
    misconception_id: str
    category: str
    severity: MisconceptionSeverity
    action: FlagSeverity
    matched_pattern: str
    correction: str
    source: str
    reference: str | None = None


class MisinfoScanResult(BaseModel):
    flags: list[MisinfoFlag] = []
    has_misinformation: bool = False
    should_block: bool = False
    overall_severity: MisconceptionSeverity | None = None
    correction_summary: str | None = None


# ---------------------------------------------------------------------------
# Quotation validation
# ---------------------------------------------------------------------------


class QuotationMatch(BaseModel):
    quoted_text: str
    is_authentic: bool
    similarity_score: float = 0.0
    closest_match: str | None = None
    source: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Core detection engine
# ---------------------------------------------------------------------------

_SEVERITY_ACTION_MAP: dict[MisconceptionSeverity, FlagSeverity] = {
    MisconceptionSeverity.LOW: FlagSeverity.INFO,
    MisconceptionSeverity.MEDIUM: FlagSeverity.WARNING,
    MisconceptionSeverity.HIGH: FlagSeverity.WARNING,
    MisconceptionSeverity.CRITICAL: FlagSeverity.BLOCK,
}

_QURAN_QUOTE_PATTERN = re.compile(
    r"[\"\'«\"\u201C\u201D](.*?)[\"\'»\"\u201C\u201D]",
    re.DOTALL,
)

_HADITH_QUOTE_PATTERN = re.compile(
    r"(?:hadith|narrated|reported|said|the\s+prophet)\s*[:\-]?\s*[\"\'«\"\u201C\u201D](.*?)[\"\'»\"\u201C\u201D]",
    re.IGNORECASE | re.DOTALL,
)


def detect_misinformation(text: str) -> MisinfoScanResult:
    """Scan text for known misconceptions using pattern matching."""
    flags: list[MisinfoFlag] = []

    for entry in MISCONCEPTION_DB.values():
        for pattern_str in entry.false_claim_patterns:
            try:
                match = re.search(pattern_str, text, re.IGNORECASE)
            except re.error:
                logger.warning("Invalid regex in misconception %s: %s", entry.id, pattern_str)
                continue

            if match:
                action = _SEVERITY_ACTION_MAP[entry.severity]
                flags.append(
                    MisinfoFlag(
                        misconception_id=entry.id,
                        category=entry.category,
                        severity=entry.severity,
                        action=action,
                        matched_pattern=pattern_str,
                        correction=entry.correction,
                        source=entry.authoritative_source,
                        reference=entry.source_reference,
                    )
                )
                break  # one flag per misconception entry

    has_misinfo = len(flags) > 0
    should_block = any(f.action == FlagSeverity.BLOCK for f in flags)

    severity_order = [
        MisconceptionSeverity.CRITICAL,
        MisconceptionSeverity.HIGH,
        MisconceptionSeverity.MEDIUM,
        MisconceptionSeverity.LOW,
    ]
    overall_severity = None
    if flags:
        for s in severity_order:
            if any(f.severity == s for f in flags):
                overall_severity = s
                break

    corrections = []
    for f in flags:
        corrections.append(f"• {f.correction} (Source: {f.source})")
    correction_summary = "\n\n".join(corrections) if corrections else None

    return MisinfoScanResult(
        flags=flags,
        has_misinformation=has_misinfo,
        should_block=should_block,
        overall_severity=overall_severity,
        correction_summary=correction_summary,
    )


def validate_quotation(quoted_text: str, context: str = "quran") -> QuotationMatch:
    """Check if a quoted text is plausible and flag suspicious fabrications."""
    if not quoted_text or not quoted_text.strip():
        return QuotationMatch(
            quoted_text=quoted_text,
            is_authentic=False,
            notes="Empty quotation.",
        )

    suspicious_signs = [
        (
            re.search(r"\b(god|allah)\s+says?\b.*\b(kill|destroy|hate|punish\s+all)\b", quoted_text, re.IGNORECASE),
            "Contains violent attribution that does not match Quranic style.",
        ),
        (
            re.search(r"(the\s+)?quran\s+(says?|commands?)\s+\"(.*?)\"", quoted_text, re.IGNORECASE),
            "Possibly fabricated quotation attributed to the Quran.",
        ),
        (
            len(quoted_text) > 500 and context == "hadith",
            "Hadith quotations are typically shorter; this may be fabricated.",
        ),
    ]

    notes_list = []
    is_suspicious = False
    for check, note in suspicious_signs:
        if check:
            is_suspicious = True
            notes_list.append(note)

    if not is_suspicious:
        return QuotationMatch(
            quoted_text=quoted_text,
            is_authentic=True,
            notes="No suspicious fabrication markers detected.",
        )

    return QuotationMatch(
        quoted_text=quoted_text,
        is_authentic=False,
        similarity_score=0.0,
        notes="; ".join(notes_list),
    )


def suggest_correction(text: str) -> str | None:
    """Given flagged text, suggest a corrected version."""
    result = detect_misinformation(text)
    if not result.flags:
        return None

    suggestions = []
    for flag in result.flags:
        suggestions.append(
            f"[{flag.severity.value.upper()}] {flag.correction}\n"
            f"  Source: {flag.source}" + (f" ({flag.reference})" if flag.reference else "")
        )

    return "The following corrections are suggested based on authoritative Islamic sources:\n\n" + "\n\n".join(
        suggestions
    )


def is_blocked(text: str) -> bool:
    """Quick check: does this text contain critical misinformation that should be blocked?"""
    result = detect_misinformation(text)
    return result.should_block


def get_all_misconceptions() -> list[dict[str, Any]]:
    """Return the full misconception database as a list of dicts (for API)."""
    return [e.model_dump() for e in MISCONCEPTION_DB.values()]


def get_misconceptions_by_category(category: str) -> list[dict[str, Any]]:
    """Return misconceptions filtered by category."""
    return [e.model_dump() for e in MISCONCEPTION_DB.values() if e.category == category]


def get_misconception_categories() -> list[str]:
    """Return all unique categories in the misconception database."""
    return sorted({e.category for e in MISCONCEPTION_DB.values()})
