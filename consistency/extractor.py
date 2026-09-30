"""Factual claim extraction and entity recognition for Islamic texts.

Parses assertions into atomic FactualClaim objects tagged with rulings,
scholarly attributions, madhhabs, conditions, citations, and semantic topics.
"""

from __future__ import annotations

import re
import uuid

from consistency.models import ClaimType, FactualClaim, RulingType

# Sentence split regex
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?\n])\s+")

# Quran and Hadith citation regexes
QURAN_REGEX = re.compile(
    r"(?:Surah|Quran|Qur\'an)?\s*\[?\b([1-9]|[1-9]\d|1[0-0]\d|11[0-4])\s*:\s*([1-9]\d*)\b\]?",
    re.IGNORECASE,
)
HADITH_REGEX = re.compile(
    r"\b(Bukhari|Muslim|Abu Dawud|Tirmidhi|Nasa\'i|Ibn Majah|Muwatta|Ahmad)\b\s*(?:hadith|no\.|number|#)?\s*(\d+)?",
    re.IGNORECASE,
)

# Ruling keyword mapping
RULING_PATTERNS: list[tuple[re.Pattern, RulingType, bool]] = [
    # Prohibitions
    (
        re.compile(
            r"\b(haram|forbidden|prohibited|unlawful|strictly forbidden|impermissible|not allowed|invalid|nullified|nullifies|breaks wudu|breaks the fast|invalidates)\b",
            re.IGNORECASE,
        ),
        RulingType.HARAM,
        False,
    ),
    # Obligations
    (
        re.compile(r"\b(fard|obligatory|mandatory|compulsory|prescribed|incumbent)\b", re.IGNORECASE),
        RulingType.FARD,
        True,
    ),
    (re.compile(r"\b(wajib|necessary|duty)\b", re.IGNORECASE), RulingType.WAJIB, True),
    # Recommendations & Sunnah
    (
        re.compile(r"\b(sunnah|mustahabb|recommended|praiseworthy|encouraged|desirable)\b", re.IGNORECASE),
        RulingType.MUSTAHABB,
        True,
    ),
    # Disliked
    (re.compile(r"\b(makruh|disliked|discouraged|reprehensible|detested)\b", re.IGNORECASE), RulingType.MAKRUH, False),
    # Permissibility
    (
        re.compile(
            r"\b(halal|permissible|allowed|lawful|valid|mubah|does not break wudu|does not invalidate|does not nullify)\b",
            re.IGNORECASE,
        ),
        RulingType.HALAL,
        True,
    ),
]

# Madhhab patterns
MADHHAB_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(hanafi|hanafees?|abu hanifa(?:h)?)\b", re.IGNORECASE), "hanafi"),
    (re.compile(r"\b(maliki|malikees?|imam malik)\b", re.IGNORECASE), "maliki"),
    (re.compile(r"\b(shafi(?:'|)i|shafi(?:'|)ees?|imam (?:al-|)shafi(?:'|)i)\b", re.IGNORECASE), "shafii"),
    (re.compile(r"\b(hanbali|hanbalees?|ahmad ibn hanbal|imam ahmad)\b", re.IGNORECASE), "hanbali"),
]

# Scholar patterns
SCHOLAR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(abu hanifa(?:h)?|imam abu hanifa(?:h)?)\b", re.IGNORECASE), "Abu Hanifa"),
    (re.compile(r"\b(malik|imam malik|malik ibn anas)\b", re.IGNORECASE), "Malik ibn Anas"),
    (re.compile(r"\b(al-shafi(?:'|)i|imam shafi(?:'|)i|muhammad ibn idris)\b", re.IGNORECASE), "al-Shafi'i"),
    (re.compile(r"\b(ahmad ibn hanbal|imam ahmad)\b", re.IGNORECASE), "Ahmad ibn Hanbal"),
    (re.compile(r"\b(ibn taymiyyah|ibn taymiya(?:h)?|shaykh al-islam)\b", re.IGNORECASE), "Ibn Taymiyyah"),
    (re.compile(r"\b(ibn al-qayyim|ibn qayyim al-jawziyya(?:h)?)\b", re.IGNORECASE), "Ibn al-Qayyim"),
    (re.compile(r"\b(al-nawawi|imam nawawi|yahya ibn sharaf)\b", re.IGNORECASE), "al-Nawawi"),
    (re.compile(r"\b(al-ghazali|imam ghazali|hujjat al-islam)\b", re.IGNORECASE), "al-Ghazali"),
    (re.compile(r"\b(al-tabari|ibn jarir al-tabari)\b", re.IGNORECASE), "al-Tabari"),
    (re.compile(r"\b(al-qurtubi|imam qurtubi)\b", re.IGNORECASE), "al-Qurtubi"),
    (re.compile(r"\b(ibn kathir|hafiz ibn kathir)\b", re.IGNORECASE), "Ibn Kathir"),
    (re.compile(r"\b(bukhari|imam bukhari)\b", re.IGNORECASE), "al-Bukhari"),
    (re.compile(r"\b(muslim|imam muslim)\b", re.IGNORECASE), "Muslim"),
]

# Conditions detection
CONDITION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(traveler|traveller|traveling|travelling|safari?|on a journey)\b", re.IGNORECASE), "traveler"),
    (re.compile(r"\b(sick|ill|illness|disease|marad|patient)\b", re.IGNORECASE), "sick"),
    (
        re.compile(r"\b(forgetful|forgetfulness|forgot|unintentionally|accidental|by mistake|nisyan)\b", re.IGNORECASE),
        "forgetfulness",
    ),
    (
        re.compile(r"\b(coerced|coercion|forced|under duress|ikrah|necessity|darurah)\b", re.IGNORECASE),
        "necessity_duress",
    ),
    (re.compile(r"\b(intentional|intentionally|deliberate|deliberately|on purpose)\b", re.IGNORECASE), "deliberate"),
    (re.compile(r"\b(with lust|with desire|with pleasure|sensually)\b", re.IGNORECASE), "with_desire"),
    (re.compile(r"\b(without lust|without desire|without pleasure)\b", re.IGNORECASE), "without_desire"),
]

# Core topics mapping
TOPIC_KEYWORDS: list[tuple[re.Pattern, str, str]] = [
    # (Regex, canonical_topic, entity_name)
    (
        re.compile(
            r"\b(touch(?:ing|ed|es)? (?:a |one\'s |the |his |her |your |skin |any )?(?:wife|spouse|woman|female|non-mahram|skin)|skin contact|skin-to-skin)\b",
            re.IGNORECASE,
        ),
        "wudu_nullification",
        "touching_opposite_gender_wudu",
    ),
    (
        re.compile(
            r"\b(bleeding|blood flow|nosebleed|wound bleeding) (?:breaks|nullifies|invalidates)?\b", re.IGNORECASE
        ),
        "wudu_nullification",
        "bleeding_wudu",
    ),
    (re.compile(r"\b(eating camel meat|camel flesh)\b", re.IGNORECASE), "wudu_nullification", "camel_meat_wudu"),
    (
        re.compile(r"\b(sleep|sleeping|dozing|deep sleep) (?:breaks|nullifies)?\b", re.IGNORECASE),
        "wudu_nullification",
        "sleep_wudu",
    ),
    (
        re.compile(r"\b(zakat on (?:women\'s |personal |worn )?jewelry|gold jewelry)\b", re.IGNORECASE),
        "zakat",
        "zakat_jewelry",
    ),
    (
        re.compile(r"\b(zakat|nisab|2\.5%|2\.5 percent|gold nisab|silver nisab)\b", re.IGNORECASE),
        "zakat",
        "zakat_rates_nisab",
    ),
    (
        re.compile(r"\b(fasting|sawm|ramadan|eating while forgetting|forgetful fast)\b", re.IGNORECASE),
        "fasting",
        "fasting_rules",
    ),
    (
        re.compile(
            r"\b(five prayers|5 daily prayers|fajr|dhuhr|asr|maghrib|isha|number of prayers|salah|salat|daily prayer|prayers)\b",
            re.IGNORECASE,
        ),
        "prayer",
        "daily_prayers_count",
    ),
    (
        re.compile(r"\b(raising hands|raf\'? al-yadayn|raising the hands in prayer)\b", re.IGNORECASE),
        "prayer",
        "raf_al_yadayn",
    ),
    (
        re.compile(r"\b(reciting (?:surah |)fatiha behind the imam|fatihah in congregation)\b", re.IGNORECASE),
        "prayer",
        "fatiha_behind_imam",
    ),
    (re.compile(r"\b(basmalah|bismillah aloud|bismillah silently)\b", re.IGNORECASE), "prayer", "basmala_in_prayer"),
    (re.compile(r"\b(qunut (?:in |at |during )?fajr|qunut du\'?a)\b", re.IGNORECASE), "prayer", "qunut_fajr"),
    (
        re.compile(r"\b(seafood|shellfish|shrimp|crab|lobster|prawns|fish)\b", re.IGNORECASE),
        "dietary",
        "seafood_permissibility",
    ),
    (
        re.compile(r"\b(tawhid|oneness of allah|shirk|associating partners)\b", re.IGNORECASE),
        "aqeedah",
        "tawhid_principles",
    ),
    (
        re.compile(r"\b(final prophet|seal of prophets|khatam al-anbiya|last messenger)\b", re.IGNORECASE),
        "aqeedah",
        "finality_of_prophethood",
    ),
    (re.compile(r"\b(riba|interest|usury|bank interest)\b", re.IGNORECASE), "transactions", "prohibition_of_riba"),
    (
        re.compile(r"\b(alcohol|khamr|wine|liquor|intoxicants)\b", re.IGNORECASE),
        "dietary",
        "prohibition_of_intoxicants",
    ),
    (re.compile(r"\b(music|musical instruments|singing)\b", re.IGNORECASE), "culture_ethics", "music_ruling"),
]

# Numerical assertion regex
NUMERICAL_REGEX = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent\b|grams?\b|g\b|rak\'?ahs?\b|times?\b|days?\b|months?\b|years?\b|dirhams?\b|dinars?\b|camels?\b|sheep\b|cattle\b)",
    re.IGNORECASE,
)


def normalize_text_for_claim(text: str) -> str:
    """Normalize text by lowering, removing diacritics, and condensing whitespace."""
    t = text.lower().strip()
    # Normalize Alef and common transliterations
    t = re.sub(r"[أإآ]", "ا", t)
    t = t.replace("’", "'").replace("`", "'").replace("–", "-")
    t = re.sub(r"[^\w\s\':\.-]", " ", t)
    return " ".join(t.split())


def extract_claims(
    text: str,
    chat_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> list[FactualClaim]:
    """Extract structured factual claims from raw text."""
    if not text or not text.strip():
        return []

    claims: list[FactualClaim] = []
    sentences = SENTENCE_SPLIT_REGEX.split(text.strip())

    for sentence in sentences:
        s = sentence.strip()
        if len(s) < 12:
            continue

        # Skip rhetorical questions and pure hedges
        if s.endswith("?") and not any(kw in s.lower() for kw in ["is it true that", "according to"]):
            continue

        # Citations
        citations: list[str] = []
        for q_match in QURAN_REGEX.finditer(s):
            citations.append(f"Quran {q_match.group(1)}:{q_match.group(2)}")
        for h_match in HADITH_REGEX.finditer(s):
            h_col = h_match.group(1)
            h_num = h_match.group(2)
            citations.append(f"{h_col}:{h_num}" if h_num else h_col)

        # Ruling & Polarity
        ruling: RulingType | None = None
        polarity = True
        claim_type = ClaimType.FACTUAL

        for pattern, r_type, pol in RULING_PATTERNS:
            if pattern.search(s):
                ruling = r_type
                polarity = pol
                claim_type = ClaimType.RULING
                break

        # Attribution & Madhhab
        attribution: str | None = None
        madhhab: str | None = None

        for pattern, m_name in MADHHAB_PATTERNS:
            if pattern.search(s):
                madhhab = m_name
                attribution = f"{m_name.capitalize()} school"
                if claim_type == ClaimType.FACTUAL:
                    claim_type = ClaimType.ATTRIBUTION
                break

        for pattern, s_name in SCHOLAR_PATTERNS:
            if pattern.search(s):
                attribution = s_name
                if claim_type == ClaimType.FACTUAL:
                    claim_type = ClaimType.ATTRIBUTION
                break

        # Condition
        condition: str | None = None
        for pattern, cond_name in CONDITION_PATTERNS:
            if pattern.search(s):
                condition = cond_name
                break

        # Topic & Entity
        topic = "general_islamic"
        entity: str | None = None
        for pattern, top_name, ent_name in TOPIC_KEYWORDS:
            if pattern.search(s):
                topic = top_name
                entity = ent_name
                break

        # Numerical checks
        num_match = NUMERICAL_REGEX.search(s)
        if num_match:
            if claim_type in (ClaimType.FACTUAL, ClaimType.ATTRIBUTION):
                claim_type = ClaimType.NUMERICAL

        # Check for core principles
        if topic == "aqeedah" or "five pillars" in s.lower() or "pillars of islam" in s.lower():
            if claim_type in (ClaimType.FACTUAL, ClaimType.RULING):
                claim_type = ClaimType.CORE_PRINCIPLE

        claim_id = str(uuid.uuid4())
        normalized_text = normalize_text_for_claim(s)

        claim = FactualClaim(
            claim_id=claim_id,
            text=s,
            normalized_text=normalized_text,
            topic=topic,
            entity=entity,
            claim_type=claim_type,
            ruling=ruling,
            polarity=polarity,
            condition=condition,
            attribution=attribution,
            madhhab=madhhab,
            citations=citations,
            session_id=session_id,
            user_id=user_id,
            chat_id=chat_id,
        )
        claims.append(claim)

    return claims
