"""Canonical core knowledge anchors and classical Ikhtilaf matrix.

Contains undisputed Islamic principles (Aqeedah, Ijma) that must remain
invariant, alongside structured classical madhhab disagreement mappings
to differentiate legitimate diversity of jurisprudence from factual error.
"""

from __future__ import annotations

import re
from typing import Any

from consistency.models import (
    ConsensusLevel,
    CorePosition,
    FactualClaim,
    RulingType,
)

# ---------------------------------------------------------------------------
# Canonical Core Knowledge Anchors (Ijma & Aqeedah Invariants)
# ---------------------------------------------------------------------------

CORE_POSITIONS: dict[str, CorePosition] = {
    "tawhid_principles": CorePosition(
        topic="aqeedah",
        title="Oneness and Uniqueness of Allah (Tawhid)",
        consensus_level=ConsensusLevel.IJMA,
        orthodox_position="Allah is the Sole Creator and Sustainer, One without partners or equals. Shirk is the gravest sin.",
        valid_perspectives=["Affirmation of Allah's Oneness in Lordship, Worship, and Names & Attributes."],
        prohibited_contradictions=[
            "polytheism is permissible",
            "shirk is acceptable",
            "god has partners or children",
            "worshiping idols is valid in islam",
        ],
        citations=["Quran 112:1-4", "Quran 2:255", "Quran 4:48"],
    ),
    "finality_of_prophethood": CorePosition(
        topic="aqeedah",
        title="Finality of Prophethood (Khatam al-Nabiyyin)",
        consensus_level=ConsensusLevel.IJMA,
        orthodox_position="Prophet Muhammad (peace be upon him) is the final Prophet and Messenger; no prophet comes after him.",
        valid_perspectives=["Unanimous consensus across all Muslims and classical schools."],
        prohibited_contradictions=[
            "new prophets can emerge after muhammad",
            "prophethood continues after muhammad",
            "muhammad is not the final messenger",
        ],
        citations=["Quran 33:40", "Bukhari:3535"],
    ),
    "daily_prayers_count": CorePosition(
        topic="prayer",
        title="Five Daily Obligatory Prayers",
        consensus_level=ConsensusLevel.IJMA,
        orthodox_position="Islam prescribes exactly five obligatory (fard) daily prayers: Fajr, Dhuhr, Asr, Maghrib, and Isha.",
        valid_perspectives=["Agreed upon by unanimous consensus (Ijma) and definitive texts."],
        prohibited_contradictions=[
            "prayers are only three per day",
            "daily prayers are not obligatory",
            "fajr prayer is optional",
            "muslims must pray six fard prayers daily",
        ],
        citations=["Quran 2:43", "Bukhari:8", "Muslim:16"],
    ),
    "zakat_rates_nisab": CorePosition(
        topic="zakat",
        title="Zakat Rate and Nisab Standard",
        consensus_level=ConsensusLevel.IJMA,
        orthodox_position="Zakat on qualifying wealth is 2.5% (one fortieth) annually once reaching the nisab (equivalent to 85g gold or 595g silver).",
        valid_perspectives=["Standard rates on cash/savings and gold/silver."],
        prohibited_contradictions=[
            "zakat rate on money is 5%",
            "zakat is 10% on cash savings",
            "zakat is not an obligation",
        ],
        citations=["Quran 9:60", "Quran 9:103", "Bukhari:1454"],
    ),
    "prohibition_of_riba": CorePosition(
        topic="transactions",
        title="Prohibition of Usury and Interest (Riba)",
        consensus_level=ConsensusLevel.IJMA,
        orthodox_position="Riba (usury / interest) is strictly prohibited by decisive Quranic revelation and consensus.",
        valid_perspectives=["Categorical prohibition with consensus across all classical madhhabs."],
        prohibited_contradictions=[
            "riba is halal",
            "usury is permissible in islam",
            "interest is recommended in islamic law",
        ],
        citations=["Quran 2:275", "Quran 2:278-279", "Muslim:1598"],
    ),
    "prohibition_of_intoxicants": CorePosition(
        topic="dietary",
        title="Prohibition of Intoxicants (Khamr)",
        consensus_level=ConsensusLevel.IJMA,
        orthodox_position="Consuming alcohol and intoxicants (khamr) is strictly forbidden (haram) in all amounts.",
        valid_perspectives=["Unanimous prohibition across all schools of Islamic thought."],
        prohibited_contradictions=[
            "drinking wine is halal in moderation",
            "alcohol consumption is permissible",
            "intoxicants are not haram",
        ],
        citations=["Quran 5:90", "Muslim:2003"],
    ),
}


# ---------------------------------------------------------------------------
# Classical Ikhtilaf Knowledge Matrix (Legitimate Multi-Madhhab Divergence)
# ---------------------------------------------------------------------------

IKHTILAF_MAP: dict[str, dict[str, Any]] = {
    "touching_opposite_gender_wudu": {
        "title": "Touching Non-Mahram / Spouse and Wudu Nullification",
        "topic": "wudu_nullification",
        "evidence_base": "Differences in interpreting 'aw lāmastumu al-nisā' in Surah al-Ma'idah 5:6.",
        "positions": {
            "shafii": {
                "ruling": RulingType.NULLIFIED,
                "summary": "Skin-to-skin touch between non-mahram adult male and female nullifies wudu unconditionally without a barrier.",
                "scholar": "Imam al-Shafi'i",
                "condition": "unconditional",
            },
            "hanafi": {
                "ruling": RulingType.NOT_NULLIFIED,
                "summary": "Touching a woman does not break wudu unless it reaches the level of direct sexual intimacy (mubasharah fahishah).",
                "scholar": "Imam Abu Hanifa",
                "condition": "unless_intimacy",
            },
            "maliki": {
                "ruling": RulingType.CONDITIONAL,
                "summary": "Touching breaks wudu only if done with sensual pleasure/desire (ladhdhah) or with the intention to experience pleasure.",
                "scholar": "Imam Malik",
                "condition": "with_desire",
            },
            "hanbali": {
                "ruling": RulingType.CONDITIONAL,
                "summary": "Touching breaks wudu if done with lust/desire (shahwah); without desire it does not break wudu.",
                "scholar": "Imam Ahmad ibn Hanbal",
                "condition": "with_desire",
            },
        },
        "reconciliation_template": (
            "Regarding whether touching a spouse or non-mahram breaks wudu: this is a well-known classical "
            "difference of opinion (ikhtilaf) among the four Sunni schools. The Shafi'i school holds that any "
            "direct skin-to-skin contact invalidates wudu; the Hanafi school holds that it does not break wudu "
            "unless accompanied by sexual intimacy; while the Maliki and Hanbali schools hold that it breaks wudu "
            "only if accompanied by desire or pleasure. All four positions are recognized classical rulings."
        ),
    },
    "bleeding_wudu": {
        "title": "Bleeding and Wudu Nullification",
        "topic": "wudu_nullification",
        "evidence_base": "Interpretation of bodily impurities and relevant hadiths.",
        "positions": {
            "hanafi": {
                "ruling": RulingType.NULLIFIED,
                "summary": "Flowing blood or pus from any part of the body that exits and flows breaks wudu.",
                "scholar": "Imam Abu Hanifa",
            },
            "hanbali": {
                "ruling": RulingType.NULLIFIED,
                "summary": "Substantial or excessive amounts of flowing blood break wudu.",
                "scholar": "Imam Ahmad ibn Hanbal",
            },
            "shafii": {
                "ruling": RulingType.NOT_NULLIFIED,
                "summary": "Blood exiting from non-private passages does not invalidate wudu, regardless of amount.",
                "scholar": "Imam al-Shafi'i",
            },
            "maliki": {
                "ruling": RulingType.NOT_NULLIFIED,
                "summary": "Bleeding from outside the two primary passages does not break wudu.",
                "scholar": "Imam Malik",
            },
        },
        "reconciliation_template": (
            "Regarding bleeding and wudu: the Hanafi and Hanbali schools hold that flowing blood breaks wudu, "
            "whereas the Shafi'i and Maliki schools hold that bleeding from elsewhere on the body does not invalidate wudu."
        ),
    },
    "seafood_permissibility": {
        "title": "Permissibility of Non-Fish Marine Creatures",
        "topic": "dietary",
        "evidence_base": "Surah al-Ma'idah 5:96 and the hadith 'Its water is pure and its dead are lawful.'",
        "positions": {
            "shafii": {
                "ruling": RulingType.HALAL,
                "summary": "All marine animals that live exclusively in water are permissible to eat.",
                "scholar": "Imam al-Shafi'i",
            },
            "maliki": {
                "ruling": RulingType.HALAL,
                "summary": "All sea creatures are completely permissible without exception.",
                "scholar": "Imam Malik",
            },
            "hanbali": {
                "ruling": RulingType.HALAL,
                "summary": "All sea creatures are permissible (with minor qualifications on noxious animals).",
                "scholar": "Imam Ahmad ibn Hanbal",
            },
            "hanafi": {
                "ruling": RulingType.MAKRUH,
                "summary": "Only true fish (samak) are halal from the sea; shellfish, crabs, and squid are disliked/impermissible.",
                "scholar": "Imam Abu Hanifa",
            },
        },
        "reconciliation_template": (
            "Regarding seafood other than fish (like shrimp, crab, or lobster): the majority of scholars "
            "(Maliki, Shafi'i, and Hanbali) view all sea creatures as completely permissible (halal). "
            "In contrast, the classical Hanafi school restricts halal marine food exclusively to fish (samak)."
        ),
    },
    "zakat_jewelry": {
        "title": "Zakat on Personal Customary Women's Jewelry",
        "topic": "zakat",
        "evidence_base": "Hadiths on gold bracelets vs traditions from Aisha and Ibn Umar.",
        "positions": {
            "hanafi": {
                "ruling": RulingType.WAJIB,
                "summary": "Zakat is mandatory on all gold and silver jewelry once reaching the nisab threshold.",
                "scholar": "Imam Abu Hanifa",
            },
            "maliki": {
                "ruling": RulingType.MUBAH,
                "summary": "No zakat is due on gold/silver jewelry designated for customary personal lawful use.",
                "scholar": "Imam Malik",
            },
            "shafii": {
                "ruling": RulingType.MUBAH,
                "summary": "No zakat is due on lawful personal jewelry of customary, non-excessive weight.",
                "scholar": "Imam al-Shafi'i",
            },
            "hanbali": {
                "ruling": RulingType.MUBAH,
                "summary": "Lawful personal jewelry for customary adornment is exempt from zakat.",
                "scholar": "Imam Ahmad ibn Hanbal",
            },
        },
        "reconciliation_template": (
            "Regarding Zakat on women's personal gold/silver jewelry: the Hanafi school obligates 2.5% Zakat if it reaches "
            "the nisab, whereas the majority (Jumhur: Maliki, Shafi'i, and Hanbali) hold that personal jewelry worn for "
            "customary adornment is exempt from Zakat."
        ),
    },
    "raf_al_yadayn": {
        "title": "Raising Hands in Prayer (Raf' al-Yadayn)",
        "topic": "prayer",
        "evidence_base": "Ibn Umar hadiths vs Ibn Mas'ud hadiths.",
        "positions": {
            "shafii": {
                "ruling": RulingType.MUSTAHABB,
                "summary": "Raising hands is recommended at opening Takbir, bowing (ruku'), and rising from ruku'.",
                "scholar": "Imam al-Shafi'i",
            },
            "hanbali": {
                "ruling": RulingType.MUSTAHABB,
                "summary": "Raising hands is recommended at opening Takbir, going into ruku', and rising from ruku'.",
                "scholar": "Imam Ahmad ibn Hanbal",
            },
            "hanafi": {
                "ruling": RulingType.MUSTAHABB,
                "summary": "Raising hands is prescribed only at the opening Takbir (Takbirat al-Ihram).",
                "scholar": "Imam Abu Hanifa",
            },
            "maliki": {
                "ruling": RulingType.MUSTAHABB,
                "summary": "The mashhur position in the Maliki school is raising hands primarily at the opening Takbir.",
                "scholar": "Imam Malik",
            },
        },
        "reconciliation_template": (
            "Regarding raising the hands (Raf' al-Yadayn) during prayer: the Shafi'i and Hanbali schools practice raising "
            "hands at the opening Takbir, before ruku', and when rising from ruku'. In the Hanafi school and the well-known "
            "Maliki position, hands are raised only at the initial opening Takbir."
        ),
    },
}


def get_core_position(entity_or_topic: str) -> CorePosition | None:
    """Retrieve canonical core position by key or topic."""
    if entity_or_topic in CORE_POSITIONS:
        return CORE_POSITIONS[entity_or_topic]
    for pos in CORE_POSITIONS.values():
        if pos.topic == entity_or_topic:
            return pos
    return None


def get_ikhtilaf_entry(entity: str | None) -> dict[str, Any] | None:
    """Retrieve classical ikhtilaf definition by entity key."""
    if not entity:
        return None
    return IKHTILAF_MAP.get(entity)


def is_core_principle_violation(claim: FactualClaim) -> tuple[bool, str | None]:
    """Check if a claim contradicts an immutable core principle / Ijma."""
    norm_text = claim.normalized_text or claim.text.lower()

    for anchor in CORE_POSITIONS.values():
        for prohibited in anchor.prohibited_contradictions:
            pattern = re.escape(prohibited)
            if re.search(r"\b" + pattern + r"\b", norm_text, re.IGNORECASE):
                return True, f"Contradicts core principle '{anchor.title}': '{prohibited}'"

    return False, None
