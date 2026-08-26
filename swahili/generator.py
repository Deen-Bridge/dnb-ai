"""Swahili Response Generation & Prompt Enhancement Engine."""

from __future__ import annotations

import logging
import re
from typing import Any

from swahili.cultural_context import cultural_context_engine
from swahili.models import SwahiliPromptEnhancement
from swahili.terminology import terminology_db

logger = logging.getLogger(__name__)

SWAHILI_SYSTEM_INSTRUCTIONS = """
MIONGOZO YA MAJIBU YA KISWAHILI (SWAHILI ISLAMIC GUIDELINES):
1. Jibu maswali yote kwa lugha safi, fasaha na sanifu ya Kiswahili chenye staha ya Kiislamu (Adabu za Kiislamu).
2. Tumia heshima na taadhima za Kiislamu kwa majina matukufu:
   - "Mwenyezi Mungu (Subhanahu wa Ta'ala / Mtukufu)" au "Allah (Subhanahu wa Ta'ala)".
   - "Mtume Muhammad (Swalla Allahu Alayhi wa Sallam / ﷺ / Rehema na Amani zimshukie)".
   - "Maswahaba (Radhi Allahu Anhum / Mwenyezi Mungu awe radhi nao)".
   - "Wanazuoni / Maimamu (Rahimahumullah / Mwenyezi Mungu awarehemu)".
3. Kila unaponukuu aya ya Qur'ani Tukufu:
   - Andika kwanza aya katika herufi asili za Kiarabu (k.m. بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ).
   - Fuatilizia na tafsiri sahihi ya Kiswahili na taja Sura na namba ya Aya (k.m. Surah Al-Baqarah 2:255).
4. Kila unaponukuu Hadithi ya Mtume ﷺ, taja mpokezi (Sahabi) na kitabu kilichopokea (k.m. Sahih al-Bukhari, Sahih Muslim) pamoja na daraja ya usahihi.
5. Zingatia muktadha wa kifiqhi wa ukanda wa Afrika Mashariki (ambapo Madhehebu ya Shafi'i ni msingi mkuu wa kihistoria) huku ukieleza pia maoni ya madhehebu mengine ya Kisunni (Hanafi, Maliki, Hanbali) panapokuwa na ikhtilafu halali za kielimu.
6. Tumia istilahi fasaha za Kiswahili: k.m. "Swala" badala ya neno la jumla tu, "Udhu/Kutawadha", "Saumu", "Zaka", "Hija", "Halali", "Haramu", "Kadhi", "Mirathi", "Ndoa/Nikahi".
"""


class SwahiliResponseEnhancer:
    """Enhances prompts and post-processes model output for natural Swahili delivery."""

    def __init__(self) -> None:
        self._term_db = terminology_db
        self._culture = cultural_context_engine

    def build_prompt_enhancement(self, user_query: str) -> SwahiliPromptEnhancement:
        """Construct prompt augmentation with detected terminology and cultural notes."""
        context = self._culture.extract_context(user_query)
        detected_terms = self._term_db.extract_terms_from_text(user_query)

        glossary: dict[str, str] = {}
        for term in detected_terms:
            glossary[term.swahili_term] = (
                f"{term.arabic_original} ({term.arabic_transliteration}) - {term.definition_sw}"
            )

        cultural_notes: list[str] = []
        if context.shafi_madhhab_relevant:
            cultural_notes.append(self._culture.get_shafi_jurisprudence_note())
        if context.local_institutions_mentioned:
            cultural_notes.append(
                f"Muktadha wa kitaasisi Afrika Mashariki: {', '.join(context.local_institutions_mentioned)}."
            )
        if context.prayer_time_context:
            cultural_notes.append(f"Wakati wa Swala: {context.prayer_time_context}.")
        if context.cultural_event_context:
            cultural_notes.append(f"Muktadha wa kiutamaduni: {context.cultural_event_context}.")

        return SwahiliPromptEnhancement(
            system_instructions=SWAHILI_SYSTEM_INSTRUCTIONS.strip(),
            contextual_glossary=glossary,
            cultural_notes=cultural_notes,
            enhanced_user_prompt=user_query,
        )

    def validate_swahili_response(self, text: str) -> dict[str, Any]:
        """Verify quality, honorifics, and citation adherence in generated response."""
        checks: dict[str, bool] = {
            "has_proper_swahili_length": len(text.strip().split()) >= 10,
            "contains_arabic_quran_quotes": bool(re.search(r"[\u0600-\u06FF]", text)),
            "mentions_prophet_honorific": bool(
                re.search(r"(ﷺ|swalla allahu alayhi wa sallam|rehema na amani)", text, re.IGNORECASE)
            )
            if "mtume" in text.lower() or "muhammad" in text.lower()
            else True,
            "mentions_allah_honorific": bool(
                re.search(r"(subhanahu wa ta'ala|mtukufu|ta'ala|jalla jalaluh)", text, re.IGNORECASE)
            )
            if "mwenyezi mungu" in text.lower() or "allah" in text.lower()
            else True,
        }

        score = sum(1 for v in checks.values() if v) / len(checks)
        return {
            "valid": score >= 0.75,
            "score": round(score, 2),
            "checks": checks,
        }


# Global singleton instance
swahili_response_enhancer = SwahiliResponseEnhancer()
