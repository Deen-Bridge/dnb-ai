"""East African Islamic Cultural Context & Legal Heritage Engine."""

from __future__ import annotations

import logging
import re

from swahili.models import CulturalContext

logger = logging.getLogger(__name__)

# East African Islamic Institutions
LOCAL_INSTITUTIONS = {
    "bakwata": "Baraza Kuu la Waislamu Tanzania (BAKWATA)",
    "kadhi mkuu": "Ofisi ya Kadhi Mkuu (Chief Kadhi Kenya / Zanzibar)",
    "mahakama ya kadhi": "Mahakama ya Kadhi (Kadhi Courts of Kenya / Zanzibar)",
    "supkem": "Supreme Council of Kenya Muslims (SUPKEM)",
    "umsc": "Uganda Muslim Supreme Council (UMSC)",
    "mufti zanzibar": "Ofisi ya Mufti Mkuu wa Zanzibar",
    "baraza la wanazuoni": "Baraza Kuu la Wanazuoni wa Kiislamu Afrika Mashariki",
    "riyadha": "Msikiti na Taasisi ya Riyadha Lamu",
}

# Swahili prayer time names
PRAYER_TIME_MAP = {
    "alfajiri": "Fajr (Alfajiri)",
    "adhuhuri": "Dhuhr (Adhuhuri)",
    "alasiri": "Asr (Alasiri)",
    "magharibi": "Maghrib (Magharibi)",
    "isha": "Isha (Isha)",
    "ijumaa": "Jumu'ah (Swala ya Ijumaa)",
}

# Cultural celebrations & practices
CULTURAL_EVENTS = {
    "maulidi": "Maulidi ya Mtume (Mawlid al-Nabi commemoration)",
    "ramadhani": "Mwezi Mtukufu wa Ramadhani (Ramadan Fasting)",
    "futari": "Kufuturu (Iftar meal tradition)",
    "daku": "Kula Daku (Suhoor pre-dawn meal)",
    "iddi ndogo": "Iddi ndogo (Eid al-Fitr)",
    "iddi kubwa": "Iddi kubwa / Sikukuu ya Kuchinja (Eid al-Adha)",
    "sikukuu ya kuchinja": "Sikukuu ya Kuchinja (Eid al-Adha)",
    "ndoa ya kiislamu": "Ndoa ya Kiislamu (Nikah traditions under Kadhi)",
    "mirathi": "Mgawanyo wa Mirathi (Estate inheritance under Kadhi courts)",
}


class EastAfricanIslamicContext:
    """Extracts and formats East African Islamic cultural context for Swahili queries."""

    def extract_context(self, text: str) -> CulturalContext:
        """Extract regional institutional, fiqh, and cultural cues."""
        lower_text = text.lower()

        # 1. Institutions
        found_institutions: list[str] = []
        for inst_key, inst_name in LOCAL_INSTITUTIONS.items():
            if re.search(r"\b" + re.escape(inst_key) + r"\b", lower_text):
                found_institutions.append(inst_name)

        # 2. Prayer times
        found_prayer: str | None = None
        for p_key, p_name in PRAYER_TIME_MAP.items():
            if re.search(r"\b" + re.escape(p_key) + r"\b", lower_text):
                found_prayer = p_name
                break

        # 3. Cultural events
        found_event: str | None = None
        for ev_key, ev_name in CULTURAL_EVENTS.items():
            if re.search(r"\b" + re.escape(ev_key) + r"\b", lower_text):
                found_event = ev_name
                break

        # 4. Shafi'i heritage relevance
        shafi_signals = [
            "shafi",
            "shafii",
            "shafi'i",
            "pwani",
            "zanzibar",
            "mombasa",
            "lamu",
            "tanga",
            "dar es salaam",
            "kadhi",
            "qunut ya alfajiri",
            "kugusa mwanamke",
            "kutawadha",
            "dagaa",
            "vyakula vya baharini",
        ]
        is_shafi_relevant = any(re.search(r"\b" + re.escape(sig) + r"\b", lower_text) for sig in shafi_signals)

        # If general fiqh query without explicit madhhab, East African default is Shafi'i
        if not is_shafi_relevant:
            fiqh_keywords = ["hukumu", "sharti", "nguzo", "batili", "swala", "udhu", "saumu", "zaka", "ndoa", "talaka"]
            if any(re.search(r"\b" + re.escape(fk) + r"\b", lower_text) for fk in fiqh_keywords):
                is_shafi_relevant = True

        # 5. Honorifics guidance
        honorifics = [
            "Mwenyezi Mungu (Subhanahu wa Ta'ala / Mtukufu)",
            "Mtume Muhammad (Swalla Allahu Alayhi wa Sallam / Rehema na amani zimshukie)",
            "Maswahaba (Radhi Allahu Anhum)",
            "Wanazuoni (Rahimahumullah / Mwenyezi Mungu awarehemu)",
        ]

        return CulturalContext(
            shafi_madhhab_relevant=is_shafi_relevant,
            local_institutions_mentioned=found_institutions,
            prayer_time_context=found_prayer,
            cultural_event_context=found_event,
            honorifics_guidance=honorifics,
            recommended_madhhab="shafii" if is_shafi_relevant else None,
        )

    def get_shafi_jurisprudence_note(self) -> str:
        """Returns standard educational context on East African Shafi'i school heritage."""
        return (
            "Katika ukanda wa Afrika Mashariki na Pwani ya Waswahili, msingi mkuu wa kifiqhi kihistoria "
            "umekuwa Madhehebu ya Imam al-Shafi'i (Rahimahullah), huku pia madhehebu mengine ya Kisunni "
            "(Hanafi, Maliki, Hanbali) yakiheshimiwa."
        )


# Global singleton instance
cultural_context_engine = EastAfricanIslamicContext()
