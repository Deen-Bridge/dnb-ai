"""Code-Switching Detection & Multi-Lingual Segmentation for Swahili-Arabic-English."""

from __future__ import annotations

import logging
import re

from swahili.models import CodeSwitchResult, CodeSwitchSegment, CodeSwitchType

logger = logging.getLogger(__name__)

_INSHA_ALLAH_GLOSS = "Mwenyezi Mungu akipenda (Insha'Allah)"
_MASHA_ALLAH_GLOSS = "Alivyopenda Mwenyezi Mungu (Masha'Allah)"
_SUBHANALLAH_GLOSS = "Ametakasika Mwenyezi Mungu (Subhanallah)"
_JAZAKALLAH_GLOSS = "Mwenyezi Mungu akulipe kheri (Jazakallahu Khayran)"
_ASSALAMU_ALAYKUM_GLOSS = "Amani iwe juu yenu (Assalamu Alaykum)"

# Common Islamic formulas and greetings in Arabic and transliteration
ISLAMIC_FORMULAS: dict[str, str] = {
    "bismillah": "Kwa jina la Mwenyezi Mungu (Bismillah)",
    "bismillahi rahmani rahiim": "Kwa jina la Mwenyezi Mungu Mwingi wa Rehema Mwenye Kurehemu",
    "alhamdulillah": "Sifa zote njema ni za Mwenyezi Mungu (Alhamdulillah)",
    "inshallah": _INSHA_ALLAH_GLOSS,
    "insha allah": _INSHA_ALLAH_GLOSS,
    "in sha allah": _INSHA_ALLAH_GLOSS,
    "masha allah": _MASHA_ALLAH_GLOSS,
    "mashallah": _MASHA_ALLAH_GLOSS,
    "subhanallah": _SUBHANALLAH_GLOSS,
    "subhan allah": _SUBHANALLAH_GLOSS,
    "allahu akbar": "Mwenyezi Mungu ni Mkubwa zaidi (Allahu Akbar)",
    "astaghfirullah": "Namwomba Mwenyezi Mungu msamaha (Astaghfirullah)",
    "astaghfirullah wa atubu ilayh": "Namwomba Mungu msamaha na kutubu Kwake",
    "jazakallah khair": _JAZAKALLAH_GLOSS,
    "jazakallahu khayran": _JAZAKALLAH_GLOSS,
    "jazakillahu khayran": _JAZAKALLAH_GLOSS,
    "jazakumullahu khayran": "Mwenyezi Mungu awalipeni kheri",
    "barakallahu feek": "Mwenyezi Mungu akubariki (Barakallahu Feek)",
    "barakallahu feekum": "Mwenyezi Mungu awabariki",
    "assalamu alaykum": _ASSALAMU_ALAYKUM_GLOSS,
    "as-salamu alaykum": _ASSALAMU_ALAYKUM_GLOSS,
    "assalamu alaikum": "Amani iwe juu yenu",
    "wa alaykumussalam": "Na nanyi amani iwe juu yenu",
    "la ilaha illallah": "Hapana mungu wa haki ila Allah (La ilaha illallah)",
    "radhi allahu anhu": "Mwenyezi Mungu awe radhi naye (Radhi Allahu Anhu)",
    "radhi allahu anha": "Mwenyezi Mungu awe radhi naye (wa kike)",
    "radhi allahu anhum": "Mwenyezi Mungu awe radhi nao (wingi)",
    "sallallahu alayhi wa sallam": "Rehema na amani za Allah zimshukie (Sallallahu Alayhi wa Sallam)",
    "alaihis salam": "Amani iwe juu yake (Alayhis Salam)",
    "subhanahu wa ta'ala": "Ametakasika na Ametukuka (Subhanahu wa Ta'ala)",
}

# English trigger words common in East African bilingual conversations
COMMON_ENGLISH_WORDS = frozenset(
    {
        "the",
        "is",
        "are",
        "what",
        "how",
        "why",
        "can",
        "should",
        "ruling",
        "about",
        "and",
        "or",
        "between",
        "difference",
        "trading",
        "crypto",
        "forex",
        "online",
        "business",
        "prayer",
        "fasting",
        "halal",
        "haram",
        "allowed",
        "forbidden",
        "according",
        "to",
        "please",
        "explain",
        "guide",
        "rules",
    }
)


class CodeSwitchingProcessor:
    """Analyzes multi-lingual code-switching in Swahili Islamic queries."""

    def _collect_segment_stats(
        self, segments: list[CodeSwitchSegment]
    ) -> tuple[list[str], int, int, int, bool, bool, bool]:
        arabic_phrases: list[str] = []
        sw_tokens = 0
        en_tokens = 0
        ar_tokens = 0
        has_arabic_formula = False
        has_english = False
        has_swahili = False

        for seg in segments:
            tokens = len(seg.text.split())
            if seg.is_islamic_formula or seg.language == "ar":
                arabic_phrases.append(seg.text)
                ar_tokens += tokens
                if seg.is_islamic_formula:
                    has_arabic_formula = True
            elif seg.language == "en":
                has_english = True
                en_tokens += tokens
            elif seg.language == "sw":
                has_swahili = True
                sw_tokens += tokens

        return (
            arabic_phrases,
            sw_tokens,
            en_tokens,
            ar_tokens,
            has_arabic_formula,
            has_english,
            has_swahili,
        )

    def _determine_switch_type(
        self,
        has_english: bool,
        has_swahili: bool,
        has_arabic_formula: bool,
        has_arabic_script: bool,
        sw_tokens: int,
        en_tokens: int,
        ar_tokens: int,
    ) -> CodeSwitchType:
        if (has_english and has_swahili and (has_arabic_formula or has_arabic_script)) or (
            en_tokens > 0 and sw_tokens > 0 and ar_tokens > 0
        ):
            return CodeSwitchType.TRILINGUAL_MIXED
        if has_english and (has_swahili or not has_arabic_script):
            return CodeSwitchType.SWAHILI_ENGLISH_MIXED
        if has_arabic_formula or has_arabic_script or ar_tokens > 0:
            return CodeSwitchType.SWAHILI_ARABIC_MIXED
        return CodeSwitchType.MONOLINGUAL_SWAHILI

    def _determine_dominant_language(self, sw_tokens: int, en_tokens: int, ar_tokens: int) -> str:
        total_tokens = sw_tokens + en_tokens + ar_tokens
        if total_tokens == 0:
            return "sw"
        if en_tokens > sw_tokens and en_tokens > ar_tokens:
            return "en"
        if ar_tokens > sw_tokens and ar_tokens > en_tokens:
            return "ar"
        return "sw"

    def analyze_code_switching(self, text: str) -> CodeSwitchResult:
        """Segment and classify language distribution in text."""
        segments = self.segment_languages(text)
        has_arabic_script = bool(re.search(r"[\u0600-\u06FF]", text))

        (
            arabic_phrases,
            sw_tokens,
            en_tokens,
            ar_tokens,
            has_arabic_formula,
            has_english,
            has_swahili,
        ) = self._collect_segment_stats(segments)

        switch_type = self._determine_switch_type(
            has_english=has_english,
            has_swahili=has_swahili,
            has_arabic_formula=has_arabic_formula,
            has_arabic_script=has_arabic_script,
            sw_tokens=sw_tokens,
            en_tokens=en_tokens,
            ar_tokens=ar_tokens,
        )

        dominant = self._determine_dominant_language(sw_tokens, en_tokens, ar_tokens)

        contains_quran_or_hadith = bool(
            re.search(r"\b(surah?|ayah?|quran|kurani|hadith|hadithi|bukhari|muslim)\b", text, re.IGNORECASE)
            or has_arabic_script
        )
        contains_dua = bool(re.search(r"\b(dua|du'a|kuomba|allahumma|rabbana|allahummaghfir)\b", text, re.IGNORECASE))

        return CodeSwitchResult(
            dominant_language=dominant,
            switch_type=switch_type,
            segments=segments,
            arabic_phrases=arabic_phrases,
            contains_quran_or_hadith=contains_quran_or_hadith,
            contains_dua=contains_dua,
        )

    def _segment_formulas(self, text: str, lower_text: str) -> list[CodeSwitchSegment]:
        segments: list[CodeSwitchSegment] = []
        for formula, gloss in ISLAMIC_FORMULAS.items():
            pattern = r"\b" + re.escape(formula) + r"\b"
            for match in re.finditer(pattern, lower_text):
                matched_text = text[match.start() : match.end()]
                segments.append(
                    CodeSwitchSegment(
                        text=matched_text,
                        language="ar",
                        is_islamic_formula=True,
                        gloss=gloss,
                    )
                )
        return segments

    def _segment_arabic_script(self, text: str) -> list[CodeSwitchSegment]:
        segments: list[CodeSwitchSegment] = []
        arabic_script_matches = list(re.finditer(r"[\u0600-\u06FF\s]+", text))
        for m in arabic_script_matches:
            ar_text = m.group().strip()
            if len(ar_text) > 1:
                segments.append(
                    CodeSwitchSegment(
                        text=ar_text,
                        language="ar",
                        is_islamic_formula=False,
                        gloss="Arabic Script Quotation",
                    )
                )
        return segments

    def _segment_words(self, text: str) -> list[CodeSwitchSegment]:
        segments: list[CodeSwitchSegment] = []
        words = text.split()
        current_en: list[str] = []
        current_sw: list[str] = []

        for word in words:
            w_clean = re.sub(r"[^\w]", "", word).lower()
            if not w_clean:
                continue

            if w_clean in COMMON_ENGLISH_WORDS:
                if current_sw:
                    segments.append(CodeSwitchSegment(text=" ".join(current_sw), language="sw"))
                    current_sw = []
                current_en.append(word)
            else:
                if current_en:
                    segments.append(CodeSwitchSegment(text=" ".join(current_en), language="en"))
                    current_en = []
                current_sw.append(word)

        if current_en:
            segments.append(CodeSwitchSegment(text=" ".join(current_en), language="en"))
        if current_sw:
            segments.append(CodeSwitchSegment(text=" ".join(current_sw), language="sw"))

        return segments

    def segment_languages(self, text: str) -> list[CodeSwitchSegment]:
        """Split text into classified language segments."""
        lower_text = text.lower()
        return self._segment_formulas(text, lower_text) + self._segment_arabic_script(text) + self._segment_words(text)


# Global singleton instance
code_switch_processor = CodeSwitchingProcessor()
