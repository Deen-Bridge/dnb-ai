"""Code-Switching Detection & Multi-Lingual Segmentation for Swahili-Arabic-English."""

from __future__ import annotations

import logging
import re

from swahili.models import CodeSwitchResult, CodeSwitchSegment, CodeSwitchType

logger = logging.getLogger(__name__)

# Common Islamic formulas and greetings in Arabic and transliteration
ISLAMIC_FORMULAS: dict[str, str] = {
    "bismillah": "Kwa jina la Mwenyezi Mungu (Bismillah)",
    "bismillahi rahmani rahiim": "Kwa jina la Mwenyezi Mungu Mwingi wa Rehema Mwenye Kurehemu",
    "alhamdulillah": "Sifa zote njema ni za Mwenyezi Mungu (Alhamdulillah)",
    "inshallah": "Mwenyezi Mungu akipenda (Insha'Allah)",
    "insha allah": "Mwenyezi Mungu akipenda (Insha'Allah)",
    "in sha allah": "Mwenyezi Mungu akipenda (Insha'Allah)",
    "masha allah": "Alivyopenda Mwenyezi Mungu (Masha'Allah)",
    "mashallah": "Alivyopenda Mwenyezi Mungu (Masha'Allah)",
    "subhanallah": "Ametakasika Mwenyezi Mungu (Subhanallah)",
    "subhan allah": "Ametakasika Mwenyezi Mungu (Subhanallah)",
    "allahu akbar": "Mwenyezi Mungu ni Mkubwa zaidi (Allahu Akbar)",
    "astaghfirullah": "Namwomba Mwenyezi Mungu msamaha (Astaghfirullah)",
    "astaghfirullah wa atubu ilayh": "Namwomba Mungu msamaha na kutubu Kwake",
    "jazakallah khair": "Mwenyezi Mungu akulipe kheri (Jazakallahu Khayran)",
    "jazakallahu khayran": "Mwenyezi Mungu akulipe kheri (Jazakallahu Khayran)",
    "jazakillahu khayran": "Mwenyezi Mungu akulipe kheri (Jazakillahu Khayran)",
    "jazakumullahu khayran": "Mwenyezi Mungu awalipeni kheri",
    "barakallahu feek": "Mwenyezi Mungu akubariki (Barakallahu Feek)",
    "barakallahu feekum": "Mwenyezi Mungu awabariki",
    "assalamu alaykum": "Amani iwe juu yenu (Assalamu Alaykum)",
    "as-salamu alaykum": "Amani iwe juu yenu (Assalamu Alaykum)",
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

    def analyze_code_switching(self, text: str) -> CodeSwitchResult:
        """Segment and classify language distribution in text."""
        segments = self.segment_languages(text)
        arabic_phrases: list[str] = []
        has_arabic_script = bool(re.search(r"[\u0600-\u06FF]", text))
        has_english = False
        has_swahili = False
        has_arabic_formula = False

        swahili_token_count = 0
        english_token_count = 0
        arabic_token_count = 0

        for seg in segments:
            if seg.is_islamic_formula or seg.language == "ar":
                arabic_phrases.append(seg.text)
                arabic_token_count += len(seg.text.split())
                if seg.is_islamic_formula:
                    has_arabic_formula = True
            elif seg.language == "en":
                has_english = True
                english_token_count += len(seg.text.split())
            elif seg.language == "sw":
                has_swahili = True
                swahili_token_count += len(seg.text.split())

        # Classify switch type
        if (has_english and has_swahili and (has_arabic_formula or has_arabic_script)) or (
            english_token_count > 0 and swahili_token_count > 0 and arabic_token_count > 0
        ):
            switch_type = CodeSwitchType.TRILINGUAL_MIXED
        elif has_english and (has_swahili or not has_arabic_script):
            switch_type = CodeSwitchType.SWAHILI_ENGLISH_MIXED
        elif has_arabic_formula or has_arabic_script or arabic_token_count > 0:
            switch_type = CodeSwitchType.SWAHILI_ARABIC_MIXED
        else:
            switch_type = CodeSwitchType.MONOLINGUAL_SWAHILI

        # Determine dominant language
        total_tokens = swahili_token_count + english_token_count + arabic_token_count
        if total_tokens == 0:
            dominant = "sw"
        elif english_token_count > swahili_token_count and english_token_count > arabic_token_count:
            dominant = "en"
        elif arabic_token_count > swahili_token_count and arabic_token_count > english_token_count:
            dominant = "ar"
        else:
            dominant = "sw"

        # Check for Quran or Hadith markers
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

    def segment_languages(self, text: str) -> list[CodeSwitchSegment]:
        """Split text into classified language segments."""
        segments: list[CodeSwitchSegment] = []
        lower_text = text.lower()

        # 1. Identify Islamic formulas with glosses
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

        # 2. Identify Arabic script passages
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

        # 3. Identify English segments
        words = text.split()
        current_en_words: list[str] = []
        current_sw_words: list[str] = []

        for word in words:
            w_clean = re.sub(r"[^\w]", "", word).lower()
            if not w_clean:
                continue

            if w_clean in COMMON_ENGLISH_WORDS:
                if current_sw_words:
                    segments.append(CodeSwitchSegment(text=" ".join(current_sw_words), language="sw"))
                    current_sw_words = []
                current_en_words.append(word)
            else:
                if current_en_words:
                    segments.append(CodeSwitchSegment(text=" ".join(current_en_words), language="en"))
                    current_en_words = []
                current_sw_words.append(word)

        if current_en_words:
            segments.append(CodeSwitchSegment(text=" ".join(current_en_words), language="en"))
        if current_sw_words:
            segments.append(CodeSwitchSegment(text=" ".join(current_sw_words), language="sw"))

        return segments


# Global singleton instance
code_switch_processor = CodeSwitchingProcessor()
