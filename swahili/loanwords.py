"""Arabic Loanword Analyzer & Phonological Transformation Engine for Swahili."""

from __future__ import annotations

import logging
import re

from swahili.models import LoanwordMatch, SwahiliToken
from swahili.terminology import terminology_db

logger = logging.getLogger(__name__)

# Common Swahili prefixes applied to Arabic roots (Bantu noun classes and verb prefixes)
BANTU_PREFIXES = [
    ("kutawa", "tawa"),  # specific to kutawadha
    ("kuta", "ta"),
    ("kusu", "su"),
    ("kuhu", "hu"),
    ("kufu", "fu"),
    ("kuto", "to"),
    ("kuji", ""),
    ("kua", "a"),
    ("ku", ""),  # Class 15 verb infinitive (kuswali, kufunga, kuhiji, kutubu)
    ("mwi", "i"),  # Class 1 human (Mwislamu)
    ("wai", "i"),  # Class 2 human plural (Waislamu)
    ("mwa", "a"),  # Class 1 (Mwadhini)
    ("waa", "a"),  # Class 2 (Waadhini)
    ("msh", "sh"),  # Class 1 (Mshirikina)
    ("wash", "sh"),  # Class 2 (Washirikina)
    ("muu", "u"),  # Class 1 (Muumini)
    ("wau", "u"),  # Class 2 (Waumini)
    ("mw", ""),  # Class 1 agent
    ("wa", ""),  # Class 2 plural
    ("ma", ""),  # Class 6 plural (Maswahaba, Madhehebu, Maulidi, Maustadhi, Masheikh)
    ("ki", ""),  # Class 7 manner/language (Kiislamu, Kiarabu, Kisheria)
    ("vi", ""),  # Class 8 plural manner
    ("u", ""),  # Class 14 abstract noun (Uislamu, Ushirikina, Utume, Uadilifu)
    ("m", ""),  # Class 1 noun prefix
]

# Arabic-to-Swahili phonological mapping patterns
PHONOLOGICAL_MAPPINGS = [
    (r"\bsw([aeiou])", r"s\1", "Emphatic sad [ص] to s"),
    (r"\bs([aeiou])", r"sw\1", "s to Swahili emphatic variant sw"),
    (r"dh([aeiou])", r"z\1", "Dhad/Zhal [ض/ذ/ظ] to z"),
    (r"z([aeiou])", r"dh\1", "z to Arabic interdental dh"),
    (r"th([aeiou])", r"s\1", "Tha [ث] to s"),
    (r"kh([aeiou])", r"h\1", "Kha [خ] to h"),
    (r"h([aeiou])", r"kh\1", "h to Arabic guttural kh"),
    (r"gh([aeiou])", r"g\1", "Ghayn [غ] to g"),
    (r"q([aeiou])", r"k\1", "Qaf [ق] to k"),
    (r"([aeiou])t\b", r"\1ti", "Final consonant epenthesis -i"),
    (r"([aeiou])d\b", r"\1di", "Final consonant epenthesis -i"),
    (r"([aeiou])m\b", r"\1mu", "Final consonant epenthesis -u"),
    (r"([aeiou])b\b", r"\1bu", "Final consonant epenthesis -u"),
]


_KNOWN_DIRECT_STEMS: list[tuple[tuple[str, ...], str, str]] = [
    (("kutawadha",), "udhu", "ku"),
    (("kuswali", "kusala"), "swala", "ku"),
    (("kufunga",), "saumu", "ku"),
    (("kuhiji",), "hija", "ku"),
    (("waislamu",), "uislamu", "wa"),
    (("muislamu", "mwislamu"), "uislamu", "m"),
    (("maswahaba",), "swahaba", "ma"),
    (("ushirikina",), "shiriki", "u"),
    (("washirikina",), "shiriki", "wa"),
    (("mshirikina",), "shiriki", "m"),
    (("madhehebu",), "madhehebu", "ma"),
    (("maustadhi",), "ustaadh", "ma"),
    (("masheikh", "mashehe"), "sheikh", "ma"),
]


class ArabicLoanwordAnalyzer:
    """Detects, normalizes, and extracts Arabic loanwords in Swahili text."""

    def __init__(self) -> None:
        self._term_db = terminology_db

    def strip_bantu_prefix(self, word: str) -> tuple[str, str | None]:
        """Strip Bantu inflectional/derivational prefixes to uncover the Arabic stem.

        Examples:
            'kuswali' -> ('swali', 'ku')
            'kutawadha' -> ('tawadha', 'ku')
            'waislamu' -> ('islamu', 'wa')
            'maswahaba' -> ('swahaba', 'ma')
            'kiislamu' -> ('islamu', 'ki')
            'uislamu' -> ('islamu', 'u')
        """
        lower_word = word.strip().lower()

        # Direct known forms
        for prefixes, stem, pfx in _KNOWN_DIRECT_STEMS:
            if lower_word.startswith(prefixes):
                return stem, pfx

        for prefix, replacement in BANTU_PREFIXES:
            if lower_word.startswith(prefix) and len(lower_word) - len(prefix) >= 3:
                stem = replacement + lower_word[len(prefix) :]
                # Check if stem is recognized in database
                if self._term_db.lookup_term(stem):
                    return stem, prefix
                # Check if stem minus trailing vowel is recognized
                if len(stem) > 3 and self._term_db.lookup_term(stem[:-1]):
                    return stem[:-1], prefix

        return lower_word, None

    def _check_direct_match(self, word: str, cleaned_word: str) -> LoanwordMatch | None:
        direct_match = self._term_db.lookup_term(cleaned_word)
        if direct_match:
            return LoanwordMatch(
                raw_word=word,
                matched_term=direct_match.swahili_term,
                arabic_original=direct_match.arabic_original,
                arabic_transliteration=direct_match.arabic_transliteration,
                category=direct_match.category.value,
                confidence=1.0,
                phonological_rule_applied="direct_match",
                morphological_prefix=None,
            )
        return None

    def _check_stem_match(self, word: str, cleaned_word: str) -> LoanwordMatch | None:
        stem, prefix = self.strip_bantu_prefix(cleaned_word)
        if prefix:
            stem_match = self._term_db.lookup_term(stem)
            if stem_match:
                return LoanwordMatch(
                    raw_word=word,
                    matched_term=stem_match.swahili_term,
                    arabic_original=stem_match.arabic_original,
                    arabic_transliteration=stem_match.arabic_transliteration,
                    category=stem_match.category.value,
                    confidence=0.95,
                    phonological_rule_applied=f"bantu_prefix_strip_{prefix}",
                    morphological_prefix=prefix,
                )
        return None

    def _check_phonological_match(self, word: str, cleaned_word: str) -> LoanwordMatch | None:
        for pattern, repl, rule_name in PHONOLOGICAL_MAPPINGS:
            variant = re.sub(pattern, repl, cleaned_word)
            if variant == cleaned_word:
                continue

            var_match = self._term_db.lookup_term(variant)
            if var_match:
                return LoanwordMatch(
                    raw_word=word,
                    matched_term=var_match.swahili_term,
                    arabic_original=var_match.arabic_original,
                    arabic_transliteration=var_match.arabic_transliteration,
                    category=var_match.category.value,
                    confidence=0.88,
                    phonological_rule_applied=rule_name,
                    morphological_prefix=None,
                )

            v_stem, v_prefix = self.strip_bantu_prefix(variant)
            if v_prefix:
                v_stem_match = self._term_db.lookup_term(v_stem)
                if v_stem_match:
                    return LoanwordMatch(
                        raw_word=word,
                        matched_term=v_stem_match.swahili_term,
                        arabic_original=v_stem_match.arabic_original,
                        arabic_transliteration=v_stem_match.arabic_transliteration,
                        category=v_stem_match.category.value,
                        confidence=0.85,
                        phonological_rule_applied=f"{rule_name}_with_prefix_{v_prefix}",
                        morphological_prefix=v_prefix,
                    )
        return None

    def _check_fuzzy_match(self, word: str, cleaned_word: str) -> LoanwordMatch | None:
        if len(cleaned_word) < 4:
            return None
        for term in self._term_db.terms:
            candidates = [term.swahili_term, term.arabic_transliteration] + term.variants_sw
            for cand in candidates:
                cand_lower = cand.lower()
                if abs(len(cleaned_word) - len(cand_lower)) <= 2:
                    dist = self._levenshtein_distance(cleaned_word, cand_lower)
                    if dist <= 1:
                        return LoanwordMatch(
                            raw_word=word,
                            matched_term=term.swahili_term,
                            arabic_original=term.arabic_original,
                            arabic_transliteration=term.arabic_transliteration,
                            category=term.category.value,
                            confidence=0.80,
                            phonological_rule_applied="fuzzy_edit_distance_1",
                            morphological_prefix=None,
                        )
        return None

    def analyze_word(self, word: str) -> LoanwordMatch | None:
        """Analyze a single word to determine if it is an Arabic loanword and map it."""
        cleaned_word = word.strip().lower()
        if len(cleaned_word) < 2:
            return None

        return (
            self._check_direct_match(word, cleaned_word)
            or self._check_stem_match(word, cleaned_word)
            or self._check_phonological_match(word, cleaned_word)
            or self._check_fuzzy_match(word, cleaned_word)
        )

    def tokenize_swahili(self, text: str) -> list[SwahiliToken]:
        """Tokenize Swahili text and enrich tokens with loanword and morphological tags."""
        raw_tokens = re.findall(r"[\w']+|[^\w\s]", text)
        enriched_tokens: list[SwahiliToken] = []

        for t in raw_tokens:
            if not re.match(r"[\w']+", t):
                continue

            cleaned = t.lower()
            stem, prefix = self.strip_bantu_prefix(cleaned)
            loan_match = self.analyze_word(t)

            is_loan = loan_match is not None
            arabic_root = loan_match.arabic_transliteration if loan_match else None
            canonical = loan_match.matched_term if loan_match else None

            token_obj = SwahiliToken(
                raw_token=t,
                normalized_token=cleaned,
                lemma=stem if prefix else cleaned,
                prefix=prefix,
                suffix=None,
                is_arabic_loanword=is_loan,
                arabic_root=arabic_root,
                canonical_term=canonical,
            )
            enriched_tokens.append(token_obj)

        return enriched_tokens

    def extract_loanwords(self, text: str) -> list[LoanwordMatch]:
        """Extract all Arabic loanwords found within the input text."""
        words = re.findall(r"[\w']+", text)
        seen_canonical: set[str] = set()
        matches: list[LoanwordMatch] = []

        # Check multi-word loanwords first
        lower_text = text.lower()
        for term in self._term_db.terms:
            sw_lower = term.swahili_term.lower()
            if " " in sw_lower and sw_lower in lower_text:
                if term.swahili_term not in seen_canonical:
                    seen_canonical.add(term.swahili_term)
                    matches.append(
                        LoanwordMatch(
                            raw_word=term.swahili_term,
                            matched_term=term.swahili_term,
                            arabic_original=term.arabic_original,
                            arabic_transliteration=term.arabic_transliteration,
                            category=term.category.value,
                            confidence=1.0,
                            phonological_rule_applied="multiword_exact",
                        )
                    )

        # Single word scan
        for w in words:
            match = self.analyze_word(w)
            if match and match.matched_term not in seen_canonical:
                seen_canonical.add(match.matched_term)
                matches.append(match)

        return matches

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


# Global singleton instance
loanword_analyzer = ArabicLoanwordAnalyzer()
