
import re
import difflib
from enum import Enum
from typing import List, Dict, Any, Optional
from corpus import corpus


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    MISMATCH = "mismatch"
    UNVERIFIED = "unverified"
    NOT_QUOTED = "not_quoted"


# Arabic Tashkeel / Diacritical Marks
TASHKEEL_REGEX = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]')

# Extraction Regex: Matches "Quran 2:255", "Surah 2:255", "[2:255]", "(2:255)", etc.
QURAN_REF_REGEX = re.compile(
    r'(?:Surah|Quran|Qur\'an)?\s*\[?\b([1-9]|[1-9]\d|1[0-0]\d|11[0-4])\s*:\s*([1-9]\d*)\b\]?'
    r'(?:\s*[\"\'«”](.*?)[\"\'»“])?',
    re.IGNORECASE | re.DOTALL
)

HADITH_REF_REGEX = re.compile(
    r'\b(Bukhari|Muslim|Abu Dawud|Tirmidhi|Nasa\'i|Ibn Majah|Muwatta|Ahmad)\b'
    r'\s*(?:hadith|no\.|number|#)?\s*(\d+)?'
    r'(?:\s*[\"\'«”](.*?)[\"\'»“])?',
    re.IGNORECASE
)


def normalize_arabic(text: str) -> str:
    """Strip tashkeel/diacritics and normalize Alef variants."""
    if not text:
        return ""
    text = TASHKEEL_REGEX.sub('', text)
    # Unify Alef forms (أ, إ, آ -> ا)
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
    return text.strip()


def normalize_english(text: str) -> str:
    """Casefold, strip punctuation, and normalize whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def calculate_similarity(generated_quote: str, corpus_text: str) -> float:
    """Calculate similarity ratio between generated quote and corpus text using stdlib difflib."""
    norm_gen = normalize_english(generated_quote)
    norm_corp = normalize_english(corpus_text)
    if not norm_gen or not norm_corp:
        return 0.0
    return difflib.SequenceMatcher(None, norm_gen, norm_corp).ratio()


def verify_quran_citation(surah: int, ayah: int, quote: Optional[str] = None) -> Dict[str, Any]:
    """Verify a single Quran reference against the corpus."""
    max_ayahs = corpus.get_ayah_count(surah)

    # 1. Check existence
    if max_ayahs is None or ayah < 1 or ayah > max_ayahs:
        return {
            "source": "quran",
            "surah": surah,
            "ayah": ayah,
            "status": VerificationStatus.MISMATCH,
            "reason": f"Surah {surah} only has {max_ayahs or 0} ayahs; ayah {ayah} does not exist."
        }

    ayah_data = corpus.get_ayah(surah, ayah)

    # If no quote is given with the reference
    if not quote or not quote.strip():
        return {
            "source": "quran",
            "surah": surah,
            "ayah": ayah,
            "status": VerificationStatus.NOT_QUOTED,
            "reason": "Reference exists; no quote provided for verification."
        }

    # 2. Check quote similarity (English translation)
    corpus_english = ayah_data.get("english", "") if ayah_data else ""
    similarity = calculate_similarity(quote, corpus_english)

    # Threshold of 0.70 accounts for variations across translation editions
    if similarity >= 0.70:
        return {
            "source": "quran",
            "surah": surah,
            "ayah": ayah,
            "status": VerificationStatus.VERIFIED,
            "similarity": round(similarity, 2)
        }
    else:
        return {
            "source": "quran",
            "surah": surah,
            "ayah": ayah,
            "status": VerificationStatus.MISMATCH,
            "similarity": round(similarity, 2),
            "correct_text": corpus_english,
            "reason": f"Quote does not match Surah {surah}:{ayah} text in corpus."
        }


def verify_hadith_citation(collection: str, number: Optional[str] = None, quote: Optional[str] = None) -> Dict[str, Any]:
    """Verification for Hadith citations (defaults to honest unverified label when corpus is unavailable)."""
    if not corpus.has_hadith_corpus():
        return {
            "source": "hadith",
            "collection": collection,
            "number": number,
            "status": VerificationStatus.UNVERIFIED,
            "reason": "Hadith corpus not available for verification."
        }
    # Future expansion for #24 when Hadith corpus lands
    return {
        "source": "hadith",
        "collection": collection,
        "number": number,
        "status": VerificationStatus.UNVERIFIED,
        "reason": "Hadith verification not implemented."
    }


def extract_and_verify_all(text: str) -> List[Dict[str, Any]]:
    """Extract all citations from text and return their verification statuses."""
    results = []

    # Extract & Verify Quran References
    for match in QURAN_REF_REGEX.finditer(text):
        surah = int(match.group(1))
        ayah = int(match.group(2))
        quote = match.group(3)
        res = verify_quran_citation(surah, ayah, quote)
        results.append(res)

    # Extract & Verify Hadith References
    for match in HADITH_REF_REGEX.finditer(text):
        collection = match.group(1)
        number = match.group(2)
        quote = match.group(3)
        res = verify_hadith_citation(collection, number, quote)
        results.append(res)

    return results

