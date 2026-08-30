import difflib
import re
from enum import Enum
from typing import Any

from corpus import corpus


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    MISMATCH = "mismatch"
    UNVERIFIED = "unverified"
    NOT_QUOTED = "not_quoted"


# Arabic Tashkeel / Diacritical Marks
TASHKEEL_REGEX = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")

# Extraction Regex: Matches "Quran 2:255", "Surah 2:255", "[2:255]", "(2:255)", etc.
QURAN_REF_REGEX = re.compile(
    r"(?:Surah|Quran|Qur\'an)?\s*\[?\b([1-9]|[1-9]\d|1[0-0]\d|11[0-4])\s*:\s*([1-9]\d*)\b\]?"
    r"(?:\s*[\"\'«”](.*?)[\"\'»“])?",
    re.IGNORECASE | re.DOTALL,
)

HADITH_REF_REGEX = re.compile(
    r"\b(Bukhari|Muslim|Abu Dawud|Tirmidhi|Nasa\'i|Ibn Majah|Muwatta|Ahmad)\b"
    r"\s*(?:hadith|no\.|number|#)?\s*(\d+)?"
    r"(?:\s*[\"\'«”](.*?)[\"\'»“])?",
    re.IGNORECASE,
)


def normalize_arabic(text: str) -> str:
    """Strip tashkeel/diacritics and normalize Alef variants."""
    if not text:
        return ""
    text = TASHKEEL_REGEX.sub("", text)
    # Unify Alef forms (أ, إ, آ -> ا)
    text = re.sub(r"[\u0622\u0623\u0625]", "\u0627", text)
    return text.strip()


def normalize_english(text: str) -> str:
    """Casefold, strip punctuation, and normalize whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def calculate_similarity(generated_quote: str, corpus_text: str) -> float:
    """Calculate similarity ratio between generated quote and corpus text using stdlib difflib."""
    norm_gen = normalize_english(generated_quote)
    norm_corp = normalize_english(corpus_text)
    if not norm_gen or not norm_corp:
        return 0.0
    return difflib.SequenceMatcher(None, norm_gen, norm_corp).ratio()


def verify_quran_citation(surah: int, ayah: int, quote: str | None = None) -> dict[str, Any]:
    """Verify a single Quran reference against the corpus."""
    max_ayahs = corpus.get_ayah_count(surah)

    # 1. Check existence
    if max_ayahs is None or ayah < 1 or ayah > max_ayahs:
        return {
            "source": "quran",
            "surah": surah,
            "ayah": ayah,
            "status": VerificationStatus.MISMATCH,
            "reason": f"Surah {surah} only has {max_ayahs or 0} ayahs; ayah {ayah} does not exist.",
        }

    ayah_data = corpus.get_ayah(surah, ayah)

    # If no quote is given with the reference
    if not quote or not quote.strip():
        return {
            "source": "quran",
            "surah": surah,
            "ayah": ayah,
            "status": VerificationStatus.NOT_QUOTED,
            "reason": "Reference exists; no quote provided for verification.",
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
            "similarity": round(similarity, 2),
        }
    else:
        return {
            "source": "quran",
            "surah": surah,
            "ayah": ayah,
            "status": VerificationStatus.MISMATCH,
            "similarity": round(similarity, 2),
            "correct_text": corpus_english,
            "reason": f"Quote does not match Surah {surah}:{ayah} text in corpus.",
        }


def verify_hadith_citation(collection: str, number: str | None = None, quote: str | None = None) -> dict[str, Any]:
    """Verification for Hadith citations (defaults to honest unverified label when corpus is unavailable)."""
    if not corpus.has_hadith_corpus():
        return {
            "source": "hadith",
            "collection": collection,
            "number": number,
            "status": VerificationStatus.UNVERIFIED,
            "reason": "Hadith corpus not available for verification.",
        }
    # Attempt to retrieve hadith text from corpus
    get_hadith = getattr(corpus, "get_hadith", None)
    if not callable(get_hadith):
        return {
            "source": "hadith",
            "collection": collection,
            "number": number,
            "status": VerificationStatus.UNVERIFIED,
            "reason": "Hadith corpus getter not available.",
        }

    hadith_data = get_hadith(collection, number)
    if hadith_data is None:
        return {
            "source": "hadith",
            "collection": collection,
            "number": number,
            "status": VerificationStatus.MISMATCH,
            "reason": f"Hadith {collection} #{number} not found in corpus.",
        }

    if not quote or not quote.strip():
        return {
            "source": "hadith",
            "collection": collection,
            "number": number,
            "status": VerificationStatus.NOT_QUOTED,
            "reason": "Reference exists; no quote provided for verification.",
        }

    corpus_text = hadith_data.get("english", "") or hadith_data.get("text", "")
    if not corpus_text:
        return {
            "source": "hadith",
            "collection": collection,
            "number": number,
            "status": VerificationStatus.UNVERIFIED,
            "reason": "Hadith text not available in corpus.",
        }

    similarity = calculate_similarity(quote, corpus_text)
    if similarity >= 0.70:
        return {
            "source": "hadith",
            "collection": collection,
            "number": number,
            "status": VerificationStatus.VERIFIED,
            "similarity": round(similarity, 2),
        }
    else:
        return {
            "source": "hadith",
            "collection": collection,
            "number": number,
            "status": VerificationStatus.MISMATCH,
            "similarity": round(similarity, 2),
            "correct_text": corpus_text,
            "reason": f"Quote does not match {collection} #{number} text in corpus.",
        }


def extract_and_verify_all(text: str) -> list[dict[str, Any]]:
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


def verify_claim(claim: str, evidence: str) -> dict[str, Any]:
    """Verify a scholarly claim against provided evidence text.

    Args:
        claim: The scholarly claim being made.
        evidence: Text containing citations (Quran, Hadith) supporting the claim.

    Returns:
        A verification report including status, support score, and audit trail.
    """
    # Extract and verify all citations from the evidence
    citation_results = extract_and_verify_all(evidence)

    # If no citations are found, the claim is unsupported
    if not citation_results:
        return {
            "claim": claim,
            "status": VerificationStatus.UNVERIFIED,
            "reason": "No primary source citations found in evidence.",
            "evidence": [],
            "support_score": 0.0,
            "audit_trail": ["No citations extracted from evidence."]
        }

    # Determine overall status based on citation verification results
    verified = [r for r in citation_results if r["status"] == VerificationStatus.VERIFIED]
    mismatched = [r for r in citation_results if r["status"] == VerificationStatus.MISMATCH]
    unverified = [r for r in citation_results if r["status"] in (VerificationStatus.UNVERIFIED, VerificationStatus.NOT_QUOTED)]

    if verified and not mismatched and not unverified:
        overall_status = VerificationStatus.VERIFIED
        reason = "All cited evidence verified successfully."
    elif mismatched:
        overall_status = VerificationStatus.MISMATCH
        reason = f"{len(mismatched)} citation(s) failed verification."
    else:
        overall_status = VerificationStatus.UNVERIFIED
        reason = "No citation could be fully verified."

    support_score = len(verified) / len(citation_results)

    # Build audit trail
    audit_trail = []
    for r in citation_results:
        source = r.get("source", "unknown")
        ref = r.get("surah", r.get("collection", ""))
        detail = r.get("ayah", r.get("number", ""))
        audit_trail.append(f"{source} {ref} {detail}: {r['status']} - {r.get('reason', '')}")

    return {
        "claim": claim,
        "status": overall_status,
        "reason": reason,
        "evidence": citation_results,
        "support_score": round(support_score, 2),
        "audit_trail": audit_trail
    }

