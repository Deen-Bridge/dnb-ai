"""Hadith authenticity grading — parsing, normalization, and policy enforcement.

Why this exists
----------------
Presenting a da'if (weak) or mawdu' (fabricated) narration as if it were
authentic is one of the most serious failure modes for an Islamic assistant.
This module makes hadith citations in a model answer grading-aware: it parses
references out of free text, normalizes collection names/numbering, looks the
reference up in a bundled grading dataset, and flags answers that lean on a
weak or unverified narration as evidence without saying so.

Data source and normalization policy
-------------------------------------
The bundled dataset (``data/hadith/*.json``) is built by
``scripts/build_hadith_data.py`` from the public-domain (Unlicense)
``fawazahmed0/hadith-api`` repository, pinned to tag ``1``. Only grading
metadata is bundled (collection, book/hadith numbers, grader, grade) — never
the translated hadith text — so translation-copyright questions never come
up. Full provenance is in ``data/hadith/PROVENANCE.md``.

Each grader's free-text grade string is decomposed into two independent
axes:

- **Strength**: MAWDU (fabricated/void) > DAIF (weak) > HASAN (good) > SAHIH
  (authentic). A composite grade like "Hasan Sahih" is treated as the
  *weaker* of the two words (HASAN), and when a hadith has multiple graders
  who disagree, the overall strength is the *weakest* strength any grader
  assigned. This is a safety-first, conservative policy: nothing is hidden
  (every individual grader's raw grade is preserved in the record), but the
  headline grade never overstates authenticity.
- **Chain type**: MARFU (attributed to the Prophet (peace be upon him) —
  the normal case) vs MAUQUF (stopped at a Companion) / MAQTU (stopped at a
  Successor) / MURSAL (a Successor narrating directly from the Prophet,
  omitting the Companion). A mauquf or maqtu report is not a saying of the
  Prophet at all, regardless of how strong its chain is, and is never
  presented as one.

Sahih al-Bukhari and Sahih Muslim carry no per-hadith grade in the source
data because classical scholarship treats both collections as authenticated
by consensus (ijma') in their entirety — that absence is the classical
position, not a gap, so both collections are graded SAHIH with grader
"Scholarly consensus (Bukhari and Muslim)".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from pathlib import Path

from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent / "data" / "hadith"


# ---------------------------------------------------------------------------
# Grade taxonomy
# ---------------------------------------------------------------------------


class Strength(str, Enum):
    SAHIH = "SAHIH"
    HASAN = "HASAN"
    DAIF = "DAIF"
    MAWDU = "MAWDU"
    UNKNOWN = "UNKNOWN"


class ChainType(str, Enum):
    MARFU = "MARFU"
    MAUQUF = "MAUQUF"
    MAQTU = "MAQTU"
    MURSAL = "MURSAL"


# Weakest-first — used both to resolve a composite raw string ("Hasan Sahih")
# and to aggregate across multiple graders (weakest applicable grade wins).
_STRENGTH_PRIORITY: tuple[Strength, ...] = (
    Strength.MAWDU,
    Strength.DAIF,
    Strength.HASAN,
    Strength.SAHIH,
)

_STRENGTH_KEYWORDS: dict[Strength, tuple[str, ...]] = {
    Strength.MAWDU: ("mawdu", "batil"),
    Strength.DAIF: ("daif", "da'if", "munkar", "shadh"),
    Strength.HASAN: ("hasan",),
    Strength.SAHIH: ("sahih",),
}

_CHAIN_PRIORITY: tuple[ChainType, ...] = (
    ChainType.MAUQUF,
    ChainType.MAQTU,
    ChainType.MURSAL,
)

_CHAIN_KEYWORDS: dict[ChainType, tuple[str, ...]] = {
    ChainType.MAUQUF: ("mauquf", "muquf"),
    ChainType.MAQTU: ("maqtu",),
    ChainType.MURSAL: ("mursal",),
}


def _contains_word(haystack: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", haystack) is not None


def parse_grade_string(raw: str) -> tuple[Strength, ChainType]:
    """Decompose one grader's free-text grade into (strength, chain_type).

    Unrecognized or blank strings (e.g. "-") yield Strength.UNKNOWN. Chain
    type defaults to MARFU (attributed to the Prophet) when no
    mauquf/maqtu/mursal keyword is present.
    """
    lowered = (raw or "").strip().lower()

    strength = Strength.UNKNOWN
    for candidate in _STRENGTH_PRIORITY:
        if any(_contains_word(lowered, kw) for kw in _STRENGTH_KEYWORDS[candidate]):
            strength = candidate
            break

    chain_type = ChainType.MARFU
    for chain_candidate in _CHAIN_PRIORITY:
        if any(_contains_word(lowered, kw) for kw in _CHAIN_KEYWORDS[chain_candidate]):
            chain_type = chain_candidate
            break

    return strength, chain_type


def aggregate_strength(strengths: list[Strength]) -> Strength:
    """Weakest-wins aggregation across multiple graders' strengths."""
    present = [s for s in strengths if s != Strength.UNKNOWN]
    if not present:
        return Strength.UNKNOWN
    for candidate in _STRENGTH_PRIORITY:
        if candidate in present:
            return candidate
    return Strength.UNKNOWN


def aggregate_chain_type(chain_types: list[ChainType]) -> ChainType:
    """If any grader flags a non-marfu chain, surface that (most cautious)."""
    for candidate in _CHAIN_PRIORITY:
        if candidate in chain_types:
            return candidate
    return ChainType.MARFU


# ---------------------------------------------------------------------------
# Collection alias normalization
# ---------------------------------------------------------------------------

COLLECTION_NAMES: dict[str, str] = {
    "bukhari": "Sahih al-Bukhari",
    "muslim": "Sahih Muslim",
    "abudawud": "Sunan Abu Dawud",
    "tirmidhi": "Jami at-Tirmidhi",
    "nasai": "Sunan an-Nasai",
    "ibnmajah": "Sunan Ibn Majah",
    "malik": "Muwatta Malik",
}

_COLLECTION_ALIASES: dict[str, str] = {
    "bukhari": "bukhari",
    "al bukhari": "bukhari",
    "al-bukhari": "bukhari",
    "sahih bukhari": "bukhari",
    "sahih al bukhari": "bukhari",
    "sahih al-bukhari": "bukhari",
    "muslim": "muslim",
    "sahih muslim": "muslim",
    "abu dawud": "abudawud",
    "abu dawood": "abudawud",
    "abudawud": "abudawud",
    "sunan abu dawud": "abudawud",
    "sunan abu dawood": "abudawud",
    "tirmidhi": "tirmidhi",
    "al tirmidhi": "tirmidhi",
    "al-tirmidhi": "tirmidhi",
    "jami at tirmidhi": "tirmidhi",
    "jami' at-tirmidhi": "tirmidhi",
    "sunan al-tirmidhi": "tirmidhi",
    "sunan tirmidhi": "tirmidhi",
    "nasai": "nasai",
    "an nasai": "nasai",
    "an-nasai": "nasai",
    "al-nasai": "nasai",
    "sunan an-nasai": "nasai",
    "sunan an nasai": "nasai",
    "sunan nasai": "nasai",
    "ibn majah": "ibnmajah",
    "ibnmajah": "ibnmajah",
    "sunan ibn majah": "ibnmajah",
    "malik": "malik",
    "muwatta": "malik",
    "muwatta malik": "malik",
    "muwatta imam malik": "malik",
    "al-muwatta": "malik",
    "al muwatta": "malik",
}


def _clean(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace("'", "").replace("’", "").replace("-", " ")
    normalized = re.sub(r"[^a-z\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# _COLLECTION_ALIASES is authored with mixed punctuation for readability;
# build a lookup keyed by the same normalization used on user input so
# "Jami' at-Tirmidhi" and "jami at tirmidhi" resolve identically.
_NORMALIZED_ALIASES: dict[str, str] = {_clean(k): v for k, v in _COLLECTION_ALIASES.items()}


def normalize_collection(name: str) -> str | None:
    """Map a free-text collection name/alias to its canonical key, or None."""
    if not name:
        return None
    cleaned = _clean(name)
    if cleaned in _NORMALIZED_ALIASES:
        return _NORMALIZED_ALIASES[cleaned]
    # Try again without a leading "sahih "/"sunan "/"jami at " qualifier some
    # inputs omit or duplicate inconsistently.
    stripped = re.sub(r"^(sahih|sunan|jami( at)?)\s+", "", cleaned).strip()
    return _NORMALIZED_ALIASES.get(stripped)


# Longest alias first so e.g. "sunan abu dawud" matches before "abu dawud".
_ALIAS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(_COLLECTION_ALIASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_NUMBER_AFTER_PATTERN = re.compile(
    r"^[^0-9]{0,40}?(?:book\s*(?P<book>\d+)\s*[,:]?\s*)?"
    r"(?:hadith|#|no\.?|number)?\s*[:#]?\s*(?P<number>\d+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawReference:
    raw: str
    collection: str
    number: int
    book: int | None
    span: tuple[int, int]


def parse_references(text: str) -> list[RawReference]:
    """Find hadith citations in *text* and normalize the collection + number.

    Handles common phrasings such as "Sahih al-Bukhari 1", "Bukhari #1", and
    "Abu Dawud, Book 1, Hadith 1". Citations that don't resolve to a
    recognized collection + number are simply not returned — they'll surface
    as ordinary text, not a false claim of grading.
    """
    references: list[RawReference] = []
    for match in _ALIAS_PATTERN.finditer(text):
        collection = _COLLECTION_ALIASES.get(match.group(1).lower())
        if collection is None:
            continue
        tail = text[match.end() : match.end() + 60]
        number_match = _NUMBER_AFTER_PATTERN.match(tail)
        if not number_match:
            continue
        number = int(number_match.group("number"))
        book = int(number_match.group("book")) if number_match.group("book") else None
        end = match.end() + number_match.end()
        while end < len(text) and text[end] in ").,;:]":
            end += 1
        references.append(
            RawReference(
                raw=text[match.start() : end].strip(),
                collection=collection,
                number=number,
                book=book,
                span=(match.start(), end),
            )
        )
    return references


# ---------------------------------------------------------------------------
# Grading dataset lookup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradeRecord:
    collection: str
    hadith_number: int
    book: int | None
    book_number: int | None
    grade: Strength
    chain_type: ChainType
    graders: list[dict[str, str]] = field(default_factory=list)


class GradingSource:
    """Pluggable lookup interface — a future RAG layer can implement this
    against retrieved hadith instead of the bundled offline dataset."""

    def get(self, collection: str, number: int, book: int | None = None) -> GradeRecord | None:
        raise NotImplementedError


class BundledGradingSource(GradingSource):
    """Looks up grades from the JSON files under data/hadith/."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self._data_dir = data_dir

    # B019: the grading sources are process-lifetime singletons, so this
    # per-instance cache is bounded by the bundled collection count, not by
    # request volume. Changing the caching strategy would be a behaviour
    # change and belongs in its own PR.
    @cache  # noqa: B019
    def _index(self, collection: str) -> dict[str, dict] | None:
        path = self._data_dir / f"{collection}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        by_number: dict[int, dict] = {}
        by_book_number: dict[tuple[int, int], dict] = {}
        for record in payload.get("hadiths", []):
            by_number[record["n"]] = record
            if record.get("book") is not None and record.get("bn") is not None:
                by_book_number[(record["book"], record["bn"])] = record
        return {"by_number": by_number, "by_book_number": by_book_number}

    def get(self, collection: str, number: int, book: int | None = None) -> GradeRecord | None:
        index = self._index(collection)
        if index is None:
            return None
        record = None
        if book is not None:
            record = index["by_book_number"].get((book, number))
        if record is None:
            record = index["by_number"].get(number)
        if record is None:
            return None
        return GradeRecord(
            collection=collection,
            hadith_number=record["n"],
            book=record.get("book"),
            book_number=record.get("bn"),
            grade=Strength(record["grade"]),
            chain_type=ChainType(record["chain"]),
            graders=record.get("graders", []),
        )


_default_source = BundledGradingSource()


def get_default_source() -> GradingSource:
    return _default_source


# ---------------------------------------------------------------------------
# Structured reference + annotation
# ---------------------------------------------------------------------------


class HadithReference(BaseModel):
    raw: str
    collection: str | None = None
    hadith_number: int | None = None
    grade: str
    grader: str | None = None
    chain_type: str | None = None
    verified: bool
    flagged: bool
    note: str | None = None


_CAVEAT_KEYWORDS = (
    "weak",
    "da'if",
    "daif",
    "fabricat",
    "not authentic",
    "not to be relied",
    "not reliable",
    "should not be relied",
    "for encouragement",
    "targhib",
    "tarhib",
    "not be used as evidence",
    "grading unverified",
    "unverified",
    "not a strong basis",
    "narrated for",
)

_CAVEAT_WINDOW = 150


def _has_nearby_caveat(text: str, span: tuple[int, int]) -> bool:
    start = max(0, span[0] - _CAVEAT_WINDOW)
    end = min(len(text), span[1] + _CAVEAT_WINDOW)
    window = text[start:end].lower()
    return any(keyword in window for keyword in _CAVEAT_KEYWORDS)


def annotate(text: str, source: GradingSource | None = None) -> list[HadithReference]:
    """Parse hadith citations out of *text* and grade each one."""
    source = source or _default_source
    results: list[HadithReference] = []
    for ref in parse_references(text):
        record = source.get(ref.collection, ref.number, ref.book)
        has_caveat = _has_nearby_caveat(text, ref.span)

        if record is None:
            results.append(
                HadithReference(
                    raw=ref.raw,
                    collection=ref.collection,
                    hadith_number=ref.number,
                    grade=Strength.UNKNOWN.value,
                    grader=None,
                    chain_type=None,
                    verified=False,
                    flagged=False,
                    note="grading unverified",
                )
            )
            continue

        weak = record.grade in (Strength.DAIF, Strength.MAWDU)
        non_marfu = record.chain_type != ChainType.MARFU
        flagged = (weak or non_marfu) and not has_caveat
        note = None
        if non_marfu:
            note = (
                f"chain is {record.chain_type.value.lower()} — attributed to a "
                "narrator, not directly to the Prophet (peace be upon him)"
            )
        elif weak:
            note = f"graded {record.grade.value.lower()} — do not treat as unqualified evidence"

        top_grader = record.graders[0]["g"] if record.graders else None
        results.append(
            HadithReference(
                raw=ref.raw,
                collection=ref.collection,
                hadith_number=ref.number,
                grade=record.grade.value,
                grader=top_grader,
                chain_type=record.chain_type.value,
                verified=True,
                flagged=flagged,
                note=note,
            )
        )
    return results


def build_caution_note(text: str, references: list[HadithReference]) -> str | None:
    """Compose a single caution suffix for the response, or None if not needed.

    Two triggers, per the grading policy:
    - a da'if/mawdu' or non-marfu hadith cited without an existing caveat
      nearby (i.e. presented as unqualified evidence), and
    - an unverified/ungraded reference that is the *sole* hadith support in
      the whole answer.
    """
    flagged = [r for r in references if r.flagged]
    sole_unverified = len(references) == 1 and not references[0].verified

    if not flagged and not sole_unverified:
        return None

    lines = ["Note on hadith authenticity in this answer:"]
    for r in flagged:
        lines.append(f'- "{r.raw}" — {r.note}.')
    if sole_unverified:
        lines.append(
            f'- "{references[0].raw}" is the only hadith cited here and its grading '
            "could not be verified against the bundled dataset — treat it as unverified, "
            "not as an authentic narration."
        )
    lines.append(
        "Please verify with a qualified scholar or a hadith reference (e.g. Sunnah.com) "
        "before relying on this as evidence."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evidence verification agent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceVerdict:
    claim: str
    references: list[HadithReference]
    supported: bool
    strength: Strength
    notes: list[str] = field(default_factory=list)


class EvidenceVerificationAgent:
    """Validates scholarly claims by tracing cited hadith evidence to its
    bundled primary-source grading record.
    """

    def __init__(self, source: GradingSource | None = None):
        self._source = source or get_default_source()

    def extract_claims(self, text: str) -> list[str]:
        """Extract candidate scholarly claims from generated response text."""
        claims = []
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            sentence = sentence.strip()
            if sentence and re.search(
                r"\b(narrated|reported|said|states|according to)\b",
                sentence,
                re.IGNORECASE,
            ):
                claims.append(sentence)
        return claims

    def verify_claims(self, text: str) -> list[EvidenceVerdict]:
        """Verify each extracted claim, falling back to the full response."""
        claims = self.extract_claims(text) or [text.strip()]
        return [self._verify(claim, text) for claim in claims]

    def _verify(self, claim: str, full_text: str) -> EvidenceVerdict:
        refs = annotate(claim, self._source) or annotate(full_text, self._source)
        notes: list[str] = []
        if not refs:
            return EvidenceVerdict(claim, [], False, Strength.UNKNOWN, ["no primary-source evidence cited"])
        if not any(ref.raw.lower() in claim.lower() for ref in refs):
            notes.append("cited evidence is not explicitly connected to this claim")
        verified = [ref for ref in refs if ref.verified]
        if not verified:
            notes.append("cited references could not be verified against primary sources")
            return EvidenceVerdict(claim, refs, False, Strength.UNKNOWN, notes)
        strengths = [Strength(ref.grade) for ref in verified]
        strength = aggregate_strength(strengths)
        if strength in (Strength.DAIF, Strength.MAWDU):
            notes.append("cited evidence is weak or fabricated; do not rely on it")
        if any(ref.flagged for ref in verified):
            notes.append("cited evidence carries an unstated authenticity caveat")
        supported = (
            bool(refs)
            and strength not in (Strength.DAIF, Strength.MAWDU, Strength.UNKNOWN)
            and not any(ref.flagged for ref in verified)
            and not any("not explicitly connected" in note for note in notes)
        )
        return EvidenceVerdict(claim, refs, supported, strength, notes)

    def trace_provenance(self, text: str) -> list[dict[str, str]]:
        """Return a verification audit trail for every citation in *text*."""
        return [
            {
                "collection": ref.collection or "",
                "hadith_number": str(ref.hadith_number or ""),
                "grade": ref.grade,
                "grader": ref.grader or "",
                "chain_type": ref.chain_type or "",
                "verified": str(ref.verified).lower(),
                "flagged": str(ref.flagged).lower(),
            }
            for ref in annotate(text, self._source)
        ]


HADITH_ADAB_CONTEXT = """

HADITH CITATION RULES:
When you cite a hadith, follow these rules:
1. Always name the collection (e.g. Sahih al-Bukhari, Sunan Abu Dawud).
2. State the authenticity grade (sahih, hasan, da'if) when you know it.
3. Never cite a hadith you cannot attribute to a specific collection and number.
4. Prefer Sahih al-Bukhari and Sahih Muslim for evidence used in rulings.
5. If a hadith is weak (da'if), only mention it for encouragement (targhib) or
   caution (tarhib) purposes, and say explicitly that it is weak — never as
   evidence for a ruling.
"""
