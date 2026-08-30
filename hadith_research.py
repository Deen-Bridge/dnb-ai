"""Hadith Research Agent — retrieval, isnad analysis, rijal cross-referencing,
grading justification, and variant alignment across ten major collections.

Why this exists
---------------
A hadith citation is only as good as its chain: the collection and number
locate the narration, the isnad establishes who transmitted it, the rijal
records say whether those transmitters are reliable, and the graders say how
strong the finished report is. This agent ties those steps together
deterministically and offline:

- **Retrieval** resolves a citation (collection + number, or free text like
  "Sahih al-Bukhari 1") against the bundled grading dataset, which now covers
  ten collections (the two Sahihs, the four Sunan, the Muwatta, al-Nawawi's
  Forty, the Forty Qudsi, and Shah Waliullah's Forty).
- **Isnad analysis** checks a supplied chain for continuity (generation
  order), flags narrators missing from the rijal reference, and surfaces
  weak or abandoned transmitters.
- **Rijal cross-referencing** matches each narrator name against a curated
  offline narrator knowledge base (``data/rijal/narrators.json``).
- **Grading justification** explains *why* a hadith carries the grade it
  does: every named grader's raw verdict is preserved, and disagreement
  between graders is surfaced rather than hidden.
- **Variant alignment** traces a famous narration across collections
  (``data/hadith/variants.json``) and grades each variant against the same
  bundled dataset, so two citations of "the same" hadith under different
  numbers are reconciled instead of presented as unrelated.

Honest limitations
------------------
The bundled dataset carries grading metadata only — never translated hadith
text — so "retrieval" returns the bibliographic + grading record, not the
matn. Every result is advisory: confirm against a full reference (e.g.
Sunnah.com) and a qualified scholar before treating any narration as
evidence. The rijal knowledge base is a curated starter subset, not an
exhaustive rijāl dictionary. All data is offline; nothing here touches the
network.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import hadith
from audio_hadith import match_score, tokenize

router = APIRouter(prefix="/hadith-research", tags=["hadith-research"])

DATA_DIR = Path(__file__).resolve().parent / "data" / "hadith"
RIJAL_PATH = Path(__file__).resolve().parent / "data" / "rijal" / "narrators.json"

# Below this token-overlap score a supplied matn is not considered a match for
# a known variant (it is far enough from every curated narration to be a
# different hadith rather than a variant of one).
TEXT_MATCH_THRESHOLD = 0.15
MAX_VARIANT_MATCHES = 5
MAX_NARRATORS_PER_CHAIN = 25

DISCLAIMER = (
    "All results are advisory: grading metadata is read from the bundled "
    "dataset (see data/hadith/PROVENANCE.md) and narrators from a curated "
    "offline rijal reference. Confirm against a full hadith reference (e.g. "
    "Sunnah.com) and a qualified scholar before relying on any narration."
)


# ---------------------------------------------------------------------------
# Offline data loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@cache
def load_collections() -> dict[str, dict[str, Any]]:
    """Every bundled collection keyed by its canonical key, with display name and size."""
    collections: dict[str, dict[str, Any]] = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        if path.name == "variants.json":
            continue
        payload = _load_json(path)
        key = payload.get("collection") or path.stem
        collections[key] = {
            "key": key,
            "name": payload.get("name") or key,
            "hadith_count": len(payload.get("hadiths", [])),
        }
    return collections


@cache
def load_variants() -> dict[str, dict[str, Any]]:
    """Curated variant map keyed by variant id (see data/hadith/variants.json)."""
    payload = _load_json(DATA_DIR / "variants.json")
    return {variant["id"]: variant for variant in payload.get("variants", [])}


@cache
def load_rijal() -> dict[str, dict[str, Any]]:
    """Curated narrator knowledge base keyed by narrator id."""
    payload = _load_json(RIJAL_PATH)
    return {narrator["id"]: narrator for narrator in payload.get("narrators", [])}


def _normalize_narrator_name(name: str) -> str:
    """Fold a narrator name for matching: case, punctuation, and bin/ibn forms."""
    lowered = re.sub(r"[^a-z' ]", "", (name or "").lower())
    lowered = lowered.replace(" bin ", " ibn ").replace(" b. ", " ibn ")
    return re.sub(r"\s+", " ", lowered).strip()


@cache
def _rijal_lookup() -> dict[str, str]:
    """Normalized name/alias -> narrator id, longest forms first."""
    index: dict[str, str] = {}
    for narrator_id, narrator in load_rijal().items():
        for label in [narrator["name"], *narrator.get("aliases", [])]:
            key = _normalize_narrator_name(label)
            if key:
                index.setdefault(key, narrator_id)
    return index


def find_narrator(name: str) -> dict[str, Any] | None:
    """Resolve a narrator name against the rijal reference, or None."""
    key = _normalize_narrator_name(name)
    if not key:
        return None
    narrator_id = _rijal_lookup().get(key)
    if narrator_id is None:
        # Substring fallback: a name may arrive embedded in a longer phrase.
        for index_key, candidate_id in _rijal_lookup().items():
            if index_key and (index_key in key or key in index_key):
                narrator_id = candidate_id
                break
    if narrator_id is None:
        return None
    return load_rijal()[narrator_id]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CollectionInfo(BaseModel):
    key: str
    name: str
    hadith_count: int


class GraderOpinion(BaseModel):
    grader: str
    raw_grade: str
    strength: str
    chain_type: str


class GradingDetail(BaseModel):
    strength: str
    chain_type: str
    disputed: bool
    graders: list[GraderOpinion]
    justification: str


class VariantReference(BaseModel):
    collection: str
    collection_name: str
    number: int
    book: int | None = None
    book_number: int | None = None
    note: str | None = None
    grade: str
    chain_type: str
    verified: bool
    citation: str


class VariantMatch(BaseModel):
    variant_id: str
    title: str
    narration: str
    score: float | None = None
    references: list[VariantReference]


class LookupResult(BaseModel):
    collection: str
    collection_name: str
    hadith_number: int
    book: int | None = None
    book_number: int | None = None
    grade: str
    chain_type: str
    verified: bool
    disputed: bool
    grading: GradingDetail
    variants: list[VariantMatch]
    citation: str


class NarratorProfile(BaseModel):
    name: str
    matched: bool
    generation: str | None = None
    death_ah: int | None = None
    reliability: str | None = None
    note: str | None = None


class ChainEntry(BaseModel):
    name: str
    profile: NarratorProfile
    warnings: list[str]


class IsnadAnalysis(BaseModel):
    narrators: list[ChainEntry]
    continuous: bool | None
    flagged: bool
    notes: list[str]
    disclaimer: str = DISCLAIMER


class LookupRequest(BaseModel):
    reference: str | None = Field(
        None,
        description="Free-text citation, e.g. 'Sahih al-Bukhari 1' or 'Sunan Abu Dawud, Book 13, Hadith 27'",
    )
    collection: str | None = Field(
        None, description="Canonical collection key or display name (see GET /hadith-research/sources)"
    )
    number: int | None = Field(None, ge=1, description="Global hadith number")
    book: int | None = Field(None, ge=1, description="Optional book number, for citations that name the book")


class ChainRequest(BaseModel):
    narrators: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_NARRATORS_PER_CHAIN,
        description="Narrator names in chain order: the person who transmitted the report down to the earliest narrator",
    )


class VariantRequest(BaseModel):
    text: str | None = Field(None, min_length=3, description="Free-text matn to match against known variants")
    collection: str | None = Field(None, description="Collection key to find variants for (requires number)")
    number: int | None = Field(None, ge=1, description="Global hadith number (requires collection)")


class VariantResponse(BaseModel):
    query: str
    matches: list[VariantMatch]
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# Grading justification and citations
# ---------------------------------------------------------------------------


def justify_grading(record: hadith.GradeRecord) -> tuple[bool, str]:
    """Explain the overall grade of a record; return (disputed, explanation).

    Every named grader's raw verdict is preserved; disagreement between
    graders on strength is surfaced as ``disputed`` and the conservative
    weakest-wins policy is stated explicitly.
    """
    opinions = list(record.graders)
    strengths = {opinion["s"] for opinion in opinions if opinion.get("s") != hadith.Strength.UNKNOWN.value}
    disputed = len(strengths) > 1

    lines: list[str] = []
    if not opinions:
        lines.append("No named grader is recorded for this hadith in the bundled dataset.")
    elif len(opinions) == 1:
        opinion = opinions[0]
        raw = opinion.get("raw") or "unstated"
        lines.append(f'Single named grader: {opinion["g"]} — "{raw}" (→ {record.grade.value}).')
    else:
        lines.append(f"{len(opinions)} named graders:")
        for opinion in opinions:
            raw = opinion.get("raw") or "unstated"
            lines.append(f'- {opinion["g"]}: "{raw}" (→ {opinion["s"]}/{opinion["c"]})')
        if disputed:
            lines.append(
                "The graders disagree on strength; the conservative weakest-wins policy "
                f"yields the overall grade {record.grade.value}."
            )
        else:
            lines.append("The graders agree on strength; the overall grade follows.")

    if record.chain_type != hadith.ChainType.MARFU:
        lines.append(
            f"Chain type {record.chain_type.value}: the report is attributed to a "
            "narrator (a Companion or Successor), not directly to the Prophet "
            "(peace be upon him), so it is not a Prophetic hadith regardless of chain strength."
        )
    return disputed, " ".join(lines)


def format_citation(
    collection: str,
    number: int,
    book: int | None = None,
    book_number: int | None = None,
) -> str:
    """Academic-style citation: collection, global number, and book-local reference."""
    collections = load_collections()
    display = collections.get(collection, {}).get("name", collection)
    parts = [f"{display}, no. {number}"]
    if book is not None and book_number is not None:
        parts.append(f"(Book {book}, Hadith {book_number})")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def lookup(collection: str, number: int, book: int | None = None) -> LookupResult | None:
    """Resolve a citation to its bundled grading record, or None when absent."""
    record = hadith.get_default_source().get(collection, number, book)
    if record is None:
        return None

    disputed, justification = justify_grading(record)
    variant_ids = _variant_ids_for(collection, number)

    return LookupResult(
        collection=record.collection,
        collection_name=load_collections().get(record.collection, {}).get("name", record.collection),
        hadith_number=record.hadith_number,
        book=record.book,
        book_number=record.book_number,
        grade=record.grade.value,
        chain_type=record.chain_type.value,
        verified=True,
        disputed=disputed,
        grading=GradingDetail(
            strength=record.grade.value,
            chain_type=record.chain_type.value,
            disputed=disputed,
            graders=[
                GraderOpinion(grader=o["g"], raw_grade=o.get("raw") or "", strength=o["s"], chain_type=o["c"])
                for o in record.graders
            ],
            justification=justification,
        ),
        variants=[build_variant_match(variant_id) for variant_id in variant_ids],
        citation=format_citation(record.collection, record.hadith_number, record.book, record.book_number),
    )


def resolve_reference(raw: str) -> tuple[str, int, int | None] | None:
    """Parse free-text citation 'Sahih al-Bukhari 1' into (collection, number, book)."""
    references = hadith.parse_references(raw or "")
    if not references:
        return None
    ref = references[0]
    return ref.collection, ref.number, ref.book


def normalize_collection(name: str) -> str | None:
    """Map a free-text collection name to its canonical key."""
    return hadith.normalize_collection(name)


# ---------------------------------------------------------------------------
# Isnad analysis
# ---------------------------------------------------------------------------


_GENERATION_RANK = {
    "Sahabah": 0,
    "Tabi'un": 1,
    "Atba' al-Tabi'in": 2,
    "Later": 3,
}


def analyze_isnad(names: list[str]) -> IsnadAnalysis:
    """Analyze a chain of narrator names for continuity and reliability.

    Narrators are matched against the curated rijal reference; names that do
    not resolve are reported as unknown rather than silently accepted. The
    generation order of adjacent *known* narrators is checked: an isnad runs
    from the latest transmitter down to the oldest (the Companion who heard
    from the Prophet), so a reversal or a multi-generation jump is flagged.
    """
    entries: list[ChainEntry] = []
    notes: list[str] = []
    flagged = False

    for name in names:
        narrator = find_narrator(name)
        warnings: list[str] = []
        if narrator is None:
            warnings.append("Not found in the bundled rijal reference — treat as unverified.")
            flagged = True
            entries.append(
                ChainEntry(
                    name=name,
                    profile=NarratorProfile(name=name, matched=False),
                    warnings=warnings,
                )
            )
            continue
        reliability = narrator.get("reliability")
        if reliability in ("daif", "matruk"):
            warnings.append(
                f"Graded {reliability} by rijal critics — do not rely on this narrator for primary evidence."
            )
            flagged = True
        elif reliability == "saduq":
            warnings.append("Graded saduq (truthful) — accepted with corroboration.")
        entries.append(
            ChainEntry(
                name=name,
                profile=NarratorProfile(
                    name=narrator["name"],
                    matched=True,
                    generation=narrator.get("generation"),
                    death_ah=narrator.get("death_ah"),
                    reliability=reliability,
                    note=narrator.get("note"),
                ),
                warnings=warnings,
            )
        )

    continuous: bool | None = None
    known = [entry for entry in entries if entry.profile.matched]
    if len(known) >= 2:
        ranks = [_GENERATION_RANK.get(entry.profile.generation or "") for entry in known]
        if any(rank is None for rank in ranks):
            notes.append("Some known narrators lack a generation label; full continuity cannot be assessed.")
        else:
            typed = [rank for rank in ranks if rank is not None]
            reversed_gap = [i for i in range(1, len(typed)) if typed[i] >= typed[i - 1]]
            if reversed_gap:
                continuous = False
                notes.append(
                    "Chain order is implausible: an earlier-generation narrator appears before a later one "
                    "(the isnad must run from the latest transmitter down to the Companion)."
                )
                flagged = True
            else:
                jumps = [i for i in range(1, len(typed)) if typed[i - 1] - typed[i] > 1]
                if jumps:
                    continuous = False
                    notes.append(
                        "A multi-generation jump appears in the chain — a missing intermediate "
                        "transmitter (an inqita') is likely."
                    )
                    flagged = True
                else:
                    continuous = True

    if not continuous and len(known) < 2 and any(not e.profile.matched for e in entries):
        notes.append("Unknown narrators prevent a continuity verdict.")

    if not flagged:
        notes.append("No reliability or continuity concerns found among the known narrators.")

    return IsnadAnalysis(narrators=entries, continuous=continuous, flagged=flagged, notes=notes)


# ---------------------------------------------------------------------------
# Variant alignment
# ---------------------------------------------------------------------------


def _variant_ids_for(collection: str, number: int) -> list[str]:
    return [
        variant_id
        for variant_id, variant in load_variants().items()
        if any(ref["collection"] == collection and ref["number"] == number for ref in variant.get("references", []))
    ]


def _grade_only(collection: str, number: int, book: int | None = None) -> tuple[str, str, bool]:
    """Grade + chain for one reference without building its variant list."""
    record = hadith.get_default_source().get(collection, number, book)
    if record is None:
        return hadith.Strength.UNKNOWN.value, "", False
    return record.grade.value, record.chain_type.value, True


def _variant_reference(ref: dict[str, Any]) -> VariantReference:
    collection = ref["collection"]
    number = ref["number"]
    grade, chain_type, verified = _grade_only(collection, number, ref.get("book"))
    return VariantReference(
        collection=collection,
        collection_name=load_collections().get(collection, {}).get("name", collection),
        number=number,
        book=ref.get("book"),
        book_number=ref.get("book_number"),
        note=ref.get("note"),
        grade=grade,
        chain_type=chain_type,
        verified=verified,
        citation=format_citation(collection, number, ref.get("book"), ref.get("book_number")),
    )


def build_variant_match(variant_id: str, score: float | None = None) -> VariantMatch | None:
    variant = load_variants().get(variant_id)
    if variant is None:
        return None
    return VariantMatch(
        variant_id=variant_id,
        title=variant.get("title", variant_id),
        narration=variant.get("narration", ""),
        score=score,
        references=[_variant_reference(ref) for ref in variant.get("references", [])],
    )


def find_variants_by_text(matn: str) -> list[VariantMatch]:
    """Match free text against the curated variants by token overlap, best first."""
    query = tokenize(matn)
    if not query:
        return []
    scored: list[tuple[float, str]] = []
    for variant_id, variant in load_variants().items():
        candidate = tokenize(f"{variant.get('title', '')} {variant.get('narration', '')}")
        score = match_score(query, candidate)
        if score >= TEXT_MATCH_THRESHOLD:
            scored.append((score, variant_id))
    scored.sort(key=lambda item: item[0], reverse=True)
    matches: list[VariantMatch] = []
    for score, variant_id in scored[:MAX_VARIANT_MATCHES]:
        match = build_variant_match(variant_id, score=score)
        if match is not None:
            matches.append(match)
    return matches


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/sources", response_model=list[CollectionInfo])
async def list_sources() -> list[CollectionInfo]:
    """Hadith collections the research agent can retrieve, with hadith counts."""
    return [
        CollectionInfo(key=info["key"], name=info["name"], hadith_count=info["hadith_count"])
        for info in load_collections().values()
    ]


@router.post("/lookup", response_model=LookupResult)
async def lookup_endpoint(request: LookupRequest) -> LookupResult:
    """Resolve a hadith citation and return its grading, justification, and variants."""
    if request.reference:
        resolved = resolve_reference(request.reference)
    elif request.collection and request.number is not None:
        resolved = (normalize_collection(request.collection) or request.collection, request.number, request.book)
    else:
        raise HTTPException(
            status_code=400, detail="Provide a free-text 'reference' or both 'collection' and 'number'."
        )

    if resolved is None:
        raise HTTPException(
            status_code=400,
            detail="Could not parse a collection + number from the reference. Try e.g. 'Sahih al-Bukhari 1'.",
        )
    collection, number, book = resolved
    result = lookup(collection, number, book)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No record for {collection} {number} in the bundled dataset.",
        )
    return result


@router.post("/chain", response_model=IsnadAnalysis)
async def chain_endpoint(request: ChainRequest) -> IsnadAnalysis:
    """Analyze a narrator chain for continuity and reliability."""
    return analyze_isnad(request.narrators)


@router.post("/variants", response_model=VariantResponse)
async def variants_endpoint(request: VariantRequest) -> VariantResponse:
    """Find known variant narrations by free-text matn or by a citation."""
    if request.collection and request.number is not None:
        collection = normalize_collection(request.collection) or request.collection
        variant_ids = _variant_ids_for(collection, request.number)
        matches: list[VariantMatch] = [
            match for match in (build_variant_match(variant_id) for variant_id in variant_ids) if match is not None
        ]
        query = f"{collection} {request.number}"
    elif request.text:
        matches = find_variants_by_text(request.text)
        query = request.text
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide 'text' to match, or both 'collection' and 'number' to trace a citation.",
        )
    return VariantResponse(query=query, matches=matches)
