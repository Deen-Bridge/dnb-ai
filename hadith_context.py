"""Contextual hadith interpretation — sharh, asbab al-wurud, and synthesis.

Why this exists
---------------
Bare hadith retrieval hands the user a translated matn and nothing else: no
occasion of narration (asbab al-wurud), no classical commentary (sharh), no
sense of which narrations complete or qualify it, and no signal about where a
madhab reads it differently. That gap is exactly where a reader draws a ruling
from a text stripped of the context the commentators always attached to it.

This module is the *contextual* layer on top of ``hadith.py`` (which grades
authenticity). It does not invent interpretation with a live model. Instead it
holds a small, curated, offline knowledge base of well-known narrations, each
annotated with the context a classical ``sharh`` supplies — historical
circumstance, scholarly commentary keyed to the work it came from, complementary
narrations, practical application, madhab-specific readings, apparent
contradictions with their reconciliation, and contemporary framing. The public
functions retrieve a record, synthesize a madhab-aware interpretation from it,
and answer plain questions by routing them to the facet of the record that
addresses them.

Design constraints
------------------
- Deterministic and dependency-free: every answer is assembled from the bundled
  records, so the same input always yields the same output and no network,
  model, or dataset download is involved.
- Attribution-first: every interpretive claim carries the name and author of the
  ``sharh`` it is drawn from. Nothing is presented as this service's own opinion.
- No import-time side effects: the knowledge base is a module-level literal, so
  ``import hadith_context`` is safe under mypy, Docker boot, and ``/ping``.
"""

from __future__ import annotations

import re
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/hadith-context", tags=["hadith-context"])


# ---------------------------------------------------------------------------
# Madhab normalization
# ---------------------------------------------------------------------------

_MADHAB_ALIASES: dict[str, str] = {
    "hanafi": "hanafi",
    "hanafee": "hanafi",
    "maliki": "maliki",
    "malikee": "maliki",
    "shafii": "shafii",
    "shafi'i": "shafii",
    "shafie": "shafii",
    "hanbali": "hanbali",
    "hanbalee": "hanbali",
}

VALID_MADHABS: tuple[str, ...] = ("hanafi", "maliki", "shafii", "hanbali")


def normalize_madhab(raw: str | None) -> str | None:
    """Fold a free-text madhab name to a canonical key, or ``None`` if absent.

    Unknown non-empty values are returned lowercased/trimmed so the caller can
    still echo them; only the four Sunni schools are treated as canonical.
    """
    if raw is None:
        return None
    key = raw.strip().lower().replace("’", "'")
    if not key:
        return None
    return _MADHAB_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Reference normalization
# ---------------------------------------------------------------------------

_COLLECTION_ALIASES: dict[str, str] = {
    "bukhari": "bukhari",
    "sahihbukhari": "bukhari",
    "sahihalbukhari": "bukhari",
    "muslim": "muslim",
    "sahihmuslim": "muslim",
    "nawawi": "nawawi",
    "arbainnawawi": "nawawi",
    "40hadithnawawi": "nawawi",
}


def normalize_reference(raw: str) -> str:
    """Normalize a hadith reference like "Sahih Bukhari 1" to "bukhari:1".

    Collection name and number are separated; the name is folded through
    ``_COLLECTION_ALIASES`` and the number is the first integer found. A
    reference already in canonical ``collection:number`` form round-trips.
    """
    text = raw.strip().lower().replace("’", "'")
    text = text.replace(":", " ")
    number_match = re.search(r"\d+", text)
    number = number_match.group(0) if number_match else ""
    name_part = re.sub(r"[^a-z]", "", re.sub(r"\d+", "", text))
    collection = _COLLECTION_ALIASES.get(name_part, name_part)
    if collection and number:
        return f"{collection}:{number}"
    return collection or number


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ScholarlyCommentary(BaseModel):
    """One strand of classical or contemporary commentary (sharh)."""

    work: str = Field(description="Title of the commentary work.")
    author: str = Field(description="Author the commentary is attributed to.")
    madhab: str | None = Field(
        default=None,
        description="Madhab lens of this reading, or null if cross-madhab.",
    )
    era: str = Field(description="'classical' or 'contemporary'.")
    note: str = Field(description="What this work says about the narration.")


class RelatedHadith(BaseModel):
    """A complementary or qualifying narration."""

    reference: str = Field(description="Canonical reference of the related hadith.")
    relation: str = Field(description="How it relates, e.g. 'completes', 'qualifies'.")
    note: str = Field(description="Why it is relevant to fuller understanding.")


class Contradiction(BaseModel):
    """An apparent tension with another text and its reconciliation."""

    apparent_tension: str = Field(description="The surface-level conflict.")
    reconciliation: str = Field(description="How the commentators reconcile it (jam').")


class HadithContext(BaseModel):
    """The full contextual record bundled for a narration."""

    reference: str
    collection: str
    text: str
    topics: list[str]
    historical_context: str
    key_points: list[str]
    applications: list[str]
    commentaries: list[ScholarlyCommentary]
    related: list[RelatedHadith]
    contradictions: list[Contradiction]
    contemporary_perspectives: list[str]


class InterpretationResponse(BaseModel):
    """A synthesized, madhab-aware interpretation of a narration."""

    reference: str
    collection: str
    text: str
    madhab: str | None = Field(default=None, description="Madhab lens applied, if any.")
    summary: str
    key_points: list[str]
    historical_context: str
    applications: list[str]
    commentaries: list[ScholarlyCommentary]
    related: list[RelatedHadith]
    contradictions: list[Contradiction]
    contemporary_perspectives: list[str]


class AskRequest(BaseModel):
    """A plain-language question about a hadith's meaning or application."""

    question: str = Field(min_length=1, description="The user's question.")
    reference: str | None = Field(
        default=None,
        description="Optional hadith reference to scope the question to.",
    )
    madhab: str | None = Field(default=None, description="Optional madhab lens for the answer.")


class AskResponse(BaseModel):
    """A deterministic answer routed to the relevant facet of the record."""

    reference: str
    matched: bool = Field(description="Whether a known narration was matched.")
    answer: str
    key_points: list[str]
    sources: list[str] = Field(description="Commentary works the answer draws on.")


# ---------------------------------------------------------------------------
# Bundled knowledge base
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: dict[str, HadithContext] = {
    "bukhari:1": HadithContext(
        reference="bukhari:1",
        collection="Sahih al-Bukhari",
        text=("Actions are but by intentions, and every person will have only what they intended."),
        topics=["intention", "niyyah", "sincerity", "deeds", "reward"],
        historical_context=(
            "Reported over the incident of the man who emigrated to Madinah to "
            "marry a woman ('the emigrant of Umm Qais'), not for the sake of God, "
            "so his hijrah was toward what he sought — the asbab al-wurud that "
            "fixes the hadith's scope to the worth of the motive behind an act."
        ),
        key_points=[
            "The moral and legal weight of an act follows the intention behind it.",
            "Identical outward acts diverge entirely in reward by their niyyah.",
            "Scholars placed it among the axes on which the whole religion turns.",
        ],
        applications=[
            "Renew intention before worship so habit does not hollow it out.",
            "Purify motive in study, work, and charity to make the mundane an act of worship.",
        ],
        commentaries=[
            ScholarlyCommentary(
                work="Fath al-Bari",
                author="Ibn Hajar al-Asqalani",
                madhab="shafii",
                era="classical",
                note=(
                    "Placed by al-Bukhari at the head of the Sahih as a preface to "
                    "the entire book; Ibn Hajar reads it as making sincerity the "
                    "precondition of every act of worship being accepted."
                ),
            ),
            ScholarlyCommentary(
                work="Al-Minhaj (Sharh Sahih Muslim)",
                author="al-Nawawi",
                madhab=None,
                era="classical",
                note=(
                    "Counts it among the hadiths the whole of Islam revolves upon "
                    "and derives that intention distinguishes worship from habit and "
                    "ranks acts of worship among themselves."
                ),
            ),
            ScholarlyCommentary(
                work="Contemporary usul framing",
                author="Modern teaching syntheses",
                madhab=None,
                era="contemporary",
                note=(
                    "Cited today to ground ethics of purpose: outward compliance "
                    "without inner intent is treated as incomplete, not sufficient."
                ),
            ),
        ],
        related=[
            RelatedHadith(
                reference="bukhari:6502",
                relation="complements",
                note=(
                    "The hadith of drawing near through voluntary acts shows the "
                    "reward that intention-shaped worship opens onto."
                ),
            ),
        ],
        contradictions=[
            Contradiction(
                apparent_tension=("If reward is by intention alone, does outward action not matter?"),
                reconciliation=(
                    "The commentators read intention as the condition of an act's "
                    "acceptance, not a substitute for it: the act is still required, "
                    "and intention determines whether the performed act is rewarded."
                ),
            ),
        ],
        contemporary_perspectives=[
            "Invoked in discussions of professional ethics and sincerity in public "
            "religious life, where visible good works can mask mixed motives.",
        ],
    ),
    "muslim:8": HadithContext(
        reference="muslim:8",
        collection="Sahih Muslim",
        text=(
            "Islam is to testify that there is no god but God and that Muhammad is "
            "the Messenger of God, to establish prayer, give zakat, fast Ramadan, "
            "and make pilgrimage; iman is to believe in God, His angels, His books, "
            "His messengers, the Last Day, and the decree; ihsan is to worship God "
            "as though you see Him."
        ),
        topics=["islam", "iman", "ihsan", "creed", "gabriel", "faith", "decree"],
        historical_context=(
            "The Hadith of Gabriel: the angel came in the form of a man and "
            "questioned the Prophet before the Companions so that the answers would "
            "teach them their religion — narrated as an occasion staged for "
            "instruction, which is why the commentators call it 'the mother of the "
            "Sunnah.'"
        ),
        key_points=[
            "Distinguishes three tiers: islam (outward submission), iman (inner "
            "belief), and ihsan (excellence in worship).",
            "Enumerates the pillars of practice and the articles of faith together.",
            "Frames the whole of the religion as a single teaching encounter.",
        ],
        applications=[
            "Use the three tiers as a self-audit: practice, belief, and quality of presence in worship.",
            "Teach foundations in this order — act, creed, then excellence.",
        ],
        commentaries=[
            ScholarlyCommentary(
                work="Al-Minhaj (Sharh Sahih Muslim)",
                author="al-Nawawi",
                madhab=None,
                era="classical",
                note=(
                    "Treats it as a comprehensive summary of worship, belief, and "
                    "excellence, and derives the layered structure of the religion "
                    "from it."
                ),
            ),
            ScholarlyCommentary(
                work="Jami' al-'Ulum wa al-Hikam",
                author="Ibn Rajab al-Hanbali",
                madhab="hanbali",
                era="classical",
                note=(
                    "Expounds it as the second of the forty and reads ihsan as the "
                    "station that perfects both islam and iman."
                ),
            ),
        ],
        related=[
            RelatedHadith(
                reference="nawawi:2",
                relation="same narration",
                note="Recorded as hadith 2 in al-Nawawi's Forty, the standard teaching text.",
            ),
        ],
        contradictions=[
            Contradiction(
                apparent_tension=("Elsewhere islam and iman are used interchangeably; here they are distinguished."),
                reconciliation=(
                    "When mentioned separately each term carries the other's "
                    "meaning; when paired, as here, islam denotes outward acts and "
                    "iman inner belief — a distinction of context, not doctrine."
                ),
            ),
        ],
        contemporary_perspectives=[
            "Widely used as the syllabus skeleton for introductory 'aqidah and worship courses.",
        ],
    ),
    "bukhari:13": HadithContext(
        reference="bukhari:13",
        collection="Sahih al-Bukhari",
        text=("None of you truly believes until he loves for his brother what he loves for himself."),
        topics=["brotherhood", "love", "faith", "ethics", "empathy", "rights"],
        historical_context=(
            "A general statement of the ethic of faith with no single occasion of "
            "narration recorded; the commentators read 'brother' expansively and "
            "'believes' as denoting completeness of faith, not its bare validity."
        ),
        key_points=[
            "Completeness of faith is tied to willing good for others as for oneself.",
            "The negation ('none truly believes') targets perfection of iman, not its existence.",
            "Establishes empathy as a measure of religious sincerity.",
        ],
        applications=[
            "Test dealings by the standard you would accept for yourself.",
            "Root out envy (hasad) as the inversion of this love.",
        ],
        commentaries=[
            ScholarlyCommentary(
                work="Fath al-Bari",
                author="Ibn Hajar al-Asqalani",
                madhab="shafii",
                era="classical",
                note=(
                    "Clarifies that the negated faith is complete faith, not the "
                    "root, and that 'what he loves' means goodness, not every desire."
                ),
            ),
            ScholarlyCommentary(
                work="Jami' al-'Ulum wa al-Hikam",
                author="Ibn Rajab al-Hanbali",
                madhab="hanbali",
                era="classical",
                note=(
                    "Reads it as requiring that one wish for fellow believers the "
                    "very good one wishes for oneself, and warns against the hasad "
                    "that contradicts it."
                ),
            ),
        ],
        related=[
            RelatedHadith(
                reference="bukhari:1",
                relation="qualifies",
                note=(
                    "Read with the intentions hadith, it locates faith's "
                    "completeness in inner disposition as well as outward act."
                ),
            ),
        ],
        contradictions=[
            Contradiction(
                apparent_tension=("Does 'none believes' expel a self-interested Muslim from faith?"),
                reconciliation=(
                    "No — the commentators take the denial to mean faith is not "
                    "perfected, harmonizing it with texts affirming the faith of "
                    "sinning believers."
                ),
            ),
        ],
        contemporary_perspectives=[
            "Cited as an Islamic articulation of the ethic of reciprocity in "
            "discussions of social solidarity and community rights.",
        ],
    ),
}

# Alias references (e.g. al-Nawawi numbering) pointing at the same record.
_REFERENCE_ALIASES: dict[str, str] = {
    "nawawi:1": "bukhari:1",
    "nawawi:2": "muslim:8",
    "nawawi:13": "bukhari:13",
}


def _resolve(reference: str) -> str:
    canonical = normalize_reference(reference)
    return _REFERENCE_ALIASES.get(canonical, canonical)


def get_hadith_context(reference: str) -> HadithContext | None:
    """Return the bundled context record for a reference, or ``None``."""
    return _KNOWLEDGE_BASE.get(_resolve(reference))


def list_references() -> list[str]:
    """Return the canonical references the knowledge base covers."""
    return sorted(_KNOWLEDGE_BASE)


def _filter_commentaries(commentaries: list[ScholarlyCommentary], madhab: str | None) -> list[ScholarlyCommentary]:
    """Order commentaries so the requested madhab (plus cross-madhab) lead.

    Nothing is dropped — hiding a school's reading would distort the ikhtilaf —
    but when a madhab is requested its readings and the cross-madhab ones are
    surfaced first, followed by the other schools for contrast.
    """
    if madhab is None:
        return list(commentaries)
    preferred = [c for c in commentaries if c.madhab in (madhab, None)]
    others = [c for c in commentaries if c.madhab not in (madhab, None)]
    return preferred + others


def synthesize_interpretation(ctx: HadithContext, madhab: str | None = None) -> InterpretationResponse:
    """Assemble a madhab-aware interpretation from a context record."""
    normalized = normalize_madhab(madhab)
    commentaries = _filter_commentaries(ctx.commentaries, normalized)
    lens = f" Read here with the {normalized} school's commentary foregrounded." if normalized else ""
    summary = (
        f"{ctx.collection} {ctx.reference.split(':')[-1]} concerns "
        f"{', '.join(ctx.topics[:3])}. {ctx.key_points[0]}{lens}"
    )
    return InterpretationResponse(
        reference=ctx.reference,
        collection=ctx.collection,
        text=ctx.text,
        madhab=normalized,
        summary=summary,
        key_points=ctx.key_points,
        historical_context=ctx.historical_context,
        applications=ctx.applications,
        commentaries=commentaries,
        related=ctx.related,
        contradictions=ctx.contradictions,
        contemporary_perspectives=ctx.contemporary_perspectives,
    )


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------

_FACET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "historical_context": (
        "history",
        "historical",
        "when",
        "why",
        "occasion",
        "context",
        "circumstance",
        "revealed",
        "narrated",
    ),
    "applications": ("apply", "application", "practice", "practical", "how do", "how should", "act", "do i", "use"),
    "contradictions": ("contradict", "conflict", "tension", "reconcile", "against", "but "),
    "contemporary_perspectives": ("today", "modern", "contemporary", "now"),
}

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "is",
        "are",
        "what",
        "does",
        "do",
        "how",
        "why",
        "when",
        "and",
        "for",
        "in",
        "on",
        "this",
        "that",
        "hadith",
        "about",
        "mean",
        "meaning",
        "it",
        "its",
        "with",
        "as",
        "be",
        "by",
    }
)


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOPWORDS}


def _best_match(question: str) -> HadithContext | None:
    """Pick the record whose topics/text overlap the question most."""
    q_tokens = _tokenize(question)
    if not q_tokens:
        return None
    best: HadithContext | None = None
    best_score = 0
    for ctx in _KNOWLEDGE_BASE.values():
        haystack = _tokenize(" ".join(ctx.topics) + " " + ctx.text)
        score = len(q_tokens & haystack)
        if score > best_score:
            best_score = score
            best = ctx
    return best if best_score > 0 else None


def answer_question(question: str, reference: str | None = None, madhab: str | None = None) -> AskResponse:
    """Answer a plain question by routing it to the relevant facet.

    If a reference is supplied it scopes the answer; otherwise the best-matching
    record is chosen by topic overlap. The question's phrasing selects which
    facet (history, application, contradiction, contemporary, or general
    interpretation) supplies the body of the answer. Everything is drawn from
    the bundled record, so the answer is deterministic and attributed.
    """
    ctx = get_hadith_context(reference) if reference else _best_match(question)
    if ctx is None:
        return AskResponse(
            reference=normalize_reference(reference) if reference else "",
            matched=False,
            answer=(
                f"No bundled contextual record matches that question. Known narrations: {', '.join(list_references())}."
            ),
            key_points=[],
            sources=[],
        )

    q_lower = question.lower()
    normalized = normalize_madhab(madhab)
    commentaries = _filter_commentaries(ctx.commentaries, normalized)
    sources = [f"{c.work} ({c.author})" for c in commentaries]

    facet = "general"
    for name, keywords in _FACET_KEYWORDS.items():
        if any(k in q_lower for k in keywords):
            facet = name
            break

    if facet == "historical_context":
        answer = ctx.historical_context
    elif facet == "applications":
        answer = "Practical guidance: " + " ".join(ctx.applications)
    elif facet == "contradictions" and ctx.contradictions:
        c = ctx.contradictions[0]
        answer = f"Apparent tension: {c.apparent_tension} Reconciliation: {c.reconciliation}"
    elif facet == "contemporary_perspectives" and ctx.contemporary_perspectives:
        answer = "Contemporary framing: " + " ".join(ctx.contemporary_perspectives)
    else:
        lead = commentaries[0] if commentaries else None
        gloss = f" {lead.work} ({lead.author}): {lead.note}" if lead else ""
        answer = f"{ctx.key_points[0]}{gloss}"

    return AskResponse(
        reference=ctx.reference,
        matched=True,
        answer=answer,
        key_points=ctx.key_points,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _references_cache() -> list[str]:
    return list_references()


@router.get("/list", response_model=list[str])
def list_endpoint() -> list[str]:
    """List the canonical hadith references the system can interpret."""
    return _references_cache()


@router.get("/interpret", response_model=InterpretationResponse)
def interpret_endpoint(reference: str, madhab: str | None = None) -> InterpretationResponse:
    """Return a contextual, madhab-aware interpretation of a narration."""
    ctx = get_hadith_context(reference)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail=f"No contextual record for '{reference}'. Known: {', '.join(list_references())}.",
        )
    return synthesize_interpretation(ctx, madhab)


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest) -> AskResponse:
    """Answer a plain-language question about a hadith's meaning or application."""
    return answer_question(payload.question, payload.reference, payload.madhab)
