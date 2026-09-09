"""Quranic Concordance — topic-based ayat discovery (#125).

A structured layer over :mod:`thematic_quran` that turns the theme taxonomy
and its verse mappings into a first-class discovery API. Where the underlying
module provides the taxonomy, the mappings and the retrieval primitives, this
module supplies the *concordance* behaviours the issue calls for:

* **Topic-based search with filtering** — find themes and their ayat by an
  English or Arabic query, filterable by relevance floor and context type;
* **Multi-topic queries with AND/OR** — a verse set is the intersection
  (``AND``) or union (``OR``) of the contributing themes' verse sets, so a
  user can ask for "verses about prayer *and* patience" or "charity *or*
  fasting";
* **Topic frequency statistics across surahs** — a per-surah histogram for a
  theme, so a user can see where a theme concentrates in the mushaf;
* **Related-topic suggestions** — co-occurrence-driven query expansion that
  recommends neighbouring themes to broaden a search;
* **Hierarchical navigation** — parent/child drill-down through the taxonomy;
* **Relevance-ranked results with context** — every returned verse carries
  its theme relevance score and context type, so a caller can order and
  explain the results rather than receiving a bare list of references.

Everything is deterministic and offline: the taxonomy and the curated
``data/theme_verses.json`` mapping (see ``scripts/build_theme_mappings.py``)
are bundled, so no API key, model call, or network is required.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from errors import APIException
from thematic_quran import ThematicRetriever, VerseThemeMapping, get_thematic_retriever

router = APIRouter(prefix="/concordance", tags=["concordance"])

# A theme must appear in at least this many verses to be worth suggesting as a
# related topic; a single shared verse is too weak a signal for expansion.
_RELATED_MIN_COOCCURRENCE = 1

# Default cap for any list endpoint.
_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class VerseHit(BaseModel):
    """One mapped verse, with the theme evidence that matched it."""

    surah: int = Field(..., ge=1, le=114)
    ayah: int = Field(..., ge=1)
    reference: str = Field(..., description="surah:ayah reference, e.g. '2:255'")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    context_type: str = Field(..., description="'primary' or 'secondary'")
    annotation: str | None = None


class ThemeSummary(BaseModel):
    """A theme as returned by search and hierarchy endpoints."""

    id: str
    name: str
    name_arabic: str
    description: str
    parent_id: str | None = None
    level: int
    verse_count: int = 0


class MultiTopicQueryRequest(BaseModel):
    """Request body for a multi-topic AND/OR query."""

    topics: list[str] = Field(..., min_length=1, description="Theme ids to combine")
    operator: str = Field("OR", description="'AND' (intersection) or 'OR' (union)")
    limit: int = Field(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT)


class MultiTopicResult(BaseModel):
    """The verse set for a multi-topic AND/OR query."""

    topics: list[str] = Field(..., description="The theme ids queried")
    operator: str = Field(..., description="'AND' or 'OR'")
    verses: list[VerseHit]
    total: int
    themes: list[ThemeSummary] = Field(default_factory=list, description="Resolved theme metadata")


class FrequencyRow(BaseModel):
    """One surah's contribution to a theme's frequency distribution."""

    surah: int
    surah_name: str | None = None
    verses: int = 0
    references: list[str] = Field(default_factory=list)


class FrequencyStats(BaseModel):
    """Per-surah frequency statistics for one theme."""

    theme_id: str
    total_verses: int
    surahs_covered: int
    by_surah: list[FrequencyRow]


class RelatedTopic(BaseModel):
    """A neighbouring theme suggested for query expansion."""

    id: str
    name: str
    name_arabic: str
    description: str
    co_occurrence: int = Field(..., ge=0, description="Number of shared verses")
    shared_verses: list[str] = Field(default_factory=list)


class RelatedTopicsResponse(BaseModel):
    """Related-topic suggestions for one theme."""

    theme_id: str
    related: list[RelatedTopic]


class ThemeBrowseResponse(BaseModel):
    """A theme with its direct children, related themes, and top verses."""

    theme: ThemeSummary
    children: list[ThemeSummary] = Field(default_factory=list)
    related: list[ThemeSummary] = Field(default_factory=list)
    verses: list[VerseHit] = Field(default_factory=list)
    total_verses: int


class VerseThemesResponse(BaseModel):
    """All themes mapped to one verse."""

    surah: int
    ayah: int
    reference: str
    themes: list[dict[str, Any]] = Field(default_factory=list)


class HierarchyNode(BaseModel):
    """One node of the topic taxonomy tree."""

    id: str
    name: str
    name_arabic: str
    description: str
    parent_id: str | None = None
    level: int
    verse_count: int
    children: list[HierarchyNode] = Field(default_factory=list)


class HierarchyResponse(BaseModel):
    """The full hierarchical topic taxonomy."""

    roots: list[HierarchyNode]


class SearchResponse(BaseModel):
    """Results of a topic search: matching themes with their top verses."""

    query: str
    themes: list[dict[str, Any]] = Field(default_factory=list)
    total: int


def _verse_hit(mapping: VerseThemeMapping) -> VerseHit:
    """Convert a mapping to the API response shape."""
    return VerseHit(
        surah=mapping.surah,
        ayah=mapping.ayah,
        reference=f"{mapping.surah}:{mapping.ayah}",
        relevance_score=mapping.relevance_score,
        context_type=mapping.context_type,
        annotation=mapping.annotation,
    )


def _theme_summary(theme: Any, verse_count: int = 0) -> ThemeSummary:
    """Convert a taxonomy theme to the API summary shape."""
    return ThemeSummary(
        id=theme.id,
        name=theme.name,
        name_arabic=theme.name_arabic,
        description=theme.description,
        parent_id=theme.parent_id,
        level=theme.level,
        verse_count=verse_count,
    )


def _resolve_retriever() -> ThematicRetriever:
    """Return the shared retriever, failing loudly if mappings are missing."""
    retriever = get_thematic_retriever()
    return retriever


def _require_theme(retriever: ThematicRetriever, theme_id: str) -> Any:
    theme = retriever.taxonomy.get_theme(theme_id)
    if theme is None:
        raise APIException(
            status_code=404,
            detail=f"Topic '{theme_id}' not found in the concordance taxonomy.",
            hint=(
                "List available topics via GET /concordance/topics, or search with "
                "GET /concordance/search?q=<query> before referencing a specific topic id."
            ),
        )
    return theme


def _verse_count(retriever: ThematicRetriever, theme_id: str) -> int:
    return len(retriever.verse_store.get_verses_for_theme(theme_id))


def _dedupe_verses(mappings: list[VerseThemeMapping]) -> list[VerseThemeMapping]:
    """Deduplicate verse mappings by (surah, ayah), keeping the strongest."""
    best: dict[tuple[int, int], VerseThemeMapping] = {}
    for mapping in mappings:
        key = (mapping.surah, mapping.ayah)
        current = best.get(key)
        if current is None or mapping.relevance_score > current.relevance_score:
            best[key] = mapping
    return sorted(best.values(), key=lambda m: (-m.relevance_score, m.surah, m.ayah))


def _surah_name(surah: int) -> str | None:
    """Transliterated surah name from the bundled index; None when unknown."""
    try:
        from corpus import corpus
    except Exception:  # noqa: BLE001 - best-effort enrichment
        return None
    info = corpus.get_surah_info(surah)
    if info is None:
        return None
    return info.get("name")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/topics", response_model=HierarchyResponse)
async def topic_hierarchy() -> HierarchyResponse:
    """Return the full hierarchical topic taxonomy (main topics → sub-topics)."""
    retriever = _resolve_retriever()

    def node(theme: Any) -> HierarchyNode:
        children = [node(child) for child in retriever.taxonomy.get_children(theme.id)]
        return HierarchyNode(
            id=theme.id,
            name=theme.name,
            name_arabic=theme.name_arabic,
            description=theme.description,
            parent_id=theme.parent_id,
            level=theme.level,
            verse_count=_verse_count(retriever, theme.id),
            children=children,
        )

    roots = [node(theme) for theme in retriever.taxonomy.get_main_themes()]
    return HierarchyResponse(roots=roots)


@router.get("/search", response_model=SearchResponse)
async def search_topics(
    q: str = Query(..., min_length=1, description="Topic query in English or Arabic"),
    include_verses: bool = Query(True, description="Attach each theme's top verses"),
    verse_limit: int = Query(10, ge=1, le=_MAX_LIMIT),
) -> SearchResponse:
    """Search the topic taxonomy by English or Arabic name and keywords.

    Returns matching themes ranked by verse coverage, each with its top
    mapped ayat when ``include_verses`` is set — the topic-discovery entry
    point: type a concept and get the verses about it.
    """
    retriever = _resolve_retriever()
    themes = retriever.taxonomy.search_themes(q)
    themes.sort(key=lambda t: -_verse_count(retriever, t.id))

    results: list[dict[str, Any]] = []
    for theme in themes:
        entry: dict[str, Any] = _theme_summary(theme, _verse_count(retriever, theme.id)).model_dump()
        if include_verses:
            mappings = retriever.verse_store.get_verses_for_theme(theme.id)[:verse_limit]
            entry["verses"] = [_verse_hit(m).model_dump() for m in mappings]
        results.append(entry)

    return SearchResponse(query=q, themes=results, total=len(results))


@router.get("/topics/{theme_id}", response_model=ThemeBrowseResponse)
async def browse_topic(
    theme_id: str,
    include_children: bool = Query(True),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    min_relevance: float = Query(0.0, ge=0.0, le=1.0),
    context_type: str | None = Query(None, description="'primary' or 'secondary'"),
) -> ThemeBrowseResponse:
    """Browse one topic: its metadata, children, related topics, and verses.

    ``min_relevance`` and ``context_type`` filter the verse list, giving the
    topic-browsing search the filtering capability the issue calls for.
    """
    retriever = _resolve_retriever()
    theme = _require_theme(retriever, theme_id)
    mappings = retriever.verse_store.get_verses_for_theme(
        theme_id,
        min_relevance=min_relevance,
        context_type=context_type,
    )[:limit]
    children = retriever.taxonomy.get_children(theme_id) if include_children else []
    related = retriever.taxonomy.get_related_themes(theme_id)

    return ThemeBrowseResponse(
        theme=_theme_summary(theme, _verse_count(retriever, theme_id)),
        children=[_theme_summary(c, _verse_count(retriever, c.id)) for c in children],
        related=[_theme_summary(r, _verse_count(retriever, r.id)) for r in related],
        verses=[_verse_hit(m) for m in mappings],
        total_verses=_verse_count(retriever, theme_id),
    )


@router.post("/query", response_model=MultiTopicResult)
async def multi_topic_query(body: MultiTopicQueryRequest) -> MultiTopicResult:
    """Combine multiple topics with AND/OR semantics.

    * ``OR`` — the union of the topics' verse sets (a verse about any topic
      is returned);
    * ``AND`` — the intersection (a verse must be about every topic).

    Results are deduplicated and ranked by the strongest per-verse relevance.
    """
    operator = body.operator
    topics = body.topics
    limit = body.limit
    if operator.upper() not in ("AND", "OR"):
        raise APIException(
            status_code=422,
            detail=f"operator must be 'AND' or 'OR', got '{operator}'.",
            hint="Use 'AND' for verses about every topic, or 'OR' for verses about any topic.",
        )
    retriever = _resolve_retriever()
    resolved: list[Any] = []
    for theme_id in topics:
        resolved.append(_require_theme(retriever, theme_id))

    per_topic: list[list[VerseThemeMapping]] = []
    for theme_id in topics:
        per_topic.append(retriever.verse_store.get_verses_for_theme(theme_id))

    if operator.upper() == "AND":
        if not per_topic:
            combined: list[VerseThemeMapping] = []
        else:
            common: set[tuple[int, int]] | None = None
            for mappings in per_topic:
                verse_set = {(m.surah, m.ayah) for m in mappings}
                common = verse_set if common is None else common & verse_set
            combined = [m for m in per_topic[0] if (m.surah, m.ayah) in (common or set())]
    else:
        combined = [m for mappings in per_topic for m in mappings]

    verses = [_verse_hit(m) for m in _dedupe_verses(combined)[:limit]]
    return MultiTopicResult(
        topics=list(topics),
        operator=operator.upper(),
        verses=verses,
        total=len(_dedupe_verses(combined)),
        themes=[_theme_summary(t, _verse_count(retriever, t.id)) for t in resolved],
    )


@router.get("/topics/{theme_id}/frequency", response_model=FrequencyStats)
async def theme_frequency(theme_id: str) -> FrequencyStats:
    """Per-surah frequency statistics for one topic.

    Shows where the topic concentrates across the mushaf: how many verses
    fall in each surah, with the references, and the overall surah coverage.
    """
    retriever = _resolve_retriever()
    _require_theme(retriever, theme_id)
    mappings = retriever.verse_store.get_verses_for_theme(theme_id)

    by_surah: dict[int, list[VerseThemeMapping]] = {}
    for mapping in mappings:
        by_surah.setdefault(mapping.surah, []).append(mapping)

    rows: list[FrequencyRow] = []
    for surah in sorted(by_surah):
        refs = sorted(f"{m.surah}:{m.ayah}" for m in by_surah[surah])
        rows.append(
            FrequencyRow(
                surah=surah,
                surah_name=_surah_name(surah),
                verses=len(refs),
                references=refs,
            )
        )
    return FrequencyStats(
        theme_id=theme_id,
        total_verses=len(mappings),
        surahs_covered=len(rows),
        by_surah=rows,
    )


@router.get("/topics/{theme_id}/related", response_model=RelatedTopicsResponse)
async def related_topics(theme_id: str) -> RelatedTopicsResponse:
    """Suggest related topics for query expansion, from co-occurrence.

    Themes that share verses with the queried topic are ranked by how many
    verses they share, so the suggestions are grounded in the mapping data
    rather than hand-picked.
    """
    retriever = _resolve_retriever()
    _require_theme(retriever, theme_id)
    cooccurrence = retriever.get_theme_cooccurrence(theme_id)

    related: list[RelatedTopic] = []
    for other_id, count in sorted(cooccurrence.items(), key=lambda item: (-item[1], item[0])):
        if count < _RELATED_MIN_COOCCURRENCE:
            continue
        other = retriever.taxonomy.get_theme(other_id)
        if other is None:
            continue
        shared = retriever.verse_store.get_verses_for_theme(other_id)
        shared_refs = sorted(f"{m.surah}:{m.ayah}" for m in shared if (m.surah, m.ayah))
        related.append(
            RelatedTopic(
                id=other.id,
                name=other.name,
                name_arabic=other.name_arabic,
                description=other.description,
                co_occurrence=count,
                shared_verses=shared_refs,
            )
        )

    return RelatedTopicsResponse(theme_id=theme_id, related=related)


@router.get("/verse/{surah}/{ayah}", response_model=VerseThemesResponse)
async def verse_themes(surah: int, ayah: int) -> VerseThemesResponse:
    """Return every topic mapped to one verse (reverse lookup)."""
    if not 1 <= surah <= 114:
        raise APIException(status_code=422, detail="surah must be between 1 and 114.")
    if ayah < 1:
        raise APIException(status_code=422, detail="ayah must be at least 1.")
    retriever = _resolve_retriever()
    mappings = retriever.verse_store.get_themes_for_verse(surah, ayah)

    themes: list[dict[str, Any]] = []
    for mapping in mappings:
        theme = retriever.taxonomy.get_theme(mapping.theme_id)
        if theme is None:
            continue
        themes.append(
            {
                "theme": _theme_summary(theme, _verse_count(retriever, theme.id)).model_dump(),
                "relevance_score": mapping.relevance_score,
                "context_type": mapping.context_type,
                "annotation": mapping.annotation,
            }
        )
    themes.sort(key=lambda t: -t["relevance_score"])
    return VerseThemesResponse(surah=surah, ayah=ayah, reference=f"{surah}:{ayah}", themes=themes)
