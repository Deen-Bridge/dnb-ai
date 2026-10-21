"""Islamic research agent orchestration framework (#126).

This module wires the generic :mod:`orchestration` engine to a set of offline,
deterministic research agents for the four classical research domains:

* ``quran``  - thematic concordance over the bundled Qur'an datasets
* ``hadith`` - collection, narration and grading detection
* ``tafsir`` - ayah reference discovery, enriched by the Qur'an agent
* ``fiqh``   - fiqh relevance and category classification

Agents are discovered through the capability registry, queries are decomposed
into research domains, the resulting plan is executed as a dependency-aware DAG
with parallel waves, and every run is retained in a bounded trace store so it
can be inspected through ``GET /orchestration/traces/{workflow_id}``.

The agents perform real work over the bundled corpora, so the whole framework
runs without an API key or network access.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from orchestration import (
    AgentContext,
    AgentRegistry,
    AgentSpec,
    AgentTask,
    ExecutionPlan,
    ExecutionPlanBuilder,
    Metrics,
    MultiAgentOrchestrator,
    QueryAnalysis,
    QueryAnalyzer,
    ResultSynthesizer,
    TaskResult,
    TaskStatus,
    WorkflowResult,
)

RESEARCH_DOMAINS: tuple[str, ...] = ("quran", "hadith", "tafsir", "fiqh")

_DOMAIN_ORDER: tuple[str, ...] = ("quran", "hadith", "fiqh", "tafsir")

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "quran": (
        "quran",
        "qur'an",
        "quranic",
        "ayah",
        "ayat",
        "verse",
        "verses",
        "surah",
        "sura",
        "allah",
        "revelation",
        "thematic",
        "theme",
        "themes",
    ),
    "hadith": (
        "hadith",
        "hadiths",
        "sunnah",
        "sunna",
        "narration",
        "narrated",
        "isnad",
        "matn",
        "sahih",
        "bukhari",
        "muslim",
        "tirmidhi",
        "nasai",
        "abu dawud",
        "ibn majah",
        "malik",
        "nawawi",
        "qudsi",
        "grading",
        "authentic",
    ),
    "tafsir": (
        "tafsir",
        "tafseer",
        "exegesis",
        "interpretation",
        "interpret",
        "explain",
        "explanation",
        "meaning",
        "commentary",
    ),
    "fiqh": (
        "fiqh",
        "ruling",
        "rulings",
        "fatwa",
        "halal",
        "haram",
        "permissible",
        "forbidden",
        "obligatory",
        "wajib",
        "fard",
        "makruh",
        "wudu",
        "salah",
        "prayer",
        "fasting",
        "zakat",
        "hajj",
        "marriage",
        "divorce",
        "inheritance",
        "riba",
        "contract",
        "madhhab",
    ),
}

_FIQH_CATEGORIES: dict[str, tuple[str, ...]] = {
    "worship": ("prayer", "salah", "wudu", "fasting", "zakat", "hajj", "worship"),
    "transactions": ("sale", "buy", "sell", "trade", "riba", "interest", "loan", "contract", "money"),
    "family": ("marriage", "divorce", "nikah", "talaq", "inheritance", "custody"),
    "diet": ("halal", "haram", "eat", "food", "meat", "alcohol"),
}

_FACET_PATTERN = re.compile(r"\s*(?:\?|;|\band\b|\bbut\b|\bhowever\b|\balso\b|\bthen\b)\s*", re.IGNORECASE)
_TERM_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "on",
        "in",
        "to",
        "for",
        "with",
        "about",
        "from",
        "by",
        "what",
        "which",
        "who",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "at",
        "between",
        "compare",
        "contrast",
        "discuss",
        "tell",
        "me",
        "please",
        "how",
        "why",
        "when",
        "where",
        "there",
        "their",
        "them",
        "they",
    }
)


def decompose_query(query: str) -> tuple[str, ...]:
    """Split a compound query into research facets."""
    parts = [part.strip() for part in _FACET_PATTERN.split(query) if part and part.strip()]
    return tuple(part for part in parts if len(part) >= 3) or (query.strip(),)


def _terms(text: str) -> set[str]:
    return {match.casefold() for match in _TERM_PATTERN.findall(text)}


def _keyword_matches(lowered: str, terms: set[str], keyword: str) -> bool:
    if " " in keyword or "'" in keyword:
        return keyword in lowered
    return keyword in terms


def _matched_domains(query: str) -> list[str]:
    lowered = query.casefold()
    terms = _terms(query)
    domains: list[str] = []
    for domain in RESEARCH_DOMAINS:
        if any(_keyword_matches(lowered, terms, keyword) for keyword in DOMAIN_KEYWORDS[domain]):
            domains.append(domain)
    return domains


class ResearchQueryAnalyzer(QueryAnalyzer):
    """Select research agents by decomposing a query into domain facets."""

    def analyze(self, query: str) -> QueryAnalysis:
        selected: list[str] = []
        capabilities: list[str] = []
        scores: dict[str, float] = {}

        for facet in decompose_query(query):
            for domain in _matched_domains(facet):
                agent = self._agent_for_domain(domain)
                if agent is None:
                    continue
                if agent.name not in selected:
                    selected.append(agent.name)
                    capabilities.append(domain)
                scores[agent.name] = scores.get(agent.name, 0.0) + 1.0

        if "tafsir" in capabilities and "quran" not in capabilities:
            quran_agent = self._agent_for_domain("quran")
            if quran_agent is not None and quran_agent.name not in selected:
                selected.insert(0, quran_agent.name)
                capabilities.insert(0, "quran")
                scores[quran_agent.name] = 0.5

        if not selected:
            candidates = [agent for agent in self._registry.all() if agent.default] or list(self._registry.all())
            if candidates:
                best = sorted(candidates, key=lambda agent: (-agent.priority, agent.name))[0]
                selected.append(best.name)
                capabilities.append(sorted(best.capabilities)[0])
                scores[best.name] = 0.1

        selected = selected[: self._max_agents]
        return QueryAnalysis(
            query=query,
            selected_agents=tuple(selected),
            capabilities=tuple(capabilities[: len(selected)]),
            sequential=False,
            scores={name: scores[name] for name in selected},
        )

    def _agent_for_domain(self, domain: str) -> AgentSpec | None:
        matches = self._registry.for_capability(domain)
        return matches[0] if matches else None


class ResearchPlanBuilder(ExecutionPlanBuilder):
    """Build a DAG where Tafsir consumes the Qur'an agent's findings."""

    def build(self, analysis: QueryAnalysis) -> ExecutionPlan:
        selected = [self._registry.get(name) for name in analysis.selected_agents]
        by_domain = {_domain_for(agent): agent for agent in selected}

        ordered: list[AgentSpec] = []
        for domain in _DOMAIN_ORDER:
            agent = by_domain.get(domain)
            if agent is not None and agent not in ordered:
                ordered.append(agent)
        for agent in selected:
            if agent not in ordered:
                ordered.append(agent)

        tasks: list[AgentTask] = []
        task_ids: dict[str, str] = {}
        for index, agent in enumerate(ordered):
            domain = _domain_for(agent)
            capability = domain if domain in agent.capabilities else sorted(agent.capabilities)[0]
            dependencies: tuple[str, ...] = ()
            if domain == "tafsir" and "quran" in task_ids:
                dependencies = (task_ids["quran"],)
            task_id = f"agent-{index + 1}-{agent.name}"
            tasks.append(
                AgentTask(
                    id=task_id,
                    agent=agent.name,
                    capability=capability,
                    payload=analysis.query,
                    dependencies=dependencies,
                    fallback_agent=agent.fallback_agent,
                )
            )
            task_ids[domain] = task_id

        plan = ExecutionPlan(tuple(tasks))
        plan.validate(self._registry)
        return plan


def _domain_for(agent: AgentSpec) -> str:
    for domain in RESEARCH_DOMAINS:
        if domain in agent.capabilities:
            return domain
    return sorted(agent.capabilities)[0]


def _query_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        return str(payload.get("query", ""))
    return str(payload)


def _search_terms(query: str) -> list[str]:
    return sorted(term for term in _terms(query) if len(term) >= 3 and term not in _STOPWORDS)


async def _quran_handler(task: AgentTask, context: AgentContext) -> dict[str, Any]:
    from thematic_quran import get_thematic_retriever

    retriever = get_thematic_retriever()
    query = _query_text(task.payload)
    themes: dict[str, Any] = {}
    for term in _search_terms(query):
        for theme in retriever.taxonomy.search_themes(term):
            themes[theme.id] = theme
    if not themes:
        for theme in retriever.taxonomy.search_themes(query.casefold()):
            themes[theme.id] = theme

    ranked = sorted(themes.values(), key=lambda theme: -len(retriever.verse_store.get_verses_for_theme(theme.id)))
    findings: list[dict[str, Any]] = []
    for theme in ranked[:3]:
        verses = retriever.verse_store.get_verses_for_theme(theme.id)[:5]
        findings.append(
            {
                "theme": theme.id,
                "name": theme.name,
                "name_arabic": theme.name_arabic,
                "verses": [f"{verse.surah}:{verse.ayah}" for verse in verses],
            }
        )

    if findings:
        parts = [
            f"theme '{finding['name']}' ({', '.join(finding['verses'][:3]) or 'no mapped verses'})"
            for finding in findings
        ]
        summary = "Qur'anic themes identified: " + "; ".join(parts) + "."
    else:
        summary = "No strong thematic match found for the query."

    result = {
        "domain": "quran",
        "query": query,
        "themes_found": len(ranked),
        "findings": findings,
        "summary": summary,
    }
    await context.publish("research.quran", result)
    return result


async def _hadith_handler(task: AgentTask, context: AgentContext) -> dict[str, Any]:
    from hadith import annotate

    query = _query_text(task.payload)
    references = annotate(query)
    collections: list[str] = []
    gradings: list[str] = []
    flagged: list[str] = []
    for reference in references:
        if reference.collection and reference.collection not in collections:
            collections.append(reference.collection)
        if reference.grade and reference.grade not in gradings:
            gradings.append(reference.grade)
        if reference.flagged:
            flagged.append(reference.raw)

    if collections or gradings:
        bits = []
        if collections:
            bits.append("collections: " + ", ".join(collections))
        if gradings:
            bits.append("gradings: " + ", ".join(gradings))
        summary = "Hadith references detected - " + "; ".join(bits) + "."
        if flagged:
            summary += " Verify flagged narrations before relying on them."
    else:
        summary = "No explicit hadith collection or grading mentioned in the query."

    result = {
        "domain": "hadith",
        "query": query,
        "collections_detected": collections,
        "gradings_detected": gradings,
        "flagged": flagged,
        "summary": summary,
    }
    await context.publish("research.hadith", result)
    return result


async def _tafsir_handler(task: AgentTask, context: AgentContext) -> dict[str, Any]:
    from tafsir import detect_ayah_references, surah_by_number

    query = _query_text(task.payload)
    references = detect_ayah_references(query)
    if not references:
        references = _references_from_dependencies(context.dependency_results)

    named: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reference in references:
        key = getattr(reference, "key", f"{reference.surah}:{reference.ayah}")
        if key in seen:
            continue
        seen.add(key)
        surah = surah_by_number(reference.surah)
        named.append({"reference": key, "surah_name": surah.name if surah else None})

    summary = (
        "Ayah references to interpret: " + ", ".join(item["reference"] for item in named) + "."
        if named
        else "No specific ayah reference detected to interpret."
    )
    result = {
        "domain": "tafsir",
        "query": query,
        "references": named,
        "summary": summary,
    }
    await context.publish("research.tafsir", result)
    return result


def _references_from_dependencies(dependency_results: Mapping[str, Any]) -> list[Any]:
    from tafsir import detect_ayah_references

    for output in dependency_results.values():
        if not isinstance(output, Mapping) or output.get("domain") != "quran":
            continue
        for finding in output.get("findings", [])[:1]:
            for verse in finding.get("verses", [])[:3]:
                references = detect_ayah_references(f"Explain verse {verse}")
                if references:
                    return references
    return []


async def _fiqh_handler(task: AgentTask, context: AgentContext) -> dict[str, Any]:
    from fiqh import classify_fiqh

    query = _query_text(task.payload)
    lowered = query.casefold()
    is_fiqh = classify_fiqh(query)
    categories = [
        category for category, keywords in _FIQH_CATEGORIES.items() if any(keyword in lowered for keyword in keywords)
    ]

    if is_fiqh or categories:
        area = ", ".join(categories) if categories else "general"
        summary = f"Fiqh question detected ({area}); classical rulings should be consulted."
    else:
        summary = "No fiqh ruling appears to be requested."

    result = {
        "domain": "fiqh",
        "query": query,
        "is_fiqh_question": is_fiqh,
        "categories": categories,
        "summary": summary,
    }
    await context.publish("research.fiqh", result)
    return result


async def _assistant_handler(task: AgentTask, context: AgentContext) -> dict[str, Any]:
    query = _query_text(task.payload)
    result = {
        "domain": "general",
        "query": query,
        "summary": "No specialised research domain matched; this is a general research fallback.",
    }
    await context.publish("research.general", result)
    return result


async def _synthesize_research(query: str, results: Mapping[str, TaskResult]) -> str:
    sections: list[str] = []
    for result in results.values():
        output = result.output if isinstance(result.output, Mapping) else {}
        summary = output.get("summary") if output else None
        if not summary:
            summary = str(result.output)
        sections.append(f"- **{result.agent.capitalize()}**: {summary}")

    if not sections:
        return "Unable to complete the research request because no agent produced a result."

    body = "\n".join(sections)
    return (
        f"Findings for: {query}\n\n"
        f"{body}\n\n"
        "These findings are structured evidence pointers from the bundled datasets; "
        "a scholar should confirm rulings and interpretations before relying on them."
    )


def create_research_registry() -> AgentRegistry:
    """Create the default registry of offline Islamic research agents."""
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="quran",
            capabilities=frozenset({"quran"}),
            keywords=frozenset(DOMAIN_KEYWORDS["quran"]),
            handler=_quran_handler,
            priority=4,
            fallback_agent="research_assistant",
        )
    )
    registry.register(
        AgentSpec(
            name="hadith",
            capabilities=frozenset({"hadith"}),
            keywords=frozenset(DOMAIN_KEYWORDS["hadith"]),
            handler=_hadith_handler,
            priority=3,
            fallback_agent="research_assistant",
        )
    )
    registry.register(
        AgentSpec(
            name="tafsir",
            capabilities=frozenset({"tafsir"}),
            keywords=frozenset(DOMAIN_KEYWORDS["tafsir"]),
            handler=_tafsir_handler,
            priority=2,
            fallback_agent="quran",
        )
    )
    registry.register(
        AgentSpec(
            name="fiqh",
            capabilities=frozenset({"fiqh"}),
            keywords=frozenset(DOMAIN_KEYWORDS["fiqh"]),
            handler=_fiqh_handler,
            priority=1,
            fallback_agent="research_assistant",
        )
    )
    registry.register(
        AgentSpec(
            name="research_assistant",
            capabilities=frozenset({"general"}),
            keywords=frozenset({"general", "research", "help"}),
            handler=_assistant_handler,
            default=True,
        )
    )
    return registry


class ResearchTraceStore:
    """Bounded, process-local store of completed research workflows."""

    def __init__(self, max_entries: int = 200) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, WorkflowResult] = OrderedDict()

    def put(self, result: WorkflowResult) -> None:
        self._entries[result.workflow_id] = result
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def get(self, workflow_id: str) -> WorkflowResult | None:
        result = self._entries.get(workflow_id)
        if result is not None:
            self._entries.move_to_end(workflow_id)
        return result


class ResearchOrchestrator:
    """Coordinates research planning, DAG execution, synthesis and tracing."""

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        *,
        trace_store: ResearchTraceStore | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self.registry = registry or create_research_registry()
        self.trace_store = trace_store or ResearchTraceStore()
        self.metrics = metrics or Metrics()
        self.engine = MultiAgentOrchestrator(
            self.registry,
            analyzer=ResearchQueryAnalyzer(self.registry),
            plan_builder=ResearchPlanBuilder(self.registry),
            synthesizer=ResultSynthesizer(_synthesize_research),
            metrics=self.metrics,
        )

    async def orchestrate(
        self,
        query: str,
        *,
        timeout_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkflowResult:
        timeout_seconds = timeout_ms / 1000.0 if timeout_ms is not None else None
        result = await self.engine.orchestrate(
            query,
            metadata=metadata,
            task_timeout_seconds=timeout_seconds,
            workflow_timeout_seconds=timeout_seconds,
        )
        self.trace_store.put(result)
        return result

    async def start(self) -> None:
        await self.engine.start()

    async def stop(self) -> None:
        await self.engine.stop()


class OrchestrationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    timeout_ms: int = Field(default=15000, ge=100, le=60000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentDescriptor(BaseModel):
    name: str
    capabilities: list[str]
    keywords: list[str]
    priority: int
    fallback_agent: str | None = None
    default: bool = False


class TaskResultView(BaseModel):
    task_id: str
    agent: str
    status: TaskStatus
    output: Any = None
    error: str | None = None
    duration_ms: float
    fallback_used: bool


class TraceSpanView(BaseModel):
    task_id: str
    agent: str
    status: TaskStatus
    duration_ms: float
    error: str | None = None


class WorkflowView(BaseModel):
    workflow_id: str
    query: str
    status: TaskStatus
    answer: str
    duration_ms: float
    partial_failure: bool
    selected_agents: list[str]
    capabilities: list[str]
    tasks: list[TaskResultView]
    traces: list[TraceSpanView]


router = APIRouter(prefix="/orchestration", tags=["orchestration"])

_research_orchestrator: ResearchOrchestrator | None = None


def get_research_orchestrator() -> ResearchOrchestrator:
    """Return the process-wide research orchestrator singleton."""
    global _research_orchestrator
    if _research_orchestrator is None:
        _research_orchestrator = ResearchOrchestrator()
    return _research_orchestrator


def reset_research_orchestrator() -> None:
    """Drop the singleton so tests can start from a clean registry and trace store."""
    global _research_orchestrator
    _research_orchestrator = None


def _to_view(result: WorkflowResult) -> WorkflowView:
    tasks = [
        TaskResultView(
            task_id=task.task_id,
            agent=task.agent,
            status=task.status,
            output=task.output,
            error=task.error,
            duration_ms=task.duration_ms,
            fallback_used=task.fallback_used,
        )
        for task in result.task_results.values()
    ]
    traces = [
        TraceSpanView(
            task_id=span.task_id,
            agent=span.agent,
            status=span.status,
            duration_ms=span.duration_ms,
            error=span.error,
        )
        for span in result.traces
    ]
    return WorkflowView(
        workflow_id=result.workflow_id,
        query=result.query,
        status=result.status,
        answer=result.answer,
        duration_ms=result.duration_ms,
        partial_failure=result.partial_failure,
        selected_agents=list(result.analysis.selected_agents),
        capabilities=list(result.analysis.capabilities),
        tasks=tasks,
        traces=traces,
    )


@router.get("/agents", response_model=list[AgentDescriptor])
async def list_agents() -> list[AgentDescriptor]:
    """Discover every registered research agent and its capabilities."""
    return [
        AgentDescriptor(
            name=agent.name,
            capabilities=sorted(agent.capabilities),
            keywords=sorted(agent.keywords),
            priority=agent.priority,
            fallback_agent=agent.fallback_agent,
            default=agent.default,
        )
        for agent in get_research_orchestrator().registry.all()
    ]


@router.get("/agents/{name}", response_model=AgentDescriptor)
async def get_agent(name: str) -> AgentDescriptor:
    """Return one agent's capability descriptor."""
    try:
        agent = get_research_orchestrator().registry.get(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not registered.") from exc
    return AgentDescriptor(
        name=agent.name,
        capabilities=sorted(agent.capabilities),
        keywords=sorted(agent.keywords),
        priority=agent.priority,
        fallback_agent=agent.fallback_agent,
        default=agent.default,
    )


@router.post("/run", response_model=WorkflowView)
async def run_orchestration(body: OrchestrationRequest) -> WorkflowView:
    """Run a complex scholarly query through the multi-agent research framework."""
    result = await get_research_orchestrator().orchestrate(
        body.query,
        timeout_ms=body.timeout_ms,
        metadata=body.metadata,
    )
    return _to_view(result)


@router.get("/traces/{workflow_id}", response_model=WorkflowView)
async def get_trace(workflow_id: str) -> WorkflowView:
    """Retrieve a completed workflow trace for transparent observability."""
    result = get_research_orchestrator().trace_store.get(workflow_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No trace with id '{workflow_id}'.")
    return _to_view(result)


@router.get("/metrics")
async def get_metrics() -> dict[str, float | int]:
    """Expose aggregate orchestration counters and latency observations."""
    return get_research_orchestrator().metrics.snapshot()
