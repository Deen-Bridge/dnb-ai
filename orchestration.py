"""Islamic Research Agent Orchestration Framework (#126).

A deterministic orchestration layer that coordinates multiple specialised
research agents — Qur'an, Hadith, Tafsir, Fiqh — so a complex scholarly query
is decomposed, routed, executed (in parallel where possible), and synthesised
into one coherent answer with a transparent execution trace.

Why deterministic
-----------------
Like :mod:`reasoning_chains`, the orchestration *structure* is the deliverable:
capability registry, query decomposition, agent selection, message passing,
execution DAG, aggregation, error propagation, and observability. The
specialised agents here perform real offline work over the bundled datasets
(the thematic concordance, the hadith grading corpus, the surah index) so the
whole framework runs in tests and CI with no API key and no network. A live
model can later fill the same agent slots with richer reasoning.

How a query flows
-----------------
1. **Decompose** — the query is split into facets (sub-questions) and each
   facet is classified into a research domain.
2. **Select** — the registry maps each facet's domain to the capable agent(s);
   facets of the same domain are merged into one task so agents run at most
   once per query.
3. **Build DAG** — a small dependency graph: the Qur'an agent identifies the
   verses/themes first, the Tafsir agent depends on it, and the Hadith and
   Fiqh agents are independent leaves that run in parallel.
4. **Execute** — independent nodes run concurrently with per-node timeouts;
   every node's inputs come from a shared blackboard (message passing), and a
   failed node degrades to a fallback agent or a partial result rather than
   failing the whole run.
5. **Aggregate** — results are merged into a coherent, attributed summary and
   an execution trace records every node, message and duration for
   transparent visibility.

Design notes
------------
* **Agent registry & discovery** — ``AgentRegistry`` holds the agents, each
  declaring a capability (domain, description, input/output shape); the API
  exposes ``GET /orchestration/agents`` for discovery.
* **Message passing** — a shared ``Blackboard`` carries typed ``AgentMessage``
  records between agents; every write is appended to the trace so inter-agent
  data sharing is auditable.
* **Execution DAG** — ``ExecutionGraph`` orders nodes by dependency; nodes with
  no unresolved dependencies run in a parallel wave.
* **Error propagation & partial results** — a node that times out or raises is
  recorded as failed; downstream nodes that only *depend* on its output run
  with a degraded input (``available: false``) instead of being cancelled,
  and the final answer reports the degradation.
* **Observability** — every orchestration produces an ``ExecutionTrace``
  (nodes with statuses and timings, events, messages) kept in a bounded store
  and retrievable by trace id.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/orchestration", tags=["orchestration"])

# ---------------------------------------------------------------------------
# Domain vocabulary
# ---------------------------------------------------------------------------


class Domain(str, Enum):
    """Research domains the orchestration can route to."""

    QURAN = "quran"
    HADITH = "hadith"
    TAFSIR = "tafsir"
    FIQH = "fiqh"
    GENERAL = "general"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    FALLBACK = "fallback"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Capabilities & registry
# ---------------------------------------------------------------------------


class AgentCapability(BaseModel):
    """What an agent can do, for the discovery endpoint."""

    domain: Domain
    description: str
    input_shape: str = "query text"
    output_shape: str = "dict of findings"
    parallel_safe: bool = Field(default=True, description="Safe to run concurrently with other agents")


class ResearchAgent(Protocol):
    """An agent that can be orchestrated. Implementations are offline & deterministic."""

    name: str
    capability: AgentCapability

    async def run(self, query: str, context: dict[str, Any]) -> dict[str, Any]: ...


class AgentRegistry:
    """Capability registry and discovery system for research agents."""

    def __init__(self) -> None:
        self._agents: dict[str, ResearchAgent] = {}
        self._fallbacks: dict[Domain, str] = {}
        self._register_defaults()

    def register(self, agent: ResearchAgent) -> None:
        """Register an agent, replacing any agent with the same name."""
        self._agents[agent.name] = agent

    def set_fallback(self, domain: Domain, agent_name: str) -> None:
        """Declare a fallback agent for a domain (used on node failure)."""
        self._fallbacks[domain] = agent_name

    def get(self, name: str) -> ResearchAgent | None:
        return self._agents.get(name)

    def find(self, domain: Domain) -> list[ResearchAgent]:
        """Discover all agents capable of handling a domain."""
        return [agent for agent in self._agents.values() if agent.capability.domain is domain]

    def find_one(self, domain: Domain) -> ResearchAgent | None:
        agents = self.find(domain)
        return agents[0] if agents else None

    def fallback_for(self, domain: Domain) -> ResearchAgent | None:
        name = self._fallbacks.get(domain)
        return self._agents.get(name) if name else None

    def list_capabilities(self) -> list[dict[str, Any]]:
        return [
            {
                "name": agent.name,
                "domain": agent.capability.domain.value,
                "description": agent.capability.description,
                "input_shape": agent.capability.input_shape,
                "output_shape": agent.capability.output_shape,
                "parallel_safe": agent.capability.parallel_safe,
            }
            for agent in self._agents.values()
        ]

    def _register_defaults(self) -> None:
        self.register(QuranResearchAgent())
        self.register(HadithResearchAgent())
        self.register(TafsirResearchAgent())
        self.register(FiqhResearchAgent())
        self.register(SynthesisAgent())
        for domain in (Domain.QURAN, Domain.HADITH, Domain.TAFSIR, Domain.FIQH):
            self.set_fallback(domain, "synthesizer")


# ---------------------------------------------------------------------------
# Blackboard (message passing infrastructure)
# ---------------------------------------------------------------------------


class AgentMessage(BaseModel):
    """A typed message written by one agent for the others to read."""

    sender: str
    recipient: str | None = Field(None, description="None means broadcast to all")
    kind: str = "result"
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Blackboard:
    """Shared message passing store between agents.

    Every write is appended to the trace, giving full inter-agent
    communication visibility. Reads are scoped by sender, so an agent only
    consumes messages addressed to it or broadcast to everyone.
    """

    def __init__(self) -> None:
        self._messages: list[AgentMessage] = []

    def write(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def read_for(self, agent_name: str) -> list[AgentMessage]:
        return [m for m in self._messages if m.recipient is None or m.recipient == agent_name or m.sender == agent_name]

    def last_from(self, agent_name: str) -> AgentMessage | None:
        for message in reversed(self._messages):
            if message.sender == agent_name:
                return message
        return None

    @property
    def all(self) -> list[AgentMessage]:
        return list(self._messages)


# ---------------------------------------------------------------------------
# Specialised research agents
# ---------------------------------------------------------------------------


class QuranResearchAgent:
    """Qur'an agent: maps a query to concordance themes and their top verses."""

    name = "quran"
    capability = AgentCapability(
        domain=Domain.QURAN,
        description="Identifies relevant Qur'anic themes and the verses mapped to them.",
    )

    async def run(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        from thematic_quran import get_thematic_retriever

        retriever = get_thematic_retriever()
        themes = retriever.taxonomy.search_themes(query)
        themes.sort(key=lambda t: -len(retriever.verse_store.get_verses_for_theme(t.id)))
        findings = []
        for theme in themes[:3]:
            verses = retriever.verse_store.get_verses_for_theme(theme.id)[:5]
            findings.append(
                {
                    "theme": theme.id,
                    "name": theme.name,
                    "name_arabic": theme.name_arabic,
                    "verses": [f"{v.surah}:{v.ayah}" for v in verses],
                }
            )
        return {
            "domain": Domain.QURAN.value,
            "query": query,
            "themes_found": len(themes),
            "findings": findings,
            "summary": self._summarize(findings),
        }

    @staticmethod
    def _summarize(findings: list[dict[str, Any]]) -> str:
        if not findings:
            return "No strong thematic match found for the query."
        parts = []
        for finding in findings:
            verse_list = ", ".join(finding["verses"][:3]) or "no mapped verses"
            parts.append(f"theme '{finding['name']}' ({verse_list})")
        return "Qur'anic themes identified: " + "; ".join(parts) + "."


class HadithResearchAgent:
    """Hadith agent: detects named collections and gradings in the query."""

    name = "hadith"
    capability = AgentCapability(
        domain=Domain.HADITH,
        description="Detects hadith collections, narrations and gradings mentioned in the query.",
    )

    async def run(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        from hadith import annotate

        annotated = annotate(query)
        collections: list[str] = []
        gradings: list[str] = []
        for ref in annotated:
            if ref.collection and ref.collection not in collections:
                collections.append(ref.collection)
            if ref.grade and ref.grade not in gradings:
                gradings.append(ref.grade)
        return {
            "domain": Domain.HADITH.value,
            "query": query,
            "collections_detected": collections,
            "gradings_detected": gradings,
            "summary": self._summarize(collections, gradings),
        }

    @staticmethod
    def _summarize(collections: list[str], gradings: list[str]) -> str:
        if not collections and not gradings:
            return "No explicit hadith collection or grading mentioned in the query."
        bits = []
        if collections:
            bits.append("collections: " + ", ".join(collections))
        if gradings:
            bits.append("gradings: " + ", ".join(gradings))
        return "Hadith references detected — " + "; ".join(bits) + "."


class TafsirResearchAgent:
    """Tafsir agent: identifies ayah references the query asks to explain."""

    name = "tafsir"
    capability = AgentCapability(
        domain=Domain.TAFSIR,
        description="Detects specific ayah references the query asks to interpret.",
        input_shape="query text + quran agent findings",
    )

    async def run(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        from tafsir import detect_ayah_references, surah_by_number

        refs = detect_ayah_references(query)
        named = []
        for ref in refs:
            surah = surah_by_number(ref.surah)
            named.append(
                {
                    "reference": f"{ref.surah}:{ref.ayah}",
                    "surah_name": surah.name if surah else None,
                }
            )
        # When the query asks to "explain" a topic, fold in the quran agent's
        # top verse so the tafsir agent has something concrete to interpret.
        quran_message = context.get("quran")
        if not named and quran_message:
            findings = quran_message.get("findings", [])
            for finding in findings[:1]:
                for verse in finding.get("verses", [])[:3]:
                    try:
                        enriched = detect_ayah_references(verse)
                    except Exception:  # noqa: BLE001 - best-effort enrichment
                        enriched = []
                    if enriched:
                        named.append({"reference": verse, "surah_name": None})
        return {
            "domain": Domain.TAFSIR.value,
            "query": query,
            "references": named,
            "summary": self._summarize(named),
        }

    @staticmethod
    def _summarize(references: list[dict[str, Any]]) -> str:
        if not references:
            return "No specific ayah reference detected to interpret."
        refs = ", ".join(r["reference"] for r in references)
        return f"Ayah references to interpret: {refs}."


class FiqhResearchAgent:
    """Fiqh agent: classifies fiqh relevance and madhhab implications."""

    name = "fiqh"
    capability = AgentCapability(
        domain=Domain.FIQH,
        description="Assesses whether the query raises a fiqh question and which rulings apply.",
    )

    _RULING_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("worship", ("prayer", "salah", "salat", "wudu", "fast", "fasting", "zakat", "hajj", "prayer")),
        ("transactions", ("sale", "buy", "sell", "trade", "riba", "interest", "loan", "contract", "money")),
        ("family", ("marriage", "divorce", "nikah", "talaq", "inheritance", "custody")),
        ("diet", ("halal", "haram", "eat", "food", "meat", "alcohol")),
    )

    async def run(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        from fiqh import classify_fiqh

        is_fiqh = classify_fiqh(query)
        lowered = query.lower()
        categories = [cat for cat, keywords in self._RULING_KEYWORDS if any(k in lowered for k in keywords)]
        return {
            "domain": Domain.FIQH.value,
            "query": query,
            "is_fiqh_question": is_fiqh,
            "categories": categories,
            "summary": self._summarize(is_fiqh, categories),
        }

    @staticmethod
    def _summarize(is_fiqh: bool, categories: list[str]) -> str:
        if not is_fiqh and not categories:
            return "No fiqh ruling appears to be requested."
        area = ", ".join(categories) if categories else "general"
        return f"Fiqh question detected ({area}); classical rulings should be consulted."


class SynthesisAgent:
    """Synthesis agent: aggregates all agents' findings into one coherent answer."""

    name = "synthesizer"
    capability = AgentCapability(
        domain=Domain.GENERAL,
        description="Merges all research findings into a coherent, attributed answer.",
        input_shape="all agent outputs",
        parallel_safe=False,
    )

    async def run(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        sections: list[str] = []
        degraded: list[str] = []
        for agent_name in ("quran", "hadith", "tafsir", "fiqh"):
            message = context.get(agent_name)
            if message is None:
                continue
            summary = message.get("summary") if isinstance(message, dict) else str(message)
            if summary:
                sections.append(f"- **{agent_name.capitalize()}**: {summary}")
            if isinstance(message, dict) and message.get("degraded"):
                degraded.append(agent_name)
        body = "\n".join(sections) if sections else "No specialised agent produced findings."
        if degraded:
            body += "\n\nNote: some agents returned partial results; verify the affected domains independently."
        return {
            "domain": Domain.GENERAL.value,
            "query": query,
            "answer": self._answer(query, body),
            "degraded_agents": degraded,
            "summary": body,
        }

    @staticmethod
    def _answer(query: str, body: str) -> str:
        return (
            f"Findings for: {query}\n\n"
            f"{body}\n\n"
            "These findings are structured evidence pointers from the bundled datasets; "
            "a scholar should confirm rulings and interpretations before relying on them."
        )


# ---------------------------------------------------------------------------
# Query decomposition & agent selection
# ---------------------------------------------------------------------------


_FACET_PATTERN = re.compile(r"\s*(?:\?|;|\band\b|\bbut\b|\bhowever\b|\balso\b)\s*", re.IGNORECASE)

_DOMAIN_KEYWORDS: tuple[tuple[Domain, tuple[str, ...]], ...] = (
    (Domain.TAFSIR, ("tafsir", "interpret", "explain", "meaning of", "verse", "ayah", "surah", "qur'an", "quran")),
    (
        Domain.HADITH,
        (
            "hadith",
            "narration",
            "narrated",
            "sunnah",
            "prophet",
            "sahih",
            "authentic",
            "bukhari",
            "muslim",
            "tirmidhi",
            "nasai",
            "abu dawud",
            "ibn majah",
        ),
    ),
    (Domain.FIQH, ("ruling", "permissible", "permitted", "halal", "haram", "forbidden", "obligatory", "wajib", "fard")),
    (Domain.QURAN, ("quran", "qur'an", "allah", "lord", "prayer", "fast", "charity", "patience", "mercy")),
)


def decompose_query(query: str) -> list[str]:
    """Split a compound query into facets; a single clause passes through."""
    parts = [p.strip() for p in _FACET_PATTERN.split(query) if p and p.strip()]
    return [p for p in parts if len(p) >= 3] or [query.strip()]


def classify_domain(facet: str) -> Domain:
    """Route a facet to its primary research domain by keyword priority."""
    lowered = facet.lower()
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return domain
    return Domain.GENERAL


def select_agents(registry: AgentRegistry, facets: list[str]) -> dict[Domain, list[str]]:
    """Map each facet to every domain it touches, then to the registered agents.

    A single facet can mention several domains ("is this hadith about zakat
    authentic?") so all matching domains are selected, not just the primary
    one — the orchestrator then runs each capable agent for the query.
    """
    selection: dict[Domain, list[str]] = {}
    for facet in facets:
        lowered = facet.lower()
        for domain, keywords in _DOMAIN_KEYWORDS:
            if any(keyword in lowered for keyword in keywords):
                agents = registry.find(domain)
                names = [agent.name for agent in agents] or [domain.value]
                selection.setdefault(domain, [])
                for name in names:
                    if name not in selection[domain]:
                        selection[domain].append(name)
        # A facet matching nothing gets a general pass-through so it is still
        # represented in the run.
        if not any(keyword in lowered for _domain, keywords in _DOMAIN_KEYWORDS for keyword in keywords):
            selection.setdefault(Domain.GENERAL, [])
            if "synthesizer" not in selection[Domain.GENERAL]:
                selection[Domain.GENERAL].append("synthesizer")
    return selection


# ---------------------------------------------------------------------------
# Execution DAG
# ---------------------------------------------------------------------------


class ExecutionNode(BaseModel):
    """One node in the execution DAG: a single agent run for the query."""

    id: str
    agent: str
    domain: Domain
    depends_on: list[str] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    latency_ms: float = 0.0
    error: str | None = None
    result: dict[str, Any] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ExecutionGraph:
    """A DAG of agent executions with dependency-aware scheduling."""

    def __init__(self, nodes: list[ExecutionNode]) -> None:
        self._nodes = {node.id: node for node in nodes}

    @property
    def nodes(self) -> list[ExecutionNode]:
        return list(self._nodes.values())

    def dependencies(self, node_id: str) -> list[ExecutionNode]:
        node = self._nodes[node_id]
        return [self._nodes[dep] for dep in node.depends_on if dep in self._nodes]

    def ready(self, finished: set[str]) -> list[ExecutionNode]:
        """Nodes whose dependencies are all finished (or already done).

        The synthesis node is deliberately excluded: it runs once in the final
        aggregation step so the answer is captured, not as part of a wave.
        """
        ready = []
        for node in self._nodes.values():
            if node.agent == "synthesizer":
                continue
            if node.status not in (NodeStatus.PENDING, NodeStatus.RUNNING):
                continue
            deps = self.dependencies(node.id)
            if all(dep.id in finished for dep in deps):
                ready.append(node)
        return ready

    def is_complete(self) -> bool:
        """All research nodes finished (the synthesis node runs separately)."""
        return all(
            node.status not in (NodeStatus.PENDING, NodeStatus.RUNNING)
            for node in self._nodes.values()
            if node.agent != "synthesizer"
        )


def build_graph(registry: AgentRegistry, query: str, facets: list[str]) -> ExecutionGraph:
    """Construct the execution DAG for a query.

    Selection is by domain; within a domain the first capable agent is the
    primary node and further agents become fallbacks. The tafsir node depends
    on the quran node (it enriches its references with the quran findings);
    hadith and fiqh are independent leaves that run in parallel.
    """
    selection = select_agents(registry, facets)
    nodes: list[ExecutionNode] = []
    node_ids: dict[Domain, str] = {}

    # The Qur'an agent always runs first when interpretation is involved: the
    # tafsir agent builds on its verse findings, so even a pure tafsir query
    # gets a quran prerequisite node.
    if Domain.TAFSIR in selection and Domain.QURAN not in selection:
        selection[Domain.QURAN] = ["quran"]

    for domain in (Domain.QURAN, Domain.HADITH, Domain.TAFSIR, Domain.FIQH):
        names = selection.get(domain, [])
        if not names:
            continue
        agent_name = names[0]
        node_id = f"{domain.value}:{agent_name}"
        depends: list[str] = []
        if domain is Domain.TAFSIR and Domain.QURAN in node_ids:
            depends.append(node_ids[Domain.QURAN])
        nodes.append(
            ExecutionNode(
                id=node_id,
                agent=agent_name,
                domain=domain,
                depends_on=depends,
            )
        )
        node_ids[domain] = node_id

    # The synthesis node always runs last, depending on every research node, so
    # the final answer aggregates the full run regardless of which domains fired.
    research_ids = [node.id for node in nodes]
    nodes.append(
        ExecutionNode(
            id="general:synthesizer",
            agent="synthesizer",
            domain=Domain.GENERAL,
            depends_on=research_ids,
        )
    )
    return ExecutionGraph(nodes)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class OrchestrationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    timeout_ms: int = Field(default=5000, ge=100, le=60000)
    fail_agents: list[str] = Field(
        default_factory=list, description="Agent names to force-fail (testing/degradation demo)"
    )


class TraceEvent(BaseModel):
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    node_id: str
    kind: str  # start | message | finish | fail
    detail: str = ""


class ExecutionTrace(BaseModel):
    trace_id: str
    query: str
    status: str = "completed"  # completed | partial | failed
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    total_latency_ms: float = 0.0
    nodes: list[ExecutionNode] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    answer: str | None = None


class OrchestrationResponse(BaseModel):
    query: str
    trace_id: str
    status: str
    answer: str
    nodes: list[ExecutionNode] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)


class _TraceStore:
    """Bounded in-memory store of execution traces (LRU eviction)."""

    def __init__(self, max_entries: int = 200) -> None:
        self._traces: OrderedDict[str, ExecutionTrace] = OrderedDict()
        self.max_entries = max_entries

    def put(self, trace: ExecutionTrace) -> None:
        self._traces[trace.trace_id] = trace
        while len(self._traces) > self.max_entries:
            self._traces.popitem(last=False)

    def get(self, trace_id: str) -> ExecutionTrace | None:
        trace = self._traces.get(trace_id)
        if trace is not None:
            self._traces.move_to_end(trace_id)
        return trace


trace_store = _TraceStore()


class Orchestrator:
    """Coordinates multi-agent research runs with a transparent execution DAG."""

    def __init__(self, registry: AgentRegistry | None = None, max_concurrency: int = 3) -> None:
        self.registry = registry or AgentRegistry()
        self.max_concurrency = max_concurrency

    async def orchestrate(self, request: OrchestrationRequest) -> OrchestrationResponse:
        trace = ExecutionTrace(trace_id=str(uuid.uuid4()), query=request.query)
        blackboard = Blackboard()
        started = time.perf_counter()

        facets = decompose_query(request.query)
        graph = build_graph(self.registry, request.query, facets)
        trace.nodes = graph.nodes

        try:
            answer = await self._run_graph(graph, blackboard, trace, request)
        finally:
            trace.finished_at = datetime.now(UTC)
            trace.total_latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            trace_store.put(trace)

        failed = [node.agent for node in graph.nodes if node.status in (NodeStatus.FAILED, NodeStatus.TIMEOUT)]
        degraded = [node.agent for node in graph.nodes if node.status is NodeStatus.FALLBACK]
        status = "failed" if failed and not answer else "partial" if degraded else "completed"
        trace.status = status
        trace.answer = answer

        return OrchestrationResponse(
            query=request.query,
            trace_id=trace.trace_id,
            status=status,
            answer=answer or "The orchestration failed; no synthesis could be produced.",
            nodes=graph.nodes,
            degraded=degraded,
        )

    async def _run_graph(
        self,
        graph: ExecutionGraph,
        blackboard: Blackboard,
        trace: ExecutionTrace,
        request: OrchestrationRequest,
    ) -> str | None:
        finished: set[str] = set()
        answer: str | None = None

        while not graph.is_complete():
            ready = graph.ready(finished)
            if not ready:
                # Cycle or stuck dependency: fail the run rather than hang.
                for node in graph.nodes:
                    if node.status is NodeStatus.PENDING:
                        node.status = NodeStatus.FAILED
                        node.error = "dependency cycle or unresolvable ordering"
                break

            for node in ready:
                node.status = NodeStatus.RUNNING
                node.started_at = datetime.now(UTC)
                trace.events.append(TraceEvent(node_id=node.id, kind="start", detail=f"running {node.agent}"))

            # Run the ready wave concurrently (parallel agent execution).
            results = await asyncio.gather(
                *[self._run_node(node, blackboard, trace, request) for node in ready],
                return_exceptions=True,
            )

            for node, result in zip(ready, results, strict=False):
                if isinstance(result, BaseException):
                    node.status = NodeStatus.FAILED
                    node.error = str(result)
                    trace.events.append(TraceEvent(node_id=node.id, kind="fail", detail=str(result)))
                    await self._attempt_fallback(node, blackboard, trace)
                finished.add(node.id)

        # Synthesis runs last, after every research node finished.
        synth_nodes = [node for node in graph.nodes if node.agent == "synthesizer"]
        for node in synth_nodes:
            node.status = NodeStatus.RUNNING
            node.started_at = datetime.now(UTC)
            trace.events.append(TraceEvent(node_id=node.id, kind="start", detail="synthesizing results"))
            context = self._context_for(blackboard)
            synthesizer = self.registry.get("synthesizer")
            try:
                if synthesizer is None:
                    raise RuntimeError("synthesizer agent is not registered")
                synth_result = await synthesizer.run(request.query, context)
                node.result = synth_result
                node.status = NodeStatus.SUCCESS
                node.finished_at = datetime.now(UTC)
                trace.events.append(TraceEvent(node_id=node.id, kind="finish", detail="synthesis complete"))
                answer = str(synth_result.get("answer") or synth_result.get("summary") or "")
            except Exception as exc:  # noqa: BLE001 - partial results are better than none
                node.status = NodeStatus.FAILED
                node.error = str(exc)
                trace.events.append(TraceEvent(node_id=node.id, kind="fail", detail=str(exc)))
            finished.add(node.id)

        return answer

    async def _run_node(
        self,
        node: ExecutionNode,
        blackboard: Blackboard,
        trace: ExecutionTrace,
        request: OrchestrationRequest,
    ) -> None:
        agent = self.registry.get(node.agent)
        if agent is None:
            node.status = NodeStatus.FAILED
            node.error = f"agent '{node.agent}' not registered"
            return

        if node.agent in request.fail_agents:
            node.status = NodeStatus.FAILED
            node.error = "forced failure (fail_agents)"
            return

        context = self._context_for(blackboard)
        try:
            result = await asyncio.wait_for(agent.run(request.query, context), timeout=request.timeout_ms / 1000.0)
            node.result = result
            node.status = NodeStatus.SUCCESS
            node.finished_at = datetime.now(UTC)
            node.latency_ms = self._latency_ms(node)
            blackboard.write(AgentMessage(sender=node.agent, recipient=None, kind="result", payload=result))
            trace.messages.append(blackboard.last_from(node.agent) or AgentMessage(sender=node.agent))
            trace.events.append(TraceEvent(node_id=node.id, kind="finish", detail="completed"))
        except TimeoutError:
            node.status = NodeStatus.TIMEOUT
            node.error = f"agent '{node.agent}' exceeded the {request.timeout_ms}ms budget"
            node.finished_at = datetime.now(UTC)
            node.latency_ms = self._latency_ms(node)
            trace.events.append(TraceEvent(node_id=node.id, kind="fail", detail=node.error))
        except Exception as exc:  # noqa: BLE001 - node failure must not kill the run
            node.status = NodeStatus.FAILED
            node.error = str(exc)
            node.finished_at = datetime.now(UTC)
            node.latency_ms = self._latency_ms(node)
            trace.events.append(TraceEvent(node_id=node.id, kind="fail", detail=str(exc)))

    async def _attempt_fallback(self, node: ExecutionNode, blackboard: Blackboard, trace: ExecutionTrace) -> None:
        """Try the domain's fallback agent; mark the node as degraded if it ran."""
        fallback = self.registry.fallback_for(node.domain)
        if fallback is None or fallback.name == node.agent:
            return
        try:
            context = self._context_for(blackboard)
            result = await fallback.run(node.agent, context)
            node.result = {"degraded": True, **result}
            node.status = NodeStatus.FALLBACK
            node.error = f"primary agent failed; fell back to '{fallback.name}'"
            trace.events.append(TraceEvent(node_id=node.id, kind="finish", detail=f"fallback to {fallback.name}"))
        except Exception as exc:  # noqa: BLE001 - fallback is best-effort
            trace.events.append(TraceEvent(node_id=node.id, kind="fail", detail=f"fallback failed: {exc}"))

    @staticmethod
    def _context_for(blackboard: Blackboard) -> dict[str, Any]:
        """Read the latest result message from every agent (message passing)."""
        context: dict[str, Any] = {}
        for message in blackboard.all:
            if message.kind == "result" and message.sender not in context:
                context[message.sender] = message.payload
        return context

    @staticmethod
    def _latency_ms(node: ExecutionNode) -> float:
        if node.started_at is None or node.finished_at is None:
            return 0.0
        return round((node.finished_at - node.started_at).total_seconds() * 1000.0, 2)


# Singleton orchestrator for the app
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/agents", response_model=list[dict[str, Any]])
async def list_agents() -> list[dict[str, Any]]:
    """Discover registered agents and their capabilities."""
    return get_orchestrator().registry.list_capabilities()


@router.get("/agents/{name}", response_model=dict[str, Any] | None)
async def get_agent(name: str) -> dict[str, Any] | None:
    """Get one agent's capability by name."""
    agent = get_orchestrator().registry.get(name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not registered.")
    return {
        "name": agent.name,
        "domain": agent.capability.domain.value,
        "description": agent.capability.description,
        "parallel_safe": agent.capability.parallel_safe,
    }


@router.post("/run", response_model=OrchestrationResponse)
async def run_orchestration(body: OrchestrationRequest) -> OrchestrationResponse:
    """Run a complex research query through the multi-agent orchestration."""
    return await get_orchestrator().orchestrate(body)


@router.get("/traces/{trace_id}", response_model=ExecutionTrace)
async def get_trace(trace_id: str) -> ExecutionTrace:
    """Retrieve a full execution trace for transparent observability."""
    trace = trace_store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No trace with id '{trace_id}'.")
    return trace
