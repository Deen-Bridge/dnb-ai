"""Asynchronous multi-agent orchestration with routing and resilient execution.

The framework is deliberately provider-independent. Agents are regular async
callables registered with capabilities and optional routing keywords. A query is
analysed into an execution plan, the plan is validated as a DAG, ready tasks are
scheduled concurrently, and successful outputs are synthesised into one result.

All state is process-local so importing this module performs no network or
storage operations. The public classes can be embedded in the FastAPI service or
used independently by tests and background workers.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

AgentHandler = Callable[["AgentTask", "AgentContext"], Awaitable[Any]]
LifecycleHook = Callable[[], Awaitable[None]]
SynthesisHandler = Callable[[str, Mapping[str, Any]], Awaitable[str]]


class PlanValidationError(ValueError):
    """Raised when an execution plan is not a valid directed acyclic graph."""


class AgentUnavailableError(RuntimeError):
    """Raised when an agent is unavailable and no fallback can serve a task."""


class CircuitOpenError(AgentUnavailableError):
    """Raised when calls to an unhealthy agent are temporarily blocked."""


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class AgentTask:
    """One node in an execution plan."""

    id: str
    agent: str
    capability: str
    payload: Any
    dependencies: tuple[str, ...] = ()
    fallback_agent: str | None = None
    timeout_seconds: float | None = None
    required: bool = True


@dataclass(frozen=True)
class AgentMessage:
    workflow_id: str
    sender: str
    topic: str
    payload: Any
    created_at: float


class MessageBroker:
    """Workflow-scoped, in-memory message broker for collaborating agents."""

    def __init__(self) -> None:
        self._messages: dict[tuple[str, str], list[AgentMessage]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(self, workflow_id: str, sender: str, topic: str, payload: Any) -> None:
        message = AgentMessage(
            workflow_id=workflow_id,
            sender=sender,
            topic=topic,
            payload=payload,
            created_at=time.time(),
        )
        async with self._lock:
            self._messages[(workflow_id, topic)].append(message)

    async def read(self, workflow_id: str, topic: str) -> tuple[AgentMessage, ...]:
        async with self._lock:
            return tuple(self._messages.get((workflow_id, topic), ()))

    async def clear_workflow(self, workflow_id: str) -> None:
        async with self._lock:
            keys = [key for key in self._messages if key[0] == workflow_id]
            for key in keys:
                del self._messages[key]


@dataclass(frozen=True)
class AgentContext:
    """Execution context supplied to every agent invocation."""

    workflow_id: str
    task_id: str
    dependency_results: Mapping[str, Any]
    broker: MessageBroker
    metadata: Mapping[str, Any]

    async def publish(self, topic: str, payload: Any) -> None:
        await self.broker.publish(self.workflow_id, self.task_id, topic, payload)

    async def messages(self, topic: str) -> tuple[AgentMessage, ...]:
        return await self.broker.read(self.workflow_id, topic)


@dataclass(frozen=True)
class AgentSpec:
    """Registered agent and the operational limits applied to it."""

    name: str
    capabilities: frozenset[str]
    handler: AgentHandler
    keywords: frozenset[str] = frozenset()
    priority: int = 0
    max_concurrency: int = 4
    rate_limit_per_second: float = 0.0
    fallback_agent: str | None = None
    default: bool = False
    start: LifecycleHook | None = None
    stop: LifecycleHook | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name must not be empty")
        if not self.capabilities:
            raise ValueError("Agent must declare at least one capability")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.rate_limit_per_second < 0:
            raise ValueError("rate_limit_per_second must not be negative")


class AgentRegistry:
    """Registry for agent discovery and capability management."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentSpec] = {}

    def register(self, agent: AgentSpec) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent already registered: {agent.name}")
        self._agents[agent.name] = agent

    def unregister(self, name: str) -> AgentSpec:
        try:
            return self._agents.pop(name)
        except KeyError as exc:
            raise KeyError(f"Unknown agent: {name}") from exc

    def get(self, name: str) -> AgentSpec:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise KeyError(f"Unknown agent: {name}") from exc

    def all(self) -> tuple[AgentSpec, ...]:
        return tuple(self._agents.values())

    def for_capability(self, capability: str) -> tuple[AgentSpec, ...]:
        normalized = _normalize_term(capability)
        matches = [agent for agent in self._agents.values() if normalized in agent.capabilities]
        return tuple(sorted(matches, key=lambda agent: (-agent.priority, agent.name)))


@dataclass(frozen=True)
class QueryAnalysis:
    query: str
    selected_agents: tuple[str, ...]
    capabilities: tuple[str, ...]
    sequential: bool
    scores: Mapping[str, float]
    complexity: float = 0.0


_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_SEQUENCE_CUES = frozenset({"after", "before", "first", "next", "then", "finally", "using"})


def _normalize_term(value: str) -> str:
    return value.strip().casefold().replace(" ", "_").replace("-", "_")


def _tokens(text: str) -> set[str]:
    return {_normalize_term(token) for token in _TOKEN_PATTERN.findall(text)}


class QueryAnalyzer:
    """Deterministic capability and agent selection engine."""

    def __init__(self, registry: AgentRegistry, max_agents: int = 5) -> None:
        if max_agents < 1:
            raise ValueError("max_agents must be at least 1")
        self._registry = registry
        self._max_agents = max_agents

    def analyze(self, query: str) -> QueryAnalysis:
        query_tokens = _tokens(query)
        scored: list[tuple[float, int, str, AgentSpec, set[str]]] = []

        for agent in self._registry.all():
            capabilities = {_normalize_term(item) for item in agent.capabilities}
            keywords = {_normalize_term(item) for item in agent.keywords}
            matched_capabilities = capabilities.intersection(query_tokens)
            matched_keywords = keywords.intersection(query_tokens)
            score = 3.0 * len(matched_capabilities) + float(len(matched_keywords))
            if score > 0:
                scored.append((score, agent.priority, agent.name, agent, matched_capabilities))

        if not scored:
            defaults = [agent for agent in self._registry.all() if agent.default]
            candidates = defaults or list(self._registry.all())
            if candidates:
                best = sorted(candidates, key=lambda agent: (-agent.priority, agent.name))[0]
                scored.append((0.1, best.priority, best.name, best, set()))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected = scored[: self._max_agents]
        selected_agents = tuple(item[3].name for item in selected)
        detected_capabilities: list[str] = []
        for _, _, _, agent, matches in selected:
            capability = sorted(matches)[0] if matches else sorted(agent.capabilities)[0]
            if capability not in detected_capabilities:
                detected_capabilities.append(capability)

        sequential = len(selected_agents) > 1 and bool(query_tokens.intersection(_SEQUENCE_CUES))
        scores = {item[3].name: item[0] for item in selected}
        complexity = min(1.0, len(query_tokens) / 40.0 + 0.15 * len(selected_agents))
        return QueryAnalysis(
            query=query,
            selected_agents=selected_agents,
            capabilities=tuple(detected_capabilities),
            sequential=sequential,
            scores=scores,
            complexity=complexity,
        )


@dataclass(frozen=True)
class ExecutionPlan:
    tasks: tuple[AgentTask, ...]

    def validate(self, registry: AgentRegistry | None = None) -> None:
        task_by_id = {task.id: task for task in self.tasks}
        if len(task_by_id) != len(self.tasks):
            raise PlanValidationError("Task identifiers must be unique")

        for task in self.tasks:
            if task.id in task.dependencies:
                raise PlanValidationError(f"Task {task.id} cannot depend on itself")
            missing = set(task.dependencies).difference(task_by_id)
            if missing:
                names = ", ".join(sorted(missing))
                raise PlanValidationError(f"Task {task.id} has missing dependencies: {names}")
            if registry is not None:
                try:
                    agent = registry.get(task.agent)
                except KeyError as exc:
                    raise PlanValidationError(str(exc)) from exc
                if _normalize_term(task.capability) not in agent.capabilities:
                    raise PlanValidationError(f"Agent {task.agent} does not provide capability {task.capability}")
                if task.fallback_agent is not None:
                    try:
                        registry.get(task.fallback_agent)
                    except KeyError as exc:
                        raise PlanValidationError(str(exc)) from exc

        indegree = {task.id: len(task.dependencies) for task in self.tasks}
        dependents: dict[str, list[str]] = defaultdict(list)
        for task in self.tasks:
            for dependency in task.dependencies:
                dependents[dependency].append(task.id)

        queue = deque(task_id for task_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            task_id = queue.popleft()
            visited += 1
            for dependent in dependents[task_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if visited != len(self.tasks):
            raise PlanValidationError("Execution plan contains a dependency cycle")


class ExecutionPlanBuilder:
    """Builds a parallel or sequential DAG from query analysis."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry
        self._granularity_threshold = 0.35

    def build(self, analysis: QueryAnalysis) -> ExecutionPlan:
        tasks: list[AgentTask] = []
        previous_id: str | None = None
        for index, agent_name in enumerate(analysis.selected_agents):
            agent = self._registry.get(agent_name)
            capability = self._choose_capability(agent, analysis.capabilities)
            task_id = f"agent-{index + 1}-{agent_name}"
            sequential = analysis.sequential or analysis.complexity >= self._granularity_threshold
            dependencies = (previous_id,) if sequential and previous_id is not None else ()
            tasks.append(
                AgentTask(
                    id=task_id,
                    agent=agent_name,
                    capability=capability,
                    payload=analysis.query,
                    dependencies=dependencies,
                )
            )
            previous_id = task_id
        plan = ExecutionPlan(tuple(tasks))
        plan.validate(self._registry)
        return plan

    @staticmethod
    def _choose_capability(agent: AgentSpec, preferred: Sequence[str]) -> str:
        normalized = {_normalize_term(item) for item in agent.capabilities}
        for candidate in preferred:
            if candidate in normalized:
                return candidate
        return sorted(normalized)[0]
