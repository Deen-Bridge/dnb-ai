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
import inspect
import re
import time
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any


AgentHandler = Callable[["AgentTask", "AgentContext"], Awaitable[Any]]
LifecycleHook = Callable[[], Awaitable[None]]
SynthesisHandler = Callable[[str, Mapping[str, "TaskResult"]], Awaitable[str]]


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
        return QueryAnalysis(
            query=query,
            selected_agents=selected_agents,
            capabilities=tuple(detected_capabilities),
            sequential=sequential,
            scores=scores,
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
                    raise PlanValidationError(
                        f"Agent {task.agent} does not provide capability {task.capability}"
                    )
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

    def build(self, analysis: QueryAnalysis) -> ExecutionPlan:
        tasks: list[AgentTask] = []
        previous_id: str | None = None
        for index, agent_name in enumerate(analysis.selected_agents):
            agent = self._registry.get(agent_name)
            capability = self._choose_capability(agent, analysis.capabilities)
            task_id = f"agent-{index + 1}-{agent_name}"
            dependencies = (previous_id,) if analysis.sequential and previous_id is not None else ()
            tasks.append(
                AgentTask(
                    id=task_id,
                    agent=agent_name,
                    capability=capability,
                    payload=analysis.query,
                    dependencies=dependencies,
                    fallback_agent=agent.fallback_agent,
                )
            )
            previous_id = task_id
        plan = ExecutionPlan(tuple(tasks))
        plan.validate(self._registry)
        return plan

    @staticmethod
    def _choose_capability(agent: AgentSpec, detected: Sequence[str]) -> str:
        for capability in detected:
            if capability in agent.capabilities:
                return capability
        return sorted(agent.capabilities)[0]


class AsyncRateLimiter:
    """Simple evenly-spaced asynchronous rate limiter."""

    def __init__(self, calls_per_second: float) -> None:
        self._interval = 1.0 / calls_per_second if calls_per_second > 0 else 0.0
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._interval == 0:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self._interval


class CircuitBreaker:
    """Failure-count circuit breaker with one half-open recovery probe."""

    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 30.0) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_seconds < 0:
            raise ValueError("recovery_seconds must not be negative")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.opened_at = 0.0
        self._probe_in_progress = False
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            if self.state is CircuitState.CLOSED:
                return True
            if self.state is CircuitState.OPEN:
                if time.monotonic() - self.opened_at < self.recovery_seconds:
                    return False
                self.state = CircuitState.HALF_OPEN
            if self._probe_in_progress:
                return False
            self._probe_in_progress = True
            return True

    async def record_success(self) -> None:
        async with self._lock:
            self.failures = 0
            self.state = CircuitState.CLOSED
            self._probe_in_progress = False

    async def record_failure(self) -> None:
        async with self._lock:
            self.failures += 1
            self._probe_in_progress = False
            if self.failures >= self.failure_threshold or self.state is CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()


@dataclass(frozen=True)
class TraceSpan:
    workflow_id: str
    task_id: str
    agent: str
    started_at: float
    ended_at: float
    status: TaskStatus
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return (self.ended_at - self.started_at) * 1000.0


class Metrics:
    """Thread-safe counters and cumulative latency measurements."""

    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()
        self._durations_ms: Counter[str] = Counter()
        self._lock = Lock()

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[key] += amount

    def observe(self, key: str, duration_ms: float) -> None:
        with self._lock:
            self._durations_ms[key] += duration_ms

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            result: dict[str, float | int] = dict(self._counters)
            result.update({f"{key}_total_ms": value for key, value in self._durations_ms.items()})
            return result


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    agent: str
    status: TaskStatus
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    fallback_used: bool = False


@dataclass(frozen=True)
class WorkflowResult:
    workflow_id: str
    query: str
    status: TaskStatus
    answer: str
    task_results: Mapping[str, TaskResult]
    traces: tuple[TraceSpan, ...]
    duration_ms: float
    analysis: QueryAnalysis

    @property
    def partial_failure(self) -> bool:
        statuses = {result.status for result in self.task_results.values()}
        return TaskStatus.SUCCEEDED in statuses and bool(
            statuses.intersection({TaskStatus.FAILED, TaskStatus.SKIPPED})
        )


class ResultSynthesizer:
    """Combines heterogeneous outputs while preserving their attribution."""

    def __init__(self, handler: SynthesisHandler | None = None) -> None:
        self._handler = handler

    async def synthesize(self, query: str, results: Mapping[str, TaskResult]) -> str:
        successful = {
            task_id: result for task_id, result in results.items() if result.status is TaskStatus.SUCCEEDED
        }
        if self._handler is not None:
            return await self._handler(query, successful)
        if not successful:
            return "Unable to complete the request because no agent produced a result."
        if len(successful) == 1:
            return str(next(iter(successful.values())).output)
        sections = [f"[{result.agent}]\n{result.output}" for result in successful.values()]
        return "\n\n".join(sections)


class MultiAgentOrchestrator:
    """Coordinates agent lifecycle, planning, execution, and synthesis."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        analyzer: QueryAnalyzer | None = None,
        plan_builder: ExecutionPlanBuilder | None = None,
        broker: MessageBroker | None = None,
        synthesizer: ResultSynthesizer | None = None,
        metrics: Metrics | None = None,
        circuit_failure_threshold: int = 3,
        circuit_recovery_seconds: float = 30.0,
        default_task_timeout: float = 8.0,
        workflow_timeout: float = 10.0,
    ) -> None:
        if default_task_timeout <= 0 or workflow_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        self.registry = registry
        self.analyzer = analyzer or QueryAnalyzer(registry)
        self.plan_builder = plan_builder or ExecutionPlanBuilder(registry)
        self.broker = broker or MessageBroker()
        self.synthesizer = synthesizer or ResultSynthesizer()
        self.metrics = metrics or Metrics()
        self.default_task_timeout = default_task_timeout
        self.workflow_timeout = workflow_timeout
        self._semaphores = {
            agent.name: asyncio.Semaphore(agent.max_concurrency) for agent in registry.all()
        }
        self._limiters = {
            agent.name: AsyncRateLimiter(agent.rate_limit_per_second) for agent in registry.all()
        }
        self._breakers = {
            agent.name: CircuitBreaker(circuit_failure_threshold, circuit_recovery_seconds)
            for agent in registry.all()
        }
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        started: list[AgentSpec] = []
        try:
            for agent in self.registry.all():
                if agent.start is not None:
                    await agent.start()
                started.append(agent)
        except Exception:
            for agent in reversed(started):
                if agent.stop is not None:
                    await agent.stop()
            raise
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        errors: list[Exception] = []
        for agent in reversed(self.registry.all()):
            if agent.stop is not None:
                try:
                    await agent.stop()
                except Exception as exc:
                    errors.append(exc)
        self._started = False
        if errors:
            raise RuntimeError(f"Failed to stop {len(errors)} agent(s)") from errors[0]

    async def __aenter__(self) -> MultiAgentOrchestrator:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.stop()

    def circuit_state(self, agent_name: str) -> CircuitState:
        return self._breakers[agent_name].state

    async def orchestrate(
        self,
        query: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        plan: ExecutionPlan | None = None,
    ) -> WorkflowResult:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if not self._started:
            await self.start()

        analysis = self.analyzer.analyze(query)
        execution_plan = plan or self.plan_builder.build(analysis)
        execution_plan.validate(self.registry)
        workflow_id = uuid.uuid4().hex
        started_at = time.monotonic()
        traces: list[TraceSpan] = []
        results: dict[str, TaskResult] = {}

        try:
            async with asyncio.timeout(self.workflow_timeout):
                await self._execute_plan(
                    workflow_id,
                    execution_plan,
                    metadata or {},
                    results,
                    traces,
                )
        except TimeoutError:
            self.metrics.increment("workflows_timed_out")
            completed = set(results)
            for task in execution_plan.tasks:
                if task.id not in completed:
                    results[task.id] = TaskResult(
                        task_id=task.id,
                        agent=task.agent,
                        status=TaskStatus.FAILED,
                        error="Workflow timed out",
                    )

        answer = await self.synthesizer.synthesize(query, results)
        statuses = {result.status for result in results.values()}
        if TaskStatus.SUCCEEDED in statuses:
            workflow_status = TaskStatus.SUCCEEDED
        else:
            workflow_status = TaskStatus.FAILED
        duration_ms = (time.monotonic() - started_at) * 1000.0
        self.metrics.increment(f"workflows_{workflow_status.value}")
        self.metrics.observe("workflow_duration", duration_ms)
        await self.broker.clear_workflow(workflow_id)
        return WorkflowResult(
            workflow_id=workflow_id,
            query=query,
            status=workflow_status,
            answer=answer,
            task_results=dict(results),
            traces=tuple(traces),
            duration_ms=duration_ms,
            analysis=analysis,
        )

    async def _execute_plan(
        self,
        workflow_id: str,
        plan: ExecutionPlan,
        metadata: Mapping[str, Any],
        results: dict[str, TaskResult],
        traces: list[TraceSpan],
    ) -> None:
        pending = {task.id: task for task in plan.tasks}
        while pending:
            skipped: list[str] = []
            for task_id, task in pending.items():
                failed_dependencies = [
                    dependency
                    for dependency in task.dependencies
                    if dependency in results and results[dependency].status is not TaskStatus.SUCCEEDED
                ]
                if failed_dependencies:
                    results[task_id] = TaskResult(
                        task_id=task_id,
                        agent=task.agent,
                        status=TaskStatus.SKIPPED,
                        error="Dependency failed: " + ", ".join(failed_dependencies),
                    )
                    skipped.append(task_id)
            for task_id in skipped:
                del pending[task_id]
            if not pending:
                break

            ready = [
                task
                for task in pending.values()
                if all(
                    dependency in results and results[dependency].status is TaskStatus.SUCCEEDED
                    for dependency in task.dependencies
                )
            ]
            if not ready:
                raise PlanValidationError("Execution plan cannot make progress")

            executions = [
                self._run_task(workflow_id, task, metadata, results, traces) for task in ready
            ]
            completed = await asyncio.gather(*executions)
            for result in completed:
                results[result.task_id] = result
                del pending[result.task_id]

    async def _run_task(
        self,
        workflow_id: str,
        task: AgentTask,
        metadata: Mapping[str, Any],
        existing_results: Mapping[str, TaskResult],
        traces: list[TraceSpan],
    ) -> TaskResult:
        dependency_outputs = {
            dependency: existing_results[dependency].output for dependency in task.dependencies
        }
        context = AgentContext(
            workflow_id=workflow_id,
            task_id=task.id,
            dependency_results=dependency_outputs,
            broker=self.broker,
            metadata=metadata,
        )
        fallback_name = task.fallback_agent or self.registry.get(task.agent).fallback_agent
        started_at = time.monotonic()
        try:
            output = await self._invoke(task.agent, task, context)
            used_agent = task.agent
            fallback_used = False
        except Exception as primary_error:
            if fallback_name is None or fallback_name == task.agent:
                return self._failed_result(task, task.agent, primary_error, started_at, traces, workflow_id)
            try:
                output = await self._invoke(fallback_name, task, context)
                used_agent = fallback_name
                fallback_used = True
                self.metrics.increment("fallbacks_succeeded")
            except Exception as fallback_error:
                combined = AgentUnavailableError(
                    f"Primary agent failed ({primary_error}); fallback failed ({fallback_error})"
                )
                return self._failed_result(
                    task, fallback_name, combined, started_at, traces, workflow_id
                )

        ended_at = time.monotonic()
        duration_ms = (ended_at - started_at) * 1000.0
        self.metrics.increment("tasks_succeeded")
        self.metrics.observe(f"agent_{used_agent}_duration", duration_ms)
        traces.append(
            TraceSpan(
                workflow_id=workflow_id,
                task_id=task.id,
                agent=used_agent,
                started_at=started_at,
                ended_at=ended_at,
                status=TaskStatus.SUCCEEDED,
            )
        )
        return TaskResult(
            task_id=task.id,
            agent=used_agent,
            status=TaskStatus.SUCCEEDED,
            output=output,
            duration_ms=duration_ms,
            fallback_used=fallback_used,
        )

    async def _invoke(self, agent_name: str, task: AgentTask, context: AgentContext) -> Any:
        agent = self.registry.get(agent_name)
        breaker = self._breakers[agent_name]
        if not await breaker.allow():
            self.metrics.increment("circuit_rejections")
            raise CircuitOpenError(f"Circuit is open for agent {agent_name}")

        limiter = self._limiters[agent_name]
        semaphore = self._semaphores[agent_name]
        await limiter.acquire()
        timeout = task.timeout_seconds or self.default_task_timeout
        try:
            async with semaphore:
                async with asyncio.timeout(timeout):
                    output = agent.handler(task, context)
                    if not inspect.isawaitable(output):
                        raise TypeError(f"Agent {agent_name} handler must return an awaitable")
                    value = await output
        except Exception:
            await breaker.record_failure()
            self.metrics.increment(f"agent_{agent_name}_failures")
            raise
        await breaker.record_success()
        return value

    def _failed_result(
        self,
        task: AgentTask,
        agent_name: str,
        error: Exception,
        started_at: float,
        traces: list[TraceSpan],
        workflow_id: str,
    ) -> TaskResult:
        ended_at = time.monotonic()
        duration_ms = (ended_at - started_at) * 1000.0
        error_text = f"{type(error).__name__}: {error}"
        self.metrics.increment("tasks_failed")
        traces.append(
            TraceSpan(
                workflow_id=workflow_id,
                task_id=task.id,
                agent=agent_name,
                started_at=started_at,
                ended_at=ended_at,
                status=TaskStatus.FAILED,
                error=error_text,
            )
        )
        return TaskResult(
            task_id=task.id,
            agent=agent_name,
            status=TaskStatus.FAILED,
            error=error_text,
            duration_ms=duration_ms,
        )
