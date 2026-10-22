import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections import defaultdict

__all__ = [
    "AgentCapability",
    "ExecutionMode",
    "AgentStatus",
    "AgentTask",
    "TaskResult",
    "CircuitBreaker",
    "SpecializedAgent",
    "MessageBroker",
    "AgentRegistry",
    "QueryAnalysisEngine",
    "DAGScheduler",
    "TaskDecompositionEngine",
    "DecompositionPlan",
    "SubTask",
    "ResultSynthesisPipeline",
    "MultiAgentOrchestrator",
]

logger = logging.getLogger(__name__)


class AgentCapability(str, Enum):
    FIQH = "fiqh"
    HADITH = "hadith"
    TAFSIR = "tafsir"
    QURAN = "quran"
    ZAKAT = "zakat"
    STELLAR = "stellar"
    GENERAL = "general"


class ExecutionMode(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


class DecompositionStrategy(str, Enum):
    """Strategy used to decompose a complex query."""

    SINGLE = "single"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HYBRID = "hybrid"


@dataclass
class AgentTask:
    task_id: str
    agent_capability: AgentCapability
    payload: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: float = 5.0


@dataclass
class SubTask:
    """A logical sub-task produced by the decomposition engine."""

    sub_task_id: str
    description: str
    capability: AgentCapability
    dependencies: list[str] = field(default_factory=list)
    priority: int = 0
    estimated_cost: float = 1.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionPlan:
    """Result of decomposing a query into sub-tasks."""

    query: str
    sub_tasks: list[SubTask] = field(default_factory=list)
    strategy: DecompositionStrategy = DecompositionStrategy.SINGLE
    parallelizable: bool = False
    complexity_score: float = 0.0
    coverage_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    agent_name: str
    success: bool
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    trace_id: str = ""


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_time_seconds: float = 10.0):
        self.failure_threshold = failure_threshold
        self.recovery_time_seconds = recovery_time_seconds
        self.failures = 0
        self.state = AgentStatus.IDLE
        self.last_failure_time = 0.0

    def record_success(self):
        self.failures = 0
        self.state = AgentStatus.IDLE

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = AgentStatus.CIRCUIT_OPEN
            logger.warning(f"Circuit breaker opened due to {self.failures} failures.")

    def allow_request(self) -> bool:
        if self.state == AgentStatus.CIRCUIT_OPEN:
            if time.time() - self.last_failure_time > self.recovery_time_seconds:
                self.state = AgentStatus.IDLE
                self.failures = 0
                return True
            return False
        return True


class SpecializedAgent:
    def __init__(self, name: str, capability: AgentCapability, handler: Callable[..., Coroutine[Any, Any, Any]]):
        self.name = name
        self.capability = capability
        self.handler = handler
        self.circuit_breaker = CircuitBreaker()
        self.status = AgentStatus.IDLE
        self.metrics = {"invocations": 0, "successes": 0, "failures": 0}

    async def execute(self, task: AgentTask, context: dict[str, Any]) -> TaskResult:
        if not self.circuit_breaker.allow_request():
            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=False,
                error="Circuit breaker open - agent throttled",
            )

        self.status = AgentStatus.BUSY
        self.metrics["invocations"] += 1
        start_time = time.time()

        try:
            async with asyncio.timeout(task.timeout_seconds):
                result_data = await self.handler(task.payload, context)
                latency = (time.time() - start_time) * 1000
                self.circuit_breaker.record_success()
                self.status = AgentStatus.IDLE
                self.metrics["successes"] += 1
                return TaskResult(
                    task_id=task.task_id, agent_name=self.name, success=True, data=result_data, latency_ms=latency
                )
        except TimeoutError:
            latency = (time.time() - start_time) * 1000
            self.circuit_breaker.record_failure()
            self.status = AgentStatus.FAILED
            self.metrics["failures"] += 1
            logger.warning(f"Agent {self.name} timed out on task {task.task_id}")
            return TaskResult(
                task_id=task.task_id, agent_name=self.name, success=False, error="Task timed out", latency_ms=latency
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.circuit_breaker.record_failure()
            self.status = AgentStatus.FAILED
            self.metrics["failures"] += 1
            logger.exception(f"Agent {self.name} failed task {task.task_id}: {e}")
            return TaskResult(
                task_id=task.task_id, agent_name=self.name, success=False, error=str(e), latency_ms=latency
            )
        finally:
            if self.status == AgentStatus.BUSY:
                self.status = AgentStatus.IDLE

    def supports_cancellation(self) -> bool:
        """Return True if the handler is a coroutine function that can be cancelled."""
        return inspect.iscoroutinefunction(self.handler)


class MessageBroker:
    def __init__(self):
        self._subscribers: dict[str, list[Callable[[Any], Coroutine[Any, Any, None]]]] = {}

    def subscribe(self, topic: str, callback: Callable[[Any], Coroutine[Any, Any, None]]):
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    async def publish(self, topic: str, message: Any):
        if topic in self._subscribers:
            await asyncio.gather(*(cb(message) for cb in self._subscribers[topic]), return_exceptions=True)


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, SpecializedAgent] = {}

    def register_agent(self, agent: SpecializedAgent):
        self._agents[agent.name] = agent

    def get_agent_by_capability(self, capability: AgentCapability) -> SpecializedAgent | None:
        for agent in self._agents.values():
            if agent.capability == capability and agent.circuit_breaker.allow_request():
                return agent
        for agent in self._agents.values():
            if agent.capability == capability:
                return agent
        return None

    def get_agent(self, name: str) -> SpecializedAgent | None:
        return self._agents.get(name)


class QueryAnalysisEngine:
    @staticmethod
    def analyze_query(query: str) -> list[AgentCapability]:
        q = query.lower()
        capabilities = []

        if any(term in q for term in ["fiqh", "halal", "haram", "ruling", "wudu", "prayer", "salah"]):
            capabilities.append(AgentCapability.FIQH)
        if any(term in q for term in ["hadith", "bukhari", "muslim", "sahih", "narration"]):
            capabilities.append(AgentCapability.HADITH)
        if any(term in q for term in ["tafsir", "exegesis", "verse meaning"]):
            capabilities.append(AgentCapability.TAFSIR)
        if any(term in q for term in ["quran", "ayah", "surah"]):
            capabilities.append(AgentCapability.QURAN)
        if any(term in q for term in ["zakat", "nisab", "charity"]):
            capabilities.append(AgentCapability.ZAKAT)
        if any(term in q for term in ["stellar", "usdc", "wallet", "balance"]):
            capabilities.append(AgentCapability.STELLAR)

        if not capabilities:
            capabilities.append(AgentCapability.GENERAL)

        return capabilities


class TaskDecompositionEngine:
    """Intelligent task decomposition engine.

    Analyzes a query, identifies logical sub-tasks, builds a dependency graph,
    and produces an execution plan optimized for parallel execution.
    """

    # Keywords that indicate a query spans multiple domains and thus benefits
    # from decomposition.
    _COMPLEXITY_MARKERS = (
        "and",
        "then",
        "after",
        "before",
        "compare",
        "combine",
        "also",
        "plus",
        "with",
    )

    def __init__(self, analyzer: "QueryAnalysisEngine | None" = None):
        self.analyzer = analyzer or QueryAnalysisEngine()
        self.metrics: dict[str, Any] = {
            "decompositions": 0,
            "parallel_plans": 0,
            "sequential_plans": 0,
            "avg_sub_tasks": 0.0,
        }

    def assess_complexity(self, query: str) -> float:
        """Return a normalized complexity score in [0.0, 1.0]."""
        q = query.lower()
        word_count = len(q.split())
        capability_count = len(self.analyzer.analyze_query(query))
        marker_count = sum(1 for marker in self._COMPLEXITY_MARKERS if marker in q.split())

        # Weighted heuristic: more capabilities and conjunction markers imply
        # higher complexity.
        score = (
            min(capability_count / 4.0, 1.0) * 0.5
            + min(marker_count / 3.0, 1.0) * 0.3
            + min(word_count / 40.0, 1.0) * 0.2
        )
        return round(min(score, 1.0), 4)

    def decompose(self, query: str, context: dict[str, Any] | None = None) -> DecompositionPlan:
        """Decompose a query into a dependency-aware plan."""
        ctx = context or {}
        capabilities = self.analyzer.analyze_query(query)
        complexity = self.assess_complexity(query)

        sub_tasks: list[SubTask] = []
        for idx, cap in enumerate(capabilities):
            sub_tasks.append(
                SubTask(
                    sub_task_id=f"sub_{idx}_{cap.value}",
                    description=f"Handle {cap.value} aspect of the query",
                    capability=cap,
                    dependencies=[],
                    priority=idx,
                    payload={"query": query, "context": ctx},
                )
            )

        # If a synthesis step is required (multiple capabilities), add a
        # dependent aggregation sub-task.
        if len(sub_tasks) > 1:
            sub_tasks.append(
                SubTask(
                    sub_task_id="sub_synthesis",
                    description="Synthesize sub-task results",
                    capability=AgentCapability.GENERAL,
                    dependencies=[st.sub_task_id for st in sub_tasks],
                    priority=len(sub_tasks),
                    payload={"query": query, "context": ctx},
                )
            )

        parallelizable = self._is_parallelizable(sub_tasks)
        if len(sub_tasks) <= 1:
            strategy = DecompositionStrategy.SINGLE
        elif parallelizable:
            strategy = DecompositionStrategy.PARALLEL
        else:
            strategy = DecompositionStrategy.SEQUENTIAL

        coverage = self._validate_coverage(query, sub_tasks, capabilities)

        plan = DecompositionPlan(
            query=query,
            sub_tasks=sub_tasks,
            strategy=strategy,
            parallelizable=parallelizable,
            complexity_score=complexity,
            coverage_score=coverage,
            metadata={"capabilities": [c.value for c in capabilities]},
        )

        self.metrics["decompositions"] += 1
        if strategy == DecompositionStrategy.PARALLEL:
            self.metrics["parallel_plans"] += 1
        elif strategy == DecompositionStrategy.SEQUENTIAL:
            self.metrics["sequential_plans"] += 1
        n = self.metrics["decompositions"]
        self.metrics["avg_sub_tasks"] = (
            (self.metrics["avg_sub_tasks"] * (n - 1) + len(sub_tasks)) / n
        )

        return plan

    def _is_parallelizable(self, sub_tasks: list[SubTask]) -> bool:
        """A plan is parallelizable if at least two sub-tasks share no deps."""
        independent = [st for st in sub_tasks if not st.dependencies]
        return len(independent) >= 2

    def _validate_coverage(
        self,
        query: str,
        sub_tasks: list[SubTask],
        required_capabilities: list[AgentCapability],
    ) -> float:
        """Return fraction of required capabilities covered by sub-tasks."""
        if not required_capabilities:
            return 1.0
        covered = {st.capability for st in sub_tasks}
        hits = sum(1 for cap in required_capabilities if cap in covered)
        return round(hits / len(required_capabilities), 4)

    def to_agent_tasks(self, plan: DecompositionPlan) -> list[AgentTask]:
        """Convert a decomposition plan into executable AgentTask objects."""
        return [
            AgentTask(
                task_id=st.sub_task_id,
                agent_capability=st.capability,
                payload=st.payload,
                dependencies=list(st.dependencies),
            )
            for st in plan.sub_tasks
        ]

    def replan(
        self,
        plan: DecompositionPlan,
        failed_task_ids: set[str],
    ) -> DecompositionPlan:
        """Produce a revised plan that drops failed sub-tasks and re-roots
        dependents so execution can continue."""
        if not failed_task_ids:
            return plan

        remaining = [st for st in plan.sub_tasks if st.sub_task_id not in failed_task_ids]
        remaining_ids = {st.sub_task_id for st in remaining}
        for st in remaining:
            st.dependencies = [d for d in st.dependencies if d in remaining_ids]

        new_plan = DecompositionPlan(
            query=plan.query,
            sub_tasks=remaining,
            strategy=plan.strategy,
            parallelizable=self._is_parallelizable(remaining),
            complexity_score=plan.complexity_score,
            coverage_score=plan.coverage_score,
            metadata={**plan.metadata, "replanned": True, "failed": sorted(failed_task_ids)},
        )
        return new_plan


class DAGScheduler:
    @staticmethod
    def build_dag(tasks: list[AgentTask]) -> list[list[AgentTask]]:
        task_map = {t.task_id: t for t in tasks}
        in_degree = {t.task_id: len(t.dependencies) for t in tasks}
        adj: dict[str, list[str]] = {t.task_id: [] for t in tasks}

        for t in tasks:
            for dep in t.dependencies:
                if dep in adj:
                    adj[dep].append(t.task_id)

        queue = [t.task_id for t in tasks if in_degree[t.task_id] == 0]
        levels = []

        while queue:
            current_level = list(queue)
            levels.append([task_map[tid] for tid in current_level])
            next_queue = []
            for tid in current_level:
                for neighbor in adj[tid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue

        return levels


class ResultSynthesisPipeline:
    @staticmethod
    def synthesize(query: str, results: list[TaskResult]) -> dict[str, Any]:
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        synthesized_text = f"Synthesis for query: '{query}'\n"
        for r in successful_results:
            synthesized_text += f"[{r.agent_name}]: {r.data}\n"

        return {
            "query": query,
            "success_count": len(successful_results),
            "failure_count": len(failed_results),
            "synthesized_response": synthesized_text.strip(),
            "raw_results": results,
            "coherence_score": 0.98 if successful_results else 0.0,
        }


class MultiAgentOrchestrator:
    def __init__(self):
        self.registry = AgentRegistry()
        self.broker = MessageBroker()
        self.analyzer = QueryAnalysisEngine()
        self.scheduler = DAGScheduler()
        self.synthesizer = ResultSynthesisPipeline()
        self.decomposer = TaskDecompositionEngine(self.analyzer)

    def register_agent(self, agent: SpecializedAgent):
        self.registry.register_agent(agent)

    async def process_query(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = context or {}
        plan = self.decomposer.decompose(query, ctx)
        tasks = self.decomposer.to_agent_tasks(plan)

        if not tasks:
            general_agent = self.registry.get_agent_by_capability(AgentCapability.GENERAL)
            if general_agent:
                tasks.append(
                    AgentTask(
                        task_id="task_0_general",
                        agent_capability=AgentCapability.GENERAL,
                        payload={"query": query, "context": ctx},
                    )
                )

        dag_levels = self.scheduler.build_dag(tasks)
        all_results: list[TaskResult] = []

        for level in dag_levels:
            coros = []
            for task in level:
                if task.task_id in ctx.get("cancelled_tasks", set()):
                    all_results.append(
                        TaskResult(task_id=task.task_id, agent_name="cancelled", success=False, error="Cancelled")
                    )
                    continue
                agent = self.registry.get_agent_by_capability(task.agent_capability)
                if agent:
                    coros.append(agent.execute(task, ctx))
                else:
                    all_results.append(
                        TaskResult(
                            task_id=task.task_id,
                            agent_name="unknown",
                            success=False,
                            error=f"No agent found for capability {task.agent_capability}",
                        )
                    )
            if coros:
                level_results = await asyncio.gather(*coros)
                all_results.extend(level_results)

        # Dynamic re-decomposition: if any tasks failed, drop them and
        # re-run the remaining dependent tasks once.
        failed_ids = {r.task_id for r in all_results if not r.success}
        if failed_ids and len(failed_ids) < len(tasks):
            revised = self.decomposer.replan(plan, failed_ids)
            retry_tasks = self.decomposer.to_agent_tasks(revised)
            retry_levels = self.scheduler.build_dag(retry_tasks)
            for level in retry_levels:
                coros = [
                    self.registry.get_agent_by_capability(t.agent_capability).execute(t, ctx)
                    for t in level
                    if self.registry.get_agent_by_capability(t.agent_capability) is not None
                ]
                if coros:
                    all_results.extend(await asyncio.gather(*coros))

        synthesis = self.synthesizer.synthesize(query, all_results)
        synthesis["decomposition"] = {
            "strategy": plan.strategy.value,
            "parallelizable": plan.parallelizable,
            "complexity_score": plan.complexity_score,
            "coverage_score": plan.coverage_score,
            "sub_task_count": len(plan.sub_tasks),
        }
        return synthesis
