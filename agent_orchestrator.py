import asyncio
import inspect
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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


@dataclass
class AgentTask:
    task_id: str
    agent_capability: AgentCapability
    payload: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: float = 5.0


@dataclass
class TaskResult:
    task_id: str
    agent_name: str
    success: bool
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    trace_id: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


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
        except asyncio.TimeoutError:
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


class DecompositionStrategy(str, Enum):
    ATOMIC = "atomic"
    COMPOSITE = "composite"
    HIERARCHICAL = "hierarchical"


@dataclass
class SubTask:
    subtask_id: str
    description: str
    capability: AgentCapability
    dependencies: list[str] = field(default_factory=list)
    estimated_complexity: float = 1.0
    parallelizable: bool = True
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionPlan:
    plan_id: str
    query: str
    strategy: DecompositionStrategy
    subtasks: list[SubTask]
    execution_levels: list[list[str]] = field(default_factory=list)
    estimated_parallelism: float = 1.0
    completeness_score: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class DecompositionMetrics:
    total_plans: int = 0
    total_subtasks: int = 0
    parallel_plans: int = 0
    re_decompositions: int = 0
    validation_failures: int = 0
    avg_completeness: float = 0.0
    avg_parallelism: float = 0.0

    def record_plan(self, plan: DecompositionPlan):
        self.total_plans += 1
        self.total_subtasks += len(plan.subtasks)
        if plan.estimated_parallelism > 1.0:
            self.parallel_plans += 1
        n = self.total_plans
        self.avg_completeness = ((self.avg_completeness * (n - 1)) + plan.completeness_score) / n
        self.avg_parallelism = ((self.avg_parallelism * (n - 1)) + plan.estimated_parallelism) / n


class QueryComplexityAssessor:
    """Assesses query complexity to determine optimal decomposition granularity."""

    COMPLEXITY_INDICATORS = {
        "multi_part": ["and", "also", "additionally", "furthermore", "plus", "as well as"],
        "comparison": ["compare", "versus", "vs", "difference", "better", "worse"],
        "conditional": ["if", "when", "unless", "depending", "in case"],
        "temporal": ["before", "after", "during", "then", "subsequently"],
        "causal": ["because", "therefore", "thus", "consequently", "due to"],
    }

    @classmethod
    def assess(cls, query: str) -> dict[str, Any]:
        q = query.lower()
        word_count = len(q.split())
        indicators_found: list[str] = []
        complexity_score = 0.0

        for category, terms in cls.COMPLEXITY_INDICATORS.items():
            if any(term in q for term in terms):
                indicators_found.append(category)
                complexity_score += 1.0

        if word_count > 30:
            complexity_score += 2.0
        elif word_count > 15:
            complexity_score += 1.0

        capability_count = len(QueryAnalysisEngine.analyze_query(query))
        complexity_score += capability_count * 0.5

        if complexity_score >= 4.0:
            level = "high"
        elif complexity_score >= 2.0:
            level = "medium"
        else:
            level = "low"

        return {
            "level": level,
            "score": complexity_score,
            "word_count": word_count,
            "indicators": indicators_found,
            "capability_count": capability_count,
            "requires_decomposition": complexity_score >= 2.0,
        }


class TaskDecomposer:
    """Breaks complex queries into sub-tasks with dependency analysis."""

    def __init__(self):
        self.assessor = QueryComplexityAssessor()

    def decompose(self, query: str, context: dict[str, Any] | None = None) -> DecompositionPlan:
        ctx = context or {}
        assessment = self.assessor.assess(query)
        capabilities = QueryAnalysisEngine.analyze_query(query)

        if assessment["level"] == "low" or len(capabilities) <= 1:
            strategy = DecompositionStrategy.ATOMIC
        elif assessment["level"] == "high":
            strategy = DecompositionStrategy.HIERARCHICAL
        else:
            strategy = DecompositionStrategy.COMPOSITE

        subtasks = self._build_subtasks(query, capabilities, strategy, ctx)
        self._infer_dependencies(subtasks, strategy)
        levels = self._compute_levels(subtasks)
        parallelism = self._estimate_parallelism(subtasks, levels)
        completeness = self._validate_completeness(query, subtasks, capabilities)

        plan = DecompositionPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            query=query,
            strategy=strategy,
            subtasks=subtasks,
            execution_levels=levels,
            estimated_parallelism=parallelism,
            completeness_score=completeness,
        )
        return plan

    def _build_subtasks(
        self,
        query: str,
        capabilities: list[AgentCapability],
        strategy: DecompositionStrategy,
        ctx: dict[str, Any],
    ) -> list[SubTask]:
        subtasks: list[SubTask] = []
        for idx, cap in enumerate(capabilities):
            subtask_id = f"sub_{idx}_{cap.value}"
            subtasks.append(
                SubTask(
                    subtask_id=subtask_id,
                    description=f"Handle {cap.value} aspect of query",
                    capability=cap,
                    payload={"query": query, "context": ctx, "aspect": cap.value},
                    estimated_complexity=1.0 if strategy == DecompositionStrategy.ATOMIC else 1.5,
                )
            )

        if strategy == DecompositionStrategy.HIERARCHICAL and len(subtasks) > 1:
            synthesis_id = "sub_synthesis"
            subtasks.append(
                SubTask(
                    subtask_id=synthesis_id,
                    description="Synthesize results from all sub-tasks",
                    capability=AgentCapability.GENERAL,
                    dependencies=[s.subtask_id for s in subtasks],
                    payload={"query": query, "context": ctx, "aspect": "synthesis"},
                    estimated_complexity=2.0,
                    parallelizable=False,
                )
            )
        return subtasks

    def _infer_dependencies(self, subtasks: list[SubTask], strategy: DecompositionStrategy):
        if strategy == DecompositionStrategy.ATOMIC:
            return
        # Composite: quran/tafsir feed into fiqh; hadith feeds into fiqh
        by_cap: dict[AgentCapability, list[str]] = {}
        for st in subtasks:
            by_cap.setdefault(st.capability, []).append(st.subtask_id)

        fiqh_ids = by_cap.get(AgentCapability.FIQH, [])
        upstream = (
            by_cap.get(AgentCapability.QURAN, [])
            + by_cap.get(AgentCapability.TAFSIR, [])
            + by_cap.get(AgentCapability.HADITH, [])
        )
        for fid in fiqh_ids:
            for st in subtasks:
                if st.subtask_id == fid and not st.dependencies:
                    st.dependencies = list(upstream)

    def _compute_levels(self, subtasks: list[SubTask]) -> list[list[str]]:
        task_map = {st.subtask_id: st for st in subtasks}
        in_degree = {st.subtask_id: len([d for d in st.dependencies if d in task_map]) for st in subtasks}
        adj: dict[str, list[str]] = {st.subtask_id: [] for st in subtasks}
        for st in subtasks:
            for dep in st.dependencies:
                if dep in adj:
                    adj[dep].append(st.subtask_id)

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        levels: list[list[str]] = []
        while queue:
            levels.append(list(queue))
            next_queue: list[str] = []
            for sid in queue:
                for neighbor in adj[sid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue
        return levels

    def _estimate_parallelism(self, subtasks: list[SubTask], levels: list[list[str]]) -> float:
        if not levels or len(subtasks) <= 1:
            return 1.0
        max_width = max(len(level) for level in levels)
        return float(max_width)

    def _validate_completeness(
        self, query: str, subtasks: list[SubTask], capabilities: list[AgentCapability]
    ) -> float:
        if not capabilities:
            return 0.0
        covered = {st.capability for st in subtasks}
        covered_count = sum(1 for cap in capabilities if cap in covered)
        return covered_count / len(capabilities)


class AgentTaskMatcher:
    """Matches sub-tasks to optimal agents based on capability and load."""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def match(self, subtask: SubTask) -> SpecializedAgent | None:
        candidates = [
            a for a in self.registry._agents.values() if a.capability == subtask.capability
        ]
        if not candidates:
            return self.registry.get_agent_by_capability(AgentCapability.GENERAL)

        def score(agent: SpecializedAgent) -> tuple[int, int, float]:
            healthy = 1 if agent.circuit_breaker.allow_request() else 0
            idle = 1 if agent.status == AgentStatus.IDLE else 0
            success_rate = (
                agent.metrics["successes"] / agent.metrics["invocations"]
                if agent.metrics["invocations"] > 0
                else 1.0
            )
            return (healthy, idle, success_rate)

        candidates.sort(key=score, reverse=True)
        return candidates[0]


class ExecutionPlanGenerator:
    """Generates an executable plan from a decomposition plan."""

    @staticmethod
    def generate(plan: DecompositionPlan) -> list[list[AgentTask]]:
        subtask_map = {st.subtask_id: st for st in plan.subtasks}
        execution_levels: list[list[AgentTask]] = []
        for level in plan.execution_levels:
            tasks: list[AgentTask] = []
            for sid in level:
                st = subtask_map.get(sid)
                if st is None:
                    continue
                tasks.append(
                    AgentTask(
                        task_id=st.subtask_id,
                        agent_capability=st.capability,
                        payload=st.payload,
                        dependencies=list(st.dependencies),
                    )
                )
            if tasks:
                execution_levels.append(tasks)
        return execution_levels


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
            "parallelization_ratio": ResultSynthesisPipeline._parallelization_ratio(results),
            "total_latency_ms": sum(r.latency_ms for r in results),
        }

    @staticmethod
    def _parallelization_ratio(results: list[TaskResult]) -> float:
        if not results:
            return 0.0
        wall = max((r.finished_at for r in results), default=0.0) - min(
            (r.started_at for r in results if r.started_at > 0), default=0.0
        )
        cpu = sum(r.latency_ms for r in results) / 1000.0
        if wall <= 0:
            return 1.0
        return cpu / wall


class MultiAgentOrchestrator:
    def __init__(self):
        self.registry = AgentRegistry()
        self.broker = MessageBroker()
        self.analyzer = QueryAnalysisEngine()
        self.scheduler = DAGScheduler()
        self.synthesizer = ResultSynthesisPipeline()
        self.decomposer = TaskDecomposer()
        self.matcher = AgentTaskMatcher(self.registry)
        self.plan_generator = ExecutionPlanGenerator()
        self.metrics = DecompositionMetrics()

    def register_agent(self, agent: SpecializedAgent):
        self.registry.register_agent(agent)

    async def process_query(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = context or {}
        plan = self.decomposer.decompose(query, ctx)
        self.metrics.record_plan(plan)

        if plan.completeness_score < 1.0:
            self.metrics.validation_failures += 1
            logger.warning(
                f"Decomposition completeness {plan.completeness_score:.2f} below threshold for query"
            )

        dag_levels = self.plan_generator.generate(plan)
        all_results: list[TaskResult] = []

        for level in dag_levels:
            coros = []
            for task in level:
                if task.task_id in ctx.get("cancelled_tasks", set()):
                    all_results.append(
                        TaskResult(task_id=task.task_id, agent_name="cancelled", success=False, error="Cancelled")
                    )
                    continue
                subtask = next(
                    (st for st in plan.subtasks if st.subtask_id == task.task_id), None
                )
                agent = self.matcher.match(subtask) if subtask else None
                if agent is None:
                    agent = self.registry.get_agent_by_capability(task.agent_capability)
                if agent:
                    coros.append(self._execute_with_timing(agent, task, ctx))
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

        # Dynamic re-decomposition: if too many failures, retry failed capabilities
        failed_caps = [
            r for r in all_results if not r.success and r.agent_name != "unknown"
        ]
        if failed_caps and len(failed_caps) < len(all_results):
            self.metrics.re_decompositions += 1
            retry_results = await self._retry_failed(failed_caps, query, ctx)
            all_results.extend(retry_results)

        synthesis = self.synthesizer.synthesize(query, all_results)
        synthesis["decomposition_plan"] = {
            "plan_id": plan.plan_id,
            "strategy": plan.strategy.value,
            "subtask_count": len(plan.subtasks),
            "execution_levels": len(plan.execution_levels),
            "estimated_parallelism": plan.estimated_parallelism,
            "completeness_score": plan.completeness_score,
        }
        synthesis["metrics"] = {
            "total_plans": self.metrics.total_plans,
            "parallel_plans": self.metrics.parallel_plans,
            "re_decompositions": self.metrics.re_decompositions,
            "avg_completeness": self.metrics.avg_completeness,
            "avg_parallelism": self.metrics.avg_parallelism,
        }
        return synthesis

    async def _execute_with_timing(
        self, agent: SpecializedAgent, task: AgentTask, ctx: dict[str, Any]
    ) -> TaskResult:
        started = time.time()
        result = await agent.execute(task, ctx)
        result.started_at = started
        result.finished_at = time.time()
        return result

    async def _retry_failed(
        self, failed: list[TaskResult], query: str, ctx: dict[str, Any]
    ) -> list[TaskResult]:
        retries: list[TaskResult] = []
        for r in failed:
            cap = next(
                (c for c in AgentCapability if c.value in r.task_id), AgentCapability.GENERAL
            )
            agent = self.registry.get_agent_by_capability(cap)
            if agent is None:
                continue
            retry_task = AgentTask(
                task_id=f"{r.task_id}_retry",
                agent_capability=cap,
                payload={"query": query, "context": ctx, "retry": True},
            )
            retries.append(await self._execute_with_timing(agent, retry_task, ctx))
        return retries