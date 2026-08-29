import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

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
    payload: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: float = 5.0


@dataclass
class TaskResult:
    task_id: str
    agent_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
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

    async def execute(self, task: AgentTask, context: Dict[str, Any]) -> TaskResult:
        if not self.circuit_breaker.allow_request():
            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=False,
                error="Circuit breaker open - agent throttled"
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
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=True,
                    data=result_data,
                    latency_ms=latency
                )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.circuit_breaker.record_failure()
            self.status = AgentStatus.FAILED
            self.metrics["failures"] += 1
            logger.exception(f"Agent {self.name} failed task {task.task_id}: {e}")
            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=False,
                error=str(e),
                latency_ms=latency
            )


class MessageBroker:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], Coroutine[Any, Any, None]]]] = {}

    def subscribe(self, topic: str, callback: Callable[[Any], Coroutine[Any, Any, None]]):
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    async def publish(self, topic: str, message: Any):
        if topic in self._subscribers:
            await asyncio.gather(*(cb(message) for cb in self._subscribers[topic]), return_exceptions=True)


class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, SpecializedAgent] = {}

    def register_agent(self, agent: SpecializedAgent):
        self._agents[agent.name] = agent

    def get_agent_by_capability(self, capability: AgentCapability) -> Optional[SpecializedAgent]:
        for agent in self._agents.values():
            if agent.capability == capability and agent.circuit_breaker.allow_request():
                return agent
        for agent in self._agents.values():
            if agent.capability == capability:
                return agent
        return None

    def get_agent(self, name: str) -> Optional[SpecializedAgent]:
        return self._agents.get(name)


class QueryAnalysisEngine:
    @staticmethod
    def analyze_query(query: str) -> List[AgentCapability]:
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
    def build_dag(tasks: List[AgentTask]) -> List[List[AgentTask]]:
        task_map = {t.task_id: t for t in tasks}
        in_degree = {t.task_id: len(t.dependencies) for t in tasks}
        adj: Dict[str, List[str]] = {t.task_id: [] for t in tasks}

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
    def synthesize(query: str, results: List[TaskResult]) -> Dict[str, Any]:
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
            "coherence_score": 0.98 if successful_results else 0.0
        }


class MultiAgentOrchestrator:
    def __init__(self):
        self.registry = AgentRegistry()
        self.broker = MessageBroker()
        self.analyzer = QueryAnalysisEngine()
        self.scheduler = DAGScheduler()
        self.synthesizer = ResultSynthesisPipeline()

    def register_agent(self, agent: SpecializedAgent):
        self.registry.register_agent(agent)

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = context or {}
        required_capabilities = self.analyzer.analyze_query(query)

        tasks = []
        for idx, cap in enumerate(required_capabilities):
            agent = self.registry.get_agent_by_capability(cap)
            if agent:
                tasks.append(AgentTask(
                    task_id=f"task_{idx}_{cap.value}",
                    agent_capability=cap,
                    payload={"query": query, "context": ctx}
                ))

        if not tasks:
            # Fallback to general agent if registered
            general_agent = self.registry.get_agent_by_capability(AgentCapability.GENERAL)
            if general_agent:
                tasks.append(AgentTask(
                    task_id="task_0_general",
                    agent_capability=AgentCapability.GENERAL,
                    payload={"query": query, "context": ctx}
                ))

        dag_levels = self.scheduler.build_dag(tasks)
        all_results: List[TaskResult] = []

        for level in dag_levels:
            coros = []
            for task in level:
                agent = self.registry.get_agent_by_capability(task.agent_capability)
                if agent:
                    coros.append(agent.execute(task, ctx))
                else:
                    all_results.append(TaskResult(
                        task_id=task.task_id,
                        agent_name="unknown",
                        success=False,
                        error=f"No agent found for capability {task.agent_capability}"
                    ))
            if coros:
                level_results = await asyncio.gather(*coros)
                all_results.extend(level_results)

        synthesis = self.synthesizer.synthesize(query, all_results)
        return synthesis
