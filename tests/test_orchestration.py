import asyncio
import time

import pytest

from orchestration import (
    AgentRegistry,
    AgentSpec,
    AgentTask,
    CircuitState,
    ExecutionPlan,
    MultiAgentOrchestrator,
    PlanValidationError,
    QueryAnalyzer,
    TaskStatus,
)


def test_query_analysis_selects_matching_agents_and_execution_mode() -> None:
    async def handler(task: AgentTask, context: object) -> str:
        return str(task.payload)

    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="hadith",
            capabilities=frozenset({"hadith"}),
            keywords=frozenset({"isnad", "narration"}),
            handler=handler,
        )
    )
    registry.register(
        AgentSpec(
            name="fiqh",
            capabilities=frozenset({"fiqh"}),
            keywords=frozenset({"ruling", "madhhab"}),
            handler=handler,
        )
    )
    registry.register(
        AgentSpec(
            name="general",
            capabilities=frozenset({"general"}),
            handler=handler,
            default=True,
        )
    )

    analyzer = QueryAnalyzer(registry)
    parallel = analyzer.analyze("Compare the hadith evidence and fiqh ruling")
    sequential = analyzer.analyze("First inspect the hadith, then explain the fiqh ruling")
    default = analyzer.analyze("Hello there")

    assert parallel.selected_agents == ("fiqh", "hadith")
    assert parallel.sequential is False
    assert sequential.sequential is True
    assert default.selected_agents == ("general",)


def test_independent_agents_execute_in_parallel_and_synthesize() -> None:
    async def slow_handler(task: AgentTask, context: object) -> str:
        await asyncio.sleep(0.08)
        return f"result from {task.agent}"

    async def run() -> None:
        registry = AgentRegistry()
        registry.register(
            AgentSpec(
                name="quran",
                capabilities=frozenset({"quran"}),
                handler=slow_handler,
            )
        )
        registry.register(
            AgentSpec(
                name="hadith",
                capabilities=frozenset({"hadith"}),
                handler=slow_handler,
            )
        )
        plan = ExecutionPlan(
            (
                AgentTask("quran-task", "quran", "quran", "query"),
                AgentTask("hadith-task", "hadith", "hadith", "query"),
            )
        )
        orchestrator = MultiAgentOrchestrator(registry)
        started = time.monotonic()
        result = await orchestrator.orchestrate("Use Quran and hadith", plan=plan)
        elapsed = time.monotonic() - started

        assert elapsed < 0.14
        assert result.status is TaskStatus.SUCCEEDED
        assert len(result.traces) == 2
        assert "[quran]" in result.answer
        assert "[hadith]" in result.answer

    asyncio.run(run())


def test_dependencies_share_results_and_broker_messages() -> None:
    async def producer(task: AgentTask, context: object) -> str:
        await context.publish("evidence", "citation")  # type: ignore[attr-defined]
        return "source material"

    async def consumer(task: AgentTask, context: object) -> str:
        messages = await context.messages("evidence")  # type: ignore[attr-defined]
        dependency = context.dependency_results["source"]  # type: ignore[attr-defined]
        return f"{dependency}: {messages[0].payload}"

    async def run() -> None:
        registry = AgentRegistry()
        registry.register(
            AgentSpec("researcher", frozenset({"research"}), producer)
        )
        registry.register(
            AgentSpec("writer", frozenset({"writing"}), consumer)
        )
        plan = ExecutionPlan(
            (
                AgentTask("source", "researcher", "research", "query"),
                AgentTask(
                    "answer",
                    "writer",
                    "writing",
                    "query",
                    dependencies=("source",),
                ),
            )
        )
        result = await MultiAgentOrchestrator(registry).orchestrate("Research then write", plan=plan)

        assert result.task_results["answer"].output == "source material: citation"
        assert result.task_results["answer"].status is TaskStatus.SUCCEEDED

    asyncio.run(run())


def test_failure_uses_fallback_and_opens_circuit() -> None:
    calls = 0

    async def failing(task: AgentTask, context: object) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("unavailable")

    async def fallback(task: AgentTask, context: object) -> str:
        return "recovered"

    async def run() -> None:
        registry = AgentRegistry()
        registry.register(
            AgentSpec(
                "primary",
                frozenset({"research"}),
                failing,
                fallback_agent="backup",
                default=True,
            )
        )
        registry.register(AgentSpec("backup", frozenset({"research"}), fallback))
        orchestrator = MultiAgentOrchestrator(
            registry,
            circuit_failure_threshold=1,
            circuit_recovery_seconds=60,
        )
        plan = ExecutionPlan((AgentTask("work", "primary", "research", "query"),))

        first = await orchestrator.orchestrate("research", plan=plan)
        second = await orchestrator.orchestrate("research", plan=plan)

        assert first.task_results["work"].fallback_used is True
        assert first.task_results["work"].output == "recovered"
        assert second.task_results["work"].output == "recovered"
        assert calls == 1
        assert orchestrator.circuit_state("primary") is CircuitState.OPEN
        assert orchestrator.metrics.snapshot()["fallbacks_succeeded"] == 2

    asyncio.run(run())


def test_partial_failure_preserves_successful_answer() -> None:
    async def success(task: AgentTask, context: object) -> str:
        return "usable evidence"

    async def failure(task: AgentTask, context: object) -> str:
        raise RuntimeError("agent failed")

    async def run() -> None:
        registry = AgentRegistry()
        registry.register(AgentSpec("good", frozenset({"general"}), success))
        registry.register(AgentSpec("bad", frozenset({"general"}), failure))
        plan = ExecutionPlan(
            (
                AgentTask("good-task", "good", "general", "query"),
                AgentTask("bad-task", "bad", "general", "query"),
            )
        )
        result = await MultiAgentOrchestrator(registry).orchestrate("query", plan=plan)

        assert result.status is TaskStatus.SUCCEEDED
        assert result.partial_failure is True
        assert "usable evidence" in result.answer
        assert result.task_results["bad-task"].status is TaskStatus.FAILED

    asyncio.run(run())


def test_lifecycle_hooks_and_plan_validation() -> None:
    events: list[str] = []

    async def start() -> None:
        events.append("start")

    async def stop() -> None:
        events.append("stop")

    async def handler(task: AgentTask, context: object) -> str:
        return "ok"

    async def run() -> None:
        registry = AgentRegistry()
        registry.register(
            AgentSpec(
                "agent",
                frozenset({"general"}),
                handler,
                default=True,
                start=start,
                stop=stop,
            )
        )
        async with MultiAgentOrchestrator(registry) as orchestrator:
            result = await orchestrator.orchestrate("hello")
            assert result.answer == "ok"
        assert events == ["start", "stop"]

        cyclic = ExecutionPlan(
            (
                AgentTask("one", "agent", "general", None, dependencies=("two",)),
                AgentTask("two", "agent", "general", None, dependencies=("one",)),
            )
        )
        with pytest.raises(PlanValidationError, match="cycle"):
            cyclic.validate(registry)

    asyncio.run(run())
