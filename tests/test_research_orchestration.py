import asyncio
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orchestration import AgentRegistry, AgentSpec, TaskStatus
from research_orchestration import (
    ResearchOrchestrator,
    ResearchPlanBuilder,
    ResearchQueryAnalyzer,
    create_research_registry,
    decompose_query,
    reset_research_orchestrator,
    router,
)

COMPLEX_QUERY = "Compare the Quranic verses and hadith on patience, explain the tafsir, and the fiqh ruling on prayer"


def test_registry_discovers_specialized_agents() -> None:
    registry = create_research_registry()
    names = {agent.name for agent in registry.all()}

    assert {"quran", "hadith", "tafsir", "fiqh", "research_assistant"} <= names
    assert registry.for_capability("tafsir")[0].name == "tafsir"
    assert registry.for_capability("quran")[0].name == "quran"


def test_query_decomposition_selects_multiple_domains() -> None:
    registry = create_research_registry()
    analyzer = ResearchQueryAnalyzer(registry)

    analysis = analyzer.analyze(COMPLEX_QUERY)

    assert set(analysis.selected_agents) >= {"quran", "hadith", "tafsir", "fiqh"}
    assert set(analysis.capabilities) >= {"quran", "hadith", "tafsir", "fiqh"}
    assert len(decompose_query(COMPLEX_QUERY)) >= 3
    assert analysis.sequential is False


def test_plan_makes_tafsir_depend_on_quran() -> None:
    registry = create_research_registry()
    analysis = ResearchQueryAnalyzer(registry).analyze(COMPLEX_QUERY)
    plan = ResearchPlanBuilder(registry).build(analysis)

    by_agent = {task.agent: task for task in plan.tasks}
    quran_task = by_agent["quran"]
    tafsir_task = by_agent["tafsir"]

    assert tafsir_task.dependencies == (quran_task.id,)
    assert by_agent["hadith"].dependencies == ()
    assert by_agent["fiqh"].dependencies == ()


async def test_default_agents_orchestrate_three_plus_domains() -> None:
    orchestrator = ResearchOrchestrator()

    result = await orchestrator.orchestrate(COMPLEX_QUERY, timeout_ms=20000)

    assert result.status is TaskStatus.SUCCEEDED
    succeeded = {task.agent for task in result.task_results.values() if task.status is TaskStatus.SUCCEEDED}
    assert {"quran", "hadith", "tafsir", "fiqh"} <= succeeded
    assert len(result.traces) == 4
    assert "Findings for:" in result.answer
    assert "a scholar should confirm" in result.answer

    stored = orchestrator.trace_store.get(result.workflow_id)
    assert stored is result


async def test_parallel_wave_executes_independent_agents_concurrently() -> None:
    async def slow(task, context):
        await asyncio.sleep(0.08)
        return {"domain": "general", "summary": task.agent}

    registry = AgentRegistry()
    for name, capability in (("quran", "quran"), ("hadith", "hadith"), ("fiqh", "fiqh")):
        registry.register(AgentSpec(name, frozenset({capability}), slow, priority=1))
    orchestrator = ResearchOrchestrator(registry)

    started = time.monotonic()
    result = await orchestrator.orchestrate("Quran, hadith and fiqh", timeout_ms=5000)
    elapsed = time.monotonic() - started

    assert result.status is TaskStatus.SUCCEEDED
    assert len(result.traces) == 3
    assert elapsed < 0.2


async def test_failed_agent_uses_registered_fallback() -> None:
    calls = 0

    async def failing(task, context):
        nonlocal calls
        calls += 1
        raise RuntimeError("primary unavailable")

    async def backup(task, context):
        return {"domain": "quran", "summary": "Recovered answer"}

    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            "quran",
            frozenset({"quran"}),
            failing,
            priority=10,
            fallback_agent="backup",
        )
    )
    registry.register(AgentSpec("backup", frozenset({"quran"}), backup, priority=1))
    orchestrator = ResearchOrchestrator(registry)

    result = await orchestrator.orchestrate("What does the Quran say about patience?", timeout_ms=5000)

    task = next(iter(result.task_results.values()))
    assert result.status is TaskStatus.SUCCEEDED
    assert task.agent == "backup"
    assert task.fallback_used is True
    assert "Recovered answer" in result.answer
    assert calls == 1


async def test_partial_failure_preserves_successful_agents() -> None:
    async def failing(task, context):
        raise RuntimeError("quran agent failed")

    async def success(task, context):
        return {"domain": "hadith", "summary": "Hadith evidence available"}

    registry = AgentRegistry()
    registry.register(AgentSpec("quran", frozenset({"quran"}), failing, priority=10))
    registry.register(AgentSpec("hadith", frozenset({"hadith"}), success, priority=5))
    orchestrator = ResearchOrchestrator(registry)

    result = await orchestrator.orchestrate("Compare the Quran and hadith", timeout_ms=5000)

    assert result.status is TaskStatus.SUCCEEDED
    assert result.partial_failure is True
    assert "Hadith evidence available" in result.answer
    assert any(task.status is TaskStatus.FAILED for task in result.task_results.values())


async def test_timeout_is_recorded_as_a_failed_task() -> None:
    async def slow(task, context):
        await asyncio.sleep(1)
        return {"domain": "quran", "summary": "too late"}

    registry = AgentRegistry()
    registry.register(AgentSpec("quran", frozenset({"quran"}), slow))
    orchestrator = ResearchOrchestrator(registry)

    result = await orchestrator.orchestrate("What does the Quran say?", timeout_ms=100)

    assert result.status is TaskStatus.FAILED
    task = next(iter(result.task_results.values()))
    assert task.status is TaskStatus.FAILED
    assert task.error is not None
    assert "timed out" in task.error.lower()


def test_api_exposes_registry_run_traces_and_metrics() -> None:
    reset_research_orchestrator()
    app = FastAPI()
    app.include_router(router)

    try:
        with TestClient(app) as client:
            agents = client.get("/orchestration/agents")
            assert agents.status_code == 200
            names = {agent["name"] for agent in agents.json()}
            assert {"quran", "hadith", "tafsir", "fiqh"} <= names

            missing_agent = client.get("/orchestration/agents/does-not-exist")
            assert missing_agent.status_code == 404

            run = client.post(
                "/orchestration/run",
                json={"query": COMPLEX_QUERY, "timeout_ms": 20000},
            )
            assert run.status_code == 200
            payload = run.json()
            assert len(payload["selected_agents"]) >= 3
            assert payload["partial_failure"] is False
            assert payload["answer"].startswith("Findings for:")

            trace = client.get(f"/orchestration/traces/{payload['workflow_id']}")
            assert trace.status_code == 200
            assert trace.json()["workflow_id"] == payload["workflow_id"]
            assert len(trace.json()["traces"]) >= 3

            missing_trace = client.get("/orchestration/traces/does-not-exist")
            assert missing_trace.status_code == 404

            metrics = client.get("/orchestration/metrics")
            assert metrics.status_code == 200
            assert metrics.json()["workflows_succeeded"] >= 1
    finally:
        reset_research_orchestrator()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What does the Quran say about mercy?", {"quran"}),
        ("Is this hadith in Bukhari authentic?", {"hadith"}),
        ("What is the fiqh ruling on wudu?", {"fiqh"}),
    ],
)
def test_single_domain_queries_route_to_one_agent(query: str, expected: set[str]) -> None:
    analysis = ResearchQueryAnalyzer(create_research_registry()).analyze(query)
    assert set(analysis.selected_agents) == expected
