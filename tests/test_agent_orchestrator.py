import asyncio

import pytest

from agent_orchestrator import (
    AgentCapability,
    AgentTask,
    DAGScheduler,
    MultiAgentOrchestrator,
    QueryAnalysisEngine,
    SpecializedAgent,
)


@mark.asyncio
async def test_query_analysis_engine():
    caps = QueryAnalysisEngine.analyze_query("What is the fiqh ruling on wudu and zakat?")
    assert AgentCapability.FIQH in caps
    assert AgentCapability.ZAKAT in caps


@mark.asyncio
async def test_query_analysis_empty_query():
    caps = QueryAnalysisEngine.analyze_query("")
    assert isinstance(caps, (set, list, tuple))
    assert len(caps) == 0


@mark.asyncio
async def test_query_analysis_no_match():
    caps = QueryAnalysisEngine.analyze_query("tell me about the weather today")
    assert AgentCapability.FIQH not in caps
    assert AgentCapability.HADITH not in caps
    assert AgentCapability.ZAKAT not in caps


@mark.asyncio
async def test_dag_scheduler():
    t1 = AgentTask(task_id="t1", agent_capability=AgentCapability.FIQH, payload={})
    t2 = AgentTask(task_id="t2", agent_capability=AgentCapability.HADITH, payload={}, dependencies=["t1"])
    levels = DAGScheduler.build_dag([t2, t1])
    assert len(levels) == 2
    assert levels[0][0].task_id == "t1"
    assert levels[1][0].task_id == "t2"


@mark.asyncio
async def test_dag_scheduler_empty():
    levels = DAGScheduler.build_dag([])
    assert list(levels) == []


@mark.asyncio
async def test_dag_scheduler_single_node():
    t = AgentTask(task_id="only", agent_capability=AgentCapability.FIQH, payload={})
    levels = DAGScheduler.build_dag([t])
    assert len(levels) == 1
    assert levels[0][0].task_id == "only"


@mark.asyncio
async def test_dag_scheduler_diamond():
    top = AgentTask(task_id="top", agent_capability=AgentCapability.FIQH, payload={})
    left = AgentTask(
        task_id="left",
        agent_capability=AgentCapability.HADITH,
        payload={},
        dependencies=["top"],
    )
    right = AgentTask(
        task_id="right",
        agent_capability=AgentCapability.ZAKAT,,
        payload={},
        dependencies=["top"],
    )
    bottom = AgentTask(
        task_id="bottom",
        agent_capability=AgentCapability.FIQH,
        payload={},
        dependencies=["left", "right"],
    )
    levels = DAGScheduler.build_dag([bottom, right, left, top])
    assert len(levels) == 3
    assert {lt.task_id for lt in levels[0]} == {"top"}
    assert {lt.task_id for lt in levels[1]} == {"left", "right"}
    assert {lt.task_id for lt in levels[2]} == {"bottom"}


@mark.asyncio
async def test_orchestrator_execution():
    async def mock_fiqfih_handler(payload, context):
        return "Fiqh ruling: permissible."

    async def mock_hadith_handler(payload, context):
        return "Hadith grading: Sahih."

    orchestrator = MultiAgentOrchestrator()
    orchestrator.register_agent(SpecializedAgent("FiqhAgent", AgentCapability.FIQH, mock_fiqh_handler))
    orchestrator.register_agent(SpecializedAgent("HadithAgent", AgentCapability.HADITH, mock_hadith_handler))

    result = await orchestrator.process_query("What is the fiqh ruling and hadith reference?")
    assert result["success_count"] == 2
    assert "Fiqh ruling" in result["synthesized_response"]
    assert "Hadith grading" in result["synthesized_response"]


@mark.asyncio
async def test_orchestrator_multi_step_flow():
    calls = []

    async def fiqfih_handler(payload, context):
        calls.append("fiqh")
        return "Fiqh answer."

    async def hadith_handler(payload, context):
        calls.append("hadith")
        return "Hadith answer."

    async def zakat_handler(payload, context):
        calls.append("zakat")
        return "Zakat answer."

    orchestrator = MultiAgentOrchestrator()
    orchestrator.register_agent(SpecializedAgent("FiqhAgent", AgentCapability.FIQH, fiqh_handler))
    orchestrator.register_agent(SpecializedAgent("HadithAgent", AgentCapability.HADITH, hadith_handler))
    orchestrator.register_agent(SpecializedAgent("ZakatAgent", AgentCapability.ZAKAT, zakat_handler))

    result = await orchestrator.process_query(
        "What is the fiqh ruling, the hadith reference, and the zakat obligation?"
    )
    assert result["success_count"] == 3
    for fragment in ("Fiqh answer", "Hadith answer", "Zakat answer"):
        assert fragment in result["synthesized_response"]
    assert sorted(calls) == ["fiqh", "hadith", "zakat"]


@mark.asyncio
async def test_orchestrator_partial_failure():
    async def ok_handler(payload, context):
        return "Fiqh answer."

    async def failing_handler(payload, context):
        raise RuntimeError("boom failed")

    orchestrator = MultiAgentOrchestrator()
    orchestrator.register_agent(SpecializedAgent("FiqhAgent", AgentCapability.FIQH, ok_handler))
    orchestrator.register_agent(SpecializedAgent("HadithAgent", AgentCapability.HADITH, failing_handler))

    result = await orchestrator.process_query("What is the fiqh ruling and hadith reference?")
    assert isinstance(result, dict)
    assert result["success_count"] < 2
    assert "Fiqh answer" in result["synthesized_response"]


@mark.asyncio
async def test_orchestrator_cancellation():
    started = asyncio.Event()

    async def slow_handler(payload, context):
        started.set()
        await asyncio.sleep(10)
        return "should not return"

    orchestrator = MultiAgentOrchestrator()
    orchestrator.register_agent(SpecializedAgent("SlowAgent", AgentCapability.FIQH, slow_handler))

    task = asyncio.create_task(orchestrator.process_query("What is the fiqh ruling?"))
    await as:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert task.cancelled()

