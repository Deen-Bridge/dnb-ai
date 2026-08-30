import asyncio
import pytest

from agent_orchestrator import (
    AgentCapability,
    AgentTask,
    DAGScheduler,
    MultiAgentOrchestrator,
    QueryAnalysisEngine,
    SpecializedAgent,
    TaskResult,
)


@pytest.mark.asyncio
async def test_query_analysis_engine():
    caps = QueryAnalysisEngine.analyze_query("What is the fiqh ruling on wudu and zakat?")
    assert AgentCapability.FIQH in caps
    assert AgentCapability.ZAKAT in caps


@pytest.mark.asyncio
async def test_dag_scheduler():
    t1 = AgentTask(task_id="t1", agent_capability=AgentCapability.FIQH, payload={})
    t2 = AgentTask(task_id="t2", agent_capability=AgentCapability.HADITH, payload={}, dependencies=["t1"])
    levels = DAGScheduler.build_dag([t2, t1])
    assert len(levels) == 2
    assert levels[0][0].task_id == "t1"
    assert levels[1][0].task_id == "t2"


@pytest.mark.asyncio
async def test_orchestrator_execution():
    async def mock_fiqh_handler(payload, context):
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
