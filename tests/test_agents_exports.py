"""Regression coverage for the public agent APIs removed by task decomposition."""

import agents
from agent_orchestrator import DAGScheduler, TaskDecompositionEngine
from model_router import ModelRouter, NoModelAvailableError
from orchestration import Metrics, MultiAgentOrchestrator, ResultSynthesizer, TaskResult, WorkflowResult


def test_agent_package_exports_implemented_symbols() -> None:
    for name in agents.__all__:
        assert getattr(agents, name) is not None
    assert agents.TaskDecompositionEngine is TaskDecompositionEngine


def test_restored_apis_coexist_with_new_decomposition() -> None:
    plan = TaskDecompositionEngine().decompose("Compare fiqh rulings and hadith evidence")
    assert plan.sub_tasks
    assert DAGScheduler.build_dag([]) == []
    assert ModelRouter().profiles
    assert issubclass(NoModelAvailableError, Exception)
    assert all(
        api is not None for api in (Metrics, MultiAgentOrchestrator, ResultSynthesizer, TaskResult, WorkflowResult)
    )
