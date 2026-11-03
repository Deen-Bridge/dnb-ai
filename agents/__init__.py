"""Agent subsystem package.

Groups agent orchestration, reasoning, synthesis, routing, and
reliability modules under a single package.

Re-export implemented decomposition APIs from their canonical modules. The
former sibling-module imports referred to files that were never committed.
"""

from agent_orchestrator import (
    AgentCapability,
    AgentRegistry,
    DecompositionPlan,
    DecompositionStrategy,
    SubTask,
    TaskDecompositionEngine,
)
from query_optimizer import DecompositionMetrics

__all__ = [
    "AgentCapability",
    "AgentRegistry",
    "DecompositionMetrics",
    "DecompositionPlan",
    "DecompositionStrategy",
    "SubTask",
    "TaskDecompositionEngine",
]
