"""Agent subsystem package.

Groups agent orchestration, reasoning, synthesis, routing, and
reliability modules under a single package.

This package also exposes the intelligent task decomposition system,
which breaks complex queries into sub-tasks, builds dependency graphs,
assigns tasks to agents, and generates parallel execution plans.
"""

from .decomposition import (
    AgentCapability,
    AgentRegistry,
    DecompositionError,
    DecompositionResult,
    ExecutionPlan,
    ExecutionStep,
    QueryComplexity,
    SubTask,
    TaskDependencyGraph,
    TaskDecomposer,
    TaskStatus,
)

from .decomposition_metrics import DecompositionMetrics, MetricsTracker

__all__ = [
    "AgentCapability",
    "AgentRegistry",
    "DecompositionError",
    "DecompositionMetrics",
    "DecompositionResult",
    "ExecutionPlan",
    "ExecutionStep",
    "MetricsTracker",
    "QueryComplexity",
    "SubTask",
    "TaskDependencyGraph",
    "TaskDecomposer",
    "TaskStatus",
]
