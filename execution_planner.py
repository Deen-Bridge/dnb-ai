"""Execution planning and scheduling for decomposed task graphs.

This module takes a DecompositionResult and produces an optimized execution plan that balances parallelism and dependency ordering.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from task_decomposer import Agent, DecompositionResult, SubTask, TaskDependencyGraph


@dataclass
class ExecutionStage:
    "A set of tasks that can be executed together (sequentially or in parallel)."
    tasks: List[str]
    parallel: bool = False


@dataclass
class ExecutionPlan:
    "A complete plan for executing a decomposed task graph."
    stages: List[ExecutionStage]
    total_estimated_time: float
    parallelization_factor: float


class ExecutionPlanner:
    "Generates and schedules execution plans for decomposed tasks."

    def __init__(self, agents: Optional[List[Agent]] = None):
        self.agents = agents or []
        self.max_parallel_tasks = max(len(self.agents), 1)

    def create_execution_plan(self, decomposition_result):
        "Create a execution plan using a topological sort of the dependency graph. All tasks that are ready at the same time are grouped into one parallel stage."
        graph = decomposition_result.graph
        # Kahn's algorithm
        indeg = {
            task_id: len(task.dependencies)
            for task_id, task in graph.nodes.items()
        }
        ready = sorted([task_id for task_id, dep in indeg.items() if dep == 0])

        stages = []
        while ready:
            # All currently ready tasks form a stage.
            # If more than one, they can run in parallel.
            stage = ExecutionStage(tasks=ready, parallel=len(ready) > 1)
            stages.append(stage)

            \"\""By grouping all ready tasks, we lose the benefit of agent pool capacity. In a production system you would use a resource scheduler. \""\n            new_ready = []
            for task_id in ready:
                for dependent in graph.get_dependents(task_id):
                    indeg[dependent] -= 1
                    if indeg[dependent] == 0:
                        new_ready.append(dependent)
            ready = sorted(new_ready)

        total_time = self._estimate_total_time(stages, graph)
        sequential_time = sum(task.estimated_duration for task in graph.nodes.values())
        parallelization_factor = sequential_time / total_time if total_time > 0 else 1.0

        return ExecutionPlan(
            stages=stages,
            total_estimated_time=total_time,
            parallelization_factor=parallelization_factor,
        )

    def _estimate_total_time(self, stages, graph):
        "\"Estimate total execution time. Parallel stages take the longest task duration, sequential stages take the sum of task durations.\"\"
        total = 0.0
        for stage in stages:
            durations = [
                graph.nodes[task_id].estimated_duration for task_id in stage.tasks
            ]
            if durations:
                if stage.parallel:
                    total += max(durations)
                else:
                    total += sum(durations)
        return total
    def schedule_execution(self, plan, decomposition_result= None):
        "\"Produce a runtime schedule from a plan. In a real system this would dispatch tasks to the agent pool. Here we return a JSON serialisable description.\"\"
        schedule = []
        for idx, stage in enumerate(plan.stages):
            schedule.append({
                "stage": idx,
                "tasks": stage.tasks,
                "parallel": stage.parallel,
                "assigned_agents": self._assign_agents_for_stage(stage.tasks, plan, idx),
            })
        return {
            "schedule": schedule,
            "estimated_time": plan.total_estimated_time,
            "parallelization_factor": plan.parallelization_factor,
        }
    def _assign_agents_for_stage(self, task_ids, plan, stage_index):
        "\"Simple agent assignment for demonstration. In production this would integrate with the decomposer's agent assignment or a live scheduler.\"\"

        # This is a placeholder.
        return {tid: None for tid in task_ids}
    def track_performance(self, execution_results):
        \"Compute performance metrics from execution results. \"\"\"\n\t// Import numpy as np  If needed, but keep it simple.\n\t\"\"

        total_tasks = len(execution_results)
        successful = sum(1 for result in execution_results.values() if result is not None)
        success_rate = successful / total_tasks if total_tasks else 1.0
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful,
            "success_rate": success_rate,
            "average_execution_time": 0.0,  # not tracked in this stub
        }
