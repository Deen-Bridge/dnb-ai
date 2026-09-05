import uuid
from dataclasses import dataclass, field
typing [Any, Dict, Iterable, List, Optional, Set]
import time

@dataclass
SubTask:
    id: str
    description: str
    required_capabilities: Set[str]
    dependencies: List[str] = field(default_factory=list)
    estimated_complexity: float = 1.0
    status: str = "pending"
    assigned_agent: Optional[str] = None
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
DependencyGraph:
    tasks: Dict[str, SubTask] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=dict)
    def add_task(self, task):
        self.tasks[task.id] = task; self.edges[task.id] = set()
    def add_dependency(self, from_task, to_task):
        self.edges[from_task].add(to_task)
    def get_dependencies_of(self, task_id):
        return {p for p, s in self.edges.items() if task_id in s}
    def get_dependents_of(self, task_id):
        return self.edges.get(task_id, set())
    def topological_sort(self):
        in_deg = {t: len(self.get_dependencies_of(t)) for t in self.tasks}
        q = [t for t, d in in_deg.items() if d == 0]
        order = []
        while q:
            t = q.pop(0); order.append(t)
            for s in self.get_dependents_of(t):
                in_deg[s] -= 1
                if in_deg[s] == 0: q.append(s)
        if len(order) != len(self.tasks): raise RuntimeError("cycle")
        return order
    def get_ready_tasks(self, completed):
        ready = []
        for t in self.tasks:
            if t in completed: continue
            if self.get_dependencies_of(t).issubset(completed): ready.append(t)
        return ready

class Agent:
    def __init__(self, agent_id, capabilities, performance=1.0):
        self.agent_id = agent_id; self.capabilities = capabilities; self.performance = performance
    def can_execute(self, task):
        return task.required_capabilities.issubset(self.capabilities)

@dataclass
ExecutionPlan:
    task_order: List[str]
    parallel_groups: List[List[str]]
    estimated_duration: float

class QueryAnalyzer:
    def analyze(self, query):
        words = query.split()
        complexity = min(10.0, max(1.0, len(words) / 10.0))
        return {"complexity": complexity, "word_count": len(words), "raw": query}

class TaskDecomposer:
    def __init__(self, agents, analyzer=None, strategy=None, validation_threshold=0.9):
        self.agents = {a.agent_id: a for a in agents}
        self.analyzer = analyzer or QueryAnalyzer()
        self.strategy = strategy or self._default_strategy
        self.validation_threshold = validation_threshold
        self.metrics = {"decompositions": 0, "successful": 0, "parallel_opportunities": 0, "agent_assignments": {}, "total_execution_time": 0.0}

    def _default_strategy(self, query, analysis):
        return[SubTask(id=str(uuid.uuid4()), description=query, required_capabilities={"general"}, estimated_complexity=analysis.get("complexity", 1.0))]

    def decompose(self, query):
        start = time.time(); self.metrics["decompositions"] += 1
        try:
            analysis = self.analyzer.analyze(query)
            subtasks = self.strategy(query, analysis)
            if not subtasks: raise RuntimeError("no subtasks")
            validation_ok = self._validate(query, subtasks)
            if not validation_ok:
                subtasks = self._default_strategy(query, analysis)
            graph = DependencyGraph()
            for t in subtasks: graph.add_task(t)
            self._assign_agents(graph)
            plan = self._create_plan(graph)
            self.metrics["successful"] += 1
            if len(plan.parallel_groups) > 1: self.metrics["parallel_opportunities"] += 1
            self.metrics["total_execution_time"] += (time.time() - start) * 1000
            return {"query": query, "analysis": analysis, "tasks": graph.tasks, "graph": graph, "plan": plan, "metrics": self.metrics.copy()}
        except Exception as e:
            self.metrics["total_execution_time"] += (time.time() - start) * 1000
            return {"query": query, "error": str(e), "tasks": {}, "graph": None, "plan": None, "metrics": self.metrics.copy()}

    def _validate(self, query, subtasks):
        full = " ".join(t.description for t in subtasks).lower()
        qw = set(query.lower().split())
        if not qw: return True
        return len(qw & set(full.split())) / len(qw) >= self.validation_threshold

    def _assign_agents(self, graph):
        for t in graph.tasks.values():
            best = None; best_score = -1
            for a in self.agents.values():
                if not a.can_execute(t): continue
                score = a.performance * (1.0 / max(1.0, t.estimated_complexity))
                if score > best_score: best_score = score; best = a.agent_id
            if best is None:
                for a in self.agents.values():
                    if "general" in a.capabilities: best = a.agent_id; break
            if best is None: raise RuntimeError(f"no agent for {t.id}")
            t.assigned_agent = best; self.metrics["agent_assignments"][t.id] = best

    def _create_plan(self, graph):
        order = graph.topological_sort()
        completed = set(); groups = []; remaining = set(order)
        while remaining:
            ready = graph.get_ready_tasks(completed)
            group = [t for t in ready if t in remaining]
            if not group: raise RuntimeError("plan failure")
            groups.append(group); completed.update(group); remaining.difference_update(group)
        task_order = [t for g in groups for t in g]
        est = sum(graph.tasks[t].estimated_complexity for t in task_order)
        return ExecutionPlan(task_order, groups, est)

    def dynamic_replan(self, original, failed_task_id, new_tasks=None):
        graph = original.get("graph")
        if graph is None or failed_task_id not in graph.tasks: return original
        affected = {failed_task_id}; q = list(graph.get_dependents_of(failed_task_id))
        while q:
            t = q.pop(0)
            if t not in affected:
                affected.add(t)
                q.extend(graph.get_dependents_of(t))
        for t in affected:
            del graph.tasks[t]
            graph.edges.pop(t, None)
        for pred in list(graph.edges):
            graph.edges[pred] = {s for s in graph.edges[pred] if s not in affected}
        if new_tasks:
            for t in new_tasks: graph.add_task(t)
        self._assign_agents(graph)
        plan = self._create_plan(graph)
        return {"query": original["query"], "analysis": original["analysis"], "tasks": graph.tasks, "graph": graph, "plan": plan, "metrics": self.metrics.copy(), "replanned": True}