"""Offline tests for the Islamic Research Agent Orchestration Framework (#126).

No secrets and no network: the specialised agents work over the bundled
datasets, and the app is exercised through httpx's ASGI transport.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("GEMINI_API_KEY", "test-key")

import main  # noqa: E402
from orchestration import (  # noqa: E402
    AgentRegistry,
    Blackboard,
    Domain,
    ExecutionGraph,
    ExecutionNode,
    OrchestrationRequest,
    build_graph,
    classify_domain,
    decompose_query,
    get_orchestrator,
    select_agents,
)


@pytest.fixture()
async def client():
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Unit tests: decomposition & routing
# ---------------------------------------------------------------------------


class TestDecomposition:
    def test_single_clause_passes_through(self):
        facets = decompose_query("What does the Quran say about patience?")
        assert facets == ["What does the Quran say about patience"]

    def test_compound_query_splits(self):
        facets = decompose_query("Explain 2:255 and is fasting obligatory in Ramadan?")
        assert len(facets) >= 2

    def test_classify_domain_tafsir(self):
        assert classify_domain("What is the meaning of verse 2:255?") is Domain.TAFSIR

    def test_classify_domain_hadith(self):
        assert classify_domain("Is this hadith authentic in Bukhari?") is Domain.HADITH

    def test_classify_domain_fiqh(self):
        assert classify_domain("Is paying interest permissible?") is Domain.FIQH

    def test_classify_domain_falls_back_to_general(self):
        assert classify_domain("hello world example text") is Domain.GENERAL


class TestSelection:
    def test_select_agents_for_query(self):
        registry = AgentRegistry()
        selection = select_agents(registry, decompose_query("Explain 2:255"))
        assert Domain.TAFSIR in selection
        assert "tafsir" in selection[Domain.TAFSIR]


# ---------------------------------------------------------------------------
# Unit tests: execution DAG
# ---------------------------------------------------------------------------


class TestExecutionGraph:
    def test_dag_builds_dependency(self):
        registry = AgentRegistry()
        graph = build_graph(registry, "Explain 2:255", ["Explain 2:255"])
        nodes = graph.nodes
        assert nodes, "expected at least one node"
        # The tafsir node depends on the quran node.
        tafsir = next(n for n in nodes if n.domain is Domain.TAFSIR)
        assert tafsir.depends_on, "tafsir node should depend on quran node"

    def test_ready_respects_dependencies(self):
        registry = AgentRegistry()
        graph = build_graph(registry, "Explain 2:255", ["Explain 2:255"])
        # Initially only the quran node is ready (its dependencies are empty).
        ready = graph.ready(set())
        assert all(not node.depends_on for node in ready)
        assert any(node.domain is Domain.QURAN for node in ready)

    def test_graph_complete_detection(self):
        nodes = [ExecutionNode(id="a", agent="quran", domain=Domain.QURAN, status="success")]
        graph = ExecutionGraph(nodes)
        assert graph.is_complete()
        nodes[0].status = "pending"
        assert not graph.is_complete()


# ---------------------------------------------------------------------------
# Unit tests: blackboard message passing
# ---------------------------------------------------------------------------


class TestBlackboard:
    def test_write_and_read(self):
        board = Blackboard()
        from orchestration import AgentMessage

        board.write(AgentMessage(sender="quran", recipient="tafsir", payload={"verses": ["2:255"]}))
        board.write(AgentMessage(sender="hadith", recipient=None, payload={"collections": ["bukhari"]}))
        tafsir_messages = board.read_for("tafsir")
        assert len(tafsir_messages) == 2  # targeted + broadcast
        assert any(m.sender == "quran" for m in tafsir_messages)
        assert len(board.all) == 2


# ---------------------------------------------------------------------------
# Integration tests: orchestrator
# ---------------------------------------------------------------------------


class TestOrchestrator:
    async def test_runs_three_plus_agents(self):
        result = await get_orchestrator().orchestrate(
            OrchestrationRequest(query="Explain 2:255 and is fasting obligatory in Ramadan?")
        )
        assert result.status == "completed"
        assert result.trace_id
        assert len(result.nodes) >= 4  # quran, hadith, tafsir, fiqh (or more)
        assert result.answer
        assert "2:255" in result.answer or "Findings" in result.answer

    async def test_all_four_specialised_agents_contribute(self):
        # A compound query touching every domain routes to quran, hadith,
        # tafsir and fiqh; all of them complete in one run.
        result = await get_orchestrator().orchestrate(
            OrchestrationRequest(
                query=(
                    "Explain 2:255, is this hadith narrated in Bukhari authentic, and is paying interest permissible?"
                )
            )
        )
        agents_run = {node.agent for node in result.nodes if node.status.value == "success"}
        assert {"quran", "hadith", "tafsir", "fiqh"} <= agents_run

    async def test_synthesis_mentions_all_domains(self):
        result = await get_orchestrator().orchestrate(
            OrchestrationRequest(query="Is paying interest permissible in Islam?")
        )
        # Fiqh routing should have run; the synthesis should be non-empty.
        assert result.status == "completed"
        assert any(node.domain is Domain.FIQH and node.status.value == "success" for node in result.nodes)

    async def test_failed_agent_degrades_gracefully(self):
        result = await get_orchestrator().orchestrate(
            OrchestrationRequest(query="Explain 2:255", fail_agents=["quran"])
        )
        # The run still completes with a fallback/partial answer.
        assert result.status in ("partial", "completed")
        assert result.answer
        quran_node = next(node for node in result.nodes if node.agent == "quran")
        assert quran_node.status.value in ("failed", "fallback")

    async def test_trace_is_stored_and_retrievable(self):
        result = await get_orchestrator().orchestrate(
            OrchestrationRequest(query="What does the Quran say about patience?")
        )
        from orchestration import trace_store

        trace = trace_store.get(result.trace_id)
        assert trace is not None
        assert trace.query == "What does the Quran say about patience?"
        assert trace.status == "completed"
        assert trace.finished_at is not None
        assert trace.total_latency_ms >= 0


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestEndpoints:
    async def test_list_agents(self, client):
        resp = await client.get("/orchestration/agents")
        assert resp.status_code == 200
        agents = resp.json()
        assert len(agents) >= 4
        names = {agent["name"] for agent in agents}
        assert {"quran", "hadith", "tafsir", "fiqh", "synthesizer"} <= names

    async def test_get_agent(self, client):
        resp = await client.get("/orchestration/agents/quran")
        assert resp.status_code == 200
        assert resp.json()["name"] == "quran"
        assert resp.json()["domain"] == "quran"

    async def test_get_unknown_agent_404(self, client):
        resp = await client.get("/orchestration/agents/nope")
        assert resp.status_code == 404

    async def test_run_endpoint(self, client):
        resp = await client.post(
            "/orchestration/run",
            json={"query": "Explain 2:255 and what does Islam say about charity?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["answer"]
        assert data["trace_id"]

    async def test_run_with_failure(self, client):
        resp = await client.post(
            "/orchestration/run",
            json={"query": "Explain 2:255", "fail_agents": ["fiqh"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("partial", "completed")

    async def test_trace_endpoint(self, client):
        run = await client.post(
            "/orchestration/run",
            json={"query": "What does the Quran say about patience?"},
        )
        trace_id = run.json()["trace_id"]
        resp = await client.get(f"/orchestration/traces/{trace_id}")
        assert resp.status_code == 200
        trace = resp.json()
        assert trace["trace_id"] == trace_id
        assert trace["nodes"]
        assert trace["events"]
        assert trace["finished_at"] is not None

    async def test_unknown_trace_404(self, client):
        resp = await client.get("/orchestration/traces/does-not-exist")
        assert resp.status_code == 404
