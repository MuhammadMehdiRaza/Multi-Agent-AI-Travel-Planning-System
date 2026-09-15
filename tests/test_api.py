"""
API tests against a stubbed graph.

Covers the parts of api.py that are logic rather than orchestration: request
validation bounds, the two-phase guards, the interrupt-to-response flattening,
and the rule that internal exception text never reaches the client.

`get_app` is patched to a graph built from stub nodes, so nothing here touches
Groq, the MCP servers or PostgreSQL.
"""

import json
import operator
from typing import Annotated, TypedDict

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import api


class StubState(TypedDict, total=False):
    user_query: str
    guardrail_blocked: bool
    guardrail_reason: str
    selected_agents: list[str]
    supervisor_reasoning: str
    trip_constraints: dict
    itinerary: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str
    llm_calls: Annotated[int, operator.add]


def _stub_app():
    """
    A two-node graph that suspends for approval, like the real one.

    The node names match the real graph. api.py special-cases the node called
    "supervisor" to emit the routing decision as its own stream event, so a stub
    with different names would not exercise that path.
    """

    def supervisor(state: StubState):
        if "blocked" in state.get("user_query", ""):
            return {
                "guardrail_blocked": True,
                "guardrail_reason": "not a travel request",
                "final_response": "not a travel request",
                "selected_agents": [],
                "llm_calls": 1,
            }
        return {
            "selected_agents": ["weather_agent", "itinerary_agent"],
            "supervisor_reasoning": "weather only",
            "trip_constraints": {"destination": "Tokyo"},
            "itinerary": "draft plan",
            "approval_request": "approve?",
            "llm_calls": 3,
        }

    def human_approval(state: StubState):
        if state.get("guardrail_blocked"):
            return {}
        answer = interrupt({"question": "approve?", "draft_itinerary": "draft plan"})
        return {
            "approved": bool(answer.get("approved")),
            "human_feedback": str(answer.get("feedback", "")),
            "final_response": "final plan",
            "llm_calls": 1,
        }

    g = StateGraph(StubState)
    g.add_node("supervisor", supervisor)
    g.add_node("human_approval", human_approval)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: "blocked" if s.get("guardrail_blocked") else "human_approval",
        {"blocked": END, "human_approval": "human_approval"},
    )
    g.add_edge("human_approval", END)
    return g.compile(checkpointer=InMemorySaver())


@pytest.fixture
def client(monkeypatch):
    stub = _stub_app()
    monkeypatch.setattr(api, "get_app", lambda: stub)
    monkeypatch.setattr(api, "shutdown", lambda: None)
    with TestClient(api.server) as test_client:
        yield test_client


# -- Validation ---------------------------------------------------------------

class TestRequestValidation:
    def test_rejects_an_empty_query(self, client):
        response = client.post(
            "/api/plan", json={"query": "", "thread_id": "t"}
        )
        assert response.status_code == 422

    def test_rejects_an_oversized_query(self, client):
        # Unbounded input on an endpoint with no rate limit is a cost
        # amplification hole: each run spends about six model calls plus paid
        # third-party requests.
        response = client.post(
            "/api/plan", json={"query": "a" * 2001, "thread_id": "t"}
        )
        assert response.status_code == 422

    def test_accepts_a_query_at_the_limit(self, client):
        response = client.post(
            "/api/plan", json={"query": "a" * 2000, "thread_id": "limit"}
        )
        assert response.status_code == 200

    def test_rejects_an_empty_thread_id(self, client):
        assert client.post("/api/plan", json={"query": "q", "thread_id": ""}).status_code == 422

    def test_rejects_an_oversized_thread_id(self, client):
        response = client.post(
            "/api/plan", json={"query": "q", "thread_id": "t" * 129}
        )
        assert response.status_code == 422

    def test_rejects_oversized_feedback(self, client):
        response = client.post(
            "/api/approve",
            json={"thread_id": "t", "approved": False, "feedback": "x" * 4001},
        )
        assert response.status_code == 422


# -- The two-phase flow -------------------------------------------------------

class TestPlanAndApprove:
    def test_plan_returns_a_draft_and_awaits_approval(self, client):
        body = client.post(
            "/api/plan", json={"query": "Tokyo trip", "thread_id": "a"}
        ).json()

        assert body["awaiting_approval"] is True
        assert body["itinerary"] == "draft plan"
        assert body["approval_request"] == "approve?"
        assert body["final_response"] == ""
        assert body["selected_agents"] == ["weather_agent", "itinerary_agent"]
        assert body["trip_constraints"] == {"destination": "Tokyo"}

    def test_approve_completes_the_run(self, client):
        client.post("/api/plan", json={"query": "Tokyo trip", "thread_id": "b"})
        body = client.post(
            "/api/approve", json={"thread_id": "b", "approved": True}
        ).json()

        assert body["awaiting_approval"] is False
        assert body["approved"] is True
        assert body["final_response"] == "final plan"

    def test_rejection_carries_feedback(self, client):
        client.post("/api/plan", json={"query": "Tokyo trip", "thread_id": "c"})
        body = client.post(
            "/api/approve",
            json={"thread_id": "c", "approved": False, "feedback": "shorter please"},
        ).json()

        assert body["approved"] is False
        assert body["human_feedback"] == "shorter please"

    def test_replanning_a_suspended_thread_is_refused(self, client):
        # Without this guard LangGraph restarts from the beginning and destroys
        # the draft the user was asked to review, with no error.
        client.post("/api/plan", json={"query": "Tokyo trip", "thread_id": "d"})
        second = client.post("/api/plan", json={"query": "Rome trip", "thread_id": "d"})

        assert second.status_code == 409
        assert "waiting for your approval" in second.json()["detail"]

    def test_the_original_draft_survives_a_refused_replan(self, client):
        client.post("/api/plan", json={"query": "Tokyo trip", "thread_id": "e"})
        client.post("/api/plan", json={"query": "Rome trip", "thread_id": "e"})
        body = client.get("/api/thread/e").json()

        assert body["awaiting_approval"] is True
        assert body["itinerary"] == "draft plan"

    def test_replanning_is_allowed_once_the_run_is_finished(self, client):
        client.post("/api/plan", json={"query": "Tokyo trip", "thread_id": "f"})
        client.post("/api/approve", json={"thread_id": "f", "approved": True})
        again = client.post("/api/plan", json={"query": "Rome trip", "thread_id": "f"})

        assert again.status_code == 200

    def test_approving_a_thread_with_no_pending_run_is_refused(self, client):
        response = client.post(
            "/api/approve", json={"thread_id": "never_planned", "approved": True}
        )
        assert response.status_code == 409

    def test_approving_twice_is_refused(self, client):
        client.post("/api/plan", json={"query": "Tokyo trip", "thread_id": "g"})
        client.post("/api/approve", json={"thread_id": "g", "approved": True})
        second = client.post("/api/approve", json={"thread_id": "g", "approved": True})

        assert second.status_code == 409


class TestStreamingPlan:
    """
    The streamed endpoint is what makes the pipeline display honest. Before it,
    the UI showed all five agents as running for the whole request, including the
    ones the supervisor had skipped.
    """

    @staticmethod
    def _events(client, **body):
        with client.stream("POST", "/api/plan/stream", json=body) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("application/x-ndjson")
            return [
                json.loads(line) for line in response.iter_lines() if line.strip()
            ]

    def test_emits_the_supervisor_decision_before_the_result(self, client):
        events = self._events(client, query="Tokyo trip", thread_id="s1")
        kinds = [event["type"] for event in events]

        assert kinds[0] == "supervisor"
        assert kinds[-1] == "done"

    def test_supervisor_event_carries_the_routing_decision(self, client):
        events = self._events(client, query="Tokyo trip", thread_id="s2")
        supervisor = events[0]

        assert supervisor["selected_agents"] == ["weather_agent", "itinerary_agent"]
        assert supervisor["trip_constraints"] == {"destination": "Tokyo"}
        assert supervisor["blocked"] is False

    def test_final_event_matches_the_non_streaming_shape(self, client):
        events = self._events(client, query="Tokyo trip", thread_id="s3")
        result = events[-1]["result"]
        reference = client.post(
            "/api/plan", json={"query": "Tokyo trip", "thread_id": "s3_ref"}
        ).json()

        assert set(result) == set(reference)
        assert result["awaiting_approval"] is True
        assert result["itinerary"] == "draft plan"

    def test_interrupt_is_not_leaked_as_a_node_event(self, client):
        # __interrupt__ arrives as its own stream chunk and is not a node.
        events = self._events(client, query="Tokyo trip", thread_id="s4")
        nodes = [event.get("node") for event in events if event["type"] == "node"]

        assert "__interrupt__" not in nodes

    def test_blocked_request_reports_through_the_supervisor_event(self, client):
        events = self._events(client, query="blocked topic", thread_id="s5")
        supervisor = events[0]

        assert supervisor["blocked"] is True
        assert supervisor["blocked_reason"] == "not a travel request"
        assert events[-1]["result"]["awaiting_approval"] is False

    def test_replanning_a_suspended_thread_streams_an_error(self, client):
        client.post("/api/plan", json={"query": "Tokyo trip", "thread_id": "s6"})
        events = self._events(client, query="Rome trip", thread_id="s6")

        assert events[-1]["type"] == "error"
        assert "waiting for your approval" in events[-1]["detail"]

    def test_a_failing_run_streams_an_error_without_internal_detail(self, client, monkeypatch):
        class Boom:
            def get_state(self, _config):
                raise RuntimeError("password=hunter2")

        monkeypatch.setattr(api, "get_app", Boom)
        events = self._events(client, query="Tokyo trip", thread_id="s7")

        assert events[-1]["type"] == "error"
        assert "hunter2" not in events[-1]["detail"]

    def test_stream_still_enforces_input_bounds(self, client):
        response = client.post(
            "/api/plan/stream", json={"query": "a" * 2001, "thread_id": "s8"}
        )
        assert response.status_code == 422


class TestBlockedRequests:
    def test_blocked_request_reports_the_reason(self, client):
        body = client.post(
            "/api/plan", json={"query": "blocked topic", "thread_id": "h"}
        ).json()

        assert body["blocked"] is True
        assert body["blocked_reason"] == "not a travel request"
        assert body["awaiting_approval"] is False
        assert body["itinerary"] == ""

    def test_blocked_request_costs_one_model_call(self, client):
        body = client.post(
            "/api/plan", json={"query": "blocked topic", "thread_id": "i"}
        ).json()
        assert body["llm_calls"] == 1


class TestThreadReadback:
    def test_unknown_thread_is_a_404(self, client):
        assert client.get("/api/thread/does_not_exist").status_code == 404

    def test_readback_recovers_a_pending_draft(self, client):
        client.post("/api/plan", json={"query": "Tokyo trip", "thread_id": "j"})
        body = client.get("/api/thread/j").json()

        assert body["awaiting_approval"] is True
        assert body["itinerary"] == "draft plan"

    def test_readback_after_completion(self, client):
        client.post("/api/plan", json={"query": "Tokyo trip", "thread_id": "k"})
        client.post("/api/approve", json={"thread_id": "k", "approved": True})
        body = client.get("/api/thread/k").json()

        assert body["awaiting_approval"] is False
        assert body["final_response"] == "final plan"


# -- Error handling and health ------------------------------------------------

class TestErrorSafety:
    def test_client_error_hides_internal_detail(self):
        # psycopg errors carry the database host and user, and httpx errors carry
        # the full request URL, which for Tavily includes the API key.
        exc = RuntimeError(
            "connection to server at 10.0.0.5, user=admin failed; "
            "url=https://mcp.tavily.com/?tavilyApiKey=tvly-SECRET"
        )
        detail = api._client_error(exc)

        assert "tvly-SECRET" not in detail
        assert "10.0.0.5" not in detail
        assert "admin" not in detail
        assert "RuntimeError" in detail

    def test_a_failing_graph_returns_500_not_a_traceback(self, client, monkeypatch):
        class Boom:
            def get_state(self, _config):
                raise RuntimeError("password=hunter2")

        monkeypatch.setattr(api, "get_app", Boom)
        response = client.post("/api/plan", json={"query": "q", "thread_id": "z"})

        assert response.status_code == 500
        assert "hunter2" not in response.json()["detail"]


class TestHealth:
    def test_root(self, client):
        assert client.get("/").status_code == 200

    def test_health_reports_model_and_server_status(self, client):
        body = client.get("/api/health").json()

        assert body["status"] == "ok"
        assert "model" in body
        assert isinstance(body["mcp_servers_ready"], list)
        assert isinstance(body["mcp_servers_unavailable"], dict)


# -- Response flattening ------------------------------------------------------

class TestToResponse:
    def test_reads_the_draft_out_of_the_interrupt_payload(self):
        class FakeInterrupt:
            value = {"draft_itinerary": "from payload", "approval_request": "ask"}

        response = api._to_response("t", {"__interrupt__": [FakeInterrupt()]})

        # The itinerary node has not returned yet, so state has no itinerary. The
        # payload is the only place the draft exists at this point.
        assert response.awaiting_approval is True
        assert response.itinerary == "from payload"
        assert response.approval_request == "ask"

    def test_falls_back_to_state_when_not_suspended(self):
        response = api._to_response("t", {"itinerary": "from state"})

        assert response.awaiting_approval is False
        assert response.itinerary == "from state"

    def test_tolerates_an_empty_interrupt_payload(self):
        class FakeInterrupt:
            value = None

        response = api._to_response("t", {"__interrupt__": [FakeInterrupt()], "itinerary": "s"})
        assert response.itinerary == "s"

    def test_missing_keys_default_rather_than_raise(self):
        response = api._to_response("t", {})

        assert response.thread_id == "t"
        assert response.selected_agents == []
        assert response.trip_constraints == {}
        assert response.llm_calls == 0
