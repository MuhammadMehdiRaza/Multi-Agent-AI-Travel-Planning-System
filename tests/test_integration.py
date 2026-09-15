"""
End-to-end tests against the live system.

Marked `integration` because they call Groq, the MCP servers and PostgreSQL.
They cost money and are excluded from CI. The behaviour that can be checked
without those services lives in test_logic.py, test_routing.py and test_api.py,
which is where most of the coverage is.

    pytest -m integration          run these
    pytest -m "not integration"    everything else, the CI default

Cases are spaced out because running them back to back exceeds the Groq free
tier's tokens-per-minute allowance. Set TEST_COOLDOWN=0 on a paid tier.
"""

import os
import time
import uuid

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

pytestmark = pytest.mark.integration

COOLDOWN = float(os.getenv("TEST_COOLDOWN", "35"))


@pytest.fixture(scope="module")
def app():
    from graph import get_app, shutdown

    try:
        yield get_app()
    finally:
        shutdown()


@pytest.fixture(autouse=True)
def cooldown():
    """Pause between cases so the suite does not spend its budget on retries."""
    yield
    if COOLDOWN:
        time.sleep(COOLDOWN)


def run(app, query: str):
    config = {"configurable": {"thread_id": "test_" + uuid.uuid4().hex[:8]}}
    state = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "user_id": "test_user",
            "user_query": query,
            "llm_calls": 0,
        },
        config=config,
    )
    return state, config


class TestGuardrail:
    def test_rejects_an_off_topic_request(self, app):
        state, _ = run(app, "Write me a Python function that reverses a linked list.")

        assert state.get("guardrail_blocked") is True
        assert state.get("guardrail_reason")
        assert state.get("selected_agents") == []
        assert not state.get("__interrupt__"), "a refused request must not reach approval"
        assert not state.get("itinerary")

    def test_rejects_empty_input_without_calling_the_model(self, app):
        state, _ = run(app, "   ")

        assert state.get("guardrail_blocked") is True
        assert state.get("llm_calls") == 0, "the length check must not charge a model call"


class TestDynamicRouting:
    def test_weather_only_request_runs_fewer_agents(self, app):
        state, _ = run(app, "What is the weather like in Tokyo right now?")
        selected = state.get("selected_agents", [])

        assert not state.get("guardrail_blocked")
        assert "weather_agent" in selected
        assert "itinerary_agent" in selected
        assert "flight_agent" not in selected
        assert not state.get("flight_results")
        assert state.get("weather_results")

    def test_full_trip_request_runs_several_and_suspends(self, app):
        state, config = run(
            app,
            "Plan a 7 day trip to Japan from Karachi under 2 lakh rupees, "
            "including flights, hotels and sightseeing.",
        )

        assert len(state.get("selected_agents", [])) >= 3
        assert state.get("flight_results")
        assert state.get("hotel_results")

        interrupts = state.get("__interrupt__")
        assert interrupts, "the graph must suspend for human approval"
        assert interrupts[0].value.get("draft_itinerary")
        assert not state.get("final_response")

        revised = app.invoke(
            Command(
                resume={
                    "approved": False,
                    "feedback": "Too rushed. Make it five days and cut one city.",
                }
            ),
            config=config,
        )

        assert not revised.get("__interrupt__")
        assert revised.get("approved") is False
        assert revised.get("human_feedback")
        assert revised.get("final_response")
        assert revised.get("final_response") != revised.get("itinerary")


class TestCheckpointing:
    def test_thread_survives_the_run(self, app):
        state, config = run(app, "What is the weather in Dubai?")
        snapshot = app.get_state(config)

        assert snapshot.created_at is not None
        assert snapshot.values.get("user_query")
        assert len(snapshot.values.get("messages", [])) >= 2


class TestMcpServers:
    def test_tools_are_discoverable(self):
        import asyncio

        from mcp_client import configured_servers, get_tools

        assert configured_servers(), "no MCP server is configured"
        tools = asyncio.run(get_tools())
        names = {tool.name for tool in tools}

        assert "tavily_search" in names
        assert "list_airports" in names
