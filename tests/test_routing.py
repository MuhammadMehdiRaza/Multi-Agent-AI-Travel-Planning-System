"""
Routing tests over the real graph with stub nodes.

This is the suite that protects the project's central claim. Dynamic routing is
the thing that makes this multi-agent rather than a fixed pipeline, and a
regression that quietly reverted it to a chain would still return a plausible
itinerary, so no output-based check would catch it.

Every test here runs the real edges and the real routing functions. Only the node
bodies are replaced, so there is no model, no MCP server and no database. An
in-memory checkpointer covers the interrupt and resume path.
"""

import operator
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command, interrupt

from agents import DATA_AGENTS, MAX_RESEARCH_STEPS
from graph import (
    POST_DATA_ROUTES,
    RESEARCH_ROUTES,
    SUPERVISOR_ROUTES,
    build_graph,
    route_after_data,
    route_after_research,
    route_from_supervisor,
)

ALL_AGENTS = [*DATA_AGENTS, "research_agent", "budget_agent", "itinerary_agent"]


# -- The routing functions in isolation ---------------------------------------

class TestRouteFromSupervisor:
    def test_blocked_request_exits_the_graph(self):
        assert route_from_supervisor({"guardrail_blocked": True}) == "blocked"
        assert SUPERVISOR_ROUTES["blocked"] is END

    def test_blocked_wins_even_if_agents_were_selected(self):
        state = {"guardrail_blocked": True, "selected_agents": ALL_AGENTS}
        assert route_from_supervisor(state) == "blocked"

    def test_fans_out_every_selected_data_agent(self):
        state = {"selected_agents": ALL_AGENTS}
        assert route_from_supervisor(state) == DATA_AGENTS

    def test_fan_out_is_a_list_not_a_single_name(self):
        # Returning a list is what makes the three data agents parallel. A string
        # here would silently restore sequential execution.
        result = route_from_supervisor({"selected_agents": ALL_AGENTS})
        assert isinstance(result, list)
        assert len(result) == 3

    def test_partial_fan_out(self):
        state = {"selected_agents": ["flight_agent", "weather_agent", "itinerary_agent"]}
        assert route_from_supervisor(state) == ["flight_agent", "weather_agent"]

    def test_single_data_agent_still_returns_a_list(self):
        state = {"selected_agents": ["weather_agent", "itinerary_agent"]}
        assert route_from_supervisor(state) == ["weather_agent"]

    def test_budget_only_skips_straight_to_budget(self):
        state = {"selected_agents": ["budget_agent", "itinerary_agent"]}
        assert route_from_supervisor(state) == "budget_agent"

    def test_itinerary_only(self):
        assert route_from_supervisor({"selected_agents": ["itinerary_agent"]}) == (
            "itinerary_agent"
        )

    @pytest.mark.parametrize("state", [{}, {"selected_agents": None}, {"selected_agents": []}])
    def test_empty_selection_still_produces_a_plan(self, state):
        assert route_from_supervisor(state) == "itinerary_agent"

    def test_every_possible_return_is_a_declared_destination(self):
        # A routing function returning a key absent from the map is a runtime
        # error, so the two must be kept in step.
        for agent in ALL_AGENTS:
            assert agent in SUPERVISOR_ROUTES
        assert "blocked" in SUPERVISOR_ROUTES


class TestRouteAfterData:
    def test_goes_to_budget_when_selected(self):
        state = {"selected_agents": ["flight_agent", "budget_agent", "itinerary_agent"]}
        assert route_after_data(state) == "budget_agent"

    def test_skips_budget_when_not_selected(self):
        state = {"selected_agents": ["flight_agent", "itinerary_agent"]}
        assert route_after_data(state) == "itinerary_agent"

    def test_all_three_data_agents_converge_on_the_same_node(self):
        # They must agree, otherwise the branches do not meet and the downstream
        # node runs more than once.
        state = {"selected_agents": [*DATA_AGENTS, "budget_agent", "itinerary_agent"]}
        assert len({route_after_data(state) for _ in DATA_AGENTS}) == 1

    def test_returns_only_declared_destinations(self):
        for selected in ([], ["budget_agent"], ALL_AGENTS):
            assert route_after_data({"selected_agents": selected}) in POST_DATA_ROUTES


# -- The whole graph, with stub nodes -----------------------------------------

class TrackedState(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    user_query: str
    guardrail_blocked: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict
    supervisor_reasoning: str
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    research_results: str
    research_messages: Annotated[list, operator.add]
    research_steps: Annotated[int, operator.add]
    itinerary: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str
    llm_calls: Annotated[int, operator.add]
    visited: Annotated[list[str], operator.add]


def _stub_nodes(plan: dict[str, Any], record: list[str]) -> dict:
    """
    Stand-ins for the ten real nodes.

    The supervisor returns `plan` verbatim, so a test states the routing decision
    directly instead of hoping a model produces it. Every other node records that
    it ran and writes the field it owns.
    """

    def supervisor(_state):
        record.append("supervisor")
        return {**plan, "visited": ["supervisor"], "llm_calls": 2}

    def make(name: str, field: str, calls: int = 1):
        def node(_state):
            record.append(name)
            return {
                field: name + " output",
                "visited": [name],
                "llm_calls": calls,
                "messages": [AIMessage(content=name)],
            }
        return node

    def itinerary(_state):
        record.append("itinerary_agent")
        return {
            "itinerary": "draft plan",
            "approval_request": "approve?",
            "visited": ["itinerary_agent"],
            "llm_calls": 1,
        }

    def approval(_state):
        record.append("human_approval")
        answer = interrupt({"question": "approve?", "draft_itinerary": "draft plan"})
        return {
            "approved": bool(answer.get("approved")),
            "human_feedback": str(answer.get("feedback", "")),
            "visited": ["human_approval"],
        }

    def final(state):
        record.append("final_response")
        verdict = "approved" if state.get("approved") else "revised"
        return {
            "final_response": "final plan (" + verdict + ")",
            "visited": ["final_response"],
            "llm_calls": 1,
        }

    def research(state):
        """
        Stands in for the tool-choosing agent.

        `research_loops` in the plan says how many tool rounds to request. The
        stub returns a message carrying tool_calls, which is exactly what
        route_after_research inspects, so the real loop condition is exercised.
        """
        record.append("research_agent")
        wanted = int(plan.get("research_loops", 0))
        spent = state.get("research_steps", 0)

        if spent < wanted:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tavily_search",
                        "args": {"query": "visa rules"},
                        "id": "call_" + str(spent),
                        "type": "tool_call",
                    }
                ],
            )
            return {
                "research_messages": [message],
                "research_steps": 1,
                "visited": ["research_agent"],
                "llm_calls": 1,
            }

        return {
            "research_messages": [AIMessage(content="findings")],
            "research_steps": 1,
            "research_results": "research findings",
            "visited": ["research_agent"],
            "llm_calls": 1,
        }

    def research_tools(_state):
        record.append("research_tools")
        return {
            "research_messages": [
                ToolMessage(content="tool output", tool_call_id="call_x", name="tavily_search")
            ],
            "visited": ["research_tools"],
        }

    return {
        "supervisor": supervisor,
        "flight_agent": make("flight_agent", "flight_results"),
        "hotel_agent": make("hotel_agent", "hotel_results", calls=0),
        "weather_agent": make("weather_agent", "weather_results", calls=0),
        "research_agent": research,
        "research_tools": research_tools,
        "budget_agent": make("budget_agent", "budget_results"),
        "itinerary_agent": itinerary,
        "human_approval": approval,
        "final_response": final,
    }


@pytest.fixture
def run_plan():
    """Run the real graph over stub nodes and return (result, visited, app, config)."""
    counter = {"n": 0}

    def _run(plan: dict[str, Any]):
        counter["n"] += 1
        record: list[str] = []
        app = build_graph(InMemorySaver(), nodes=_stub_nodes(plan, record))
        config = {"configurable": {"thread_id": "t" + str(counter["n"])}}
        result = app.invoke(
            {"user_query": "q", "llm_calls": 0, "visited": [], "messages": []},
            config=config,
        )
        return result, record, app, config

    return _run


class TestGraphRouting:
    def test_full_trip_runs_every_specialist_once(self, run_plan):
        result, visited, _, _ = run_plan({"selected_agents": ALL_AGENTS})

        for agent in ALL_AGENTS:
            assert visited.count(agent) == 1, agent + " ran " + str(visited.count(agent)) + " times"
        assert result.get("__interrupt__")

    def test_data_agents_converge_on_budget_exactly_once(self, run_plan):
        # This is the fan-out's real risk: three branches meeting at one node.
        _, visited, _, _ = run_plan({"selected_agents": ALL_AGENTS})
        assert visited.count("budget_agent") == 1
        assert visited.count("itinerary_agent") == 1

    def test_data_agents_run_before_budget(self, run_plan):
        _, visited, _, _ = run_plan({"selected_agents": ALL_AGENTS})
        for agent in DATA_AGENTS:
            assert visited.index(agent) < visited.index("budget_agent")

    def test_weather_only_request_skips_three_agents(self, run_plan):
        result, visited, _, _ = run_plan(
            {"selected_agents": ["weather_agent", "itinerary_agent"]}
        )
        assert "weather_agent" in visited
        assert "flight_agent" not in visited
        assert "hotel_agent" not in visited
        assert "budget_agent" not in visited
        assert not result.get("flight_results")
        assert not result.get("hotel_results")

    def test_two_of_three_data_agents(self, run_plan):
        _, visited, _, _ = run_plan(
            {"selected_agents": ["flight_agent", "weather_agent", "itinerary_agent"]}
        )
        assert "flight_agent" in visited
        assert "weather_agent" in visited
        assert "hotel_agent" not in visited

    def test_budget_without_any_data_agent(self, run_plan):
        _, visited, _, _ = run_plan(
            {"selected_agents": ["budget_agent", "itinerary_agent"]}
        )
        assert visited == ["supervisor", "budget_agent", "itinerary_agent", "human_approval"]

    def test_itinerary_only(self, run_plan):
        _, visited, _, _ = run_plan({"selected_agents": ["itinerary_agent"]})
        assert visited == ["supervisor", "itinerary_agent", "human_approval"]

    def test_blocked_request_stops_after_the_supervisor(self, run_plan):
        result, visited, _, _ = run_plan(
            {
                "guardrail_blocked": True,
                "guardrail_reason": "not a travel request",
                "selected_agents": [],
                "final_response": "not a travel request",
            }
        )
        assert visited == ["supervisor"]
        assert not result.get("__interrupt__")
        assert not result.get("itinerary")
        assert result["final_response"] == "not a travel request"

    def test_blocked_request_never_reaches_approval(self, run_plan):
        # The bug this guards: routing a refused request onward would suspend the
        # graph at the approval node and ask a human to review nothing.
        _, visited, _, _ = run_plan(
            {"guardrail_blocked": True, "selected_agents": [], "final_response": "no"}
        )
        assert "human_approval" not in visited
        assert "itinerary_agent" not in visited


class TestApprovalFlow:
    def test_graph_suspends_before_the_final_node(self, run_plan):
        result, visited, _, _ = run_plan({"selected_agents": ALL_AGENTS})
        assert result.get("__interrupt__")
        assert "final_response" not in visited
        assert not result.get("final_response")

    def test_interrupt_payload_carries_the_draft(self, run_plan):
        result, _, _, _ = run_plan({"selected_agents": ["itinerary_agent"]})
        assert result["__interrupt__"][0].value["draft_itinerary"] == "draft plan"

    def test_resume_with_approval_completes_the_run(self, run_plan):
        _, visited, app, config = run_plan({"selected_agents": ["itinerary_agent"]})
        result = app.invoke(
            Command(resume={"approved": True, "feedback": ""}), config=config
        )
        assert not result.get("__interrupt__")
        assert result["approved"] is True
        assert result["final_response"] == "final plan (approved)"
        assert "final_response" in visited

    def test_resume_with_rejection_carries_feedback_through(self, run_plan):
        _, _, app, config = run_plan({"selected_agents": ["itinerary_agent"]})
        result = app.invoke(
            Command(resume={"approved": False, "feedback": "make it five days"}),
            config=config,
        )
        assert result["approved"] is False
        assert result["human_feedback"] == "make it five days"
        assert result["final_response"] == "final plan (revised)"

    def test_checkpoint_survives_between_the_two_calls(self, run_plan):
        result, _, app, config = run_plan({"selected_agents": ALL_AGENTS})
        snapshot = app.get_state(config)
        assert snapshot.next == ("human_approval",)
        assert snapshot.values["itinerary"] == "draft plan"
        app.invoke(Command(resume={"approved": True}), config=config)
        assert app.get_state(config).next == ()


class TestLlmCallAccounting:
    def test_counter_sums_across_parallel_branches(self, run_plan):
        # A plain int here raises InvalidUpdateError the moment two parallel
        # branches write it, which is why state.llm_calls is a reducer.
        result, _, _, _ = run_plan({"selected_agents": ALL_AGENTS})
        # supervisor 2 + flight 1 + hotel 0 + weather 0 + research 1 + budget 1
        # + itinerary 1
        assert result["llm_calls"] == 6

    def test_counter_reflects_the_narrower_route(self, run_plan):
        result, _, _, _ = run_plan({"selected_agents": ["weather_agent", "itinerary_agent"]})
        # supervisor 2 + weather 0 + itinerary 1
        assert result["llm_calls"] == 3

    def test_resume_adds_the_final_call(self, run_plan):
        _, _, app, config = run_plan({"selected_agents": ["itinerary_agent"]})
        before = app.get_state(config).values["llm_calls"]
        result = app.invoke(Command(resume={"approved": True}), config=config)
        assert result["llm_calls"] == before + 1


class TestRouteAfterResearch:
    """The back edge that makes this the only cycle in the graph."""

    @staticmethod
    def _state(tool_calls: bool, steps: int, selected=None):
        message = AIMessage(
            content="",
            tool_calls=(
                [{"name": "tavily_search", "args": {}, "id": "c1", "type": "tool_call"}]
                if tool_calls
                else []
            ),
        )
        return {
            "research_messages": [message],
            "research_steps": steps,
            "selected_agents": selected if selected is not None else ALL_AGENTS,
        }

    def test_loops_when_a_tool_was_requested_and_budget_remains(self):
        assert route_after_research(self._state(True, 0)) == "research_tools"

    def test_leaves_when_no_tool_was_requested(self):
        assert route_after_research(self._state(False, 1)) == "budget_agent"

    def test_leaves_once_the_step_budget_is_spent(self):
        # Without this the loop would run until the recursion limit.
        state = self._state(True, MAX_RESEARCH_STEPS)
        assert route_after_research(state) == "budget_agent"

    def test_skips_budget_on_the_way_out_when_not_selected(self):
        state = self._state(False, 1, selected=["research_agent", "itinerary_agent"])
        assert route_after_research(state) == "itinerary_agent"

    def test_empty_history_does_not_raise(self):
        assert route_after_research({"selected_agents": ALL_AGENTS}) == "budget_agent"

    def test_returns_only_declared_destinations(self):
        for calls in (True, False):
            for steps in (0, MAX_RESEARCH_STEPS):
                assert route_after_research(self._state(calls, steps)) in RESEARCH_ROUTES


class TestResearchLoop:
    def test_no_tool_request_means_no_loop(self, run_plan):
        _, visited, _, _ = run_plan(
            {"selected_agents": ALL_AGENTS, "research_loops": 0}
        )
        assert visited.count("research_agent") == 1
        assert "research_tools" not in visited

    def test_one_tool_request_goes_round_once(self, run_plan):
        _, visited, _, _ = run_plan(
            {"selected_agents": ALL_AGENTS, "research_loops": 1}
        )
        assert visited.count("research_tools") == 1
        assert visited.count("research_agent") == 2

    def test_two_tool_requests_go_round_twice(self, run_plan):
        _, visited, _, _ = run_plan(
            {"selected_agents": ALL_AGENTS, "research_loops": 2}
        )
        assert visited.count("research_tools") == 2
        assert visited.count("research_agent") == 3

    def test_the_step_budget_stops_a_runaway_loop(self, run_plan):
        # The stub asks for a tool every single pass. Only the budget can end it.
        result, visited, _, _ = run_plan(
            {"selected_agents": ALL_AGENTS, "research_loops": 99}
        )
        # MAX counts model passes and the last is reserved for the answer, so
        # the agent gets one fewer round of tool calls than that.
        assert visited.count("research_tools") == MAX_RESEARCH_STEPS - 1
        assert result.get("__interrupt__"), "the run must still reach approval"

    def test_loop_output_reaches_the_plan(self, run_plan):
        result, _, _, _ = run_plan(
            {"selected_agents": ALL_AGENTS, "research_loops": 1}
        )
        assert result["research_results"] == "research findings"

    def test_tool_results_accumulate_in_the_private_scratchpad(self, run_plan):
        result, _, _, _ = run_plan(
            {"selected_agents": ALL_AGENTS, "research_loops": 2}
        )
        history = result["research_messages"]
        tool_messages = [m for m in history if isinstance(m, ToolMessage)]

        # The model must see what its earlier calls returned in order to decide
        # what to do next, which is why this key accumulates.
        assert len(tool_messages) == 2

    def test_research_can_run_without_any_data_agent(self, run_plan):
        _, visited, _, _ = run_plan(
            {"selected_agents": ["research_agent", "itinerary_agent"], "research_loops": 1}
        )
        assert visited[:2] == ["supervisor", "research_agent"]
        assert "flight_agent" not in visited

    def test_research_is_skipped_when_not_selected(self, run_plan):
        _, visited, _, _ = run_plan(
            {"selected_agents": [*DATA_AGENTS, "itinerary_agent"]}
        )
        assert "research_agent" not in visited
        assert "research_tools" not in visited

    def test_research_runs_after_the_fan_out_joins(self, run_plan):
        # Its branch length is unpredictable, so it must not be inside the
        # fan-out: a cyclic branch there makes the join node run twice.
        _, visited, _, _ = run_plan(
            {"selected_agents": ALL_AGENTS, "research_loops": 1}
        )
        for agent in DATA_AGENTS:
            assert visited.index(agent) < visited.index("research_agent")
        assert visited.count("budget_agent") == 1


def test_graph_compiles_without_a_checkpointer():
    # The Mermaid diagram and any structural check must not need a database.
    assert build_graph() is not None


def test_graph_exposes_the_expected_nodes():
    nodes = set(build_graph().get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {
        "supervisor",
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "research_agent",
        "research_tools",
        "budget_agent",
        "itinerary_agent",
        "human_approval",
        "final_response",
    }
