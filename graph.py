"""
graph.py - Graph assembly, dynamic routing, and the PostgreSQL checkpointer.

The shape of this graph is what makes the system multi-agent rather than a fixed
pipeline. Nothing is hard-wired. The supervisor decides which specialists a
request needs, and the conditional edges follow that decision:

    START -> supervisor -> (blocked) ------------------------------------> END
                        -> flight_agent  ┐
                        -> hotel_agent   ├ in parallel, one superstep
                        -> weather_agent ┘
                             -> budget_agent (if selected)
                                  -> itinerary -> human_approval -> final -> END

The three data-gathering specialists fan out. They read only `user_query` and
`trip_constraints`, so they are genuinely independent, and running them
sequentially cost the full sum of three network round trips for no reason. The
supervisor's conditional edge returns a *list* of node names, which LangGraph
schedules in a single superstep. Their outgoing edges then converge on one
downstream node, which runs once rather than once per branch.

Two things this required:

  - `llm_calls` had to become a reducer in state.py. Two parallel nodes writing a
    plain key in the same superstep raises InvalidUpdateError.
  - The old AGENT_ORDER walk, which returned one node name at a time, is gone
    along with the per-node route maps it needed.

Nothing here runs at import. `get_app()` builds the graph and opens the database
pool on first use, so this module can be imported, and the graph built with stub
nodes, without credentials or a running PostgreSQL.
"""

import logging
from collections.abc import Hashable

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from agents import (
    DATA_AGENTS,
    MAX_RESEARCH_STEPS,
    budget_agent,
    final_response_agent,
    flight_agent,
    hotel_agent,
    human_approval_agent,
    itinerary_agent,
    research_agent,
    research_tools_agent,
    supervisor_agent,
    weather_agent,
)
from config import DATABASE_URL
from state import TravelState

log = logging.getLogger(__name__)

# Node name -> implementation. Passing a replacement dict to build_graph is how
# the tests exercise routing without calling a model.
NODES = {
    "supervisor": supervisor_agent,
    "flight_agent": flight_agent,
    "hotel_agent": hotel_agent,
    "weather_agent": weather_agent,
    "research_agent": research_agent,
    "research_tools": research_tools_agent,
    "budget_agent": budget_agent,
    "itinerary_agent": itinerary_agent,
    "human_approval": human_approval_agent,
    "final_response": final_response_agent,
}

# Destinations the supervisor's edge can return. "blocked" maps to END so a
# request the guardrail refused stops immediately, instead of running the
# itinerary and approval nodes over an empty state.
SUPERVISOR_ROUTES: dict[Hashable, str] = {
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "research_agent": "research_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
    "blocked": END,
}

# Where a data specialist can go once the fan-out joins.
POST_DATA_ROUTES: dict[Hashable, str] = {
    "research_agent": "research_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}

# The research agent's own exits. "research_tools" is the back edge that makes
# this the one cycle in the graph.
RESEARCH_ROUTES: dict[Hashable, str] = {
    "research_tools": "research_tools",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}

# Retry the nodes that call the model.
#
# Without this, one transient Groq failure at the itinerary or final node threw
# away every call the run had already paid for. LangGraph retries the node from
# its own checkpoint, so completed nodes are not re-run and nothing is charged
# twice. The approval node is deliberately excluded: retrying an interrupt would
# re-ask the human.
LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)

RETRY_NODES = {
    "supervisor",
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "research_agent",
    "budget_agent",
    "itinerary_agent",
    "final_response",
}


# -- Routing ------------------------------------------------------------------

def _selected(state: TravelState) -> list[str]:
    return list(state.get("selected_agents") or [])


def _next_after_gathering(state: TravelState) -> str:
    """The first non-gathering stage the supervisor selected."""
    return "budget_agent" if "budget_agent" in _selected(state) else "itinerary_agent"


def route_from_supervisor(state: TravelState) -> str | list[str]:
    """
    Leave the supervisor for every selected data specialist at once.

    Returning a list is the fan-out. When no data agent was selected, fall
    through to whichever later stage was, so a budget-only or itinerary-only
    request still reaches the node that answers it.
    """
    if state.get("guardrail_blocked"):
        return "blocked"

    selected = _selected(state)

    parallel = [agent for agent in DATA_AGENTS if agent in selected]
    if parallel:
        return parallel

    # No data agent selected, so fall through to the first later stage that was.
    # research_agent has to be checked here too: without it a request needing
    # only research skipped straight past the agent that would have answered it.
    if "research_agent" in selected:
        return "research_agent"

    return _next_after_gathering(state)


def route_after_data(state: TravelState) -> str:
    """
    Where the data specialists converge.

    All three return the same answer, so they meet at one node and LangGraph runs
    it once. The research agent comes next when selected, then budget, which
    reasons over everything gathered. The itinerary node is the guaranteed
    terminus either way.
    """
    if "research_agent" in _selected(state):
        return "research_agent"

    return _next_after_gathering(state)


def route_after_research(state: TravelState) -> str:
    """
    Close the research loop, or leave it.

    The agent goes round again whenever it asked for a tool and has budget left.
    The budget is also enforced inside the node, by invoking the model with no
    tools bound on the final pass, so the loop cannot spin against the recursion
    limit even if this check were wrong.
    """
    history = state.get("research_messages") or []
    last = history[-1] if history else None

    requested = getattr(last, "tool_calls", None) or []
    spent = state.get("research_steps", 0)

    if requested and spent < MAX_RESEARCH_STEPS:
        return "research_tools"

    return _next_after_gathering(state)


# -- Assembly -----------------------------------------------------------------

def build_graph(checkpointer=None, nodes: dict | None = None):
    """
    Wire the nodes and edges and compile.

    `nodes` overrides the implementations by name. That is how the routing tests
    run the real edges over stub nodes, with no model, no MCP servers and an
    in-memory checkpointer.
    """
    implementations = {**NODES, **(nodes or {})}

    graph = StateGraph(TravelState)

    for name, node in implementations.items():
        if name in RETRY_NODES:
            graph.add_node(name, node, retry_policy=LLM_RETRY)
        else:
            graph.add_node(name, node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, SUPERVISOR_ROUTES)

    for name in DATA_AGENTS:
        graph.add_conditional_edges(name, route_after_data, POST_DATA_ROUTES)

    graph.add_conditional_edges("research_agent", route_after_research, RESEARCH_ROUTES)
    graph.add_edge("research_tools", "research_agent")

    graph.add_edge("budget_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "human_approval")
    graph.add_edge("human_approval", "final_response")
    graph.add_edge("final_response", END)

    return graph.compile(checkpointer=checkpointer)


# -- Checkpointer and application factory -------------------------------------

_app = None
_pool: ConnectionPool[Connection[DictRow]] | None = None


def build_checkpointer() -> PostgresSaver | None:
    """
    Open the PostgreSQL checkpointer that backs memory and human-in-the-loop.

    A connection pool rather than a single connection, because FastAPI serves
    requests from a thread pool and psycopg connections are not safe to share
    across threads. autocommit is required by PostgresSaver, and disabling
    prepared statements keeps it working through external connection poolers.
    """
    global _pool

    if not DATABASE_URL:
        log.warning(
            "DATABASE_URL is not set. Running without checkpoints, so approval "
            "cannot be resumed and a suspended run cannot be recovered."
        )
        return None

    _pool = ConnectionPool[Connection[DictRow]](
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=10,
        open=True,
        # row_factory matches what PostgresSaver expects of a pool it is
        # handed. Every query it runs sets dict_row on its own cursor, so
        # this is belt and braces rather than a fix, but it keeps the pool
        # type aligned with the saver's signature.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )

    checkpointer = PostgresSaver(_pool)
    checkpointer.setup()

    return checkpointer


def get_app():
    """The compiled graph, built on first use. Safe to call repeatedly."""
    global _app

    if _app is None:
        _app = build_graph(build_checkpointer())

    return _app


def shutdown() -> None:
    """
    Close the pool while the interpreter is still healthy.

    Left to garbage collection at exit, the pool's worker threads cannot be
    joined and Python reports a finalization error on every shutdown.
    """
    global _app, _pool

    if _pool is not None:
        _pool.close()
        _pool = None

    _app = None


if __name__ == "__main__":
    # The diagram needs no database, so build without a checkpointer.
    print(build_graph().get_graph().draw_mermaid())
