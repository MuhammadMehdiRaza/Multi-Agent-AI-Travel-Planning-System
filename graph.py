"""
graph.py - Graph assembly, dynamic routing, and the PostgreSQL checkpointer.

The shape of this graph is what makes the system multi-agent rather than a fixed
pipeline. There is no hard-coded chain of nodes. Every specialist leaves through a
conditional edge that asks the same question: given the supervisor's plan, which
agent should run next? A request for "what is the weather in Tokyo" runs two nodes;
a full trip request runs all five.

    START -> supervisor -> (blocked) -----------------------------> END
                        -> first selected specialist
                             -> next selected specialist ...
                                  -> itinerary -> human_approval -> final -> END
"""

import atexit
import logging

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import ConnectionPool

from agents import (
    AGENT_ORDER,
    budget_agent,
    final_response_agent,
    flight_agent,
    hotel_agent,
    human_approval_agent,
    itinerary_agent,
    supervisor_agent,
    weather_agent,
)
from config import DATABASE_URL
from state import TravelState

log = logging.getLogger(__name__)

NODES = {
    "flight_agent": flight_agent,
    "hotel_agent": hotel_agent,
    "weather_agent": weather_agent,
    "budget_agent": budget_agent,
    "itinerary_agent": itinerary_agent,
}

# Each conditional edge declares only the destinations its routing function can
# actually return. Handing every node the full map would still run correctly, but
# it would claim edges that can never be taken, and those phantom edges show up
# in the rendered diagram and in any reachability check over the graph.
#
# "blocked" is the exit the guardrail uses. It maps to END so a rejected request
# stops immediately instead of running the itinerary and approval nodes on
# nothing, which is what a single shared route map would have caused.
SUPERVISOR_ROUTES = {name: name for name in NODES} | {"blocked": END}


def _routes_after(current_agent: str) -> dict[str, str]:
    """Only agents later in the running order, plus the itinerary terminus."""
    position = AGENT_ORDER.index(current_agent)
    reachable = AGENT_ORDER[position + 1 :]
    return {name: name for name in reachable} | {"itinerary_agent": "itinerary_agent"}


# -- Routing ------------------------------------------------------------------

def _selected(state: TravelState) -> list[str]:
    """The supervisor's chosen agents, restored to canonical execution order."""
    chosen = state.get("selected_agents") or []
    return [agent for agent in AGENT_ORDER if agent in chosen]


def route_from_supervisor(state: TravelState) -> str:
    """Leave the supervisor for the first selected specialist, or stop."""
    if state.get("guardrail_blocked"):
        return "blocked"

    selected = _selected(state)
    return selected[0] if selected else "itinerary_agent"


def route_after(current_agent: str):
    """
    Build the routing function for one specialist.

    It walks forward through AGENT_ORDER from the current node and returns the
    next agent the supervisor actually selected, skipping the rest. The itinerary
    node is the guaranteed terminus of the specialist chain.
    """

    def route(state: TravelState) -> str:
        selected = _selected(state)
        position = AGENT_ORDER.index(current_agent)

        for candidate in AGENT_ORDER[position + 1 :]:
            if candidate in selected:
                return candidate

        return "itinerary_agent"

    return route


# -- Assembly -----------------------------------------------------------------

def build_graph(checkpointer=None):
    """Wire the nodes and edges. Kept separate from compilation so tests can
    build the graph without needing a database."""
    graph = StateGraph(TravelState)

    graph.add_node("supervisor", supervisor_agent)
    for name, node in NODES.items():
        graph.add_node(name, node)
    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("final_response", final_response_agent)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, SUPERVISOR_ROUTES)

    # Every specialist except the itinerary node routes dynamically.
    for name in AGENT_ORDER:
        if name != "itinerary_agent":
            graph.add_conditional_edges(name, route_after(name), _routes_after(name))

    graph.add_edge("itinerary_agent", "human_approval")
    graph.add_edge("human_approval", "final_response")
    graph.add_edge("final_response", END)

    return graph.compile(checkpointer=checkpointer)


def build_checkpointer() -> PostgresSaver | None:
    """
    Open the PostgreSQL checkpointer used for memory and for human-in-the-loop.

    A connection pool rather than a single connection, because FastAPI serves each
    request on a worker thread and psycopg connections are not safe to share
    across threads. autocommit is required by PostgresSaver, and disabling
    prepared statements keeps the saver working through connection poolers.
    """
    if not DATABASE_URL:
        log.warning("DATABASE_URL is not set. Running without memory or HITL resume.")
        return None

    pool = ConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=10,
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )

    # Close the pool while the interpreter is still healthy. Left to garbage
    # collection at shutdown, the pool's worker threads cannot be joined and
    # Python reports a finalization error on every exit.
    atexit.register(pool.close)

    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    return checkpointer


app = build_graph(build_checkpointer())


if __name__ == "__main__":
    # Print the graph as a Mermaid diagram, which renders directly on GitHub.
    print(app.get_graph().draw_mermaid())
