"""
state.py - The shared state object that every agent in the graph reads and writes.

LangGraph passes one TravelState dictionary through the whole workflow. Each node
returns a partial dictionary, and LangGraph merges it into the running state. Only
`messages` accumulates (via operator.add); every other key is overwritten by the
node that produces it.

`total=False` means no key is mandatory, which matters here because the supervisor
decides at runtime which agents run. A weather-only request never populates
`flight_results`, so every reader must use .get() rather than indexing.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage


class TravelState(TypedDict, total=False):
    # Conversation history. Annotated with operator.add so each node appends
    # rather than replacing the list.
    messages: Annotated[list[AnyMessage], operator.add]

    # Request identity
    user_id: str
    user_query: str

    # Input guardrail
    guardrail_blocked: bool
    guardrail_reason: str

    # Supervisor planning output
    trip_constraints: dict[str, Any]
    selected_agents: list[str]
    supervisor_reasoning: str

    # Specialist agent output
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str

    # Research agent. Unlike every other specialist, this one chooses its own
    # tool calls, so it needs a scratchpad of its own and a step budget.
    #
    # research_messages is the agent's private conversation: its own replies and
    # the tool results it asked for. It accumulates, because the model has to see
    # what previous calls returned in order to decide what to do next. It is kept
    # separate from `messages` so the agent's intermediate reasoning does not end
    # up in the user-facing transcript.
    research_results: str
    research_messages: Annotated[list[AnyMessage], operator.add]
    research_steps: Annotated[int, operator.add]

    # Draft plan and human review
    itinerary: str
    approval_request: str
    approved: bool
    human_feedback: str

    # Final answer
    final_response: str

    # Model call counter.
    #
    # A reducer, not a plain int, because the three data-gathering specialists run
    # in parallel. Two nodes writing a plain key in the same superstep raises
    # InvalidUpdateError, and the old read-modify-write pattern,
    # state.get("llm_calls", 0) + 1, only worked because execution was serial.
    # Nodes now return the number of calls they made and LangGraph sums them.
    llm_calls: Annotated[int, operator.add]
