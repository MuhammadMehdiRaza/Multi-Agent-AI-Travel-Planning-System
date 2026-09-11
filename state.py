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

    # Draft plan and human review
    itinerary: str
    approval_request: str
    approved: bool
    human_feedback: str

    # Final answer
    final_response: str

    # Simple observability counter
    llm_calls: int
