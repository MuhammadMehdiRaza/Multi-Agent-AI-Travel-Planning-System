"""
main.py - Command line entry point for the travel planning graph.

Useful for exercising the whole workflow without the web stack. It shows the two
phases that human-in-the-loop introduces: the first invoke runs until the approval
node suspends the graph, and a second invoke resumes it with the answer.

    python main.py

Pass a thread id to continue an earlier conversation, since the PostgreSQL
checkpointer keeps history per thread:

    python main.py --thread mehdi

The graph itself lives in graph.py, the nodes in agents.py, and the shared state
in state.py. This file only handles input and output.
"""

import argparse
import logging
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

RULE = "=" * 72


def _section(title: str, body: str) -> None:
    if not body:
        return
    print("")
    print(RULE)
    print(title)
    print(RULE)
    print(body.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent travel planner")
    parser.add_argument(
        "--thread",
        default=None,
        help="Thread id to resume. Defaults to a fresh random thread.",
    )
    parser.add_argument("--user", default="cli_user", help="User id recorded in state.")
    args = parser.parse_args()

    thread_id = args.thread or "cli_" + uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}

    print("Thread: " + thread_id)
    query = input("Enter travel request: ").strip()

    result = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "user_id": args.user,
            "user_query": query,
            "llm_calls": 0,
        },
        config=config,
    )

    # The guardrail rejected the request, so the graph routed straight to END
    # without scheduling any specialist.
    if result.get("guardrail_blocked"):
        _section("REQUEST BLOCKED", result.get("guardrail_reason", ""))
        return

    _section("SUPERVISOR PLAN", result.get("supervisor_reasoning", ""))
    print("")
    print("Selected agents: " + ", ".join(result.get("selected_agents", [])))

    _section("FLIGHTS", result.get("flight_results", ""))
    _section("HOTELS", result.get("hotel_results", ""))
    _section("WEATHER", result.get("weather_results", ""))
    _section("BUDGET", result.get("budget_results", ""))

    # An interrupt means the approval node suspended the run. The draft arrives in
    # the interrupt payload, and the graph's position is already saved in
    # PostgreSQL, so this process could exit here and the run would still resume.
    interrupts = result.get("__interrupt__")

    if not interrupts:
        _section("FINAL PLAN", result.get("final_response", ""))
        return

    payload = interrupts[0].value
    _section("DRAFT ITINERARY", payload.get("draft_itinerary", ""))

    print("")
    print(payload.get("approval_request", ""))
    answer = input("Approve this plan? [y/N]: ").strip().lower()
    approved = answer in {"y", "yes"}

    feedback = ""
    if not approved:
        feedback = input("What should change? ").strip()

    result = app.invoke(
        Command(resume={"approved": approved, "feedback": feedback}),
        config=config,
    )

    _section("FINAL PLAN", result.get("final_response", ""))
    print("")
    print("LLM calls this run: " + str(result.get("llm_calls", 0)))
    print("Resume this conversation with: python main.py --thread " + thread_id)


if __name__ == "__main__":
    main()
