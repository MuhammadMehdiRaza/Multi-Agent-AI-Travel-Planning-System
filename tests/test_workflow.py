"""
tests/test_workflow.py - End-to-end checks over the compiled graph.

These are integration tests: they hit Groq, the MCP servers, and PostgreSQL, so
they cost API calls and take a minute or two. Run them after changing routing,
the guardrail, or the approval flow.

    python tests/test_workflow.py
"""

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app

PASS = "PASS"
FAIL = "FAIL"

# Seconds to wait between test cases, to stay inside the Groq free tier's
# tokens-per-minute limit. Set to 0 on a paid tier.
COOLDOWN_SECONDS = 35

results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))
    print(("  " + PASS if condition else "  " + FAIL) + "  " + name + (("  " + detail) if detail else ""))


def run(query: str) -> tuple[dict, dict]:
    thread_id = "test_" + uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}
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


def test_guardrail_blocks_off_topic() -> None:
    print("\n1. Guardrail rejects an off-topic request")
    state, _ = run("Write me a Python function that reverses a linked list.")

    check("request is marked blocked", bool(state.get("guardrail_blocked")))
    check("a reason is given", bool(state.get("guardrail_reason")))
    check("no specialist was scheduled", state.get("selected_agents") == [])
    check("graph stopped, no approval pending", not state.get("__interrupt__"))
    check("no itinerary was generated", not state.get("itinerary"))


def test_guardrail_blocks_empty() -> None:
    print("\n2. Guardrail rejects an empty request without calling the model")
    state, _ = run("  ")
    check("request is marked blocked", bool(state.get("guardrail_blocked")))
    check("only one llm_call was counted", state.get("llm_calls") == 1, str(state.get("llm_calls")))


def test_supervisor_routes_narrow_request() -> None:
    print("\n3. Supervisor routes a weather-only request to fewer agents")
    state, _ = run("What is the weather like in Tokyo right now?")

    selected = state.get("selected_agents", [])
    check("request was allowed", not state.get("guardrail_blocked"))
    check("weather agent selected", "weather_agent" in selected, str(selected))
    check("itinerary agent always selected", "itinerary_agent" in selected)
    check("flight agent skipped", "flight_agent" not in selected, str(selected))
    check("no flight output produced", not state.get("flight_results"))
    check("weather section present", bool(state.get("weather_results")))


def test_full_trip_and_approval() -> None:
    print("\n4. Full trip request runs the specialists and suspends for approval")
    state, config = run(
        "Plan a 7 day trip to Japan from Karachi under 2 lakh rupees, "
        "including flights, hotels and sightseeing."
    )

    selected = state.get("selected_agents", [])
    check("request was allowed", not state.get("guardrail_blocked"))
    check("several agents selected", len(selected) >= 3, str(selected))
    check("flight section present", bool(state.get("flight_results")))
    check("hotel section present", bool(state.get("hotel_results")))

    interrupts = state.get("__interrupt__")
    check("graph suspended at approval node", bool(interrupts))

    if not interrupts:
        return

    payload = interrupts[0].value
    check("draft itinerary in interrupt payload", bool(payload.get("draft_itinerary")))
    check("no final response yet", not state.get("final_response"))

    print("\n5. Resuming the suspended run with a rejection and feedback")
    revised = app.invoke(
        Command(
            resume={
                "approved": False,
                "feedback": "Too rushed. Make it five days and cut one city.",
            }
        ),
        config=config,
    )

    check("run completed", not revised.get("__interrupt__"))
    check("approval recorded as rejected", revised.get("approved") is False)
    check("feedback stored in state", bool(revised.get("human_feedback")))
    check("final response produced", bool(revised.get("final_response")))
    check(
        "final response differs from the draft",
        revised.get("final_response") != revised.get("itinerary"),
    )


def test_memory_persists_per_thread() -> None:
    print("\n6. PostgreSQL checkpoint retains the thread after the run")
    state, config = run("What is the weather in Dubai?")
    snapshot = app.get_state(config)

    check("checkpoint was written", snapshot.created_at is not None)
    check("user query survived in the checkpoint", snapshot.values.get("user_query") is not None)
    check("messages accumulated", len(snapshot.values.get("messages", [])) >= 2)


def main() -> int:
    print("=" * 72)
    print("Multi-agent travel planner, end-to-end checks")
    print("=" * 72)

    tests = (
        test_guardrail_blocks_off_topic,
        test_guardrail_blocks_empty,
        test_supervisor_routes_narrow_request,
        test_full_trip_and_approval,
        test_memory_persists_per_thread,
    )

    for index, test in enumerate(tests):
        # Running the whole suite back to back exceeds the Groq free tier's
        # tokens-per-minute allowance. The agents retry on their own, but pausing
        # between cases keeps the suite from spending its budget on retries.
        if index:
            time.sleep(COOLDOWN_SECONDS)

        try:
            test()
        except Exception as exc:
            check(test.__name__ + " raised", False, type(exc).__name__ + ": " + str(exc))

    failures = [row for row in results if row[0] == FAIL]

    print("")
    print("=" * 72)
    print(str(len(results) - len(failures)) + " passed, " + str(len(failures)) + " failed")
    print("=" * 72)

    for _, name, detail in failures:
        print("  FAILED  " + name + (("  " + detail) if detail else ""))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
