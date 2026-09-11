"""
api.py - FastAPI server wrapping the LangGraph travel planning app.

Human-in-the-loop makes this a two-call API rather than one. A single request
cannot both produce a draft and collect a person's verdict on it, so the run is
split at the approval node:

  POST /api/plan      run the graph until the approval node suspends it
  POST /api/approve   resume the same run with the reviewer's answer
  GET  /api/thread    read back where a thread currently stands
  GET  /api/health    which MCP servers came up on this machine

The link between the two calls is thread_id. LangGraph writes the suspended run
to PostgreSQL, so the second call can arrive minutes later, from a different
worker thread, and still resume exactly where the first one stopped.
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from config import GROQ_MODEL
from graph import app as travel_app
from mcp_client import configured_servers, missing_servers

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


# -- Request and response shapes ----------------------------------------------

class PlanRequest(BaseModel):
    query: str = Field(min_length=1, description="The traveller's request.")
    thread_id: str = Field(min_length=1, description="Conversation id, used for memory.")
    user_id: str = "web_user"


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


class PlanResponse(BaseModel):
    thread_id: str

    # Guardrail
    blocked: bool = False
    blocked_reason: str = ""

    # Supervisor
    selected_agents: list[str] = []
    supervisor_reasoning: str = ""
    trip_constraints: dict = {}

    # Specialists
    flight_results: str = ""
    hotel_results: str = ""
    weather_results: str = ""
    budget_results: str = ""

    # Draft and human review
    itinerary: str = ""
    approval_request: str = ""
    awaiting_approval: bool = False

    # Outcome
    approved: bool = False
    human_feedback: str = ""
    final_response: str = ""

    llm_calls: int = 0


# -- App setup ----------------------------------------------------------------

server = FastAPI(
    title="Multi-Agent Travel Planning API",
    description="Supervisor routing, input guardrails, and human-in-the-loop over MCP servers.",
    version="3.0.0",
)

# The Vite dev server runs on 5173. Without this the browser blocks the request.
server.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_response(thread_id: str, state: dict) -> PlanResponse:
    """
    Flatten a graph result into the API shape.

    When the graph is suspended, the draft itinerary lives in the interrupt
    payload rather than in the merged state, because the node that would have
    written it has not returned yet. Read the payload first and fall back to
    state, so both the suspended and the completed case populate the same field.
    """
    interrupts = state.get("__interrupt__") or []
    awaiting = bool(interrupts)

    itinerary = state.get("itinerary", "")
    approval_request = state.get("approval_request", "")

    if awaiting:
        payload = interrupts[0].value or {}
        itinerary = payload.get("draft_itinerary") or itinerary
        approval_request = payload.get("approval_request") or approval_request

    return PlanResponse(
        thread_id=thread_id,
        blocked=bool(state.get("guardrail_blocked")),
        blocked_reason=state.get("guardrail_reason", ""),
        selected_agents=state.get("selected_agents", []),
        supervisor_reasoning=state.get("supervisor_reasoning", ""),
        trip_constraints=state.get("trip_constraints") or {},
        flight_results=state.get("flight_results", ""),
        hotel_results=state.get("hotel_results", ""),
        weather_results=state.get("weather_results", ""),
        budget_results=state.get("budget_results", ""),
        itinerary=itinerary,
        approval_request=approval_request,
        awaiting_approval=awaiting,
        approved=bool(state.get("approved")),
        human_feedback=state.get("human_feedback", ""),
        final_response=state.get("final_response", ""),
        llm_calls=state.get("llm_calls", 0),
    )


# -- Endpoints ----------------------------------------------------------------

@server.get("/")
def root() -> dict:
    return {"status": "Multi-agent travel planning API is running"}


@server.get("/api/health")
def health() -> dict:
    """Report which MCP servers are usable, so a missing key is visible in the UI."""
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "mcp_servers_ready": configured_servers(),
        "mcp_servers_unavailable": missing_servers(),
    }


@server.post("/api/plan", response_model=PlanResponse)
def create_plan(request: PlanRequest) -> PlanResponse:
    """
    Phase one. Runs the guardrail, the supervisor, and whichever specialists the
    supervisor selected, then stops at the approval node and returns the draft.
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        state = travel_app.invoke(
            {
                "messages": [HumanMessage(content=request.query)],
                "user_id": request.user_id,
                "user_query": request.query,
                "llm_calls": 0,
            },
            config=config,
        )
    except Exception as exc:
        log.exception("Planning run failed for thread %s", request.thread_id)
        raise HTTPException(
            status_code=500, detail=type(exc).__name__ + ": " + str(exc)
        ) from exc

    return _to_response(request.thread_id, state)


@server.post("/api/approve", response_model=PlanResponse)
def approve_plan(request: ApprovalRequest) -> PlanResponse:
    """
    Phase two. Resumes the suspended run with the reviewer's verdict. The resume
    value is what the approval node's interrupt() call returns.
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    snapshot = travel_app.get_state(config)

    if not snapshot.next:
        raise HTTPException(
            status_code=409,
            detail="No run is waiting for approval on this thread. Create a plan first.",
        )

    try:
        state = travel_app.invoke(
            Command(
                resume={"approved": request.approved, "feedback": request.feedback}
            ),
            config=config,
        )
    except Exception as exc:
        log.exception("Approval resume failed for thread %s", request.thread_id)
        raise HTTPException(
            status_code=500, detail=type(exc).__name__ + ": " + str(exc)
        ) from exc

    return _to_response(request.thread_id, state)


@server.get("/api/thread/{thread_id}", response_model=PlanResponse)
def read_thread(thread_id: str) -> PlanResponse:
    """Read a thread's latest checkpoint, so a reload does not lose the draft."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = travel_app.get_state(config)

    if not snapshot.created_at:
        raise HTTPException(status_code=404, detail="Unknown thread id.")

    state = dict(snapshot.values)

    # get_state exposes pending interrupts on the snapshot rather than inside
    # values, so re-attach them in the shape _to_response expects.
    if snapshot.interrupts:
        state["__interrupt__"] = list(snapshot.interrupts)

    return _to_response(thread_id, state)
