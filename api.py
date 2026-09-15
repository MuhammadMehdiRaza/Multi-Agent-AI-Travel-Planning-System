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

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from config import GROQ_MODEL
from graph import get_app, shutdown
from mcp_client import configured_servers, missing_servers

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


# -- Request and response shapes ----------------------------------------------

class PlanRequest(BaseModel):
    # max_length is a cost control, not cosmetics. Each run spends about six
    # Groq calls plus paid AviationStack, Tavily and OpenWeatherMap requests, so
    # an unbounded query field is an invitation to drain four free tiers.
    query: str = Field(min_length=1, max_length=2000, description="The traveller's request.")
    thread_id: str = Field(min_length=1, max_length=128, description="Conversation id.")
    user_id: str = Field(default="web_user", max_length=128)


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    approved: bool
    feedback: str = Field(default="", max_length=4000)


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

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Build the graph and open the database pool on startup, close on shutdown.

    Doing this here rather than at import keeps api.py importable without
    credentials or a live PostgreSQL, which is what lets the unit tests run with
    no external services. It also means a bad DATABASE_URL fails at startup with
    a clear error instead of during the first request.
    """
    get_app()
    yield
    shutdown()


server = FastAPI(
    title="Multi-Agent Travel Planning API",
    description="Supervisor routing, input guardrails, and human-in-the-loop over MCP servers.",
    version="3.0.0",
    lifespan=lifespan,
)

# The Vite dev server runs on 5173. Without this the browser blocks the request.
server.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client_error(exc: Exception) -> str:
    """
    A safe 500 body.

    Raw exception text is not safe to return. A psycopg failure carries the
    database host and user, and an httpx failure carries the full request URL,
    which for the Tavily MCP endpoint includes the API key. The detail goes to
    the log; the client gets the type only.
    """
    return (
        type(exc).__name__
        + ": the planning service failed. Check the server log for details."
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
        # Starting a fresh run on a thread that is already suspended does not
        # error: LangGraph restarts the graph from the beginning, overwrites
        # user_query, and replaces the pending interrupt. The earlier draft is
        # destroyed with no warning. Since the session name is hand-typed and
        # persists across submissions, pressing the button twice is the normal
        # way to hit this, so it has to be refused rather than absorbed.
        snapshot = get_app().get_state(config)
        if snapshot.next:
            raise HTTPException(
                status_code=409,
                detail=(
                    "This thread already has a plan waiting for your approval. "
                    "Answer it first, or use a different session name."
                ),
            )

        state = get_app().invoke(
            {
                "messages": [HumanMessage(content=request.query)],
                "user_id": request.user_id,
                "user_query": request.query,
                "llm_calls": 0,
            },
            config=config,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Planning run failed for thread %s", request.thread_id)
        raise HTTPException(status_code=500, detail=_client_error(exc)) from exc

    return _to_response(request.thread_id, state)


@server.post("/api/plan/stream")
def create_plan_stream(request: PlanRequest) -> StreamingResponse:
    """
    Phase one, streamed as newline-delimited JSON.

    Same work as POST /api/plan, but it emits an event as each node finishes
    instead of returning once at the end. That is what lets the UI light up the
    agents that are actually running. Without it the client waits a minute or
    more with nothing to show, which is why the interface used to display every
    agent as running, including the ones the supervisor had skipped.

    NDJSON over a POST body rather than server-sent events, because EventSource
    is GET-only and the traveller's request does not belong in a URL.

    Event shapes:
      {"type": "supervisor", "selected_agents": [...], ...}
      {"type": "node", "node": "flight_agent"}
      {"type": "done", "result": {...PlanResponse...}}
      {"type": "error", "detail": "..."}
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    def event(payload: dict) -> str:
        return json.dumps(payload) + "\n"

    def generate() -> Iterator[str]:
        app = get_app()

        try:
            # Same guard as the non-streaming endpoint. A second run on a
            # suspended thread would restart the graph and destroy the draft.
            if app.get_state(config).next:
                yield event(
                    {
                        "type": "error",
                        "detail": (
                            "This thread already has a plan waiting for your approval. "
                            "Answer it first, or use a different session name."
                        ),
                    }
                )
                return

            for chunk in app.stream(
                {
                    "messages": [HumanMessage(content=request.query)],
                    "user_id": request.user_id,
                    "user_query": request.query,
                    "llm_calls": 0,
                },
                config=config,
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        continue

                    if node == "supervisor" and isinstance(update, dict):
                        # Emitted separately so the client can grey out the
                        # skipped agents immediately, rather than after the run.
                        yield event(
                            {
                                "type": "supervisor",
                                "selected_agents": update.get("selected_agents", []),
                                "supervisor_reasoning": update.get(
                                    "supervisor_reasoning", ""
                                ),
                                "trip_constraints": update.get("trip_constraints") or {},
                                "blocked": bool(update.get("guardrail_blocked")),
                                "blocked_reason": update.get("guardrail_reason", ""),
                            }
                        )
                        continue

                    yield event({"type": "node", "node": node})

            # Read the settled state rather than accumulating the updates, so the
            # final payload is identical to what /api/plan would have returned.
            snapshot = app.get_state(config)
            state = dict(snapshot.values)
            if snapshot.interrupts:
                state["__interrupt__"] = list(snapshot.interrupts)

            yield event(
                {
                    "type": "done",
                    "result": _to_response(request.thread_id, state).model_dump(),
                }
            )

        except Exception as exc:
            log.exception("Streamed planning run failed for thread %s", request.thread_id)
            yield event({"type": "error", "detail": _client_error(exc)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@server.post("/api/approve", response_model=PlanResponse)
def approve_plan(request: ApprovalRequest) -> PlanResponse:
    """
    Phase two. Resumes the suspended run with the reviewer's verdict. The resume
    value is what the approval node's interrupt() call returns.
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        # get_state belongs inside the try. It reaches PostgreSQL, so a pool
        # timeout or a connection error here would otherwise escape as an
        # unhandled 500 with nothing in the log.
        snapshot = get_app().get_state(config)

        if not snapshot.next:
            raise HTTPException(
                status_code=409,
                detail="No run is waiting for approval on this thread. Create a plan first.",
            )

        state = get_app().invoke(
            Command(
                resume={"approved": request.approved, "feedback": request.feedback}
            ),
            config=config,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Approval resume failed for thread %s", request.thread_id)
        raise HTTPException(status_code=500, detail=_client_error(exc)) from exc

    return _to_response(request.thread_id, state)


@server.get("/api/thread/{thread_id}", response_model=PlanResponse)
def read_thread(thread_id: str) -> PlanResponse:
    """Read a thread's latest checkpoint, so a reload does not lose the draft."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot = get_app().get_state(config)
    except Exception as exc:
        log.exception("Could not read thread %s", thread_id)
        raise HTTPException(status_code=500, detail=_client_error(exc)) from exc

    if not snapshot.created_at:
        raise HTTPException(status_code=404, detail="Unknown thread id.")

    state = dict(snapshot.values)

    # get_state exposes pending interrupts on the snapshot rather than inside
    # values, so re-attach them in the shape _to_response expects.
    if snapshot.interrupts:
        state["__interrupt__"] = list(snapshot.interrupts)

    return _to_response(thread_id, state)
