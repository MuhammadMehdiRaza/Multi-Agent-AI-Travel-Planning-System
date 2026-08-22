"""
api.py — FastAPI server that wraps the LangGraph travel planning app.

This file turns the agent graph into an HTTP API so the React frontend
can send travel queries and receive results from each agent.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the compiled LangGraph app and message types from main.py
from main import app as travel_app
from langchain_core.messages import HumanMessage


# ── Request / Response shapes ─────────────────────────────────────────────────

class TravelRequest(BaseModel):
    """What the frontend sends to us."""
    query: str          # The user's travel request, e.g. "Plan a trip to Tokyo"
    thread_id: str      # A unique session name so LangGraph can remember history


class TravelResponse(BaseModel):
    """What we send back to the frontend."""
    flight_results: str
    hotel_results: str
    itinerary: str
    final_response: str
    llm_calls: int


# ── FastAPI app setup ─────────────────────────────────────────────────────────

server = FastAPI(title="Travel Planning API")

# Allow the React dev server (localhost:5173) to call this API.
# Without this, the browser would block the request (CORS policy).
server.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@server.get("/")
def health_check():
    """Simple check to confirm the server is running."""
    return {"status": "Travel Planning API is running"}


@server.post("/api/chat", response_model=TravelResponse)
def chat(request: TravelRequest):
    """
    Main endpoint. The frontend calls this with a travel query.
    We run the full LangGraph agent pipeline and return the results.
    """

    # thread_id lets LangGraph remember the conversation history for this user.
    # Each user gets their own "memory" based on the name they typed in the UI.
    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    # Run all four agents: Flight → Hotel → Itinerary → Final
    try:
        result = travel_app.invoke(
            {
                "messages": [HumanMessage(content=request.query)],
                "user_query": request.query,
                "flight_results": "",
                "hotel_results": "",
                "itinerary": "",
                "llm_calls": 0,
            },
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # The final agent's message is the last one in the messages list.
    # It contains the complete, polished travel response.
    final_response = result["messages"][-1].content if result["messages"] else ""

    return TravelResponse(
        flight_results=result.get("flight_results", ""),
        hotel_results=result.get("hotel_results", ""),
        itinerary=result.get("itinerary", ""),
        final_response=final_response,
        llm_calls=result.get("llm_calls", 0),
    )
