"""
mcp_client.py - A single MCP client fronting all three Model Context Protocol servers.

The agents in this project never call a vendor REST API directly. They call the
thin wrappers at the bottom of this file, which resolve to MCP tools discovered at
runtime from three servers:

  tavily         remote, streamable HTTP    web and hotel search
  aviationstack  local subprocess, stdio    airports, airlines, routes, schedules
  weather        local subprocess, stdio    current weather and forecast

Because tool discovery is dynamic, a server can be added or removed without
touching any agent. A server whose credentials are missing is simply left out of
the client, and the wrappers that depend on it report themselves unavailable
instead of raising, so a partial configuration still produces a usable plan.
"""

import asyncio
import concurrent.futures
import json
import re
import threading
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection

from config import (
    AVIATION_STACK_API_KEY,
    AVIATIONSTACK_MCP_PYTHON,
    OPENWEATHER_API_KEY,
    TAVILY_API_KEY,
    TAVILY_MCP_URL,
    WEATHER_MCP_PYTHON,
    WEATHER_MCP_SCRIPT,
)


class ToolUnavailable(RuntimeError):
    """Raised when a requested MCP tool is not exposed by any configured server."""


# Query parameters whose values must never appear in a log line, an error
# message, or anything stored in graph state.
_SECRET_PARAMS = re.compile(
    r"((?:api[-_]?key|apikey|access[-_]?key|token|tavilyApiKey)=)[^&\s'\"]+",
    re.IGNORECASE,
)


def redact(text: Any) -> str:
    """
    Strip credentials out of text before it is logged or stored.

    The Tavily MCP endpoint takes its key as a URL query parameter, and httpx
    puts the full request URL into HTTPStatusError. Without this, a Tavily 401
    or 429 would write the live API key into graph state, from there into the
    PostgreSQL checkpoint, into the next agent's prompt, and onto the screen of
    anyone watching the demo.
    """
    return _SECRET_PARAMS.sub(r"\1[REDACTED]", str(text))


# -- Server registry ----------------------------------------------------------

def _build_server_config() -> dict[str, Connection]:
    """Register only the MCP servers this machine can actually reach."""
    servers: dict[str, Connection] = {}

    if TAVILY_API_KEY:
        servers["tavily"] = {
            "transport": "streamable_http",
            "url": f"{TAVILY_MCP_URL}?tavilyApiKey={TAVILY_API_KEY}",
        }

    if AVIATION_STACK_API_KEY and AVIATIONSTACK_MCP_PYTHON.exists():
        servers["aviationstack"] = {
            "transport": "stdio",
            "command": str(AVIATIONSTACK_MCP_PYTHON),
            "args": ["-m", "aviationstack_mcp", "mcp", "run"],
            "env": {"AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY},
        }

    if OPENWEATHER_API_KEY and WEATHER_MCP_SCRIPT.exists():
        servers["weather"] = {
            "transport": "stdio",
            "command": str(WEATHER_MCP_PYTHON),
            "args": [str(WEATHER_MCP_SCRIPT)],
            "env": {"OPENWEATHER_API_KEY": OPENWEATHER_API_KEY},
        }

    return servers


SERVERS = _build_server_config()

client = MultiServerMCPClient(SERVERS) if SERVERS else None


def configured_servers() -> list[str]:
    """Names of the MCP servers that were registered on this machine."""
    return sorted(SERVERS)


def missing_servers() -> dict[str, str]:
    """Explain which servers were skipped and why, for the health endpoint."""
    missing: dict[str, str] = {}

    if "tavily" not in SERVERS:
        missing["tavily"] = "TAVILY_API_KEY is not set."

    if "aviationstack" not in SERVERS:
        if not AVIATION_STACK_API_KEY:
            missing["aviationstack"] = "AVIATIONSTACK_API_KEY is not set."
        else:
            missing["aviationstack"] = (
                "Interpreter not found at "
                + str(AVIATIONSTACK_MCP_PYTHON)
                + ". Run 'uv sync' inside aviationstack-mcp/."
            )

    if "weather" not in SERVERS:
        if not OPENWEATHER_API_KEY:
            missing["weather"] = "OPENWEATHER_API_KEY is not set."
        else:
            missing["weather"] = "Server script not found at " + str(WEATHER_MCP_SCRIPT)

    return missing


# -- Tool discovery -----------------------------------------------------------

# MultiServerMCPClient.get_tools() returns stateless tool objects: each call to a
# tool opens its own short-lived MCP session. That makes the discovered list safe
# to cache and reuse across event loops, which matters because FastAPI runs each
# request in a worker thread with a fresh loop.
_tools_cache: list[Any] | None = None

# A threading lock, not an asyncio one. FastAPI serves each request on its own
# worker thread with its own event loop, so an asyncio.Lock created here would
# belong to whichever loop touched it first. Two requests arriving with the cache
# cold would otherwise both run full discovery, which means four extra
# subprocess launches and a wasted Tavily round trip.
_tools_lock = threading.Lock()


async def get_tools(refresh: bool = False) -> list[Any]:
    """Discover every tool across all configured servers, caching the result."""
    global _tools_cache

    if client is None:
        return []

    if _tools_cache is not None and not refresh:
        return _tools_cache

    discovered = await client.get_tools()

    with _tools_lock:
        if _tools_cache is None or refresh:
            _tools_cache = discovered

    return _tools_cache


async def list_tool_names() -> list[str]:
    return [tool.name for tool in await get_tools()]


async def call_tool(tool_name: str, args: dict[str, Any] | None = None) -> Any:
    """Invoke an MCP tool by name, whichever server happens to provide it."""
    tools = await get_tools()

    tool = next((candidate for candidate in tools if candidate.name == tool_name), None)

    if tool is None:
        available = ", ".join(sorted(candidate.name for candidate in tools)) or "none"
        raise ToolUnavailable(
            "MCP tool '" + tool_name + "' is not available. Discovered tools: " + available
        )

    return await tool.ainvoke(args or {})


# -- Response shaping ---------------------------------------------------------

def flatten_content(value: Any) -> str:
    """
    Turn an MCP tool result into plain text.

    The protocol returns a list of typed content blocks, so a tool response
    arrives as something like [{"type": "text", "text": "..."}]. Passing that
    straight through str() leaves a Python repr full of escaped newlines in both
    the UI and the next agent's prompt, which wastes tokens and reads badly.
    This unwraps the blocks and keeps only their text.
    """
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        if value.get("type") == "text" and "text" in value:
            return str(value["text"])
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, (list, tuple)):
        return "\n".join(flatten_content(item) for item in value)

    return str(value)


def describe_api_error(data: Any) -> str | None:
    """
    Recognise an AviationStack error envelope and explain it in one line.

    The upstream server answers HTTP 200 with {"ok": false, "error": ...} when the
    account's plan does not cover an endpoint. Pasting that JSON into a prompt
    reads to the model as data rather than as a failure, so it gets translated
    here. Returns None when the payload is not an error.
    """
    if not isinstance(data, dict) or data.get("ok") is not False:
        return None

    error = str(data.get("error", "unknown error"))

    if "function_access_restricted" in error:
        return (
            "[Not available on this AviationStack plan. Free keys cover live "
            "flight schedules only, not the airport, airline or route directories.]"
        )

    return "[AviationStack error: " + error[:200] + "]"


def summarise_schedule(payload: Any, limit: int = 6) -> str:
    """
    Project a flight schedule response down to the fields a planner needs.

    Each record carries eighteen fields including gates, terminals and three
    separate timestamps per leg. Pasting the raw JSON meant the prompt budget
    truncated it mid-value, handing the model syntactically broken input. This
    keeps the airline, flight number, route and times, and drops the rest.
    """
    text = flatten_content(payload).strip()

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return text

    described = describe_api_error(data)
    if described:
        return described

    if not isinstance(data, list):
        return text

    def clock(value: Any) -> str:
        raw = str(value or "")
        return raw[11:16] if len(raw) >= 16 else "--:--"

    lines: list[str] = []

    for item in data[:limit]:
        if not isinstance(item, dict):
            continue

        airline = str(item.get("airline") or "Unknown airline")
        number = str(item.get("flight_number") or "")
        arrives = str(item.get("arrival_airport_code") or "")
        delay = item.get("departure_delay")

        parts = [
            (airline + " " + number).strip(),
            "to " + arrives if arrives else "",
            "dep " + clock(item.get("departure_scheduled_time")),
            "arr " + clock(item.get("arrival_scheduled_time")),
            "delayed " + str(delay) + " min" if delay else "",
        ]
        lines.append("  " + "  ".join(part for part in parts if part))

    return "\n".join(lines) if lines else "No scheduled flights returned."


def format_search_results(payload: Any, limit: int = 5, snippet_chars: int = 320) -> str:
    """
    Render a Tavily search response as a readable numbered list.

    Falls back to the flattened text when the payload is not a search response,
    so this is safe to call on any tool result.
    """
    text = flatten_content(payload).strip()

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return text

    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return text

    entries: list[str] = []

    for index, item in enumerate(data["results"][:limit], start=1):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "Untitled").strip()
        url = str(item.get("url") or "").strip()
        snippet = " ".join(str(item.get("content") or "").split())

        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars].rsplit(" ", 1)[0] + "..."

        entries.append(
            str(index) + ". **" + title + "**\n   " + url + "\n   " + snippet
        )

    if not entries:
        return text

    answer = str(data.get("answer") or "").strip()
    heading = (answer + "\n\n") if answer else ""

    return heading + "\n\n".join(entries)


# -- Bridge from synchronous graph nodes to async MCP calls -------------------

def run_sync(coro) -> Any:
    """
    Run an MCP coroutine from a synchronous LangGraph node.

    Graph nodes are ordinary functions, but MCP is async. When no event loop is
    running in this thread we drive the coroutine with asyncio.run. If a loop is
    already running we hand the coroutine to a dedicated thread, so we never try
    to nest one event loop inside another.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# -- Tool wrappers used by the agents -----------------------------------------

async def tavily_search(query: str) -> Any:
    return await call_tool("tavily_search", {"query": query})


async def flight_schedule(
    airport_iata: str, schedule_type: str = "departure", limit: int = 6
) -> Any:
    """
    Live same-day arrivals or departures for one airport.

    This is the only AviationStack endpoint a free key can reach. The airport,
    airline and route directories all answer with function_access_restricted, so
    the flight agent is built on this one and derives real carriers from it
    rather than asking the model to recall which airlines fly a route.
    """
    return await call_tool(
        "flight_arrival_departure_schedule",
        {
            "airport_iata_code": airport_iata,
            "schedule_type": schedule_type,
            "number_of_flights": limit,
        },
    )


async def current_weather(city: str) -> Any:
    return await call_tool("get_current_weather", {"city": city})


async def forecast(city: str) -> Any:
    return await call_tool("get_forecast", {"city": city})
