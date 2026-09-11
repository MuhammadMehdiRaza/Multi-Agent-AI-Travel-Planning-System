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
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

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


# -- Server registry ----------------------------------------------------------

def _build_server_config() -> dict[str, dict[str, Any]]:
    """Register only the MCP servers this machine can actually reach."""
    servers: dict[str, dict[str, Any]] = {}

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


async def get_tools(refresh: bool = False) -> list[Any]:
    """Discover every tool across all configured servers, caching the result."""
    global _tools_cache

    if client is None:
        return []

    if _tools_cache is None or refresh:
        _tools_cache = await client.get_tools()

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


async def list_airports(search: str = "", limit: int = 10) -> Any:
    return await call_tool(
        "list_airports", {"search": search, "limit": limit, "offset": 0}
    )


async def list_airlines(search: str = "", limit: int = 10) -> Any:
    return await call_tool(
        "list_airlines", {"search": search, "limit": limit, "offset": 0}
    )


async def list_routes(dep_iata: str = "", arr_iata: str = "", limit: int = 10) -> Any:
    return await call_tool(
        "list_routes",
        {"dep_iata": dep_iata, "arr_iata": arr_iata, "limit": limit, "offset": 0},
    )


async def current_weather(city: str) -> Any:
    return await call_tool("get_current_weather", {"city": city})


async def forecast(city: str) -> Any:
    return await call_tool("get_forecast", {"city": city})
