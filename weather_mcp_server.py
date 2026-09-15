"""
weather_mcp_server.py - A custom MCP server exposing OpenWeatherMap as two tools.

This is the one MCP server in the project that is written here rather than
installed. It speaks the Model Context Protocol over stdio, so the LangGraph
process launches it as a subprocess and calls `get_current_weather` and
`get_forecast` the same way it calls the remote Tavily server or the local
AviationStack server. The agent code never sees an HTTP call.

Run it directly to check it starts:  python weather_mcp_server.py
"""

import os

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("Weather Server")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

TIMEOUT = 15


@mcp.tool()
def get_current_weather(city: str) -> dict:
    """Get the current weather for a city. Returns temperature in Celsius."""
    if not OPENWEATHER_API_KEY:
        return {"error": "OPENWEATHER_API_KEY is not set for the weather MCP server."}

    try:
        response = requests.get(
            CURRENT_URL,
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=TIMEOUT,
        )
        data = response.json()
    except requests.RequestException as exc:
        return {"error": f"Weather request failed: {exc}"}

    if response.status_code != 200:
        return {"error": data.get("message", "Unknown OpenWeatherMap error")}

    # .get() throughout, matching the rule the agent layer follows. A 200 with a
    # partial body is uncommon but real, and indexing would turn it into a
    # KeyError inside the MCP server rather than a usable answer.
    main = data.get("main") or {}
    weather = (data.get("weather") or [{}])[0]
    wind = data.get("wind") or {}

    return {
        "city": data.get("name", city),
        "temperature_c": main.get("temp"),
        "feels_like_c": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "condition": weather.get("description"),
        "wind_speed": wind.get("speed"),
    }


@mcp.tool()
def get_forecast(city: str, entries: int = 5) -> dict:
    """Get an upcoming weather forecast for a city, in three-hour steps."""
    if not OPENWEATHER_API_KEY:
        return {"error": "OPENWEATHER_API_KEY is not set for the weather MCP server."}

    try:
        response = requests.get(
            FORECAST_URL,
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=TIMEOUT,
        )
        data = response.json()
    except requests.RequestException as exc:
        return {"error": f"Forecast request failed: {exc}"}

    if response.status_code != 200:
        return {"error": data.get("message", "Unknown OpenWeatherMap error")}

    forecast = [
        {
            "datetime": item.get("dt_txt"),
            "temperature_c": (item.get("main") or {}).get("temp"),
            "condition": ((item.get("weather") or [{}])[0]).get("description"),
        }
        for item in (data.get("list") or [])[: max(1, entries)]
    ]

    return {"city": city, "forecast": forecast}


if __name__ == "__main__":
    mcp.run()
