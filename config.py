"""
config.py - Environment configuration, MCP server locations, and the shared LLM factory.

Every filesystem path used by this project is derived from PROJECT_ROOT, so the
repository can be cloned into any directory on any machine without editing source
files. This replaces the hard-coded absolute paths that the earlier prototype used.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env", override=True)

_BIN = "Scripts" if os.name == "nt" else "bin"
_PY = "python.exe" if os.name == "nt" else "python"

# ── Credentials ───────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# The upstream AviationStack MCP server reads AVIATION_STACK_API_KEY from its own
# environment, while this project's .env has always used AVIATIONSTACK_API_KEY.
# Accept either spelling so neither file has to change.
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATIONSTACK_API_KEY")
    or os.getenv("AVIATION_STACK_API_KEY")
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

DATABASE_URL = os.getenv("DATABASE_URL")

# ── Model ─────────────────────────────────────────────────────────────────────

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ── MCP server locations ──────────────────────────────────────────────────────

# Remote MCP server, reached over streamable HTTP.
TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"

# Local MCP server, cloned from github.com/Pradumnasaraf/aviationstack-mcp and
# installed into its own virtual environment (it requires Python >= 3.13).
AVIATIONSTACK_MCP_DIR = PROJECT_ROOT / "aviationstack-mcp"
AVIATIONSTACK_MCP_PYTHON = AVIATIONSTACK_MCP_DIR / ".venv" / _BIN / _PY

# Local MCP server written for this project. It runs under the same interpreter
# that is running the graph, so it needs no separate environment.
WEATHER_MCP_SCRIPT = PROJECT_ROOT / "weather_mcp_server.py"
WEATHER_MCP_PYTHON = Path(sys.executable)


def get_llm(temperature: float = 0.3) -> ChatGroq:
    """
    Return a configured Groq chat model. One place to change the model.

    max_retries is raised above the default of 2 because a full trip request makes
    five or six model calls in quick succession, which is enough to trip the free
    tier's tokens-per-minute limit. Groq returns a retry-after hint with its 429,
    and the client honours it, so a brief pause is far better than failing a run
    that is most of the way finished.
    """
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=4096,
        max_retries=5,
    )
