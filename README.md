# Multi-Agent Travel Planning System

A stateful multi-agent system for travel planning built with LangGraph. Given a user query, the system coordinates between a flight lookup agent and a hotel/web search agent, then generates a complete travel itinerary using the Llama 3.3 70B model via Groq.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | ChatGroq — `llama-3.3-70b-versatile` |
| Flight Data | [Aviationstack API](https://aviationstack.com/) |
| Web / Hotel Search | [Tavily Search API](https://tavily.com/) |
| State Persistence | PostgreSQL via `PostgresSaver` |

---

## Project Structure

```
Multi Agent System Demo/
├── tools/
│   ├── flight_tool.py       # Fetches live flight data from Aviationstack
│   └── tavily_tool.py       # Web and hotel search via Tavily
├── main.py                  # LangGraph state graph and agent workflow
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not committed)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Prerequisites

Before running this project, make sure you have the following:

- Python 3.10 or higher
- A running PostgreSQL instance
- API keys from:
  - [Groq Cloud](https://console.groq.com/) — for LLM inference
  - [Tavily](https://tavily.com/) — for web and hotel search
  - [Aviationstack](https://aviationstack.com/) — for flight data

---

## Setup

**1. Clone the repository**

```bash
git clone <repository-url>
cd "Multi Agent System Demo"
```

**2. Create a virtual environment**

```bash
python -m venv LangGraphenv
```

Activate it:

```bash
# Windows
LangGraphenv\Scripts\activate

# Linux / macOS
source LangGraphenv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Create a `.env` file**

In the root directory, create a `.env` file with the following variables:

```env
GROQ_API_KEY=your_groq_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=postgresql://username:password@localhost:5432/your_database
```

---

## Running

```bash
python main.py
```

---

## State Schema

The agent graph passes a shared `TravelState` object through each node:

| Field | Type | Description |
|---|---|---|
| `messages` | `list[AnyMessage]` | Full conversation history |
| `user_query` | `str` | The raw travel query from the user |
| `flight_results` | `str` | Output from the flight lookup tool |
| `hotel_results` | `str` | Output from the hotel/web search tool |
| `itinerary` | `str` | Final generated travel itinerary |
| `llm_calls` | `int` | Counter tracking the number of LLM calls made |

---

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
