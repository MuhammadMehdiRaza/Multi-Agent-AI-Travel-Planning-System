# Multi-Agent AI Travel Planning System

A full-stack travel planning application powered by a multi-agent AI pipeline. The system coordinates four specialized agents — flight lookup, hotel search, itinerary generation, and final compilation — and presents results through a modern web interface built with React and TypeScript.

---

## How It Works

```
Browser (React + TypeScript)
        ↓  POST /api/chat
FastAPI Server (api.py)
        ↓  app.invoke(...)
LangGraph Agent Graph (main.py)
   ├── Flight Agent   →  Aviationstack API
   ├── Hotel Agent    →  Tavily Search API
   ├── Itinerary Agent →  Groq LLM (llama / gpt-oss)
   └── Final Agent    →  Groq LLM
        ↓
PostgreSQL  (conversation memory via PostgresSaver)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | ChatGroq — `openai/gpt-oss-120b` |
| Flight Data | [Aviationstack API](https://aviationstack.com/) |
| Web / Hotel Search | [Tavily Search API](https://tavily.com/) |
| State Persistence | PostgreSQL via `PostgresSaver` |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 19 + TypeScript (Vite) |
| Icons | Lucide React |

---

## Project Structure

```
Multi Agent System Demo/
├── frontend/                      # React + TypeScript UI
│   ├── src/
│   │   ├── api/
│   │   │   └── travel.ts          # API helper — calls the FastAPI backend
│   │   ├── components/
│   │   │   ├── AgentStep.tsx      # Vertical stepper card per agent
│   │   │   ├── ResultsPanel.tsx   # Pipeline view + final plan card
│   │   │   └── SearchBar.tsx      # Session name + query form
│   │   ├── App.tsx                # Root layout and state
│   │   ├── index.css              # Design tokens + animations
│   │   └── main.tsx               # React entry point
│   ├── index.html
│   └── package.json
├── tools/
│   ├── flight_tool.py             # Aviationstack API client
│   └── tavily_tool.py             # Tavily Search API client
├── api.py                         # FastAPI server wrapping the LangGraph app
├── main.py                        # Agent graph definition and state schema
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (not committed)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- A running PostgreSQL instance
- API keys from:
  - [Groq Cloud](https://console.groq.com/)
  - [Tavily](https://tavily.com/)
  - [Aviationstack](https://aviationstack.com/)

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/MuhammadMehdiRaza/Multi-Agent-AI-Travel-Planning-System.git
cd "Multi-Agent-AI-Travel-Planning-System"
```

**2. Create and activate a Python virtual environment**

```bash
python -m venv LangGraphenv
```

```bash
# Windows
LangGraphenv\Scripts\activate

# Linux / macOS
source LangGraphenv/bin/activate
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**4. Create a `.env` file in the project root**

```env
GROQ_API_KEY=your_groq_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=postgresql://username:password@localhost:5432/your_database
```

**5. Install frontend dependencies**

```bash
cd frontend
npm install
```

---

## Running the Application

You need two terminals running simultaneously.

**Terminal 1 — Backend API** (from the project root, with venv active):

```bash
uvicorn api:server --reload --port 8000
```

**Terminal 2 — Frontend** (from the `frontend/` directory):

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Using the App

1. Enter a **session name** (e.g. your name). Using the same name on future visits lets the system remember your conversation history.
2. Type a **travel request** — be as specific as you like (destination, dates, budget, preferences).
3. Click **Plan trip** and watch the four agents work through the pipeline.
4. The final itinerary appears in the **Your travel plan** section once all agents complete.

---

## Agent State Schema

The LangGraph graph passes a shared `TravelState` object through each node:

| Field | Type | Description |
|---|---|---|
| `messages` | `list[AnyMessage]` | Full conversation history |
| `user_query` | `str` | The raw travel query from the user |
| `flight_results` | `str` | Output from the flight lookup agent |
| `hotel_results` | `str` | Output from the hotel search agent |
| `itinerary` | `str` | Generated day-by-day itinerary |
| `llm_calls` | `int` | Number of LLM calls made in this run |

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
