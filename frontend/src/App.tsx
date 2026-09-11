/**
 * App.tsx
 * Shell layout and the two-phase planning flow.
 *
 * Planning is two requests, not one, because the graph suspends at its human
 * approval node. `handlePlan` starts the run and receives a draft;
 * `handleApproval` resumes the same run on the same thread. The thread id is the
 * only thing that links them, and it doubles as the memory key in PostgreSQL.
 */

import { useEffect, useState } from "react";
import type { CSSProperties } from "react";
import { AlertTriangle, Plane } from "lucide-react";
import SearchBar from "./components/SearchBar";
import ResultsPanel from "./components/ResultsPanel";
import { createPlan, readHealth, submitApproval } from "./api/travel";
import type { HealthResponse, PlanResponse } from "./api/travel";
import "./index.css";

const s: Record<string, CSSProperties> = {
  navbar: {
    position: "sticky",
    top: 0,
    zIndex: 50,
    height: "56px",
    backgroundColor: "rgba(248,250,252,0.85)",
    backdropFilter: "blur(16px)",
    WebkitBackdropFilter: "blur(16px)",
    borderBottom: "1px solid var(--border)",
    display: "flex",
    alignItems: "center",
    padding: "0 24px",
  },
  navInner: {
    maxWidth: "760px",
    margin: "0 auto",
    width: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  navLogo: { display: "flex", alignItems: "center", gap: "8px" },
  navBadge: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "28px",
    height: "28px",
    borderRadius: "8px",
    background: "linear-gradient(135deg, var(--indigo-600), var(--indigo-500))",
    color: "#fff",
    flexShrink: 0,
  },
  navBrand: {
    fontSize: "14px",
    fontWeight: 600,
    color: "var(--text-900)",
    letterSpacing: "-0.01em",
  },
  navPill: {
    fontSize: "11px",
    fontWeight: 500,
    color: "var(--indigo-600)",
    backgroundColor: "var(--indigo-50)",
    border: "1px solid var(--indigo-100)",
    borderRadius: "99px",
    padding: "3px 10px",
    letterSpacing: "0.01em",
  },

  main: { maxWidth: "760px", margin: "0 auto", padding: "52px 24px 96px" },

  heading: { marginBottom: "28px" },
  h1: {
    fontSize: "24px",
    fontWeight: 700,
    color: "var(--text-900)",
    letterSpacing: "-0.025em",
    lineHeight: 1.25,
    marginBottom: "8px",
  },
  sub: { fontSize: "14px", color: "var(--text-500)", lineHeight: 1.7 },

  warning: {
    display: "flex",
    gap: "10px",
    alignItems: "flex-start",
    backgroundColor: "var(--amber-50)",
    border: "1px solid #fde68a",
    borderRadius: "var(--r-md)",
    padding: "12px 14px",
    marginBottom: "24px",
    fontSize: "13px",
    color: "var(--text-700)",
    lineHeight: 1.65,
  },
};

export default function App() {
  const [query, setQuery] = useState("");
  const [threadId, setThreadId] = useState("");

  const [result, setResult] = useState<PlanResponse | null>(null);
  const [isPlanning, setIsPlanning] = useState(false);
  const [isApproving, setIsApproving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  // Surface a half-configured backend up front rather than letting it show up
  // as an empty weather section halfway through a run.
  useEffect(() => {
    readHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  async function handlePlan() {
    setResult(null);
    setError(null);
    setIsPlanning(true);

    try {
      setResult(await createPlan(query, threadId));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      setIsPlanning(false);
    }
  }

  async function handleApproval(approved: boolean, feedback: string) {
    setError(null);
    setIsApproving(true);

    try {
      setResult(await submitApproval(threadId, approved, feedback));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not submit your review.");
    } finally {
      setIsApproving(false);
    }
  }

  const unavailable = Object.entries(health?.mcp_servers_unavailable ?? {});

  return (
    <div className="dot-grid" style={{ minHeight: "100vh" }}>
      <header style={s.navbar}>
        <div style={s.navInner}>
          <div style={s.navLogo}>
            <div style={s.navBadge}>
              <Plane size={14} strokeWidth={2.5} />
            </div>
            <span style={s.navBrand}>Travel Planner</span>
          </div>
          <span style={s.navPill}>
            {health ? `Supervisor · MCP · HITL · ${health.model}` : "Supervisor · MCP · HITL"}
          </span>
        </div>
      </header>

      <main style={s.main}>
        <div style={s.heading}>
          <h1 style={s.h1}>Plan your next trip</h1>
          <p style={s.sub}>
            Describe the trip you want. A supervisor agent decides which specialists to
            run, they gather live data through MCP servers, and you review the draft plan
            before it is finalised.
          </p>
        </div>

        {unavailable.length > 0 && (
          <div style={s.warning}>
            <AlertTriangle size={16} color="var(--amber-600)" style={{ flexShrink: 0, marginTop: "1px" }} />
            <div>
              <strong>Some MCP servers are not configured.</strong> Those sections of the
              plan will be marked unavailable instead of failing the run.
              <ul style={{ margin: "6px 0 0 18px" }}>
                {unavailable.map(([name, reason]) => (
                  <li key={name}>
                    <code>{name}</code> — {reason}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        <SearchBar
          query={query}
          sessionId={threadId}
          isLoading={isPlanning || isApproving}
          onQueryChange={setQuery}
          onSessionIdChange={setThreadId}
          onSubmit={handlePlan}
        />

        <ResultsPanel
          isPlanning={isPlanning}
          isApproving={isApproving}
          result={result}
          error={error}
          onApproval={handleApproval}
        />
      </main>
    </div>
  );
}
