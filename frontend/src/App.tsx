/**
 * App.tsx
 * Shell layout: dot-grid page, glassmorphism navbar, centred 680px column.
 */

import { useState } from "react";
import type { CSSProperties } from "react";
import { Plane } from "lucide-react";
import SearchBar from "./components/SearchBar";
import ResultsPanel from "./components/ResultsPanel";
import { sendTravelQuery } from "./api/travel";
import type { TravelResponse } from "./api/travel";
import "./index.css";

const s: Record<string, CSSProperties> = {
  /* Navbar */
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
    maxWidth: "680px",
    margin: "0 auto",
    width: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  navLogo: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
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

  /* Main column */
  main: {
    maxWidth: "680px",
    margin: "0 auto",
    padding: "52px 24px 96px",
  },

  /* Page heading */
  heading: { marginBottom: "36px" },
  h1: {
    fontSize: "24px",
    fontWeight: 700,
    color: "var(--text-900)",
    letterSpacing: "-0.025em",
    lineHeight: 1.25,
    marginBottom: "8px",
  },
  sub: {
    fontSize: "14px",
    color: "var(--text-500)",
    lineHeight: 1.7,
  },
};

export default function App() {
  const [query, setQuery]           = useState("");
  const [sessionId, setSessionId]   = useState("");
  const [result, setResult]         = useState<TravelResponse | null>(null);
  const [isLoading, setIsLoading]   = useState(false);
  const [error, setError]           = useState<string | null>(null);

  async function handleSubmit() {
    setResult(null); setError(null); setIsLoading(true);
    try {
      setResult(await sendTravelQuery(query, sessionId));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "An unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="dot-grid" style={{ minHeight: "100vh" }}>

      {/* Navbar */}
      <header style={s.navbar}>
        <div style={s.navInner}>
          <div style={s.navLogo}>
            <div style={s.navBadge}>
              <Plane size={14} strokeWidth={2.5} />
            </div>
            <span style={s.navBrand}>Travel Planner</span>
          </div>
          <span style={s.navPill}>Multi-Agent AI</span>
        </div>
      </header>

      <main style={s.main}>
        <div style={s.heading}>
          <h1 style={s.h1}>Plan your next trip</h1>
          <p style={s.sub}>
            Describe where you want to go. Our AI agents will search for flights,
            find hotels, and build a full itinerary automatically.
          </p>
        </div>

        <SearchBar
          query={query} sessionId={sessionId} isLoading={isLoading}
          onQueryChange={setQuery} onSessionIdChange={setSessionId}
          onSubmit={handleSubmit}
        />

        <ResultsPanel isLoading={isLoading} result={result} error={error} />
      </main>
    </div>
  );
}
