/**
 * ResultsPanel.tsx
 * Section labels with dividers, vertical stepper pipeline, premium final plan card.
 */

import type { CSSProperties } from "react";
import { Sparkles } from "lucide-react";
import AgentStep from "./AgentStep";
import type { TravelResponse } from "../api/travel";

interface ResultsPanelProps {
  isLoading: boolean;
  result: TravelResponse | null;
  error: string | null;
}

const AGENTS = [
  {
    title: "Flight Agent",
    subtitle: "Fetches live flight data via Aviationstack",
    key: "flight_results" as keyof TravelResponse,
  },
  {
    title: "Hotel Agent",
    subtitle: "Finds hotels and accommodation via Tavily Search",
    key: "hotel_results" as keyof TravelResponse,
  },
  {
    title: "Itinerary Agent",
    subtitle: "Builds a day-by-day plan using Groq LLM",
    key: "itinerary" as keyof TravelResponse,
  },
  {
    title: "Final Agent",
    subtitle: "Compiles and polishes the complete travel response",
    key: "final_response" as keyof TravelResponse,
  },
];

const s: Record<string, CSSProperties> = {
  sectionRow: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    marginBottom: "16px",
  },
  sectionLabel: {
    fontSize: "11px",
    fontWeight: 600,
    color: "var(--text-400)",
    textTransform: "uppercase" as const,
    letterSpacing: "0.08em",
    whiteSpace: "nowrap" as const,
    flexShrink: 0,
  },
  sectionLine: {
    flex: 1,
    height: "1px",
    backgroundColor: "var(--border)",
  },

  /* Error */
  errorBox: {
    backgroundColor: "var(--red-50)",
    border: "1px solid #fecaca",
    borderRadius: "var(--r-md)",
    padding: "12px 16px",
    marginBottom: "20px",
    fontSize: "13px",
    color: "var(--red-600)",
    lineHeight: 1.6,
  },

  /* Final plan */
  finalWrap: {
    marginTop: "28px",
  },
  finalCard: {
    backgroundColor: "var(--white)",
    border: "1px solid var(--border)",
    borderRadius: "var(--r-2xl)",
    overflow: "hidden",
    boxShadow: "var(--shadow-lg)",
    animation: "fadeUp 0.4s cubic-bezier(0.16,1,0.3,1) both",
  },
  finalHeader: {
    padding: "16px 24px",
    borderBottom: "1px solid var(--border)",
    backgroundColor: "var(--bg-muted)",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  finalHeadLeft: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  finalIconWrap: {
    width: "28px",
    height: "28px",
    borderRadius: "var(--r-sm)",
    backgroundColor: "var(--indigo-50)",
    border: "1px solid var(--indigo-100)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "var(--indigo-600)",
  },
  finalTitle: {
    fontSize: "13px",
    fontWeight: 600,
    color: "var(--text-900)",
  },
  finalMeta: {
    fontSize: "12px",
    color: "var(--text-400)",
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  metaDot: {
    width: "4px",
    height: "4px",
    borderRadius: "50%",
    backgroundColor: "var(--text-300)",
    display: "inline-block",
  },
  finalBody: {
    padding: "24px",
    fontSize: "14px",
    color: "var(--text-700)",
    lineHeight: 1.85,
    whiteSpace: "pre-wrap" as const,
  },
};

export default function ResultsPanel({ isLoading, result, error }: ResultsPanelProps) {
  if (!isLoading && !result && !error) return null;

  return (
    <div>
      {/* Error */}
      {error && <div style={s.errorBox}>{error}</div>}

      {/* Agent pipeline section */}
      <div style={s.sectionRow}>
        <span style={s.sectionLabel}>Agent pipeline</span>
        <div style={s.sectionLine} />
      </div>

      {/* Vertical stepper */}
      <div style={{ paddingLeft: "0" }}>
        {AGENTS.map((a, i) => (
          <AgentStep
            key={a.key}
            stepNumber={i + 1}
            title={a.title}
            subtitle={a.subtitle}
            content={result ? String(result[a.key] ?? "") : ""}
            isLoading={isLoading}
            isComplete={!!result}
            isLast={i === AGENTS.length - 1}
          />
        ))}
      </div>

      {/* Final travel plan */}
      {result?.final_response && (
        <div style={s.finalWrap}>
          <div style={s.sectionRow}>
            <span style={s.sectionLabel}>Your travel plan</span>
            <div style={s.sectionLine} />
          </div>

          <div style={s.finalCard}>
            <div style={s.finalHeader}>
              <div style={s.finalHeadLeft}>
                <div style={s.finalIconWrap}>
                  <Sparkles size={13} strokeWidth={2.5} />
                </div>
                <span style={s.finalTitle}>Complete Itinerary</span>
              </div>
              <div style={s.finalMeta}>
                <span>{result.llm_calls} LLM call{result.llm_calls !== 1 ? "s" : ""}</span>
                <span style={s.metaDot} />
                <span>4 agents</span>
              </div>
            </div>
            <div style={s.finalBody}>{result.final_response}</div>
          </div>
        </div>
      )}
    </div>
  );
}
