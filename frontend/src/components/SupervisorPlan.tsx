/**
 * SupervisorPlan.tsx
 * Shows the routing decision: which specialists the supervisor picked, which it
 * skipped, and the constraints it pulled out of the request. This is the part of
 * the UI that makes dynamic routing visible rather than something you have to
 * read the logs to believe.
 */

import type { CSSProperties } from "react";
import { GitBranch, ShieldCheck } from "lucide-react";
import { AGENT_KEYS } from "../api/travel";
import type { AgentKey, TripConstraints } from "../api/travel";

interface SupervisorPlanProps {
  selectedAgents: AgentKey[];
  reasoning: string;
  constraints: TripConstraints;
}

const AGENT_LABELS: Record<AgentKey, string> = {
  flight_agent: "Flight",
  hotel_agent: "Hotel",
  weather_agent: "Weather",
  research_agent: "Research",
  budget_agent: "Budget",
  itinerary_agent: "Itinerary",
};

const CONSTRAINT_LABELS: [keyof TripConstraints, string][] = [
  ["origin", "From"],
  ["destination", "To"],
  ["destination_iata", "Airport"],
  ["duration", "Duration"],
  ["budget", "Budget"],
  ["travel_style", "Style"],
];

const s: Record<string, CSSProperties> = {
  card: {
    backgroundColor: "var(--white)",
    border: "1px solid var(--border)",
    borderRadius: "var(--r-lg)",
    boxShadow: "var(--shadow-sm)",
    overflow: "hidden",
    marginBottom: "24px",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    padding: "13px 16px",
    borderBottom: "1px solid var(--border)",
  },
  iconWrap: {
    width: "28px",
    height: "28px",
    borderRadius: "var(--r-sm)",
    backgroundColor: "var(--indigo-50)",
    border: "1px solid var(--indigo-100)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "var(--indigo-600)",
    flexShrink: 0,
  },
  title: { fontSize: "13px", fontWeight: 600, color: "var(--text-900)" },
  subtitle: { fontSize: "12px", color: "var(--text-400)", marginTop: "1px" },
  body: { padding: "16px" },
  chipRow: { display: "flex", flexWrap: "wrap", gap: "6px", marginBottom: "14px" },
  chipOn: {
    fontSize: "12px",
    fontWeight: 500,
    color: "var(--indigo-600)",
    backgroundColor: "var(--indigo-50)",
    border: "1px solid var(--indigo-100)",
    borderRadius: "99px",
    padding: "3px 11px",
  },
  chipOff: {
    fontSize: "12px",
    fontWeight: 500,
    color: "var(--text-400)",
    backgroundColor: "var(--bg-muted)",
    border: "1px dashed var(--border-input)",
    borderRadius: "99px",
    padding: "3px 11px",
    textDecoration: "line-through",
  },
  reasoning: {
    fontSize: "13px",
    color: "var(--text-700)",
    lineHeight: 1.75,
    marginBottom: "14px",
  },
  constraintGrid: {
    display: "flex",
    flexWrap: "wrap",
    gap: "8px 20px",
    paddingTop: "12px",
    borderTop: "1px solid var(--border)",
  },
  constraintItem: { display: "flex", flexDirection: "column", gap: "2px" },
  constraintLabel: {
    fontSize: "10px",
    fontWeight: 600,
    color: "var(--text-400)",
    textTransform: "uppercase",
    letterSpacing: "0.06em",
  },
  constraintValue: { fontSize: "13px", color: "var(--text-900)", fontWeight: 500 },
};

export default function SupervisorPlan({
  selectedAgents,
  reasoning,
  constraints,
}: SupervisorPlanProps) {
  const skipped = AGENT_KEYS.filter((key) => !selectedAgents.includes(key));

  const shownConstraints = CONSTRAINT_LABELS.filter(
    ([key]) => typeof constraints[key] === "string" && constraints[key]
  );

  const preferences = constraints.special_preferences ?? [];

  return (
    <div style={s.card} className="fade-up">
      <div style={s.header}>
        <div style={s.iconWrap}>
          <GitBranch size={14} strokeWidth={2} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={s.title}>Supervisor routing decision</div>
          <div style={s.subtitle}>
            {selectedAgents.length} of {AGENT_KEYS.length} specialists scheduled for this request
          </div>
        </div>
        <div style={{ ...s.iconWrap, backgroundColor: "var(--green-50)", borderColor: "#bbf7d0", color: "var(--green-600)" }}>
          <ShieldCheck size={14} strokeWidth={2} />
        </div>
      </div>

      <div style={s.body}>
        <div style={s.chipRow}>
          {selectedAgents.map((key) => (
            <span key={key} style={s.chipOn}>
              {AGENT_LABELS[key]}
            </span>
          ))}
          {skipped.map((key) => (
            <span key={key} style={s.chipOff}>
              {AGENT_LABELS[key]}
            </span>
          ))}
        </div>

        {reasoning && <p style={s.reasoning}>{reasoning}</p>}

        {(shownConstraints.length > 0 || preferences.length > 0) && (
          <div style={s.constraintGrid}>
            {shownConstraints.map(([key, label]) => (
              <div key={key} style={s.constraintItem}>
                <span style={s.constraintLabel}>{label}</span>
                <span style={s.constraintValue}>{String(constraints[key])}</span>
              </div>
            ))}
            {preferences.length > 0 && (
              <div style={s.constraintItem}>
                <span style={s.constraintLabel}>Preferences</span>
                <span style={s.constraintValue}>{preferences.join(", ")}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
