/**
 * ResultsPanel.tsx
 * Lays out one planning run: the guardrail verdict, the supervisor's routing
 * decision, the agent pipeline, the human approval gate, and the final plan.
 */

import type { CSSProperties } from "react";
import { ShieldAlert, Sparkles } from "lucide-react";
import AgentStep from "./AgentStep";
import type { StepState } from "./AgentStep";
import ApprovalPanel from "./ApprovalPanel";
import SupervisorPlan from "./SupervisorPlan";
import type { AgentKey, PlanResponse, SupervisorEvent, TripConstraints } from "../api/travel";

interface ResultsPanelProps {
  isPlanning: boolean;
  isApproving: boolean;
  result: PlanResponse | null;
  /** The supervisor's decision, streamed in before the specialists start. */
  livePlan: SupervisorEvent | null;
  /** Node names that have actually finished, in order of completion. */
  completedNodes: string[];
  error: string | null;
  onApproval: (approved: boolean, feedback: string) => void;
}

// Mirrors DATA_AGENTS in agents.py. These three run in a single parallel
// superstep, so while any of them is outstanding all three are genuinely active.
const DATA_AGENT_KEYS: AgentKey[] = ["flight_agent", "hotel_agent", "weather_agent"];

const PIPELINE: {
  key: AgentKey;
  title: string;
  subtitle: string;
  icon: "plane" | "hotel" | "cloud" | "wallet" | "map";
  field: keyof PlanResponse;
}[] = [
  {
    key: "flight_agent",
    title: "Flight Agent",
    subtitle: "Airports and airlines via the AviationStack MCP server",
    icon: "plane",
    field: "flight_results",
  },
  {
    key: "hotel_agent",
    title: "Hotel Agent",
    subtitle: "Accommodation search via the Tavily MCP server",
    icon: "hotel",
    field: "hotel_results",
  },
  {
    key: "weather_agent",
    title: "Weather Agent",
    subtitle: "Conditions and forecast via the weather MCP server",
    icon: "cloud",
    field: "weather_results",
  },
  {
    key: "budget_agent",
    title: "Budget Agent",
    subtitle: "Cost feasibility across the other agents' findings",
    icon: "wallet",
    field: "budget_results",
  },
  {
    key: "itinerary_agent",
    title: "Itinerary Agent",
    subtitle: "Assembles the draft plan for human review",
    icon: "map",
    field: "itinerary",
  },
];

const s: Record<string, CSSProperties> = {
  sectionRow: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    marginBottom: "16px",
    marginTop: "28px",
  },
  sectionLabel: {
    fontSize: "11px",
    fontWeight: 600,
    color: "var(--text-400)",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    whiteSpace: "nowrap",
    flexShrink: 0,
  },
  sectionLine: { flex: 1, height: "1px", backgroundColor: "var(--border)" },

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

  blockedCard: {
    backgroundColor: "var(--white)",
    border: "1px solid #fecaca",
    borderLeft: "3px solid var(--red-600)",
    borderRadius: "var(--r-lg)",
    padding: "18px 20px",
    display: "flex",
    gap: "12px",
    alignItems: "flex-start",
    boxShadow: "var(--shadow-sm)",
  },
  blockedTitle: { fontSize: "13px", fontWeight: 600, color: "var(--text-900)", marginBottom: "4px" },
  blockedBody: { fontSize: "13px", color: "var(--text-700)", lineHeight: 1.7 },
  blockedNote: { fontSize: "12px", color: "var(--text-400)", marginTop: "8px", lineHeight: 1.6 },

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
    gap: "12px",
    flexWrap: "wrap",
  },
  finalHeadLeft: { display: "flex", alignItems: "center", gap: "8px" },
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
  finalTitle: { fontSize: "13px", fontWeight: 600, color: "var(--text-900)" },
  finalMeta: {
    fontSize: "12px",
    color: "var(--text-400)",
    display: "flex",
    alignItems: "center",
    gap: "6px",
    flexWrap: "wrap",
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
    whiteSpace: "pre-wrap",
  },
  revisionBadge: {
    fontSize: "11px",
    fontWeight: 500,
    color: "var(--indigo-600)",
    backgroundColor: "var(--indigo-50)",
    border: "1px solid var(--indigo-100)",
    borderRadius: "99px",
    padding: "3px 10px",
  },
};

function SectionLabel({ children }: { children: string }) {
  return (
    <div style={s.sectionRow}>
      <span style={s.sectionLabel}>{children}</span>
      <div style={s.sectionLine} />
    </div>
  );
}

export default function ResultsPanel({
  isPlanning,
  isApproving,
  result,
  livePlan,
  completedNodes,
  error,
  onApproval,
}: ResultsPanelProps) {
  if (!isPlanning && !result && !error) return null;

  // The routing decision, from the finished run if there is one, otherwise from
  // the streamed supervisor event.
  const selected: AgentKey[] | null =
    result?.selected_agents ?? livePlan?.selected_agents ?? null;

  const blocked = result?.blocked ?? livePlan?.blocked ?? false;
  const blockedReason = result?.blocked_reason ?? livePlan?.blocked_reason ?? "";

  /**
   * Honest per-agent state.
   *
   * This used to return "running" for all five agents for the whole request,
   * which misrepresented a sequential run and showed skipped agents as active.
   * Now it is driven by the stream: an agent is running only once the supervisor
   * has selected it and it is the next one not yet reported complete.
   */
  const stepState = (key: AgentKey, hasContent: boolean): StepState => {
    if (result && !isPlanning) {
      if (!result.selected_agents.includes(key)) return "skipped";
      return hasContent ? "complete" : "waiting";
    }

    // Still streaming.
    if (!selected) return "waiting";
    if (!selected.includes(key)) return "skipped";
    if (completedNodes.includes(key)) return "complete";

    const pending = selected.filter((agent) => !completedNodes.includes(agent));

    // The three data agents run in one parallel superstep, so if any of them is
    // pending they are all genuinely in flight together.
    if (DATA_AGENT_KEYS.includes(key)) {
      return pending.some((agent) => DATA_AGENT_KEYS.includes(agent))
        ? "running"
        : "waiting";
    }

    return pending[0] === key ? "running" : "waiting";
  };

  return (
    <div>
      {error && <div style={s.errorBox}>{error}</div>}

      {blocked && (
        <>
          <SectionLabel>Input guardrail</SectionLabel>
          <div style={s.blockedCard} className="fade-up">
            <ShieldAlert size={18} color="var(--red-600)" style={{ flexShrink: 0, marginTop: "1px" }} />
            <div>
              <div style={s.blockedTitle}>Request rejected before any agent ran</div>
              <div style={s.blockedBody}>{blockedReason}</div>
              <div style={s.blockedNote}>
                The guardrail runs ahead of the supervisor, so a rejected request costs one
                model call instead of a full pipeline of external API requests.
              </div>
            </div>
          </div>
        </>
      )}

      {selected && !blocked && (
        <>
          <SectionLabel>Supervisor</SectionLabel>
          <SupervisorPlan
            selectedAgents={selected}
            reasoning={
              (result?.supervisor_reasoning || livePlan?.supervisor_reasoning) ?? ""
            }
            constraints={
              (result?.trip_constraints ??
                livePlan?.trip_constraints ??
                {}) as TripConstraints
            }
          />
        </>
      )}

      {!blocked && (isPlanning || result) && (
        <>
          <SectionLabel>Agent pipeline</SectionLabel>
          <div>
            {PIPELINE.map((step, index) => {
              const content = result ? String(result[step.field] ?? "") : "";
              return (
                <AgentStep
                  key={step.key}
                  title={step.title}
                  subtitle={step.subtitle}
                  icon={step.icon}
                  content={content}
                  state={stepState(step.key, Boolean(content))}
                  isLast={index === PIPELINE.length - 1}
                />
              );
            })}
          </div>
        </>
      )}

      {result?.awaiting_approval && (
        <ApprovalPanel
          prompt={result.approval_request}
          isSubmitting={isApproving}
          onSubmit={onApproval}
        />
      )}

      {result?.final_response && (
        <>
          <SectionLabel>Your travel plan</SectionLabel>
          <div style={s.finalCard}>
            <div style={s.finalHeader}>
              <div style={s.finalHeadLeft}>
                <div style={s.finalIconWrap}>
                  <Sparkles size={13} strokeWidth={2.5} />
                </div>
                <span style={s.finalTitle}>
                  {result.approved ? "Approved itinerary" : "Revised itinerary"}
                </span>
                {!result.approved && <span style={s.revisionBadge}>Rewritten from your feedback</span>}
              </div>
              <div style={s.finalMeta}>
                <span>
                  {result.llm_calls} model call{result.llm_calls === 1 ? "" : "s"}
                </span>
                <span style={s.metaDot} />
                <span>
                  {result.selected_agents.length} agent
                  {result.selected_agents.length === 1 ? "" : "s"} run
                </span>
              </div>
            </div>
            <div style={s.finalBody}>{result.final_response}</div>
          </div>
        </>
      )}
    </div>
  );
}
