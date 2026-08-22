/**
 * AgentStep.tsx
 * Vertical stepper card: numbered step connector, pulsing status pill,
 * structured output rendering (tables/badges vs raw text).
 */

import type { CSSProperties } from "react";
import { CheckCircle2, Circle, Plane, Hotel, Map, FileText } from "lucide-react";

interface AgentStepProps {
  title: string;
  subtitle: string;
  content: string;
  isLoading: boolean;
  isComplete: boolean;
  stepNumber: number;
  isLast: boolean;
}

/* ── Icon per agent ───────────────────────────────────────────────────────── */
const ICONS = [
  <Plane    size={14} strokeWidth={2} />,
  <Hotel    size={14} strokeWidth={2} />,
  <Map      size={14} strokeWidth={2} />,
  <FileText size={14} strokeWidth={2} />,
];

/* ── Parse flight text into rows ─────────────────────────────────────────── */
function parseFlightBlocks(text: string): Record<string, string>[] {
  if (!text) return [];
  return text.split(/\n(?=Airline:)/g)
    .map(block => {
      const obj: Record<string, string> = {};
      block.split("\n").forEach(line => {
        const [k, ...v] = line.split(":");
        if (k && v.length) obj[k.trim()] = v.join(":").trim();
      });
      return obj;
    })
    .filter(o => Object.keys(o).length > 1);
}

/* ── Parse numbered list items from Tavily output ─────────────────────────── */
function parseListItems(text: string): { num: string; title: string; url: string; snippet: string }[] {
  if (!text) return [];
  const regex = /(\d+)\.\s+\*\*(.+?)\*\*\s+(https?:\/\/[^\s]+)\s+([\s\S]+?)(?=\n\d+\.|$)/g;
  const results = [];
  let m;
  while ((m = regex.exec(text)) !== null) {
    results.push({ num: m[1], title: m[2], url: m[3], snippet: m[4].trim() });
  }
  return results;
}

/* ── Render structured flight cards ────────────────────────────────────────── */
function FlightOutput({ text }: { text: string }) {
  const flights = parseFlightBlocks(text);
  if (flights.length === 0) return <RawText text={text} />;

  const cardStyle: CSSProperties = {
    backgroundColor: "var(--white)",
    border: "1px solid var(--border)",
    borderRadius: "var(--r-md)",
    padding: "12px 14px",
    marginBottom: "8px",
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  };
  const rowStyle: CSSProperties = { display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap" as const };
  const chip = (_label: string, _value: string, color: string): CSSProperties => ({
    fontSize: "12px",
    color,
    backgroundColor: color === "var(--green-600)" ? "var(--green-50)"
      : color === "var(--amber-600)" ? "var(--amber-50)"
      : "var(--bg-muted)",
    border: `1px solid ${color === "var(--green-600)" ? "#bbf7d0"
      : color === "var(--amber-600)" ? "#fde68a"
      : "var(--border)"}`,
    borderRadius: "99px",
    padding: "2px 8px",
    fontWeight: 500,
  });

  return (
    <div>
      {flights.map((f, i) => {
        const statusOk = (f.Status || "").toLowerCase() === "active" || (f.Status || "").toLowerCase() === "landed";
        const statusColor = statusOk ? "var(--green-600)" : "var(--amber-600)";
        return (
          <div key={i} style={cardStyle}>
            <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-900)" }}>
              {f.Airline || "Unknown Airline"}
            </div>
            <div style={rowStyle}>
              <span style={chip("Departure", "", "var(--text-700)")}>
                ✈ {f.Departure || "—"}
              </span>
              <span style={{ color: "var(--text-400)", fontSize: "12px" }}>→</span>
              <span style={chip("Arrival", "", "var(--text-700)")}>
                {f.Arrival || "—"}
              </span>
              {f.Status && (
                <span style={chip("Status", f.Status, statusColor)}>{f.Status}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Render structured hotel/web search items ───────────────────────────────── */
function SearchOutput({ text }: { text: string }) {
  const items = parseListItems(text);
  if (items.length === 0) return <RawText text={text} />;

  const itemStyle: CSSProperties = {
    backgroundColor: "var(--white)",
    border: "1px solid var(--border)",
    borderRadius: "var(--r-md)",
    padding: "12px 14px",
    marginBottom: "8px",
  };
  return (
    <div>
      {items.map((item, i) => (
        <div key={i} style={itemStyle}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "8px", marginBottom: "4px" }}>
            <span style={{
              flexShrink: 0,
              fontSize: "11px", fontWeight: 700,
              color: "var(--indigo-600)", backgroundColor: "var(--indigo-50)",
              border: "1px solid var(--indigo-100)",
              borderRadius: "99px", padding: "2px 7px",
            }}>#{item.num}</span>
            <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-900)" }}>{item.title}</div>
          </div>
          <a href={item.url} target="_blank" rel="noreferrer" style={{
            fontSize: "11px", color: "var(--indigo-500)",
            textDecoration: "none", display: "block", marginBottom: "4px",
          }}>{item.url}</a>
          <div style={{ fontSize: "13px", color: "var(--text-500)", lineHeight: 1.65 }}>{item.snippet}</div>
        </div>
      ))}
    </div>
  );
}

/* ── Fallback: plain readable text ────────────────────────────────────────── */
function RawText({ text }: { text: string }) {
  return (
    <div style={{
      fontSize: "13px",
      color: "var(--text-700)",
      lineHeight: 1.8,
      whiteSpace: "pre-wrap",
      backgroundColor: "var(--bg-muted)",
      border: "1px solid var(--border)",
      borderRadius: "var(--r-md)",
      padding: "14px 16px",
      maxHeight: "280px",
      overflowY: "auto",
    }}>
      {text}
    </div>
  );
}

/* ── Main component ───────────────────────────────────────────────────────── */
export default function AgentStep({
  title, subtitle, content, isLoading, isComplete, stepNumber, isLast,
}: AgentStepProps) {
  const state = isLoading ? "loading" : isComplete ? "complete" : "idle";
  const idx   = stepNumber - 1;

  /* Status pill */
  const pillStyles: Record<string, CSSProperties> = {
    idle: {
      fontSize: "11px", fontWeight: 500,
      color: "var(--text-400)", backgroundColor: "var(--bg-muted)",
      border: "1px solid var(--border)",
      borderRadius: "99px", padding: "3px 10px",
      display: "flex", alignItems: "center", gap: "5px",
    },
    loading: {
      fontSize: "11px", fontWeight: 500,
      color: "var(--amber-600)", backgroundColor: "var(--amber-50)",
      border: "1px solid #fde68a",
      borderRadius: "99px", padding: "3px 10px",
      display: "flex", alignItems: "center", gap: "5px",
    },
    complete: {
      fontSize: "11px", fontWeight: 500,
      color: "var(--green-600)", backgroundColor: "var(--green-50)",
      border: "1px solid #bbf7d0",
      borderRadius: "99px", padding: "3px 10px",
      display: "flex", alignItems: "center", gap: "5px",
    },
  };

  const dotPulse: CSSProperties = {
    width: "6px", height: "6px", borderRadius: "50%", flexShrink: 0,
    backgroundColor: state === "loading" ? "var(--amber-500)"
      : state === "complete" ? "var(--green-500)"
      : "var(--text-300)",
    animation: state === "loading" ? "pulse-ring 1s ease-in-out infinite" : undefined,
  };

  /* Step number indicator */
  const numBg: Record<string, { bg: string; border: string; color: string }> = {
    idle:     { bg: "var(--white)",    border: "var(--border)",  color: "var(--text-400)" },
    loading:  { bg: "var(--amber-50)", border: "#fde68a",         color: "var(--amber-600)" },
    complete: { bg: "var(--indigo-50)",border: "var(--indigo-100)",color: "var(--indigo-600)" },
  };

  /* Decide renderer */
  function renderContent() {
    if (!content) return null;
    if (idx === 0) return <FlightOutput text={content} />;
    if (idx === 1) return <SearchOutput text={content} />;
    return <RawText text={content} />;
  }

  return (
    <div style={{ display: "flex", gap: "0", marginBottom: isLast ? 0 : "4px" }}>

      {/* Left: connector line + step circle */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginRight: "14px" }}>
        <div style={{
          width: "32px", height: "32px", borderRadius: "50%", flexShrink: 0,
          backgroundColor: numBg[state].bg,
          border: `1.5px solid ${numBg[state].border}`,
          color: numBg[state].color,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "13px", fontWeight: 700,
          transition: "all 0.25s ease",
          boxShadow: state === "complete" ? "0 0 0 4px rgba(99,102,241,0.08)" : undefined,
        }}>
          {isComplete
            ? <CheckCircle2 size={16} color="var(--indigo-600)" strokeWidth={2.5} />
            : state === "loading"
              ? <Circle size={16} color="var(--amber-500)" strokeWidth={2.5} style={{ animation: "pulse-ring 1s ease-in-out infinite" }} />
              : <span style={{ fontSize: "12px" }}>{stepNumber}</span>
          }
        </div>
        {/* Vertical connector */}
        {!isLast && (
          <div style={{
            width: "1.5px", flex: 1, minHeight: "16px", marginTop: "4px",
            backgroundColor: isComplete ? "var(--indigo-100)" : "var(--border)",
          }} />
        )}
      </div>

      {/* Right: card */}
      <div style={{
        flex: 1, paddingBottom: isLast ? 0 : "16px",
        animation: isComplete ? "fadeUp 0.3s ease both" : undefined,
      }}>
        <div style={{
          backgroundColor: "var(--white)",
          border: `1px solid ${state === "complete" ? "var(--border)" : "var(--border)"}`,
          borderRadius: "var(--r-lg)",
          overflow: "hidden",
          boxShadow: "var(--shadow-sm)",
        }}>
          {/* Card header */}
          <div style={{
            display: "flex", alignItems: "center", gap: "10px",
            padding: "13px 16px",
            borderBottom: (isComplete && content) ? "1px solid var(--border)" : "none",
          }}>
            {/* Icon badge */}
            <div style={{
              width: "28px", height: "28px", borderRadius: "var(--r-sm)",
              backgroundColor: state === "complete" ? "var(--indigo-50)" : "var(--bg-muted)",
              border: `1px solid ${state === "complete" ? "var(--indigo-100)" : "var(--border)"}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              color: state === "complete" ? "var(--indigo-600)" : "var(--text-400)",
              transition: "all 0.2s ease",
            }}>
              {ICONS[idx]}
            </div>

            <div style={{ flex: 1 }}>
              <div style={{ fontSize: "13px", fontWeight: 600, color: state === "idle" ? "var(--text-400)" : "var(--text-900)", transition: "color 0.2s" }}>
                {title}
              </div>
              <div style={{ fontSize: "12px", color: "var(--text-400)", marginTop: "1px" }}>
                {subtitle}
              </div>
            </div>

            {/* Status pill */}
            <div style={pillStyles[state]}>
              <div style={dotPulse} />
              {state === "idle" ? "Waiting" : state === "loading" ? "Running" : "Completed"}
            </div>
          </div>

          {/* Content body */}
          {isComplete && content && (
            <div style={{ padding: "14px 16px" }} className="slide-in">
              {renderContent()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
