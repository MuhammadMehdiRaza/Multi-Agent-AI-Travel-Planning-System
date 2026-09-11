/**
 * AgentStep.tsx
 * One step in the vertical agent pipeline.
 *
 * A step has four possible states. "skipped" is the one that matters most here:
 * the supervisor routes dynamically, so an agent that was not scheduled has to
 * read as deliberately skipped rather than as failed or still pending.
 */

import type { CSSProperties, ReactNode } from "react";
import {
  Ban,
  CheckCircle2,
  Circle,
  FileText,
  Hotel,
  Loader2,
  Map,
  Plane,
  Wallet,
} from "lucide-react";

export type StepState = "waiting" | "running" | "complete" | "skipped";

interface AgentStepProps {
  title: string;
  subtitle: string;
  icon: "plane" | "hotel" | "cloud" | "wallet" | "map";
  content: string;
  state: StepState;
  isLast: boolean;
}

const ICONS: Record<AgentStepProps["icon"], ReactNode> = {
  plane: <Plane size={14} strokeWidth={2} />,
  hotel: <Hotel size={14} strokeWidth={2} />,
  cloud: <Map size={14} strokeWidth={2} />,
  wallet: <Wallet size={14} strokeWidth={2} />,
  map: <FileText size={14} strokeWidth={2} />,
};

/* -- Content renderers ----------------------------------------------------- */

/** Parse the numbered "1. **Title** / url / snippet" list the search tools emit. */
function parseSearchResults(text: string) {
  const pattern =
    /(\d+)\.\s+\*\*(.+?)\*\*\s*\n\s*(https?:\/\/\S*)\s*\n\s*([\s\S]*?)(?=\n\s*\d+\.\s+\*\*|$)/g;

  const items: { num: string; title: string; url: string; snippet: string }[] = [];
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    items.push({
      num: match[1],
      title: match[2],
      url: match[3],
      snippet: match[4].trim(),
    });
  }

  return items;
}

function SearchOutput({ text }: { text: string }) {
  const items = parseSearchResults(text);
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
      {items.map((item) => (
        <div key={item.num + item.url} style={itemStyle}>
          <div
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: "8px",
              marginBottom: "4px",
            }}
          >
            <span
              style={{
                flexShrink: 0,
                fontSize: "11px",
                fontWeight: 700,
                color: "var(--indigo-600)",
                backgroundColor: "var(--indigo-50)",
                border: "1px solid var(--indigo-100)",
                borderRadius: "99px",
                padding: "2px 7px",
              }}
            >
              #{item.num}
            </span>
            <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-900)" }}>
              {item.title}
            </div>
          </div>
          <a
            href={item.url}
            target="_blank"
            rel="noreferrer"
            style={{
              fontSize: "11px",
              color: "var(--indigo-500)",
              textDecoration: "none",
              display: "block",
              marginBottom: "4px",
              wordBreak: "break-all",
            }}
          >
            {item.url}
          </a>
          <div style={{ fontSize: "13px", color: "var(--text-500)", lineHeight: 1.65 }}>
            {item.snippet}
          </div>
        </div>
      ))}
    </div>
  );
}

/** Highlight the "[X unavailable: ...]" notes a degraded MCP server produces. */
function RawText({ text }: { text: string }) {
  const degraded = /^\[.+(unavailable|failed):/m.test(text);

  return (
    <div
      style={{
        fontSize: "13px",
        color: degraded ? "var(--amber-600)" : "var(--text-700)",
        lineHeight: 1.8,
        whiteSpace: "pre-wrap",
        backgroundColor: degraded ? "var(--amber-50)" : "var(--bg-muted)",
        border: `1px solid ${degraded ? "#fde68a" : "var(--border)"}`,
        borderRadius: "var(--r-md)",
        padding: "14px 16px",
        maxHeight: "320px",
        overflowY: "auto",
      }}
    >
      {text}
    </div>
  );
}

/* -- Status presentation --------------------------------------------------- */

const PILL_TEXT: Record<StepState, string> = {
  waiting: "Waiting",
  running: "Running",
  complete: "Completed",
  skipped: "Skipped by supervisor",
};

const PALETTE: Record<StepState, { fg: string; bg: string; border: string; dot: string }> = {
  waiting: { fg: "var(--text-400)", bg: "var(--bg-muted)", border: "var(--border)", dot: "var(--text-300)" },
  running: { fg: "var(--amber-600)", bg: "var(--amber-50)", border: "#fde68a", dot: "var(--amber-500)" },
  complete: { fg: "var(--green-600)", bg: "var(--green-50)", border: "#bbf7d0", dot: "var(--green-500)" },
  skipped: { fg: "var(--text-400)", bg: "var(--bg-muted)", border: "var(--border-input)", dot: "var(--text-300)" },
};

/* -- Component ------------------------------------------------------------- */

export default function AgentStep({
  title,
  subtitle,
  icon,
  content,
  state,
  isLast,
}: AgentStepProps) {
  const palette = PALETTE[state];
  const showBody = state === "complete" && Boolean(content);
  const useSearchRenderer = icon === "hotel";

  return (
    <div style={{ display: "flex", marginBottom: isLast ? 0 : "4px" }}>
      {/* Connector column */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          marginRight: "14px",
        }}
      >
        <div
          style={{
            width: "32px",
            height: "32px",
            borderRadius: "50%",
            flexShrink: 0,
            backgroundColor: state === "complete" ? "var(--indigo-50)" : palette.bg,
            border: `1.5px solid ${state === "complete" ? "var(--indigo-100)" : palette.border}`,
            color: state === "complete" ? "var(--indigo-600)" : palette.fg,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "12px",
            fontWeight: 700,
            transition: "all 0.25s ease",
            boxShadow: state === "complete" ? "0 0 0 4px rgba(99,102,241,0.08)" : undefined,
          }}
        >
          {state === "complete" && <CheckCircle2 size={16} strokeWidth={2.5} />}
          {state === "running" && (
            <Loader2 size={16} strokeWidth={2.5} style={{ animation: "spin 0.7s linear infinite" }} />
          )}
          {state === "skipped" && <Ban size={15} strokeWidth={2.5} />}
          {state === "waiting" && <Circle size={8} strokeWidth={3} />}
        </div>

        {!isLast && (
          <div
            style={{
              width: "1.5px",
              flex: 1,
              minHeight: "16px",
              marginTop: "4px",
              backgroundColor: state === "complete" ? "var(--indigo-100)" : "var(--border)",
            }}
          />
        )}
      </div>

      {/* Card */}
      <div
        style={{
          flex: 1,
          paddingBottom: isLast ? 0 : "16px",
          animation: state === "complete" ? "fadeUp 0.3s ease both" : undefined,
          opacity: state === "skipped" ? 0.62 : 1,
        }}
      >
        <div
          style={{
            backgroundColor: "var(--white)",
            border: "1px solid var(--border)",
            borderRadius: "var(--r-lg)",
            overflow: "hidden",
            boxShadow: "var(--shadow-sm)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
              padding: "13px 16px",
              borderBottom: showBody ? "1px solid var(--border)" : "none",
            }}
          >
            <div
              style={{
                width: "28px",
                height: "28px",
                borderRadius: "var(--r-sm)",
                backgroundColor: state === "complete" ? "var(--indigo-50)" : "var(--bg-muted)",
                border: `1px solid ${state === "complete" ? "var(--indigo-100)" : "var(--border)"}`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: state === "complete" ? "var(--indigo-600)" : "var(--text-400)",
                flexShrink: 0,
              }}
            >
              {ICONS[icon]}
            </div>

            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  fontSize: "13px",
                  fontWeight: 600,
                  color: state === "complete" ? "var(--text-900)" : "var(--text-500)",
                  textDecoration: state === "skipped" ? "line-through" : "none",
                }}
              >
                {title}
              </div>
              <div style={{ fontSize: "12px", color: "var(--text-400)", marginTop: "1px" }}>
                {subtitle}
              </div>
            </div>

            <div
              style={{
                fontSize: "11px",
                fontWeight: 500,
                color: palette.fg,
                backgroundColor: palette.bg,
                border: `1px solid ${palette.border}`,
                borderRadius: "99px",
                padding: "3px 10px",
                display: "flex",
                alignItems: "center",
                gap: "5px",
                whiteSpace: "nowrap",
                flexShrink: 0,
              }}
            >
              <span
                style={{
                  width: "6px",
                  height: "6px",
                  borderRadius: "50%",
                  backgroundColor: palette.dot,
                  flexShrink: 0,
                  animation: state === "running" ? "pulse-ring 1s ease-in-out infinite" : undefined,
                }}
              />
              {PILL_TEXT[state]}
            </div>
          </div>

          {showBody && (
            <div style={{ padding: "14px 16px" }} className="slide-in">
              {useSearchRenderer ? <SearchOutput text={content} /> : <RawText text={content} />}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
