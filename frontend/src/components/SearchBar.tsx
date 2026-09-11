/**
 * SearchBar.tsx
 * Elevated white card, icon-enhanced inputs, indigo gradient button.
 */

import type { CSSProperties, FormEvent } from "react";
import { User, MapPin, Loader2 } from "lucide-react";

interface SearchBarProps {
  query: string;
  sessionId: string;
  isLoading: boolean;
  onQueryChange: (v: string) => void;
  onSessionIdChange: (v: string) => void;
  onSubmit: () => void;
}

const s: Record<string, CSSProperties> = {
  card: {
    backgroundColor: "var(--white)",
    border: "1px solid var(--border)",
    borderRadius: "var(--r-2xl)",
    boxShadow: "var(--shadow-lg)",
    overflow: "hidden",
    marginBottom: "28px",
  },
  body: {
    padding: "28px 28px 24px",
    display: "grid",
    gridTemplateColumns: "220px 1fr",
    gap: "20px",
    alignItems: "start",
  },
  field: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "7px",
  },
  labelRow: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  label: {
    fontSize: "13px",
    fontWeight: 600,
    color: "var(--text-700)",
    letterSpacing: "-0.005em",
  },
  labelIcon: {
    color: "var(--text-400)",
    flexShrink: 0,
  },
  inputWrap: {
    position: "relative" as const,
  },
  inputIcon: {
    position: "absolute" as const,
    left: "11px",
    top: "50%",
    transform: "translateY(-50%)",
    color: "var(--text-400)",
    pointerEvents: "none" as const,
  },
  input: {
    width: "100%",
    backgroundColor: "var(--white)",
    border: "1px solid var(--border-input)",
    borderRadius: "var(--r-md)",
    color: "var(--text-900)",
    fontSize: "14px",
    padding: "9px 12px 9px 34px",
    fontFamily: "inherit",
    boxShadow: "var(--shadow-sm)",
    transition: "border-color 0.15s, box-shadow 0.15s",
  },
  textareaWrap: {
    position: "relative" as const,
  },
  textareaIcon: {
    position: "absolute" as const,
    left: "11px",
    top: "11px",
    color: "var(--text-400)",
    pointerEvents: "none" as const,
  },
  textarea: {
    width: "100%",
    backgroundColor: "var(--white)",
    border: "1px solid var(--border-input)",
    borderRadius: "var(--r-md)",
    color: "var(--text-900)",
    fontSize: "14px",
    padding: "9px 12px 9px 34px",
    fontFamily: "inherit",
    lineHeight: 1.65,
    resize: "vertical" as const,
    minHeight: "100px",
    boxShadow: "var(--shadow-sm)",
    transition: "border-color 0.15s, box-shadow 0.15s",
  },
  hint: {
    fontSize: "12px",
    color: "var(--text-400)",
    lineHeight: 1.5,
  },

  /* Footer strip */
  footer: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "16px",
    padding: "16px 28px",
    backgroundColor: "var(--bg-muted)",
    borderTop: "1px solid var(--border)",
    flexWrap: "wrap" as const,
  },
  footerNote: {
    fontSize: "12px",
    color: "var(--text-400)",
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  footerDot: {
    width: "4px",
    height: "4px",
    borderRadius: "50%",
    backgroundColor: "var(--text-300)",
  },

  /* Button — indigo gradient, hover lift */
  button: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "9px 22px",
    background: "linear-gradient(135deg, var(--indigo-600) 0%, var(--indigo-500) 100%)",
    color: "#fff",
    border: "1px solid var(--indigo-700)",
    borderRadius: "var(--r-md)",
    fontSize: "13px",
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "inherit",
    letterSpacing: "-0.005em",
    boxShadow: "0 1px 2px rgba(79,70,229,0.25), inset 0 1px 0 rgba(255,255,255,0.12)",
    transition: "filter 0.15s, transform 0.1s, box-shadow 0.15s",
    flexShrink: 0,
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: "not-allowed",
    filter: "none",
  },
  spinner: {
    animation: "spin 0.7s linear infinite",
    flexShrink: 0,
  },
};

const STEPS = ["Guardrail", "Supervisor", "Specialists", "Your review"];

export default function SearchBar({
  query, sessionId, isLoading,
  onQueryChange, onSessionIdChange, onSubmit,
}: SearchBarProps) {
  const disabled = isLoading || !query.trim() || !sessionId.trim();

  return (
    <form onSubmit={(e: FormEvent) => { e.preventDefault(); onSubmit(); }} style={s.card}>
      <div style={s.body}>

        {/* Session Name */}
        <div style={s.field}>
          <div style={s.labelRow}>
            <User size={13} style={s.labelIcon} />
            <label htmlFor="session-id" style={s.label}>Session name</label>
          </div>
          <div style={s.inputWrap}>
            <User size={14} style={s.inputIcon} />
            <input
              id="session-id"
              type="text"
              placeholder="e.g. mehdi"
              value={sessionId}
              onChange={(e) => onSessionIdChange(e.target.value)}
              disabled={isLoading}
              style={s.input}
              required
            />
          </div>
          <span style={s.hint}>
            Also the thread id. Reuse it to reattach to an existing thread.
          </span>
        </div>

        {/* Travel Request */}
        <div style={s.field}>
          <div style={s.labelRow}>
            <MapPin size={13} style={s.labelIcon} />
            <label htmlFor="query" style={s.label}>Travel request</label>
          </div>
          <div style={s.textareaWrap}>
            <MapPin size={14} style={s.textareaIcon} />
            <textarea
              id="query"
              placeholder="e.g. Plan a 5-day trip to Tokyo in December, budget $2,000"
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
              disabled={isLoading}
              style={s.textarea}
              required
            />
          </div>
        </div>

      </div>

      {/* Footer */}
      <div style={s.footer}>
        <div style={s.footerNote}>
          {STEPS.map((step, i) => (
            <span key={step} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              {i > 0 && <div style={s.footerDot} />}
              {step}
            </span>
          ))}
        </div>

        <button
          type="submit"
          disabled={disabled}
          style={{ ...s.button, ...(disabled ? s.buttonDisabled : {}) }}
        >
          {isLoading
            ? <Loader2 size={14} style={s.spinner} />
            : null}
          {isLoading ? "Planning…" : "Create draft plan"}
        </button>
      </div>
    </form>
  );
}
