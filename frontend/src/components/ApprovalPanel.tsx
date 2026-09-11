/**
 * ApprovalPanel.tsx
 * The human-in-the-loop gate. While the graph is suspended at its approval node,
 * this is the only thing that can move the run forward: approve the draft, or
 * send feedback and have the final agent rewrite it.
 */

import { useState } from "react";
import type { CSSProperties } from "react";
import { Check, Loader2, PenLine, UserCheck } from "lucide-react";

interface ApprovalPanelProps {
  prompt: string;
  isSubmitting: boolean;
  onSubmit: (approved: boolean, feedback: string) => void;
}

const s: Record<string, CSSProperties> = {
  card: {
    backgroundColor: "var(--white)",
    border: "1px solid var(--amber-500)",
    borderRadius: "var(--r-lg)",
    boxShadow: "var(--shadow-md)",
    overflow: "hidden",
    marginTop: "24px",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    padding: "13px 16px",
    backgroundColor: "var(--amber-50)",
    borderBottom: "1px solid #fde68a",
  },
  iconWrap: {
    width: "28px",
    height: "28px",
    borderRadius: "var(--r-sm)",
    backgroundColor: "var(--white)",
    border: "1px solid #fde68a",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "var(--amber-600)",
    flexShrink: 0,
  },
  title: { fontSize: "13px", fontWeight: 600, color: "var(--text-900)" },
  subtitle: { fontSize: "12px", color: "var(--amber-600)", marginTop: "1px" },
  body: { padding: "16px" },
  prompt: {
    fontSize: "13px",
    color: "var(--text-700)",
    lineHeight: 1.75,
    marginBottom: "14px",
  },
  choiceRow: { display: "flex", gap: "8px", marginBottom: "14px", flexWrap: "wrap" },
  choice: {
    display: "flex",
    alignItems: "center",
    gap: "7px",
    fontSize: "13px",
    fontWeight: 500,
    padding: "8px 16px",
    borderRadius: "var(--r-md)",
    cursor: "pointer",
    fontFamily: "inherit",
    transition: "all 0.15s",
    backgroundColor: "var(--white)",
    border: "1px solid var(--border-input)",
    color: "var(--text-700)",
  },
  choiceActiveApprove: {
    backgroundColor: "var(--green-50)",
    border: "1px solid var(--green-500)",
    color: "var(--green-600)",
  },
  choiceActiveRevise: {
    backgroundColor: "var(--indigo-50)",
    border: "1px solid var(--indigo-500)",
    color: "var(--indigo-600)",
  },
  label: {
    display: "block",
    fontSize: "12px",
    fontWeight: 600,
    color: "var(--text-700)",
    marginBottom: "6px",
  },
  textarea: {
    width: "100%",
    backgroundColor: "var(--white)",
    border: "1px solid var(--border-input)",
    borderRadius: "var(--r-md)",
    color: "var(--text-900)",
    fontSize: "14px",
    padding: "9px 12px",
    fontFamily: "inherit",
    lineHeight: 1.65,
    resize: "vertical",
    minHeight: "80px",
    marginBottom: "14px",
  },
  submit: {
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
    boxShadow: "0 1px 2px rgba(79,70,229,0.25)",
  },
  submitDisabled: { opacity: 0.5, cursor: "not-allowed" },
  spinner: { animation: "spin 0.7s linear infinite", flexShrink: 0 },
  hint: { fontSize: "12px", color: "var(--text-400)", marginTop: "10px", lineHeight: 1.6 },
};

export default function ApprovalPanel({
  prompt,
  isSubmitting,
  onSubmit,
}: ApprovalPanelProps) {
  const [approved, setApproved] = useState(true);
  const [feedback, setFeedback] = useState("");

  // Rejecting without saying why gives the final agent nothing to act on, so the
  // submit button stays disabled until there is feedback to send.
  const disabled = isSubmitting || (!approved && !feedback.trim());

  return (
    <div style={s.card} className="fade-up">
      <div style={s.header}>
        <div style={s.iconWrap}>
          <UserCheck size={14} strokeWidth={2} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={s.title}>Human approval required</div>
          <div style={s.subtitle}>The workflow is paused until you respond</div>
        </div>
      </div>

      <div style={s.body}>
        {prompt && <p style={s.prompt}>{prompt}</p>}

        <div style={s.choiceRow}>
          <button
            type="button"
            onClick={() => setApproved(true)}
            disabled={isSubmitting}
            style={{ ...s.choice, ...(approved ? s.choiceActiveApprove : {}) }}
          >
            <Check size={14} strokeWidth={2.5} />
            Approve as is
          </button>
          <button
            type="button"
            onClick={() => setApproved(false)}
            disabled={isSubmitting}
            style={{ ...s.choice, ...(!approved ? s.choiceActiveRevise : {}) }}
          >
            <PenLine size={14} strokeWidth={2.5} />
            Request changes
          </button>
        </div>

        {!approved && (
          <div className="slide-in">
            <label htmlFor="feedback" style={s.label}>
              What should change?
            </label>
            <textarea
              id="feedback"
              value={feedback}
              onChange={(event) => setFeedback(event.target.value)}
              placeholder="e.g. Too rushed. Make it five days and drop one city."
              disabled={isSubmitting}
              style={s.textarea}
            />
          </div>
        )}

        <button
          type="button"
          onClick={() => onSubmit(approved, feedback)}
          disabled={disabled}
          style={{ ...s.submit, ...(disabled ? s.submitDisabled : {}) }}
        >
          {isSubmitting && <Loader2 size={14} style={s.spinner} />}
          {isSubmitting
            ? "Finalising…"
            : approved
              ? "Approve and finalise"
              : "Send feedback and revise"}
        </button>

        <p style={s.hint}>
          Your answer resumes the suspended run on this thread. The draft is saved in
          PostgreSQL, so it survives a page reload.
        </p>
      </div>
    </div>
  );
}
