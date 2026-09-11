/**
 * travel.ts - The only module that talks to the FastAPI backend.
 *
 * Human-in-the-loop makes planning a two-call flow. `createPlan` runs the graph
 * until the approval node suspends it and returns the draft. `submitApproval`
 * resumes that same run with the reviewer's verdict. Both calls are tied together
 * by threadId, which the backend uses to find the suspended run in PostgreSQL.
 */

const API_BASE = "http://localhost:8000";

/** The five specialists the supervisor can schedule, in running order. */
export const AGENT_KEYS = [
  "flight_agent",
  "hotel_agent",
  "weather_agent",
  "budget_agent",
  "itinerary_agent",
] as const;

export type AgentKey = (typeof AGENT_KEYS)[number];

export interface TripConstraints {
  destination?: string;
  origin?: string;
  duration?: string;
  budget?: string;
  travel_style?: string;
  special_preferences?: string[];
}

/** One shape covers every phase, so the UI reads the same fields throughout. */
export interface PlanResponse {
  thread_id: string;

  /** Set when the input guardrail refused the request. */
  blocked: boolean;
  blocked_reason: string;

  /** The supervisor's routing decision. */
  selected_agents: AgentKey[];
  supervisor_reasoning: string;
  trip_constraints: TripConstraints;

  /** Specialist output. Empty for any agent the supervisor skipped. */
  flight_results: string;
  hotel_results: string;
  weather_results: string;
  budget_results: string;

  /** Draft plan and the pending review. */
  itinerary: string;
  approval_request: string;
  awaiting_approval: boolean;

  /** Outcome after review. */
  approved: boolean;
  human_feedback: string;
  final_response: string;

  llm_calls: number;
}

export interface HealthResponse {
  status: string;
  model: string;
  mcp_servers_ready: string[];
  mcp_servers_unavailable: Record<string, string>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // fetch only rejects when the request never reached the server, so this is
    // a connectivity problem rather than an application error.
    throw new Error(
      "Could not reach the planning API. Start it with: uvicorn api:server --reload --port 8000"
    );
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed with status ${response.status}.`);
  }

  return response.json() as Promise<T>;
}

/** Phase one: guardrail, supervisor, specialists, then stop for review. */
export function createPlan(query: string, threadId: string): Promise<PlanResponse> {
  return request<PlanResponse>("/api/plan", {
    method: "POST",
    body: JSON.stringify({ query, thread_id: threadId, user_id: threadId }),
  });
}

/** Phase two: resume the suspended run with the reviewer's answer. */
export function submitApproval(
  threadId: string,
  approved: boolean,
  feedback: string
): Promise<PlanResponse> {
  return request<PlanResponse>("/api/approve", {
    method: "POST",
    body: JSON.stringify({ thread_id: threadId, approved, feedback }),
  });
}

/** Read a thread's latest checkpoint, so a page reload does not lose a draft. */
export function readThread(threadId: string): Promise<PlanResponse> {
  return request<PlanResponse>(`/api/thread/${encodeURIComponent(threadId)}`);
}

/** Which MCP servers came up, used to warn about a missing API key. */
export function readHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}
