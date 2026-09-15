/**
 * travel.ts - The only module that talks to the FastAPI backend.
 *
 * Human-in-the-loop makes planning a two-call flow. `createPlan` runs the graph
 * until the approval node suspends it and returns the draft. `submitApproval`
 * resumes that same run with the reviewer's verdict. Both calls are tied together
 * by threadId, which the backend uses to find the suspended run in PostgreSQL.
 */

// Read from the build environment so `npm run build` produces a deployable
// bundle. Hardcoding localhost meant the production build could only ever talk
// to the developer's own machine.
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

// A full trip request runs five agents and can take well over a minute, so the
// timeout is generous. Without one, a hung backend leaves the UI spinning
// forever with no way back.
const REQUEST_TIMEOUT_MS = Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS ?? 180_000);

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

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...init,
    });
  } catch (caught) {
    if (caught instanceof DOMException && caught.name === "AbortError") {
      throw new Error(
        `The planning API did not respond within ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s. ` +
          "It may still be working; reuse the same session name to recover the draft."
      );
    }
    // fetch only rejects when the request never reached the server, so anything
    // else here is a connectivity problem rather than an application error.
    throw new Error(
      "Could not reach the planning API. Start it with: uvicorn api:server --reload --port 8000"
    );
  } finally {
    clearTimeout(timer);
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

/* -- Streaming ------------------------------------------------------------- */

/** What the supervisor decided, delivered before the specialists start. */
export interface SupervisorEvent {
  selected_agents: AgentKey[];
  supervisor_reasoning: string;
  trip_constraints: TripConstraints;
  blocked: boolean;
  blocked_reason: string;
}

export interface StreamHandlers {
  /** The routing decision. Arrives first, so skipped agents can be greyed out. */
  onSupervisor?: (event: SupervisorEvent) => void;
  /** One node finished. Used to advance the pipeline display. */
  onNode?: (node: string) => void;
}

/**
 * Phase one, streamed.
 *
 * Reads newline-delimited JSON off the response body so the caller learns which
 * agent is running as it happens. The non-streaming `createPlan` is kept for
 * callers that only want the final answer, and the two return the same shape.
 */
export async function streamPlan(
  query: string,
  threadId: string,
  handlers: StreamHandlers = {}
): Promise<PlanResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/plan/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, thread_id: threadId, user_id: threadId }),
      signal: controller.signal,
    });
  } catch (caught) {
    clearTimeout(timer);
    if (caught instanceof DOMException && caught.name === "AbortError") {
      throw new Error(
        `The planning API did not respond within ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s. ` +
          "It may still be working; reuse the same session name to recover the draft."
      );
    }
    throw new Error(
      "Could not reach the planning API. Start it with: uvicorn api:server --reload --port 8000"
    );
  }

  if (!response.ok || !response.body) {
    clearTimeout(timer);
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed with status ${response.status}.`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";
  let result: PlanResponse | null = null;
  let failure: string | null = null;

  /** Handle one complete NDJSON line. */
  function consume(line: string) {
    const trimmed = line.trim();
    if (!trimmed) return;

    let event: Record<string, unknown>;
    try {
      event = JSON.parse(trimmed);
    } catch {
      // A partial or malformed line is not worth failing the run over.
      return;
    }

    if (event.type === "supervisor") {
      handlers.onSupervisor?.(event as unknown as SupervisorEvent);
    } else if (event.type === "node") {
      handlers.onNode?.(String(event.node));
    } else if (event.type === "done") {
      result = event.result as PlanResponse;
    } else if (event.type === "error") {
      failure = String(event.detail ?? "The planning run failed.");
    }
  }

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Keep the trailing fragment; it is the start of the next line.
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      lines.forEach(consume);
    }
    consume(buffer);
  } finally {
    clearTimeout(timer);
  }

  if (failure) throw new Error(failure);
  if (!result) throw new Error("The planning run ended without returning a plan.");

  return result;
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
