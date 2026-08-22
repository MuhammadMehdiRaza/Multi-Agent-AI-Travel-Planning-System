/**
 * travel.ts — API helper
 *
 * This file contains the single function that talks to our FastAPI backend.
 * Keeping API logic here (separate from components) makes the code cleaner
 * and easier to change later.
 */

// The shape of data we send TO the backend
export interface TravelRequest {
  query: string;
  thread_id: string;
}

// The shape of data we get BACK from the backend
export interface TravelResponse {
  flight_results: string;
  hotel_results: string;
  itinerary: string;
  final_response: string;
  llm_calls: number;
}

// The URL of our FastAPI server (running locally on port 8000)
const API_BASE = "http://localhost:8000";

/**
 * sendTravelQuery — calls the backend with the user's query.
 *
 * @param query     - The travel request typed by the user
 * @param thread_id - The session name the user entered (for memory/history)
 * @returns         - The full response from all agents
 */
export async function sendTravelQuery(
  query: string,
  thread_id: string
): Promise<TravelResponse> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, thread_id }),
  });

  // If the server returned an error, throw it so our UI can show an error message
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Something went wrong. Please try again.");
  }

  return response.json();
}
