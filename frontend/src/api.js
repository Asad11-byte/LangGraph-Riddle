/**
 * api.js
 * ------
 * Thin fetch wrapper around the FastAPI backend. Every function returns
 * the parsed JSON body or throws an Error with the backend's detail
 * message so components can surface it directly.
 *
 * Set VITE_API_BASE_URL in a .env file (or Vercel project env vars) to
 * point at the deployed backend, e.g.:
 *   VITE_API_BASE_URL=https://your-backend.vercel.app
 */
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* response wasn't JSON; keep statusText */
    }
    throw new Error(detail);
  }

  return res.json();
}

/** Start a new game. Returns the initial GameResponse (riddle + thread_id). */
export function startGame({ playerId, difficulty = "medium", theme = "general", maxAttempts = 3 }) {
  return request("/start-game", {
    method: "POST",
    body: JSON.stringify({
      player_id: playerId,
      difficulty,
      theme,
      max_attempts: maxAttempts,
    }),
  });
}

/** Submit a guess for an in-progress (paused) thread. Resumes the graph. */
export function submitGuess({ threadId, guess }) {
  return request("/submit-guess", {
    method: "POST",
    body: JSON.stringify({ thread_id: threadId, guess }),
  });
}

/** Read-only snapshot of a thread's current state. */
export function getGameState(threadId) {
  return request(`/game-state/${threadId}`);
}

/** Player's persisted score/streak. */
export function getPlayerScore(playerId) {
  return request(`/player-score/${playerId}`);
}
