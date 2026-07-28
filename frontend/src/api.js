/**
 * api.js
 * ------
 * Thin fetch wrapper around the FastAPI backend. Every function returns
 * the parsed JSON body or throws an Error with the backend's detail
 * message so components can surface it directly.
 *
 * All backend routes live under /api/* (see backend/app/main.py's
 * `router = APIRouter(prefix="/api")`), and in the single-project
 * (monorepo) Vercel deployment, frontend and backend share one origin --
 * root vercel.json rewrites /api/:path* to the Python function. So the
 * default here is a same-origin relative path, NOT an absolute URL.
 *
 * Only set VITE_API_BASE_URL if you're running the backend separately
 * from the frontend, e.g. local dev with `uvicorn` on :8000 and the Vite
 * dev server on :5173 (different origins):
 *   VITE_API_BASE_URL=http://localhost:8000
 * Leave it unset (or empty) for the deployed single-project setup.
 */
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}/api${path}`, {
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