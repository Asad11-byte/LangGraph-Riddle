# AI Riddle & Logic Game Master

An interactive lateral-thinking riddle game built with **LangGraph**
(Evaluator-Optimizer pattern + Human-in-the-Loop breakpoints), a
**FastAPI** backend running on **Python 3.14**, **Groq** for LLM
inference, and a **React (Vite)** client.

```
riddle-game/
├── backend/
│   ├── app/
│   │   ├── state.py     # GameState TypedDict
│   │   ├── tools.py     # 4 tools: generate/evaluate/hint/tally (Groq)
│   │   ├── db.py        # Supabase (or in-memory) score persistence
│   │   ├── nodes.py     # graph node functions (1 tool call each)
│   │   ├── graph.py     # StateGraph wiring + HITL breakpoint + routing
│   │   └── main.py      # FastAPI app: /start-game, /submit-guess, ...
│   ├── api/index.py     # Vercel Python entry point (re-exports `app`)
│   ├── requirements.txt
│   ├── .python-version  # pins Vercel + local tooling to Python 3.14
│   └── vercel.json      # maxDuration + catch-all rewrite only
└── frontend/
    ├── src/
    │   ├── api.js        # fetch wrapper around the backend
    │   ├── App.jsx        # state management + UI
    │   ├── main.jsx
    │   └── index.css
    ├── package.json
    └── vercel.json
```

## 1. Why Groq

Every LLM call in this app (`generate_riddle_seed`, `evaluate_user_guess`,
`provide_dynamic_hint`) sits directly in the request/response path of a
synchronous HTTP call — the player is waiting on the other end. Groq's
LPU inference is very low-latency compared to typical GPU-hosted
inference, which keeps `/start-game` and `/submit-guess` feeling snappy
even though each one does a real model call.

`langchain-groq`'s `ChatGroq` is a drop-in `BaseChatModel`, so
`with_structured_output(...)` works the same way it would with any other
provider — `tools.py` uses it for both `RiddleSeed` (riddle + answer) and
`GuessEvaluation` (correct/incorrect + confidence + rationale).

**Model:** defaults to `openai/gpt-oss-120b`, Groq's current
general-purpose recommendation (Groq deprecated `llama-3.3-70b-versatile`
and `llama-3.1-8b-instant` in June 2026). Override with `GROQ_MODEL_NAME`
if you want a different Groq-hosted model — check Groq's console/docs
for what's currently available, since their lineup changes fairly often.

If `GROQ_API_KEY` isn't set, every tool falls back to a small local
riddle bank + fuzzy string matching, so the whole HITL loop is testable
without any credentials.

## 2. How the LangGraph HITL loop works

The graph implements a static breakpoint via
`graph.compile(checkpointer=..., interrupt_before=["evaluate_guess"])`.

```
START -> generate_riddle -> [PAUSE] -> evaluate_guess -> route_after_evaluation()
                                                                |
                              +---------------------------------+---------------------+
                              |                                 |                     |
                              v                                 v                     v
                         tally_node                        hint_node             failed_node
                        (correct guess)               (incorrect, attempts     (incorrect, no
                              |                          remain -> loops           attempts left)
                             END                        back to evaluate_guess,       |
                                                          re-arming [PAUSE])          END
```

Because `interrupt_before` is declared on the **node name**, LangGraph
pauses execution every time that node is *about to run* — including on
the second, third, etc. pass through the loop. This is what lets the
same mechanism serve every guess in the game, not just the first one.

1. `POST /start-game` creates a fresh `thread_id`, calls
   `graph.invoke(initial_state, config)`. The graph runs
   `generate_riddle`, then halts before `evaluate_guess`. The paused
   state (which includes the riddle) is persisted by the checkpointer,
   keyed on `thread_id`.
2. The riddle is returned to the client; the server holds nothing else
   in memory beyond what the checkpointer already stores.
3. `POST /submit-guess` calls
   `graph.update_state(config, {"user_guess": guess})` to write the
   player's answer into the paused checkpoint, then
   `graph.invoke(None, config)` to resume from exactly where execution
   left off.
4. `evaluate_guess` runs, the conditional edge routes to `tally_node`
   (correct), `hint_node` (incorrect, attempts remain — loops back and
   re-pauses), or `failed_node` (incorrect, no attempts left).

## 3. Important production note: checkpointer choice

`InMemorySaver` (used by default) only lives for the lifetime of one
Python process. That's fine for local development (`uvicorn` keeps
running), but **on serverless platforms like Vercel, each request may
hit a different, freshly cold-started function instance**, so a thread
paused in `/start-game` could be gone by the time `/submit-guess`
arrives.

`backend/app/graph.py::_build_checkpointer()` already handles this: if
a `DATABASE_URL` environment variable is set (a Postgres/Supabase
connection string), it uses `langgraph.checkpoint.postgres.PostgresSaver`
instead, which persists checkpoints outside the process. **Set
`DATABASE_URL` before deploying the backend to Vercel** — you can point
it at the same Supabase project you use for scores (Supabase is
Postgres under the hood, so one connection string covers both) or a
separate Postgres instance.

## 4. Backend setup (local)

Requires **Python 3.14** (see `backend/.python-version`). If you don't
have it yet: `pyenv install 3.14` / `uv python install 3.14`, or grab it
from python.org.

```bash
cd backend
python3.14 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# required for real LLM-generated riddles / semantic grading;
# without it, tools.py falls back to a small local riddle bank +
# fuzzy string matching so the app still runs end-to-end.
export GROQ_API_KEY=gsk_...

# optional: pick a different Groq-hosted model (default: openai/gpt-oss-120b)
export GROQ_MODEL_NAME=openai/gpt-oss-120b

# optional: persistent scores via Supabase (else in-memory)
export SUPABASE_URL=https://xxxx.supabase.co
export SUPABASE_KEY=your-service-role-or-anon-key

# optional locally, but required for correctness once deployed to Vercel
export DATABASE_URL=postgresql://user:pass@host:5432/postgres

uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger UI, or
`http://localhost:8000/health` for a quick liveness check.

Supabase schema for scores (run in the SQL editor):

```sql
create table player_scores (
    player_id text primary key,
    score integer not null default 0,
    streak integer not null default 0,
    games_played integer not null default 0,
    updated_at timestamptz not null default now()
);
```

### API summary

| Method | Path                       | Purpose                                   |
|--------|-----------------------------|--------------------------------------------|
| POST   | `/start-game`                | Create a thread, generate riddle, pause    |
| POST   | `/submit-guess`               | Inject guess into paused thread, resume    |
| GET    | `/game-state/{thread_id}`     | Read-only snapshot (reconnects/debugging)  |
| GET    | `/player-score/{player_id}`   | Current score/streak                        |
| GET    | `/health`                     | Liveness check                              |

## 5. Frontend setup (local)

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_BASE_URL to your backend URL
npm run dev
```

The UI is intentionally minimal — it exists to exercise state
management and the API contract, not to be a polished visual design.
Only color/contrast styling was applied; no new flexbox/grid
containers, padding, or margin structures should be introduced beyond
what's here without deliberate design review. Nothing here changes with
the Groq/Python 3.14 switch — the frontend only ever talks to the
FastAPI JSON API.

## 6. Deploying to Vercel

Vercel's Python runtime is effectively zero-config as of 2026: it
auto-detects an ASGI `app` object at `api/index.py` (which just
re-exports the FastAPI app), reads dependencies from `requirements.txt`,
and reads the Python version from `.python-version` (already set to
`3.14` here). Deploy the backend and frontend as **two separate Vercel
projects**:

**Backend** (`/backend`):
1. Import the repo in Vercel, set root directory to `backend`. No
   framework preset needed — Vercel detects FastAPI automatically.
2. Set env vars in the Vercel dashboard: `GROQ_API_KEY`,
   `DATABASE_URL` (required — see note above), `SUPABASE_URL`,
   `SUPABASE_KEY`, and optionally `GROQ_MODEL_NAME`.
3. Deploy. `vercel.json` here only sets `maxDuration` (Groq is fast, but
   30-60s of headroom is cheap insurance) and a catch-all rewrite to
   `api/index.py` — it's not doing routing/build work itself, since the
   runtime handles that automatically.
4. Locally, you can exercise the exact Vercel runtime with `vercel dev`
   instead of `uvicorn` if you want to test cold-start behavior.

**Frontend** (`/frontend`):
1. Import the repo, set root directory to `frontend`, framework
   preset `Vite`.
2. Set `VITE_API_BASE_URL` to your deployed backend's URL
   (e.g. `https://your-backend.vercel.app`).
3. Deploy — `vercel.json` already configures the build command,
   output directory, and SPA rewrite.

If you'd rather run the backend on a long-lived server (Render,
Railway, Fly.io, a VM) instead of serverless, `InMemorySaver` is fine
and `DATABASE_URL` becomes optional — long-lived processes don't have
the cold-start state-loss problem serverless does.

## 7. Known limitations / next steps

- `evaluate_user_guess` and `provide_dynamic_hint` fall back to simple
  local heuristics when `GROQ_API_KEY` isn't set, so the full loop is
  testable without credentials — this is by design (keeps local dev and
  grading friction-free), but production games should always have a key
  set for real semantic grading and hints.
- CORS is currently wide open (`allow_origins=["*"]`); restrict this to
  your deployed frontend origin before going to production.
- No auth on `/start-game` / `/submit-guess` — add a session or API key
  check before exposing this publicly.
- **Testing caveat:** this was built in a sandbox with no outbound
  network access and only Python 3.12 installed (no 3.14 available to
  install). The Python backend was verified with `py_compile` (all
  modules compile cleanly under 3.12; nothing in the code uses syntax
  exclusive to a specific 3.1x version, so this is a reasonable proxy)
  rather than an actual install + run under 3.14. Before you deploy: run
  `pip install -r requirements.txt` and start the app under a real
  Python 3.14 interpreter, and run `npm install && npm run build` in
  `frontend/`, to catch anything environment-specific.
