# AI Riddle & Logic Game Master

An interactive lateral-thinking riddle engine built around LangGraph's
**Evaluator–Optimizer** agentic pattern with a **Human-in-the-Loop (HITL)**
interrupt at its core. A LangGraph agent generates a puzzle, pauses
execution and hands control back to a human player, evaluates their guess
semantically, and either rewards them or loops back with a progressively
stronger hint — entirely driven by a checkpointed, resumable graph rather
than ad-hoc request/response glue code.

**Stack:** LangGraph + LangChain · FastAPI (Python 3.14) · Groq (LLM
inference) · React/Vite · Supabase (Postgres) · Vercel

---

## Table of contents

1. [Why this is a genuine LangGraph / AI-agent use case](#1-why-this-is-a-genuine-langgraph--ai-agent-use-case)
2. [Architecture](#2-architecture)
3. [Project structure](#3-project-structure)
4. [Graph state](#4-graph-state)
5. [Backend setup (local)](#5-backend-setup-local)
6. [Frontend setup (local)](#6-frontend-setup-local)
7. [API reference](#7-api-reference)
8. [Deploying to Vercel](#8-deploying-to-vercel)
9. [Environment variables](#9-environment-variables)
10. [Limitations & suggested improvements](#10-limitations--suggested-improvements)

---

## 1. Why this is a genuine LangGraph / AI-agent use case

It's easy to bolt an LLM onto a CRUD endpoint and call it "agentic." This
project is deliberately structured to exercise the parts of LangGraph that
a plain request/response API can't easily replicate:

- **Evaluator–Optimizer loop.** The riddle generator (optimizer) and the
  guess evaluator sit in a real feedback loop: a wrong guess doesn't just
  return an error, it routes to a *hint generator* that conditions on
  `failed_attempts` to produce a strictly-more-revealing clue each time,
  then re-enters the same evaluation step. That loop — generate, judge,
  refine, re-judge — is the canonical evaluator-optimizer shape, and it's
  expressed as graph topology (nodes + conditional edges) rather than
  nested `if/else` in a route handler.
- **Long-running, interruptible state, not a single LLM call.** A riddle
  "session" can span an arbitrary number of HTTP requests over an
  arbitrary amount of wall-clock time (a player can walk away and come
  back). LangGraph's checkpointer persists the *entire* agent state —
  message history, attempt count, target answer — outside the process, so
  execution can be suspended and resumed from an arbitrary point without
  the backend holding anything in memory between requests. This is the
  problem LangGraph's persistence layer exists to solve; a stateless REST
  handler would have to reinvent it.
- **True human-in-the-loop, not just a chat turn.** `interrupt()` doesn't
  just "wait for the next message" — it suspends the graph *mid-execution*
  at a specific node, with the ability to resume with a payload
  (`Command(resume=...)`) that gets injected back into that exact point in
  the control flow. The graph doesn't know or care whether it was resumed
  a second later or a day later. This models a real class of agent
  problems — approval gates, clarification requests, "ask the user"
  steps — that a single-shot prompt can't represent.
- **Tools as the unit of capability, not prompt strings.** Each of the
  four required behaviors (generate, evaluate, hint, persist score) is a
  standalone `@tool`-decorated function with a typed signature and
  docstring — the same shape LangChain uses for agent tool-calling. Here
  they're invoked deterministically by graph nodes rather than chosen by
  an LLM's tool-calling decision, but the tools themselves are
  interchangeable with a future version where an LLM *does* pick between
  them (e.g. a "hint vs. re-explain vs. skip" decision), because the
  interface is already agent-compatible.
- **Conditional routing as first-class control flow.** `route_after_evaluation`
  and `route_after_hint` are pure functions of state that LangGraph uses
  to pick the next node. That keeps "what happens next" fully declarative
  and testable in isolation from the LLM calls themselves — you can unit
  test the routing logic with zero API calls.

In short: this app has the three ingredients that make LangGraph the
right tool instead of overhead — **a multi-step feedback loop, state that
outlives a single request, and a real pause/resume boundary controlled by
a human** — rather than using it as a fancy wrapper around one prompt.

## 2. Architecture

```
                         ┌────────────────────┐
   POST /start-game ───▶ │  generate_riddle    │  Groq LLM → RiddleSeed
                         └─────────┬───────────┘
                                   ▼
                         ┌────────────────────┐
                    ┌───▶│    await_guess      │  interrupt() — graph pauses,
                    │    │  (HITL breakpoint)  │  riddle/hint returned to client
                    │    └─────────┬───────────┘
                    │              ▼  Command(resume=guess) from /submit-guess
                    │    ┌────────────────────┐
                    │    │   evaluate_guess     │  Groq LLM → semantic judge
                    │    └─────────┬───────────┘
                    │              ▼
                    │     route_after_evaluation
                    │        │            │
                    │    correct      incorrect
                    │        ▼            ▼
                    │  ┌───────────┐ ┌───────────┐
                    │  │update_tally│ │   hint     │  Groq LLM → progressive clue
                    │  └─────┬─────┘ └─────┬─────┘
                    │        ▼             ▼
                    │       END      route_after_hint
                    │                 │          │
                    │              retry   out_of_attempts
                    └──────────────┘              ▼
                                                  END
```

The FastAPI layer never keeps game state in process memory — every
`/start-game` and `/submit-guess` call is a fresh `graph.invoke()` /
`graph.invoke(Command(resume=...))` keyed on a `thread_id`, and the
LangGraph checkpointer is the sole source of truth for "where is this
game right now."

## 3. Project structure

```
ai-riddle-game/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app: /start-game, /submit-guess
│   │   ├── models.py               # Pydantic request/response schemas
│   │   ├── graph/
│   │   │   ├── state.py            # GameState TypedDict + factory
│   │   │   ├── tools.py            # 4 tools (Groq-backed)
│   │   │   ├── nodes.py            # node functions
│   │   │   ├── edges.py            # conditional routing functions
│   │   │   └── graph.py            # StateGraph wiring + checkpointer selection
│   │   └── db/
│   │       └── supabase_client.py  # score persistence (Supabase or in-memory)
│   ├── api/index.py                # Vercel serverless entrypoint
│   ├── requirements.txt
│   ├── pyproject.toml / .python-version   # pins Python 3.14
│   └── vercel.json
└── frontend/
    ├── src/
    │   ├── api/client.js           # fetch wrapper around the backend
    │   ├── components/RiddleGame.jsx
    │   ├── App.jsx / main.jsx / index.css
    ├── package.json
    └── vercel.json
```

## 4. Graph state

```python
class GameState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    current_riddle: str
    target_answer: str
    difficulty: str
    theme: str
    failed_attempts: int
    max_attempts: int
    player_score: int
    points_per_solve: int
    player_id: str
    user_guess: str
    last_hint: str
    last_evaluation_confidence: float
    game_status: Literal["generating", "waiting_for_user", "evaluating", "solved", "failed"]
```

## 5. Backend setup (local)

Requires **Python 3.14** (see `backend/.python-version`).

```bash
cd backend
python3.14 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# at minimum, set GROQ_API_KEY in .env

uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger UI, or
`http://localhost:8000/health` for a liveness check.

Supabase schema for scores (run in the SQL editor — optional, falls back
to an in-memory store if unset):

```sql
create table if not exists player_scores (
    player_id  text primary key,
    score      integer not null default 0,
    streak     integer not null default 0,
    updated_at timestamptz not null default now()
);
```

## 6. Frontend setup (local)

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api/*` to `http://localhost:8000` (see
`vite.config.js`), so no `.env` is required locally. Set
`VITE_API_BASE_URL` only when pointing at a deployed backend.

## 7. API reference

| Method | Path            | Body                                                  | Purpose                                              |
|--------|-----------------|--------------------------------------------------------|-------------------------------------------------------|
| GET    | `/health`        | —                                                        | Liveness check                                         |
| POST   | `/start-game`     | `player_id, difficulty, theme, max_attempts`             | Creates a thread, generates a riddle, pauses (HITL)     |
| POST   | `/submit-guess`   | `thread_id, guess`                                       | Resumes the paused thread with the player's guess       |

`/submit-guess` returns one of three shapes depending on `game_status`:

- **`waiting_for_user`** — wrong guess, attempts remain: `hint`,
  `attempts_left`, `riddle`.
- **`solved`** — correct guess: `player_score`.
- **`failed`** — out of attempts: `target_answer`.

## 8. Deploying to Vercel

Deploy backend and frontend as **two separate Vercel projects**.

**Backend** (`root directory: backend`):
1. Vercel auto-detects the Python runtime from `api/index.py` and reads
   the interpreter version from `.python-version` (`3.14`).
2. Set env vars: `GROQ_API_KEY`, `CHECKPOINTER=postgres`,
   `SUPABASE_DB_URL`, `SUPABASE_URL`, `SUPABASE_KEY`,
   `CORS_ALLOW_ORIGINS=<your frontend's deployed URL>`.
3. **`CHECKPOINTER=postgres` is not optional in production.** See
   [§10](#10-limitations--suggested-improvements) below.

**Frontend** (`root directory: frontend`, framework preset `Vite`):
1. Set `VITE_API_BASE_URL` to the deployed backend's URL.
2. Deploy — `vercel.json` handles the build command, output directory,
   and SPA rewrite.

## 9. Environment variables

| Variable                | Where          | Required                        | Purpose                                       |
|--------------------------|-----------------|----------------------------------|-------------------------------------------------|
| `GROQ_API_KEY`            | backend         | yes                               | LLM inference                                    |
| `GROQ_MODEL`              | backend         | no (has default)                 | Riddle/eval model override                       |
| `CHECKPOINTER`            | backend         | no (`memory` default)            | `memory` locally, `postgres` on Vercel           |
| `SUPABASE_DB_URL`         | backend         | yes if `CHECKPOINTER=postgres`   | Checkpoint persistence                           |
| `SUPABASE_URL` / `_KEY`   | backend         | no (falls back to in-memory)     | Score persistence                                |
| `CORS_ALLOW_ORIGINS`      | backend         | recommended in prod              | Restrict to your frontend origin                 |
| `VITE_API_BASE_URL`       | frontend        | yes when deployed                | Points the client at the deployed backend        |

## 10. Limitations & suggested improvements

**Correctness / robustness**
- No auth on `/start-game` or `/submit-guess` — anyone with a `thread_id`
  can resume someone else's game. Add a session token or signed
  `player_id` check before exposing this publicly.
- `evaluate_user_guess` relies entirely on the LLM's judgement with no
  fallback; an API outage mid-game currently surfaces as a 500 rather
  than degrading gracefully. A pragmatic improvement: fall back to
  fuzzy/string-distance matching against `target_answer` and
  `acceptable_variants` if the Groq call fails or times out.
- `RiddleSeed.answer` and `acceptable_variants` are generated once and
  never re-validated — nothing currently checks that the LLM's own
  riddle isn't ambiguous or has an unintended second valid answer. Worth
  adding a lightweight self-check pass (a second LLM call or a rule-based
  sanity filter) before a riddle is served.
- CORS defaults to `*` unless `CORS_ALLOW_ORIGINS` is explicitly set —
  fine for local dev, must be locked down before production traffic.

**Architecture**
- `compile_graph()`'s Postgres path opens a connection synchronously at
  import time. On Vercel's per-invocation model this reconnects on every
  cold start; migrating to `AsyncPostgresSaver` with proper connection
  pooling (or a managed pooler like Supabase's PgBouncer endpoint) would
  reduce cold-start latency under real traffic.
- The evaluator-optimizer loop currently has a hard-coded 3-tier hint
  "strength" ladder (`0 / 1 / 2+ failed attempts`). A more general
  version would let `provide_dynamic_hint` see the full hint history so
  it can guarantee monotonically increasing specificity rather than
  relying on prompt instructions alone.
- `update_player_tally` is invoked directly rather than exposed to the
  graph as an LLM-selectable action — appropriate for this fixed-flow
  game, but if the game grows branching outcomes (bonus rounds, streak
  multipliers, etc.), it's a natural candidate to become part of a
  tool-calling node instead of a hard-wired edge target.

**Testing**
- This was built and syntax-checked (`py_compile`) in a sandboxed
  environment without outbound network access, so **no dependency was
  actually installed or run** — `pip install -r requirements.txt`,
  `uvicorn app.main:app`, and `npm install && npm run dev` should all be
  run for the first time in your own environment before relying on this.
- There are no automated tests yet. The routing functions in `edges.py`
  are pure functions of `GameState` and are the highest-value place to
  start (they can be tested with zero LLM calls); node functions would
  need Groq calls mocked out.

**Product**
- Difficulty scaling is currently just a string passed into the
  generation prompt (`"easy" | "medium" | "hard"`) with no feedback loop
  — the system doesn't yet adapt difficulty based on a player's actual
  win rate, which would be a natural next iteration on the
  evaluator-optimizer theme already present in the graph.
- No persistence of riddle history per player, so there's currently no
  way to guarantee a player doesn't see a repeat riddle across sessions.