"""
main.py
-------
FastAPI application exposing the riddle game as an HTTP API.

Endpoints:
    POST /start-game     -> creates a new checkpointed thread, runs the
                             graph up to the HITL breakpoint, returns the
                             riddle.
    POST /submit-guess   -> injects the player's guess into the paused
                             thread's state and resumes the graph.
    GET  /game-state/{thread_id} -> read-only snapshot of a thread (handy
                             for reconnects / debugging).
    GET  /player-score/{player_id} -> current score/streak lookup.

The HITL mechanics:
    1. `riddle_graph.invoke(initial_state, config)` runs until the
       `interrupt_before=["evaluate_guess"]` breakpoint fires, then
       returns control to us with the paused state.
    2. We hand the riddle to the client and store nothing else server
       side -- the checkpointer (InMemorySaver locally, PostgresSaver in
       production) already persists the paused state keyed by `thread_id`.
    3. On `/submit-guess`, we call `riddle_graph.update_state(config, ...)`
       to write the guess into the checkpoint, then
       `riddle_graph.invoke(None, config)` to resume execution from
       exactly where it left off.

Note: "the server holds nothing else server side" below assumes a
persistent checkpointer (Postgres) is configured -- see
graph.py::_build_checkpointer for why that matters on Vercel.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .db import get_player_score
from .graph import riddle_graph

app = FastAPI(title="AI Riddle & Logic Game Master", version="1.0.0")

# CORS: allow the React client (local dev + Vercel deployment) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class StartGameRequest(BaseModel):
    player_id: str = Field(..., description="Unique player identifier")
    difficulty: str = Field("medium", description="easy | medium | hard")
    theme: str = Field("general", description="Loose riddle theme/subject")
    max_attempts: int = Field(3, ge=1, le=10)


class SubmitGuessRequest(BaseModel):
    thread_id: str
    guess: str


class GameResponse(BaseModel):
    thread_id: str
    game_status: str
    current_riddle: Optional[str] = None
    last_hint: Optional[str] = None
    failed_attempts: int = 0
    max_attempts: int = 3
    target_answer: Optional[str] = None  # only populated once the game ends
    points_awarded: Optional[int] = None
    player_score: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _snapshot_to_response(thread_id: str, values: dict) -> GameResponse:
    status = values.get("game_status", "unknown")
    reveal_answer = status in ("solved", "failed")
    return GameResponse(
        thread_id=thread_id,
        game_status=status,
        current_riddle=values.get("current_riddle"),
        last_hint=values.get("last_hint"),
        failed_attempts=values.get("failed_attempts", 0),
        max_attempts=values.get("max_attempts", 3),
        target_answer=values.get("target_answer") if reveal_answer else None,
        points_awarded=values.get("points_awarded"),
        player_score=values.get("player_score"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/start-game", response_model=GameResponse)
def start_game(req: StartGameRequest):
    thread_id = str(uuid.uuid4())
    config = _config_for(thread_id)

    initial_state = {
        "messages": [],
        "difficulty": req.difficulty,
        "theme": req.theme,
        "current_riddle": "",
        "target_answer": "",
        "user_guess": None,
        "failed_attempts": 0,
        "max_attempts": req.max_attempts,
        "last_hint": None,
        "last_evaluation": None,
        "player_id": req.player_id,
        "player_score": 0,
        "points_awarded": 0,
        "game_status": "generating",
    }

    riddle_graph.invoke(initial_state, config)  # runs generate_riddle, then pauses
    snapshot = riddle_graph.get_state(config)
    return _snapshot_to_response(thread_id, snapshot.values)


@app.post("/api/submit-guess", response_model=GameResponse)
def submit_guess(req: SubmitGuessRequest):
    config = _config_for(req.thread_id)

    snapshot = riddle_graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    if snapshot.values.get("game_status") in ("solved", "failed"):
        raise HTTPException(status_code=400, detail="This game has already ended")

    # Inject the player's guess into the paused checkpoint, then resume.
    riddle_graph.update_state(config, {"user_guess": req.guess})
    riddle_graph.invoke(None, config)

    snapshot = riddle_graph.get_state(config)
    return _snapshot_to_response(req.thread_id, snapshot.values)


@app.get("/api/game-state/{thread_id}", response_model=GameResponse)
def game_state(thread_id: str):
    config = _config_for(thread_id)
    snapshot = riddle_graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    return _snapshot_to_response(thread_id, snapshot.values)


@app.get("/api/player-score/{player_id}")
def player_score(player_id: str):
    return get_player_score(player_id)


@app.get("/api/health")
def health():
    return {"status": "ok"}
