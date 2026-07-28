"""
state.py
--------
Defines the shared graph state for the AI Riddle & Logic Game Master.

The state is a TypedDict (LangGraph's preferred schema type for
`StateGraph`) so every node receives and returns a partial update that
LangGraph merges into the running state. `messages` uses the
`add_messages` reducer so chat history is appended rather than
overwritten across HITL resumes.
"""
from __future__ import annotations

from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

GameStatus = Literal[
    "generating",       # riddle is being generated
    "waiting_for_user",  # graph is paused at the HITL breakpoint
    "evaluating",        # user's guess is being scored
    "hint_given",         # incorrect guess, hint issued, waiting again
    "solved",             # user guessed correctly
    "failed",              # user exhausted all attempts
]


class GameState(TypedDict):
    # --- conversation / audit trail -----------------------------------
    messages: Annotated[List[BaseMessage], add_messages]

    # --- riddle content --------------------------------------------------
    difficulty: str
    theme: str
    current_riddle: str
    target_answer: str

    # --- HITL / gameplay bookkeeping -------------------------------------
    user_guess: Optional[str]
    failed_attempts: int
    max_attempts: int
    last_hint: Optional[str]
    last_evaluation: Optional[dict]

    # --- scoring -----------------------------------------------------------
    player_id: str
    player_score: int
    points_awarded: int

    # --- control flag --------------------------------------------------------
    game_status: GameStatus
