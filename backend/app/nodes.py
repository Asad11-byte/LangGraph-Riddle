"""
nodes.py
--------
Node functions for the Evaluator-Optimizer riddle graph.

Each node takes the current `GameState`, invokes exactly one tool, and
returns a partial state update (LangGraph merges this into the running
state). Keeping one tool call per node keeps the graph auditable and
each step independently testable.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from .state import GameState
from .tools import (
    evaluate_user_guess,
    generate_riddle_seed,
    provide_dynamic_hint,
    update_player_tally,
)

POINTS_PER_SOLVE = 100
HINT_PENALTY_PER_ATTEMPT = 15


# ---------------------------------------------------------------------------
# Node: generate_riddle
# ---------------------------------------------------------------------------
def generate_riddle_node(state: GameState) -> dict:
    seed = generate_riddle_seed.invoke(
        {"difficulty": state.get("difficulty", "medium"), "theme": state.get("theme", "general")}
    )
    return {
        "current_riddle": seed["riddle"],
        "target_answer": seed["answer"],
        "failed_attempts": 0,
        "last_hint": None,
        "game_status": "waiting_for_user",
        "messages": [AIMessage(content=f"Riddle: {seed['riddle']}")],
    }


# ---------------------------------------------------------------------------
# Node: evaluate_guess  (the graph is paused with `interrupt_before` right
# before this node runs, which is how we implement the HITL wait step)
# ---------------------------------------------------------------------------
def evaluate_guess_node(state: GameState) -> dict:
    guess = state.get("user_guess") or ""
    result = evaluate_user_guess.invoke(
        {"user_input": guess, "target_answer": state["target_answer"]}
    )
    return {
        "game_status": "evaluating",
        "messages": [HumanMessage(content=guess)],
        # stash the raw evaluation for the conditional router to consult
        "last_evaluation": result,
    }


# ---------------------------------------------------------------------------
# Node: award / tally on a correct solve
# ---------------------------------------------------------------------------
def tally_node(state: GameState) -> dict:
    attempts_penalty = state.get("failed_attempts", 0) * HINT_PENALTY_PER_ATTEMPT
    points = max(POINTS_PER_SOLVE - attempts_penalty, 10)
    tally = update_player_tally.invoke({"player_id": state["player_id"], "points": points})
    return {
        "game_status": "solved",
        "points_awarded": points,
        "player_score": tally["score"],
        "messages": [AIMessage(content=f"Correct! The answer was '{state['target_answer']}'. +{points} points.")],
    }


# ---------------------------------------------------------------------------
# Node: dynamic hint on an incorrect guess (more attempts remain)
# ---------------------------------------------------------------------------
def hint_node(state: GameState) -> dict:
    new_attempts = state.get("failed_attempts", 0) + 1
    hint = provide_dynamic_hint.invoke(
        {
            "failed_attempts": new_attempts,
            "current_riddle": state["current_riddle"],
            "target_answer": state["target_answer"],
        }
    )
    return {
        "failed_attempts": new_attempts,
        "last_hint": hint,
        "game_status": "waiting_for_user",
        "messages": [AIMessage(content=f"Not quite. Hint: {hint}")],
    }


# ---------------------------------------------------------------------------
# Node: out of attempts
# ---------------------------------------------------------------------------
def failed_node(state: GameState) -> dict:
    # Record the loss (0 points) so streak resets, still counts a game played.
    tally = update_player_tally.invoke({"player_id": state["player_id"], "points": 0})
    return {
        "game_status": "failed",
        "points_awarded": 0,
        "player_score": tally["score"],
        "messages": [
            AIMessage(
                content=f"Out of attempts! The answer was '{state['target_answer']}'."
            )
        ],
    }
