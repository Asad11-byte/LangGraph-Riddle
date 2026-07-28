"""
graph.py
--------
Wires nodes.py functions into a LangGraph `StateGraph` implementing the
Evaluator-Optimizer pattern with a Human-in-the-Loop breakpoint.

Flow:

    START -> generate_riddle -> [BREAKPOINT] -> evaluate_guess -> route()
                                                                     |
                          +------------------------------------------+
                          |                    |                     |
                          v                    v                     v
                     tally_node           hint_node             failed_node
                          |                    |                     |
                         END      (loops back to evaluate_guess,     END
                                    which re-arms the breakpoint)

The breakpoint is implemented with `interrupt_before=["evaluate_guess"]`
at compile time, combined with a checkpointer (`InMemorySaver` here; swap
for a persistent checkpointer such as `PostgresSaver` in production --
this module does that automatically when `DATABASE_URL` is set, see
`_build_checkpointer()` below). Because the interrupt is declared on the *node*, not a
single traversal, the graph re-pauses every time it is about to run
`evaluate_guess` again -- which is exactly what we want for repeated
guesses/hints within one riddle.
"""
from __future__ import annotations

import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from .nodes import (
    evaluate_guess_node,
    failed_node,
    generate_riddle_node,
    hint_node,
    tally_node,
)
from .state import GameState

NODE_GENERATE = "generate_riddle"
NODE_EVALUATE = "evaluate_guess"
NODE_TALLY = "tally"
NODE_HINT = "hint"
NODE_FAILED = "failed"


def route_after_evaluation(state: GameState) -> str:
    """Conditional edge: decide what happens after a guess is scored."""
    evaluation = state.get("last_evaluation") or {}
    if evaluation.get("is_correct"):
        return NODE_TALLY

    attempts = state.get("failed_attempts", 0)
    max_attempts = state.get("max_attempts", 3)
    if attempts + 1 >= max_attempts:
        # this incorrect guess was the last one allowed
        return NODE_FAILED
    return NODE_HINT


def build_graph():
    graph = StateGraph(GameState)

    graph.add_node(NODE_GENERATE, generate_riddle_node)
    graph.add_node(NODE_EVALUATE, evaluate_guess_node)
    graph.add_node(NODE_TALLY, tally_node)
    graph.add_node(NODE_HINT, hint_node)
    graph.add_node(NODE_FAILED, failed_node)

    graph.set_entry_point(NODE_GENERATE)
    graph.add_edge(NODE_GENERATE, NODE_EVALUATE)

    graph.add_conditional_edges(
        NODE_EVALUATE,
        route_after_evaluation,
        {
            NODE_TALLY: NODE_TALLY,
            NODE_HINT: NODE_HINT,
            NODE_FAILED: NODE_FAILED,
        },
    )

    # hint_node loops back to evaluate_guess, re-arming the breakpoint
    graph.add_edge(NODE_HINT, NODE_EVALUATE)
    graph.add_edge(NODE_TALLY, END)
    graph.add_edge(NODE_FAILED, END)

    checkpointer = _build_checkpointer()
    compiled = graph.compile(checkpointer=checkpointer, interrupt_before=[NODE_EVALUATE])
    return compiled


def _build_checkpointer():
    """InMemorySaver only persists for the lifetime of one Python process.
    That's fine for local `uvicorn` dev, but on serverless (Vercel), each
    invocation may be a fresh cold-started process, so the paused thread
    from /start-game could vanish before /submit-guess resumes it. If
    DATABASE_URL (a Postgres/Supabase connection string) is set, use a
    persistent Postgres checkpointer instead.
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            saver_ctx = PostgresSaver.from_conn_string(database_url)
            saver = saver_ctx.__enter__()
            saver.setup()
            return saver
        except Exception as exc:  # pragma: no cover
            print(f"[graph] Falling back to InMemorySaver -- Postgres checkpointer unavailable: {exc}")
    return InMemorySaver()


# Singleton compiled graph used by the FastAPI app.
riddle_graph = build_graph()
