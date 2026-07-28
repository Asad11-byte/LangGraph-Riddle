"""
tools.py
--------
The four core tools used by the graph nodes.

Each tool is decorated with @tool so it is independently unit-testable
and could also be bound to a tool-calling agent if desired, but in this
graph they are invoked directly from node functions (see nodes.py) for
deterministic, auditable control flow -- appropriate for a game loop
where we always want a specific tool to run at a specific step.

All LLM calls go through `get_llm()` so the model can be swapped via the
MODEL_NAME env var. If no ANTHROPIC_API_KEY is configured, each tool
falls back to a small local heuristic so the app still runs end-to-end
for local/demo purposes.
"""
from __future__ import annotations

import difflib
import json
import os
import random
from typing import Optional

from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "llama-3.3-70b-versatile",
)

_llm = None


def get_llm():
    """
    Lazily initialize the Groq LLM.

    Returns:
        ChatGroq instance if GROQ_API_KEY exists,
        otherwise None so the application can fall
        back to local heuristics.
    """
    global _llm

    if _llm is not None:
        return _llm

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("⚠️ GROQ_API_KEY not found. Using fallback implementation.")
        return None

    _llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.7,
        api_key=api_key,
    )

    return _llm
# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------
class RiddleSeed(BaseModel):
    riddle: str = Field(description="The riddle text presented to the player.")
    answer: str = Field(description="The single, definitive correct answer.")


class GuessEvaluation(BaseModel):
    is_correct: bool = Field(description="True if the guess semantically matches the answer.")
    confidence: float = Field(description="0-1 confidence in the judgement.")
    rationale: str = Field(description="One short sentence explaining the judgement.")


# Local fallback riddle bank, keyed loosely by theme, used only when no
# LLM credentials are available.
_FALLBACK_RIDDLES = [
    {"riddle": "The more of me there is, the less you see. What am I?", "answer": "darkness"},
    {"riddle": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?", "answer": "an echo"},
    {"riddle": "What has keys but no locks, space but no room, and you can enter but not go inside?", "answer": "a keyboard"},
    {"riddle": "The person who makes it sells it. The person who buys it never uses it. The person who uses it never knows they're using it. What is it?", "answer": "a coffin"},
    {"riddle": "I am not alive, but I grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I?", "answer": "fire"},
]


# ---------------------------------------------------------------------------
# Tool 1: generate_riddle_seed
# ---------------------------------------------------------------------------
@tool
def generate_riddle_seed(difficulty: str, theme: str) -> dict:
    """Generate a unique lateral-thinking riddle constrained by difficulty and theme.

    Args:
        difficulty: one of "easy", "medium", "hard".
        theme: a loose subject/theme for the riddle, e.g. "nature", "objects", "abstract".

    Returns:
        dict with keys `riddle` and `answer`.
    """
    llm = get_llm()
    if llm is None:
        seed = random.choice(_FALLBACK_RIDDLES)
        return seed

    structured_llm = llm.with_structured_output(RiddleSeed)
    prompt = (
        "You are a game master who crafts short, clever lateral-thinking riddles "
        "and word riddles (NOT trivia questions).\n"
        f"Difficulty: {difficulty}\n"
        f"Theme: {theme}\n"
        "Constraints:\n"
        "- The riddle must have exactly one clear, defensible answer.\n"
        "- Keep it to 1-3 sentences.\n"
        "- Do not reveal or hint at the answer inside the riddle text.\n"
        "- The answer should be a short word or phrase (1-4 words).\n"
    )
    result: RiddleSeed = structured_llm.invoke(prompt)
    return {"riddle": result.riddle, "answer": result.answer}


# ---------------------------------------------------------------------------
# Tool 2: evaluate_user_guess
# ---------------------------------------------------------------------------
@tool
def evaluate_user_guess(user_input: str, target_answer: str) -> dict:
    """Semantically compare a user's guess against the target answer, tolerating
    phrasing variation (synonyms, articles, plurals) rather than requiring an
    exact string match.

    Args:
        user_input: the player's guess.
        target_answer: the definitive correct answer.

    Returns:
        dict with keys `is_correct` (bool), `confidence` (float), `rationale` (str).
    """
    llm = get_llm()
    if llm is None:
        # Local fallback: normalized fuzzy string match.
        norm_a = _normalize(user_input)
        norm_b = _normalize(target_answer)
        ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
        is_correct = norm_a == norm_b or ratio > 0.82 or norm_a in norm_b or norm_b in norm_a
        return {
            "is_correct": is_correct,
            "confidence": round(ratio, 2),
            "rationale": "Local fuzzy-match fallback (no LLM configured).",
        }

    structured_llm = llm.with_structured_output(GuessEvaluation)
    prompt = (
        "Judge whether the player's guess is semantically equivalent to the "
        "correct riddle answer. Accept synonyms, minor spelling issues, "
        "singular/plural variants, and paraphrases. Reject guesses that are "
        "a different concept.\n\n"
        f"Correct answer: {target_answer}\n"
        f"Player's guess: {user_input}\n"
    )
    result: GuessEvaluation = structured_llm.invoke(prompt)
    return {
        "is_correct": result.is_correct,
        "confidence": result.confidence,
        "rationale": result.rationale,
    }


def _normalize(text: str) -> str:
    text = text.lower().strip()
    for article in ("a ", "an ", "the "):
        if text.startswith(article):
            text = text[len(article):]
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace()).strip()


# ---------------------------------------------------------------------------
# Tool 3: provide_dynamic_hint
# ---------------------------------------------------------------------------
@tool
def provide_dynamic_hint(failed_attempts: int, current_riddle: str, target_answer: str) -> str:
    """Generate a contextual hint whose specificity scales with the number of
    failed attempts so far (more attempts -> more revealing hint).

    Args:
        failed_attempts: number of incorrect guesses so far (>=1).
        current_riddle: the active riddle text, for context.
        target_answer: the correct answer (used to ground the hint, never
            revealed verbatim).

    Returns:
        A single hint string.
    """
    llm = get_llm()
    if llm is None:
        return _fallback_hint(failed_attempts, target_answer)

    strength = {
        1: "vague, only nudges the player's thinking in the right direction",
        2: "moderate, narrows the category or domain of the answer",
        3: "strong, almost gives it away but still requires the final leap",
    }.get(min(failed_attempts, 3), "very strong, nearly reveals the answer")

    prompt = (
        f"Riddle: {current_riddle}\n"
        f"Correct answer (do not state it verbatim): {target_answer}\n"
        f"The player has failed {failed_attempts} time(s).\n"
        f"Write ONE short hint sentence. Hint strength should be: {strength}.\n"
        "Never say the answer word itself."
    )
    response = llm.invoke(prompt)
    return response.content if isinstance(response.content, str) else str(response.content)


def _fallback_hint(failed_attempts: int, target_answer: str) -> str:
    if failed_attempts <= 1:
        return "Think about everyday, non-physical things -- the answer isn't an object you can hold."
    if failed_attempts == 2:
        length = len(target_answer.replace(" ", ""))
        return f"The answer has {length} letters (ignoring spaces)."
    return f"It starts with the letter '{target_answer.strip()[0].upper()}'."


# ---------------------------------------------------------------------------
# Tool 4: update_player_tally
# ---------------------------------------------------------------------------
@tool
def update_player_tally(player_id: str, points: int) -> dict:
    """Persist the player's updated score/streak to the database.

    Args:
        player_id: unique player identifier.
        points: points to add (can be 0 for a loss, to still record the attempt).

    Returns:
        dict with the player's new total score and streak.
    """
    from .db import upsert_player_score

    return upsert_player_score(player_id, points)
