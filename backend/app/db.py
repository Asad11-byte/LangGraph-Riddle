"""
db.py
-----
Player score persistence.

Uses Supabase (Postgres) when SUPABASE_URL / SUPABASE_KEY are configured.
Falls back to an in-memory dictionary otherwise.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -------------------------------------------------------------------
# In-memory fallback
# -------------------------------------------------------------------
_memory_store: dict[str, dict[str, Any]] = {}

# Cached Supabase client
_supabase_client = None


# -------------------------------------------------------------------
# Supabase Client
# -------------------------------------------------------------------
def _get_supabase():
    """
    Return a cached Supabase client.

    Returns:
        Supabase client if configured, otherwise None.
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        return None

    try:
        from supabase import create_client

        _supabase_client = create_client(url, key)
        return _supabase_client

    except ImportError:
        print("⚠️ supabase package is not installed.")
        return None

    except Exception as e:
        print(f"⚠️ Failed to initialize Supabase: {e}")
        return None


# -------------------------------------------------------------------
# Update Score
# -------------------------------------------------------------------
def upsert_player_score(player_id: str, points: int) -> dict:
    """
    Update a player's score, streak, and games played.
    """

    client = _get_supabase()

    if client is not None:
        try:
            response = (
                client.table("player_scores")
                .select("*")
                .eq("player_id", player_id)
                .execute()
            )

            if response.data:
                row = response.data[0]
            else:
                row = {
                    "player_id": player_id,
                    "score": 0,
                    "streak": 0,
                    "games_played": 0,
                }

            row["score"] += points
            row["streak"] = row["streak"] + 1 if points > 0 else 0
            row["games_played"] += 1

            client.table("player_scores").upsert(row).execute()

            return row

        except Exception as e:
            print(f"⚠️ Supabase error: {e}")

    # ---------------- In-memory fallback ----------------
    row = _memory_store.setdefault(
        player_id,
        {
            "player_id": player_id,
            "score": 0,
            "streak": 0,
            "games_played": 0,
        },
    )

    row["score"] += points
    row["streak"] = row["streak"] + 1 if points > 0 else 0
    row["games_played"] += 1

    return dict(row)


# -------------------------------------------------------------------
# Get Score
# -------------------------------------------------------------------
def get_player_score(player_id: str) -> dict:
    """
    Return a player's current score.
    """

    client = _get_supabase()

    if client is not None:
        try:
            response = (
                client.table("player_scores")
                .select("*")
                .eq("player_id", player_id)
                .execute()
            )

            if response.data:
                return response.data[0]

        except Exception as e:
            print(f"⚠️ Supabase error: {e}")

    return dict(
        _memory_store.get(
            player_id,
            {
                "player_id": player_id,
                "score": 0,
                "streak": 0,
                "games_played": 0,
            },
        )
    )