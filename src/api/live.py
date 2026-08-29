"""
live.py
-------
API routes for the Live tab. Thin wrappers around src/live/espn.py:
they fetch from ESPN (with the short in-process cache) and return the
parsed shapes the frontend renders.

These routes are public like /recommend - they cost nothing per call
(no Claude involved) and the espn module's cache keeps ESPN traffic to
at most one request every few seconds no matter how many viewers.
"""

import urllib.error

from fastapi import APIRouter, HTTPException

from src.live import espn

router = APIRouter(prefix="/live", tags=["live"])


def _espn_call(fn, *args):
    """Run an espn fetch, converting network failures into a clean 502."""
    try:
        return fn(*args)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HTTPException(status_code=404, detail="Game not found.")
        raise HTTPException(status_code=502, detail=f"ESPN returned HTTP {e.code}.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid game id.")
    except (urllib.error.URLError, TimeoutError):
        raise HTTPException(status_code=502, detail="Could not reach ESPN.")


@router.get("/games")
def live_games():
    """Today's NFL games (live, upcoming and final) from the scoreboard."""
    data = _espn_call(espn.get_scoreboard)
    return {"games": espn.parse_scoreboard(data)}


@router.get("/games/{event_id}")
def live_game_state(event_id: str):
    """
    One game's full live state: score/clock, the current situation shaped
    like a /recommend request (or null between plays), and the most
    recent scrimmage plays already classified into model concepts.
    """
    data = _espn_call(espn.get_game_summary, event_id)
    return espn.parse_game_state(data)
