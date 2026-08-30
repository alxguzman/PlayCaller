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

from fastapi import APIRouter, HTTPException, Request

from src.live import espn

router = APIRouter(prefix="/live", tags=["live"])


def _espn_call(fn, *args):
    """Run an espn fetch, converting network failures into a clean 502."""
    try:
        return fn(*args)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HTTPException(status_code=404, detail="Game not found.")
        if e.code in (401, 403):
            # ESPN filters by headers and by client IP; cloud/datacenter
            # ranges get blocked even with browser-like headers. Nothing
            # the user can fix, so say so plainly instead of "HTTP 403".
            raise HTTPException(
                status_code=502,
                detail="ESPN is blocking requests from this server, so live "
                "game data isn't available here. The rest of the app "
                "(play caller, season scorecard) works normally.",
            )
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


# --- browser-relayed variants -------------------------------------------
# ESPN rate-limits by IP and blocks shared cloud ranges, so the server
# often can't fetch at all from a host like Render. These endpoints take
# the raw ESPN payload that the VIEWER's browser fetched (their own IP,
# their own quota - the same request espn.com makes) and run it through
# the identical parser, so the concept-mapping logic stays in one place
# instead of being duplicated in JavaScript. No outbound network here.

@router.post("/parse/scoreboard")
async def parse_scoreboard(request: Request):
    """Parse a raw ESPN scoreboard payload relayed by the browser."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be ESPN scoreboard JSON.")
    try:
        return {"games": espn.parse_scoreboard(data)}
    except (AttributeError, TypeError, KeyError, ValueError):
        raise HTTPException(status_code=422, detail="Unrecognized ESPN scoreboard payload.")


@router.post("/parse/game")
async def parse_game(request: Request):
    """Parse a raw ESPN game-summary payload relayed by the browser."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be ESPN summary JSON.")
    try:
        return espn.parse_game_state(data)
    except (AttributeError, TypeError, KeyError, ValueError, IndexError):
        raise HTTPException(status_code=422, detail="Unrecognized ESPN summary payload.")


@router.get("/games/{event_id}")
def live_game_state(event_id: str):
    """
    One game's full live state: score/clock, the current situation shaped
    like a /recommend request (or null between plays), and the most
    recent scrimmage plays already classified into model concepts.
    """
    data = _espn_call(espn.get_game_summary, event_id)
    return espn.parse_game_state(data)
