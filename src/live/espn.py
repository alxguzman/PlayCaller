"""
espn.py
-------
Fetches and parses live NFL game data from ESPN's public (unofficial)
JSON API. No auth needed. This is what powers the Live tab: nflverse
data only updates nightly, but ESPN updates within seconds of each play.

Two endpoints:
  - scoreboard: today's games + their status
  - summary:    one game's drives/plays + current situation

Everything here is a pure fetch/parse function - no FastAPI, no model.
The API routes live in src/api/live.py and the offline replay tool in
src/live/replay.py both build on this module.

Honesty note on the play-text heuristics: ESPN's play descriptions can
tell us run direction and pass depth, but NOT play action (pa_* concepts)
or reliably screens. So live grading happens at the "family" level -
pa_deep_pass and deep_pass collapse into one family. The rigorous
comparison still comes from nflverse data after each week (see
recommendation_agreement in src/model/evaluate.py).
"""

import json
import threading
import time
import urllib.request

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"

# ESPN team abbreviations that differ from nflverse's (the model was
# trained on nflverse codes, so every team must map to one of those).
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}

# The model's 10 concepts collapse to 8 "families" for live grading,
# because play action can't be detected from ESPN play text.
CONCEPT_FAMILY = {
    "pa_deep_pass": "deep_pass",
    "pa_short_pass": "short_pass",
}

# ---------------------------------------------------------------------------
# Fetching (with a tiny in-process cache so many viewers = one ESPN call)
# ---------------------------------------------------------------------------

_cache = {}  # url -> (fetched_at, parsed_json)
_cache_lock = threading.Lock()

# ESPN filters on request headers: a custom User-Agent ("playcaller/1.0")
# gets a flat 403, and a bare "Mozilla/5.0" is enough from a home IP but
# not from a datacenter one like Render's. These are the headers a real
# browser sends when the espn.com scoreboard page fetches this same
# endpoint - same public data, just asked for the way the site asks.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/nfl/scoreboard",
    "Origin": "https://www.espn.com",
    "Connection": "keep-alive",
}


def fetch_json(url, ttl=8):
    """
    GET a JSON url, caching the result for `ttl` seconds.

    The cache means that no matter how many browser tabs are polling the
    Live tab, this server sends ESPN at most one request per ttl per url.
    """
    now = time.time()
    with _cache_lock:
        hit = _cache.get(url)
        if hit and now - hit[0] < ttl:
            return hit[1]

    req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.load(resp)

    with _cache_lock:
        _cache[url] = (now, data)
    return data


def get_scoreboard():
    return fetch_json(SCOREBOARD_URL, ttl=20)


def get_game_summary(event_id):
    if not str(event_id).isdigit():
        raise ValueError(f"bad event id: {event_id!r}")
    return fetch_json(SUMMARY_URL.format(event_id=event_id), ttl=8)


def to_nflverse(abbr):
    """ESPN team code -> the code the model was trained on."""
    return ESPN_TO_NFLVERSE.get(abbr, abbr)


# ---------------------------------------------------------------------------
# Scoreboard parsing
# ---------------------------------------------------------------------------

def parse_scoreboard(data):
    """
    Flatten the scoreboard into a simple list the frontend can render.

    state is "pre" (not started), "in" (live) or "post" (final).
    """
    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = event.get("status", {})
        teams = {"home": None, "away": None}
        for c in comp.get("competitors", []):
            side = c.get("homeAway")
            if side in teams:
                teams[side] = {
                    "abbr": to_nflverse(c.get("team", {}).get("abbreviation", "")),
                    "score": int(c.get("score") or 0),
                }
        games.append({
            "id": event.get("id"),
            "name": event.get("name"),
            "short_name": event.get("shortName"),
            "state": status.get("type", {}).get("state", "pre"),
            "detail": status.get("type", {}).get("shortDetail", ""),
            "home": teams["home"],
            "away": teams["away"],
        })
    return games


# ---------------------------------------------------------------------------
# Play classification
# ---------------------------------------------------------------------------

# Play types that are not offensive scrimmage snaps - never graded.
_SKIP_TYPE_WORDS = (
    "kickoff", "punt", "field goal", "extra point", "timeout",
    "end period", "end of", "two-minute", "coin toss", "penalty",
    "official", "delay", "period start", "game start",
)


def classify_play(type_text, text):
    """
    Map an ESPN play (its type label + description text) onto one of the
    model's play concepts, or None when the concept can't be determined.

    Returns (concept, is_scrimmage):
      - ("inside_run", True)   normal graded snap
      - (None, True)           scrimmage snap we can't grade (sack, scramble)
      - (None, False)          not a scrimmage play at all (kick, penalty...)
    """
    tt = (type_text or "").lower()
    tx = (text or "").lower()

    if any(w in tt for w in _SKIP_TYPE_WORDS):
        return None, False
    if "kneel" in tt or "kneel" in tx or "spike" in tt or "spiked" in tx:
        return None, False

    if "sneak" in tx:
        return "qb_sneak", True

    if "sack" in tt:
        return None, True  # pass play, but depth unknowable - excluded (matches training)

    is_rush = "rush" in tt
    is_pass = "pass" in tt or "interception" in tt or "reception" in tt or "incompletion" in tt

    if is_rush:
        if "scramble" in tx:
            return None, True  # scrambles are unlabeled in training too
        if any(w in tx for w in ("up the middle", "left guard", "right guard")):
            return "inside_run", True
        if "left tackle" in tx or "right tackle" in tx:
            return "off_tackle_run", True
        if "left end" in tx or "right end" in tx:
            return "outside_run", True
        return "inside_run", True  # direction unknown - most common run type

    if is_pass:
        if "screen" in tx:
            return "screen_pass", True
        if "deep" in tx:
            return "deep_pass", True
        return "short_pass", True

    if "fumble" in tt:
        # Fumble on a scrimmage play; the text usually still says how the
        # play started ("up the middle", "pass short left"...).
        if "pass" in tx:
            return ("deep_pass" if "deep" in tx else "short_pass"), True
        if any(w in tx for w in ("up the middle", "left guard", "right guard")):
            return "inside_run", True
        if "left tackle" in tx or "right tackle" in tx:
            return "off_tackle_run", True
        if "left end" in tx or "right end" in tx:
            return "outside_run", True
        return None, True

    return None, False


def concept_family(concept):
    """Collapse pa_* concepts into their gradable family."""
    if concept is None:
        return None
    return CONCEPT_FAMILY.get(concept, concept)


def compute_success(down, distance, yards, turnover, touchdown):
    """
    The standard 40/60/100 success rule (a stand-in for the model's real
    training target, EPA > 0, which can't be computed live):
      1st down: gain at least 40% of the distance
      2nd down: gain at least 60%
      3rd/4th : convert (gain all of it)
    Turnovers always fail; offensive touchdowns always succeed.
    """
    if turnover:
        return False
    if touchdown:
        return True
    if distance <= 0:
        return yards > 0
    if down == 1:
        return yards >= 0.4 * distance
    if down == 2:
        return yards >= 0.6 * distance
    return yards >= distance


def map_play(play, team_abbrs):
    """
    Convert one ESPN play dict into a graded row, or None if it isn't a
    scrimmage play.

    team_abbrs: {espn_team_id: nflverse_abbr} built from the game header,
    used to resolve which team had the ball (play.start.team.id).
    """
    type_text = play.get("type", {}).get("text", "")
    text = play.get("text", "")
    concept, is_scrimmage = classify_play(type_text, text)
    if not is_scrimmage:
        return None

    start = play.get("start", {})
    down = start.get("down") or 0
    distance = start.get("distance") or 0
    yardline_100 = start.get("yardsToEndzone") or 0
    if not (1 <= down <= 4) or not (1 <= yardline_100 <= 99):
        return None

    tt = type_text.lower()
    yards = play.get("statYardage") or 0
    turnover = "interception" in tt or "fumble recovery (opponent)" in tt
    touchdown = bool(play.get("scoringPlay")) and "touchdown" in tt and not turnover

    posteam = team_abbrs.get(str(start.get("team", {}).get("id", "")), "")

    return {
        "id": play.get("id"),
        "text": text,
        "concept": concept,
        "family": concept_family(concept),
        "down": down,
        "distance": distance,
        "yardline_100": yardline_100,
        "yards": yards,
        "turnover": turnover,
        "touchdown": touchdown,
        "success": compute_success(down, distance, yards, turnover, touchdown),
        "posteam": posteam,
        "qtr": play.get("period", {}).get("number"),
        "clock": play.get("clock", {}).get("displayValue"),
    }


# ---------------------------------------------------------------------------
# Full game-state parsing
# ---------------------------------------------------------------------------

def _clock_to_seconds(display_clock, qtr):
    """'9:33' in Q2 -> seconds remaining in regulation (capped for OT)."""
    try:
        mins, secs = str(display_clock).split(":")
        in_period = int(mins) * 60 + int(secs)
    except (ValueError, AttributeError):
        return None
    if qtr is None:
        return None
    if qtr >= 5:  # overtime: model expects 0-3600, so just the period clock
        return in_period
    return (4 - qtr) * 900 + in_period


def _all_plays(drives):
    plays = []
    for drive in drives.get("previous", []) or []:
        plays.extend(drive.get("plays", []) or [])
    current = drives.get("current")
    if current:
        # The current drive can also appear at the end of "previous" -
        # don't double count.
        seen = {p.get("id") for p in plays}
        plays.extend(p for p in current.get("plays", []) or [] if p.get("id") not in seen)
    return plays


def parse_game_state(summary):
    """
    Turn a summary response into everything the Live tab needs:

    {
      game: {state, detail, period, clock, home: {abbr, score}, away: {...}},
      situation: dict shaped like the /recommend GameSituation, or None,
      reason: why situation is None ("" when it isn't),
      last_play_id: id of the newest play of any kind,
      recent_plays: last 10 scrimmage plays, newest first
    }
    """
    comp = summary.get("header", {}).get("competitions", [{}])[0]
    status = comp.get("status", {})
    state = status.get("type", {}).get("state", "pre")

    team_abbrs = {}   # espn numeric id -> nflverse abbr
    sides = {}
    scores = {}
    for c in comp.get("competitors", []):
        abbr = to_nflverse(c.get("team", {}).get("abbreviation", ""))
        team_abbrs[str(c.get("team", {}).get("id", ""))] = abbr
        sides[c.get("homeAway")] = abbr
        scores[abbr] = int(c.get("score") or 0)

    plays = _all_plays(summary.get("drives", {}) or {})
    last_play = plays[-1] if plays else None

    mapped = [m for m in (map_play(p, team_abbrs) for p in plays) if m is not None]
    recent = list(reversed(mapped[-10:]))

    # Current situation = the end state of the newest play.
    situation = None
    reason = ""
    if state != "in":
        reason = "Game is not in progress."
    elif last_play is None:
        reason = "No plays yet."
    else:
        end = last_play.get("end", {})
        down = end.get("down") or 0
        distance = end.get("distance") or 0
        yardline_100 = end.get("yardsToEndzone") or 0
        posteam = team_abbrs.get(str(end.get("team", {}).get("id", "")), "")
        defteam = next((a for a in scores if a != posteam), "")

        qtr = status.get("period") or last_play.get("period", {}).get("number")
        clock = status.get("displayClock") or last_play.get("clock", {}).get("displayValue")
        seconds = _clock_to_seconds(clock, qtr)

        if not (1 <= down <= 4) or not (1 <= yardline_100 <= 99) or not posteam:
            reason = "Between plays (kickoff, PAT, or change of possession)."
        elif seconds is None:
            reason = "No game clock available."
        else:
            situation = {
                "posteam": posteam,
                "defteam": defteam,
                "down": down,
                "ydstogo": max(1, min(99, distance)),
                "yardline_100": yardline_100,
                "qtr": min(qtr, 5),
                "game_seconds_remaining": seconds,
                "score_differential": scores.get(posteam, 0) - scores.get(defteam, 0),
            }

    game = {
        "state": state,
        "detail": status.get("type", {}).get("shortDetail", ""),
        "period": status.get("period"),
        "clock": status.get("displayClock"),
        "home": {"abbr": sides.get("home", ""), "score": scores.get(sides.get("home", ""), 0)},
        "away": {"abbr": sides.get("away", ""), "score": scores.get(sides.get("away", ""), 0)},
    }

    return {
        "game": game,
        "situation": situation,
        "reason": reason,
        "last_play_id": last_play.get("id") if last_play else None,
        "recent_plays": recent,
    }
