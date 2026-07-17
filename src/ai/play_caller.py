"""
play_caller.py (Phase 4)
------------------------
Claude API layer: turns the model's numeric output into a natural-language
play call, like a coordinator talking into the headset.

Example output:
  "It's 3rd & 7 from their 35, down 4 in the 4th quarter. Their defense
   has been giving up 0.15 EPA per pass over the last month - air it out.
   Recommendation: PA Short Pass (62% success probability)."

Requires ANTHROPIC_API_KEY in a .env file (see .env.example).
"""

import json
import os
from collections import OrderedDict
from threading import Lock

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Haiku: fast + cheap, ideal for short coordinator-style explanations.
# Override with CLAUDE_MODEL in .env to try a bigger model.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")

# Short answers only - this also caps the cost of a runaway response.
MAX_TOKENS = 400

# Same display names the frontend uses (frontend/src/lib/labels.js), so
# Claude talks about "PA Deep Pass", not "pa_deep_pass".
CONCEPT_LABELS = {
    "deep_pass": "Deep Pass",
    "short_pass": "Short Pass",
    "screen_pass": "Screen Pass",
    "pa_deep_pass": "Play-Action Deep Pass",
    "pa_short_pass": "Play-Action Short Pass",
    "inside_run": "Inside Run",
    "off_tackle_run": "Off-Tackle Run",
    "outside_run": "Outside Run",
    "qb_sneak": "QB Sneak",
    "trick_play": "Trick Play",
}

SYSTEM_PROMPT = """\
You are an NFL offensive coordinator talking into the headset. You are given
one game situation, a ranked call sheet from a statistical model (success
probability and expected EPA per play concept), and both teams' recent form.

Explain WHY the recommended call is the right one, in 2-4 punchy sentences:
- Open with the situation the way a coordinator would read it.
- Ground every claim in the numbers you were given (success probability,
  EPA edge over the alternative, team form ranks). Never invent stats,
  players, or injuries.
- If a tempting alternative was ruled out (e.g. a deep shot that failed the
  success floor), say so in one clause.
- End with the call, stated with conviction.

Plain text only - no markdown, no bullet lists, no headers."""


class PlayCallerError(RuntimeError):
    """Raised when the Claude explanation can't be produced. `status` hints
    at the right HTTP status for the API layer (503 = misconfigured/missing
    key, 502 = upstream Claude API failure)."""

    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


# One client per process (it's thread-safe and holds the connection pool).
_client = None


def _get_client():
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise PlayCallerError(
                "ANTHROPIC_API_KEY is not set - copy .env.example to .env "
                "and add your key.",
                status=503,
            )
        _client = anthropic.Anthropic()
    return _client


# Tiny LRU cache so replaying the same situation (demo clicks, dashboard
# refreshes) doesn't burn API credits. Keyed on everything that shapes the
# prompt; bounded so a long-running server can't grow without limit.
_CACHE_MAX = 256
_cache = OrderedDict()
_cache_lock = Lock()


def _cache_key(*parts):
    return json.dumps(parts, sort_keys=True, default=str)


def _label(concept):
    return CONCEPT_LABELS.get(concept, concept)


def _format_situation(situation):
    down_names = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    down = down_names.get(situation.get("down"), f"{situation.get('down')}th")
    diff = situation.get("score_differential", 0)
    score = (
        "tied" if diff == 0
        else f"up {diff}" if diff > 0
        else f"down {-diff}"
    )
    secs = situation.get("game_seconds_remaining", 0)
    return (
        f"{down} & {situation.get('ydstogo')}, "
        f"{situation.get('yardline_100')} yards from the end zone. "
        f"Q{situation.get('qtr')}, {secs // 60}:{secs % 60:02d} left in the game, "
        f"{situation.get('posteam')} (offense) {score} against "
        f"{situation.get('defteam')} (defense)."
    )


def _format_ranking(ranking):
    """The call sheet as compact text rows Claude can read at a glance.
    `ranking` is a list of dicts: play_concept, success_prob, expected_epa,
    meets_floor."""
    lines = []
    for row in ranking:
        epa = row.get("expected_epa")
        epa_txt = f"{epa:+.3f}" if epa is not None else "n/a"
        floor = "yes" if row.get("meets_floor") else "NO (failed the success floor)"
        lines.append(
            f"- {_label(row['play_concept'])}: "
            f"success {row['success_prob']:.0%}, expected EPA {epa_txt}, "
            f"eligible: {floor}"
        )
    return "\n".join(lines)


def _format_form(team, ranks, side):
    """One line of team form, e.g. 'KC offense: #3 overall (+0.12 EPA/play)...'.
    `ranks` is the /offense-ranks or /defense-ranks entry for the team."""
    if not ranks:
        return None
    unit = "offense" if side == "off" else "defense"
    suffix = "EPA/play" if side == "off" else "EPA/play allowed"
    return (
        f"{team} {unit} (last 4 weeks): "
        f"#{ranks['overall_rank']} overall ({ranks['overall_epa']:+.3f} {suffix}), "
        f"#{ranks['pass_rank']} vs pass ({ranks['pass_epa']:+.3f}), "
        f"#{ranks['run_rank']} vs run ({ranks['run_epa']:+.3f})"
    )


def explain_recommendation(situation, result, off_ranks=None, def_ranks=None):
    """
    Generate a coordinator-style natural-language explanation with Claude.

    Parameters
    ----------
    situation : dict
        The game situation as sent to /recommend (down, ydstogo, teams...).
    result : dict
        The recommendation payload: best_call, best_prob, best_epa, and
        ranking (list of dicts).
    off_ranks, def_ranks : dict, optional
        The offense/defense rank entries for the two teams (overall_rank,
        overall_epa, pass_rank, ... as served by /offense-ranks).

    Returns
    -------
    str : the explanation text.
    """
    key = _cache_key(situation, result["best_call"], result["ranking"])
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

    form_lines = [
        line
        for line in (
            _format_form(situation.get("posteam"), off_ranks, "off"),
            _format_form(situation.get("defteam"), def_ranks, "def"),
        )
        if line
    ]
    form_block = "\n".join(form_lines) if form_lines else "(no recent form data)"

    prompt = (
        f"SITUATION: {_format_situation(situation)}\n\n"
        f"TEAM FORM:\n{form_block}\n\n"
        f"MODEL CALL SHEET (ranked by expected EPA; a play must clear the "
        f"success floor to be recommendable):\n{_format_ranking(result['ranking'])}\n\n"
        f"RECOMMENDED CALL: {_label(result['best_call'])} "
        f"(success {result['best_prob']:.0%}"
        + (f", expected EPA {result['best_epa']:+.3f}" if result.get("best_epa") is not None else "")
        + ")\n\nExplain this call."
    )

    client = _get_client()
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        raise PlayCallerError("Claude API key was rejected - check .env.", status=503)
    except anthropic.RateLimitError:
        raise PlayCallerError("Claude API rate limit hit - try again shortly.", status=429)
    except anthropic.APIError as e:
        raise PlayCallerError(f"Claude API error: {e}", status=502)

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise PlayCallerError("Claude returned an empty response.", status=502)

    with _cache_lock:
        _cache[key] = text
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return text


if __name__ == "__main__":
    # Demo: run the recommender, then have Claude explain the pick.
    from src.model.recommend import recommend_play

    situation = {
        "posteam": "KC", "defteam": "BUF",
        "down": 3, "ydstogo": 7, "yardline_100": 35,
        "qtr": 4, "game_seconds_remaining": 240,
        "score_differential": -4,
    }
    result = recommend_play(dict(situation))
    result["ranking"] = result["ranking"].to_dict(orient="records")
    print(f"Model pick: {result['best_call']}\n")
    print(explain_recommendation(situation, result))
