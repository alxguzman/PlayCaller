"""
replay.py
---------
Replays a COMPLETED game through the live pipeline, so the whole
model-vs-coach flow can be tested without waiting for a live game.

ESPN's summary endpoint serves finished games too, so this fetches every
play, classifies each one exactly like the Live tab does (map_play),
asks the model what it would have called in that situation
(recommend_play), and prints the same scorecard the Live tab keeps.

Find an event id on the scoreboard (python -m src.live.replay with no
args lists today's games) or in an ESPN game page URL.

Run with:  python -m src.live.replay <event_id>
           python -m src.live.replay 401873298
"""

import sys

from src.live.espn import (
    concept_family,
    get_game_summary,
    get_scoreboard,
    map_play,
    parse_scoreboard,
    to_nflverse,
    _all_plays,
    _clock_to_seconds,
)
from src.model.evaluate import load_artifact
from src.model.recommend import recommend_play

RUN_FAMILIES = {"inside_run", "off_tackle_run", "outside_run", "qb_sneak"}


def _run_or_pass(family):
    return "run" if family in RUN_FAMILIES else "pass"


def replay(event_id):
    artifact = load_artifact()
    summary = get_game_summary(event_id)

    comp = summary["header"]["competitions"][0]
    team_abbrs = {}
    scores_by_id = {}
    for c in comp["competitors"]:
        abbr = to_nflverse(c["team"].get("abbreviation", ""))
        team_abbrs[str(c["team"].get("id", ""))] = abbr
        scores_by_id[str(c["team"].get("id", ""))] = 0
    teams = list(team_abbrs.values())
    away = next((to_nflverse(c["team"]["abbreviation"]) for c in comp["competitors"]
                 if c.get("homeAway") == "away"), teams[0])
    home = next(a for a in teams if a != away)
    print(f"Replaying {away} @ {home} (event {event_id})\n")

    # Running score: scoringPlays carries both teams' totals after each
    # score, so build a play_id -> {abbr: score} lookup and update as we
    # walk the plays in order.
    score_after = {}
    for sp in summary.get("scoringPlays", []):
        score_after[str(sp.get("id"))] = {
            to_nflverse(c["team"]["abbreviation"]): (
                sp.get("homeScore") if c.get("homeAway") == "home" else sp.get("awayScore")
            )
            for c in comp["competitors"]
        }

    rows = []
    current_score = {a: 0 for a in teams}
    for play in _all_plays(summary.get("drives", {}) or {}):
        mapped = map_play(play, team_abbrs)
        if str(play.get("id")) in score_after:
            current_score = score_after[str(play.get("id"))]
        if mapped is None or mapped["concept"] is None or not mapped["posteam"]:
            continue

        posteam = mapped["posteam"]
        defteam = next(a for a in teams if a != posteam)
        qtr = mapped["qtr"] or 1
        seconds = _clock_to_seconds(mapped["clock"], qtr) or 0

        situation = {
            "posteam": posteam,
            "defteam": defteam,
            "down": mapped["down"],
            "ydstogo": mapped["distance"],
            "yardline_100": mapped["yardline_100"],
            "qtr": min(qtr, 5),
            "game_seconds_remaining": seconds,
            "score_differential": current_score.get(posteam, 0) - current_score.get(defteam, 0),
        }
        rec = recommend_play(situation, artifact=artifact)
        model_family = concept_family(rec["best_call"])
        agreed = model_family == mapped["family"]

        rows.append({
            "model": rec["best_call"],
            "model_family": model_family,
            "coach": mapped["concept"],
            "coach_family": mapped["family"],
            "agreed": agreed,
            "rp_agreed": _run_or_pass(model_family) == _run_or_pass(mapped["family"]),
            "yards": mapped["yards"],
            "success": mapped["success"],
        })

    if not rows:
        print("No gradable scrimmage plays found.")
        return

    n = len(rows)
    agreed = [r for r in rows if r["agreed"]]
    disagreed = [r for r in rows if not r["agreed"]]

    def rate(subset, key):
        return sum(r[key] for r in subset) / len(subset) if subset else float("nan")

    print(f"Snaps graded          : {n}")
    print(f"Family agreement      : {len(agreed) / n:.1%}")
    print(f"Run/pass agreement    : {sum(r['rp_agreed'] for r in rows) / n:.1%}")
    print()
    print(f"{'':28}{'plays':>6}{'success (40/60/100)':>22}{'avg yards':>11}")
    print(f"{'coach agreed with model':28}{len(agreed):>6}{rate(agreed, 'success'):>21.1%} {rate(agreed, 'yards'):>10.1f}")
    print(f"{'coach called differently':28}{len(disagreed):>6}{rate(disagreed, 'success'):>21.1%} {rate(disagreed, 'yards'):>10.1f}")

    print("\nCoach concept counts (from ESPN play text):")
    counts = {}
    for r in rows:
        counts[r["coach"]] = counts.get(r["coach"], 0) + 1
    for concept, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {concept:<20}{count}")


def list_games():
    print("Today's games (pass an id to replay a finished one):\n")
    for g in parse_scoreboard(get_scoreboard()):
        print(f"  {g['id']}  {g['short_name']:<12} {g['state']:<5} {g['detail']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_games()
    else:
        replay(sys.argv[1])
