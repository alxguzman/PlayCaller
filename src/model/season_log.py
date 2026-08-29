"""
season_log.py
-------------
Builds a per-play "model vs coach" log for the whole test season
(TEST_SEASON in train.py) and saves it as a CSV.

For EVERY play the model gets only what it would have known before the
snap - situation + team form. It never sees the coach's call or the
outcome; those columns are joined in afterwards purely for comparison.
Because the model is deterministic, regenerating this log after the
weekly data refresh gives the exact same recommendations it would have
made live during the game.

A note on reading the numbers: when the coach called something else, we
never observe how the model's play would have gone - so disagreement is
NOT evidence the model was wrong. The informative comparison is whether
plays where the coach happened to agree with the model out-performed the
rest (and even that is evidence, not proof: agreement isn't random).

Weekly routine addition (after refresh + train + evaluate):

    python -m src.model.season_log

Output: data/processed/season_log_<TEST_SEASON>.csv - one row per play
(game, situation, model_call, coach_call, agreed, actual epa/success/yards)
plus a per-week summary printed to the terminal.
"""

import json
import os
import time

import pandas as pd

from src.model.evaluate import load_artifact, model_recommendations
from src.model.train import (
    MODEL_DIR,
    PROJECT_ROOT,
    TEST_SEASON,
    load_processed,
    prepare_data,
)

LOG_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", f"season_log_{TEST_SEASON}.csv"
)

# The compact week-by-week summary lives in model/ (NOT data/, which is
# gitignored) so committing the weekly retrain also ships the scorecard
# to the deployed app - GET /season-scorecard serves this file and the
# frontend's Season tab renders it.
SUMMARY_PATH = os.path.join(MODEL_DIR, "season_scorecard.json")

# Play facts worth carrying into the log (all joined AFTER the model has
# made its call - none of these are model inputs beyond the situation).
PLAY_COLUMNS = [
    "game_id", "week", "posteam", "defteam",
    "qtr", "down", "ydstogo", "yardline_100",
    "play_concept", "yards_gained", "success", "epa",
]


def build_log(artifact=None):
    """Return the full per-play log as a DataFrame (or None pre-season)."""
    artifact = artifact or load_artifact()

    df = load_processed()
    X, y, meta = prepare_data(df)
    is_test = (meta["season"] == TEST_SEASON).to_numpy()
    if is_test.sum() == 0:
        print(f"No {TEST_SEASON} plays in the dataset yet - nothing to log.")
        return None

    X_test = X[is_test]
    print(f"Scoring {len(X_test):,} plays from the {TEST_SEASON} season ...")
    recommended = model_recommendations(X_test, artifact)

    # prepare_data preserved the original row index, so we can pull the
    # play facts straight from the processed dataframe.
    log = df.loc[X_test.index, PLAY_COLUMNS].copy()
    log = log.rename(columns={"play_concept": "coach_call"})
    log.insert(len(log.columns) - 4, "model_call", recommended)
    log["agreed"] = log["model_call"] == log["coach_call"]
    return log.reset_index(drop=True)


def weekly_summary(log):
    """Per-week agreement + outcome table, plus a season-total row."""
    def summarize(g):
        agreed = g[g["agreed"]]
        other = g[~g["agreed"]]
        return pd.Series({
            "plays": len(g),
            "agreement": g["agreed"].mean(),
            "agreed_epa": agreed["epa"].mean(),
            "other_epa": other["epa"].mean(),
            "agreed_success": agreed["success"].mean(),
            "other_success": other["success"].mean(),
        })

    weekly = log.groupby("week").apply(summarize, include_groups=False)
    weekly.loc["season"] = summarize(log)
    weekly["plays"] = weekly["plays"].astype(int)
    return weekly


def write_summary_json(weekly):
    """
    Save the weekly table as model/season_scorecard.json for the API.

    NaN (a week where the model and coach never agreed, early on) becomes
    null so the JSON stays valid.
    """
    def row_dict(label, row):
        d = {"week": label}
        for col, val in row.items():
            d[col] = None if pd.isna(val) else round(float(val), 4)
        d["plays"] = int(row["plays"])
        return d

    weeks = [row_dict(int(w), weekly.loc[w]) for w in weekly.index if w != "season"]
    payload = {
        "season": TEST_SEASON,
        "generated_at": time.strftime("%Y-%m-%d"),
        "weeks": weeks,
        "totals": row_dict("season", weekly.loc["season"]),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


def main():
    log = build_log()
    if log is None:
        return

    log.to_csv(LOG_PATH, index=False)
    print(f"Saved {len(log):,} plays -> {LOG_PATH}\n")

    weekly = weekly_summary(log)
    write_summary_json(weekly)
    print(f"Saved week-by-week summary -> {SUMMARY_PATH}\n")

    print("=" * 72)
    print(f"MODEL vs COACH - {TEST_SEASON} season, week by week")
    print("=" * 72)
    print(weekly.round(3).to_string())
    print(
        "\nReminder: 'other' rows are plays where the coach chose differently -"
        "\nthe model's play never ran, so those rows don't grade the model wrong."
    )


if __name__ == "__main__":
    main()
