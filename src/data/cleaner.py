"""
cleaner.py
----------
Takes the raw combined dataframe from loader.py and:
  1. Drops columns that are 95%+ null (mostly punt/kick/penalty details).
  2. Filters to real offensive plays only (runs and passes).
  3. Removes kneel-downs and spikes (those aren't real play-calling decisions).
  4. Keeps only the columns we need for modeling.

The result is a much smaller, focused dataframe: every row is a genuine
"the offense chose to run or pass" decision.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Columns to keep, organized by what they're used for.
# Note: the raw data calls yards-after-catch "yards_after_catch"; we rename
# it to "yac" at the end to match the shorter name used in this project.
# ---------------------------------------------------------------------------
GAME_STATE_COLS = [
    "game_id", "play_id", "season", "week", "game_date",
    "home_team", "away_team", "posteam", "defteam",
    "down", "ydstogo", "yardline_100",
    "score_differential", "qtr", "game_seconds_remaining",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "shotgun", "no_huddle", "goal_to_go",
    "wp", "ep", "spread_line",
    "home_coach", "away_coach", "roof", "surface", "temp", "wind",
]

FORMATION_COLS = [
    # From participation data (all seasons; download per season from nflverse):
    "offense_formation", "offense_personnel", "defenders_in_box",
    "defense_personnel", "number_of_pass_rushers",
    "route", "defense_man_zone_type", "defense_coverage_type",
    "was_pressure", "time_to_throw",
    # From FTN charting (2022-present, all our seasons):
    "qb_location", "n_offense_backfield", "n_defense_box",
    "is_motion", "is_play_action", "is_screen_pass", "is_rpo",
    "is_trick_play", "is_qb_sneak", "n_blitzers", "n_pass_rushers",
]

OUTCOME_COLS = [
    "play_type", "success", "epa", "wpa", "yards_gained",
    "air_yards", "yards_after_catch", "pass_location", "pass_length",
    "run_location", "run_gap", "qb_scramble", "sack",
]

QB_COLS = [
    # Who dropped back, for the rolling QB-form features in features.py.
    "passer_player_id", "passer", "qb_dropback",
]

SCORING_PROB_COLS = [
    "td_prob", "fg_prob", "no_score_prob", "opp_td_prob",
]

DRIVE_COLS = [
    "series_success", "drive_ended_with_score", "drive_inside20",
    "third_down_converted", "fourth_down_converted",
]

ADVANCED_COLS = [
    "xyac_epa", "xyac_success", "xpass", "pass_oe", "cp", "cpoe",
    "qb_epa", "air_epa", "yac_epa",
    "total_home_rush_epa", "total_away_rush_epa",
    "total_home_pass_epa", "total_away_pass_epa",
]

KEEP_COLS = (
    GAME_STATE_COLS + FORMATION_COLS + OUTCOME_COLS + QB_COLS
    + SCORING_PROB_COLS + DRIVE_COLS + ADVANCED_COLS
)


def clean_data(df):
    """
    Clean the raw play-by-play dataframe down to modeling-ready rows/columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw combined dataframe from loader.load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe: run/pass plays only, focused column set.
    """
    print("=" * 60)
    print("CLEANING DATA")
    print("=" * 60)
    rows_before, cols_before = df.shape

    # --- Step 1: drop columns that are 95%+ null ---------------------------
    # These are things like punt return details that don't apply to most
    # plays. They just waste memory.
    null_frac = df.isna().mean()              # fraction of nulls per column
    too_null = null_frac[null_frac >= 0.95].index
    df = df.drop(columns=too_null)
    print(f"Dropped {len(too_null)} columns that were 95%+ null.")

    # --- Step 2: keep only real offensive plays ----------------------------
    # play_type covers kickoffs, punts, field goals, penalties... we only
    # care about plays where the offense chose to run or pass.
    df = df[df["play_type"].isin(["run", "pass"])]

    # Some non-plays (kickoffs, extra points) sneak through with a null
    # down. A real offensive snap always has a down (1-4).
    df = df[df["down"].notna()]

    # Kneel-downs and spikes are technically run/pass plays, but they're
    # clock management, not play-calling decisions. Remove them.
    if "qb_kneel" in df.columns:
        df = df[df["qb_kneel"] != 1]
    if "qb_spike" in df.columns:
        df = df[df["qb_spike"] != 1]

    # --- Step 3: keep only the columns we need ------------------------------
    # Intersect our wish-list with what actually survived step 1
    # (a column we want could theoretically have been 95% null).
    keep = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"NOTE: these requested columns are not in the data: {missing}")
    df = df[keep].copy()

    # Rename to the short name used throughout this project.
    if "yards_after_catch" in df.columns:
        df = df.rename(columns={"yards_after_catch": "yac"})

    # --- Summary printout ----------------------------------------------------
    print("-" * 60)
    print(f"Rows    : {rows_before:,} -> {len(df):,}")
    print(f"Columns : {cols_before} -> {df.shape[1]}")
    print("\nplay_type value counts (should be only run/pass):")
    print(df["play_type"].value_counts().to_string())
    print("=" * 60)

    return df


if __name__ == "__main__":
    # Quick smoke test: python -m src.data.cleaner
    from src.data.loader import load_data

    raw = load_data()
    clean = clean_data(raw)
