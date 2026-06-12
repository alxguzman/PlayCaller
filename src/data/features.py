"""
features.py
-----------
Builds aggregated TEAM-level features and joins them onto every play.

The idea: before predicting/recommending a play, the model should know how
good each offense and defense has been recently - like the defensive
rankings you see on the Sleeper app.

To avoid DATA LEAKAGE we use a rolling window of the PREVIOUS 4 weeks:
the stats attached to a week-10 play only use weeks 6-9. We never let
the model peek at the future (or even at the current week, since the
play we're predicting is part of it).
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

ROLLING_WEEKS = 4  # how many previous weeks of form to average over


def _rolling_previous(group):
    """
    For one (season, team) group sorted by week: average each stat over the
    previous ROLLING_WEEKS weeks, NOT including the current week.

    .shift(1) pushes everything down one week, so week N sees weeks
    N-4..N-1. Week 1 has no history and comes out NaN (filled later).
    """
    return group.shift(1).rolling(ROLLING_WEEKS, min_periods=1).mean()


def build_offense_stats(df):
    """
    Build weekly offensive stats per team, then convert them to rolling
    "recent form" averages.

    Returns
    -------
    pd.DataFrame with columns:
        season, week, posteam,
        off_epa_per_play, off_epa_pass, off_epa_run,
        off_success_rate, off_pass_rate, off_yards_per_play
    """
    # Helper columns: epa_pass is the play's EPA if it was a pass, else NaN
    # (NaNs are ignored by .mean(), so this gives us "mean EPA on passes").
    tmp = df[["season", "week", "posteam", "epa", "success", "yards_gained"]].copy()
    tmp["epa_pass"] = df["epa"].where(df["play_type"] == "pass")
    tmp["epa_run"] = df["epa"].where(df["play_type"] == "run")
    tmp["is_pass"] = (df["play_type"] == "pass").astype(float)

    # Situational splits - how good is this offense in money situations?
    tmp["epa_third_down"] = df["epa"].where(df["down"] >= 3)
    tmp["epa_redzone"] = df["epa"].where(df["yardline_100"] <= 20)
    # pass_oe = pass rate over expected: how much more often the team
    # passes than the situation predicts. Positive = pass-happy identity.
    if "pass_oe" in df.columns:
        tmp["pass_oe"] = df["pass_oe"]
    else:
        tmp["pass_oe"] = np.nan

    # First: raw per-week numbers for every (season, week, team).
    weekly = (
        tmp.groupby(["season", "week", "posteam"])
        .agg(
            off_epa_per_play=("epa", "mean"),
            off_epa_pass=("epa_pass", "mean"),
            off_epa_run=("epa_run", "mean"),
            off_success_rate=("success", "mean"),
            off_pass_rate=("is_pass", "mean"),
            off_yards_per_play=("yards_gained", "mean"),
            off_epa_third_down=("epa_third_down", "mean"),
            off_epa_redzone=("epa_redzone", "mean"),
            off_pass_oe=("pass_oe", "mean"),
        )
        .reset_index()
        .sort_values(["season", "posteam", "week"])
    )

    # Then: turn raw weekly numbers into rolling previous-4-week averages.
    stat_cols = [c for c in weekly.columns if c.startswith("off_")]
    weekly[stat_cols] = weekly.groupby(["season", "posteam"])[stat_cols].transform(
        _rolling_previous
    )
    return weekly


def build_defense_stats(df):
    """
    Same as build_offense_stats but from the DEFENSE's point of view:
    EPA *allowed*, success rate *allowed*, etc. Grouped by defteam.

    Returns
    -------
    pd.DataFrame with columns:
        season, week, defteam,
        def_epa_allowed_per_play, def_epa_allowed_pass, def_epa_allowed_run,
        def_success_rate_allowed, def_yards_allowed_per_play
    """
    # Same helper-column trick as the offense version (see above).
    tmp = df[["season", "week", "defteam", "epa", "success", "yards_gained"]].copy()
    tmp["epa_pass"] = df["epa"].where(df["play_type"] == "pass")
    tmp["epa_run"] = df["epa"].where(df["play_type"] == "run")

    # Situational splits from the defense's point of view.
    tmp["epa_third_down"] = df["epa"].where(df["down"] >= 3)
    tmp["epa_redzone"] = df["epa"].where(df["yardline_100"] <= 20)
    # Blitz rate: on charted pass plays, how often does this defense send
    # 5+ rushers? (FTN's n_blitzers counts extra rushers beyond the front.)
    if "n_blitzers" in df.columns:
        tmp["blitzed"] = (df["n_blitzers"] > 0).astype(float).where(
            (df["play_type"] == "pass") & df["n_blitzers"].notna()
        )
    else:
        tmp["blitzed"] = np.nan

    weekly = (
        tmp.groupby(["season", "week", "defteam"])
        .agg(
            def_epa_allowed_per_play=("epa", "mean"),
            def_epa_allowed_pass=("epa_pass", "mean"),
            def_epa_allowed_run=("epa_run", "mean"),
            def_success_rate_allowed=("success", "mean"),
            def_yards_allowed_per_play=("yards_gained", "mean"),
            def_epa_third_down_allowed=("epa_third_down", "mean"),
            def_epa_redzone_allowed=("epa_redzone", "mean"),
            def_blitz_rate=("blitzed", "mean"),
        )
        .reset_index()
        .sort_values(["season", "defteam", "week"])
    )

    stat_cols = [c for c in weekly.columns if c.startswith("def_")]
    weekly[stat_cols] = weekly.groupby(["season", "defteam"])[stat_cols].transform(
        _rolling_previous
    )
    return weekly


def build_qb_stats(df):
    """
    Rolling form for each team's CURRENT starting quarterback.

    Why: team form smears over the actual humans. If the starter goes
    down, off_epa_pass takes ~4 weeks to absorb the backup; the QB's own
    rolling stats switch over immediately.

    Two steps:
      1. Per-QB form: for every (season, QB, week) with at least one
         dropback, average qb_epa / cpoe / sack over his dropbacks that
         week, then roll over his previous ROLLING_WEEKS games played
         (same shift(1) trick as team form - never the current week).
      2. Who's the QB this week: each (season, week, team)'s primary
         passer = whoever took the most dropbacks that week. Knowing
         who starts is legitimate pre-game information; his STATS still
         come only from previous weeks. (Edge case: if the starter is
         hurt mid-game, the reliever can own the whole week's plays -
         rare enough to accept as noise.)

    The features are prefixed qbf_ (QB form), NOT qb_, because raw
    outcome columns like qb_epa / qb_scramble already use that prefix
    and must never be auto-included as model features.

    Returns
    -------
    pd.DataFrame with columns:
        season, week, posteam,
        qbf_epa_per_dropback, qbf_cpoe, qbf_sack_rate
    """
    drops = df[(df["qb_dropback"] == 1) & df["passer_player_id"].notna()]

    # Step 1: per-QB weekly numbers -> rolling previous-games form.
    weekly = (
        drops.groupby(["season", "passer_player_id", "week"])
        .agg(
            qbf_epa_per_dropback=("qb_epa", "mean"),
            qbf_cpoe=("cpoe", "mean"),
            qbf_sack_rate=("sack", "mean"),
        )
        .reset_index()
        .sort_values(["season", "passer_player_id", "week"])
    )
    stat_cols = [c for c in weekly.columns if c.startswith("qbf_")]
    weekly[stat_cols] = weekly.groupby(["season", "passer_player_id"])[
        stat_cols
    ].transform(_rolling_previous)

    # Step 2: each team-week's primary passer (most dropbacks that week).
    counts = (
        drops.groupby(["season", "week", "posteam", "passer_player_id"])
        .size()
        .reset_index(name="n_dropbacks")
        .sort_values("n_dropbacks")
        .drop_duplicates(subset=["season", "week", "posteam"], keep="last")
    )

    qb_form = counts.merge(
        weekly, on=["season", "passer_player_id", "week"], how="left"
    )
    return qb_form[["season", "week", "posteam"] + stat_cols]


def add_team_features(df, save=True):
    """
    Build rolling team stats and join them onto every play.

    - Offensive stats join on (season, week, posteam)
    - Defensive stats join on (season, week, defteam)
    - NaNs (mostly week-1 games with no history) are filled with the
      season-wide mean for that column.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from cleaner.clean_data().
    save : bool
        If True, write the result to data/processed/pbp_features.csv.

    Returns
    -------
    pd.DataFrame
        df with ~11 new team-form columns, ready for model training.
    """
    print("=" * 60)
    print("BUILDING TEAM FEATURES (rolling previous-4-week form)")
    print("=" * 60)

    off = build_offense_stats(df)
    print(f"Offense stats : {off.shape[0]:,} team-week rows")
    de = build_defense_stats(df)
    print(f"Defense stats : {de.shape[0]:,} team-week rows")
    qb = build_qb_stats(df)
    print(f"QB form stats : {qb.shape[0]:,} team-week rows")

    rows_before = len(df)
    df = df.merge(off, on=["season", "week", "posteam"], how="left")
    df = df.merge(de, on=["season", "week", "defteam"], how="left")
    df = df.merge(qb, on=["season", "week", "posteam"], how="left")
    assert len(df) == rows_before, "Join created duplicate rows!"

    # Fill NaNs with the season mean for that column (week-1 games and
    # first-start QBs have no rolling history yet, so we substitute "an
    # average team" / "an average QB").
    feature_cols = [c for c in df.columns if c.startswith(("off_", "def_", "qbf_"))]
    for col in feature_cols:
        season_means = df.groupby("season")[col].transform("mean")
        df[col] = df[col].fillna(season_means)

    print(f"Added {len(feature_cols)} team-form feature columns.")

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, "pbp_features.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")

    print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 60)
    return df


if __name__ == "__main__":
    # Full pipeline smoke test: python -m src.data.features
    # load -> clean -> play concepts -> team features -> save
    from src.data.cleaner import clean_data
    from src.data.concepts import add_play_concepts
    from src.data.loader import load_data

    final = add_team_features(add_play_concepts(clean_data(load_data())))
