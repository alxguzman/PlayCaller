"""
concepts.py
-----------
Upgrades the target from "run or pass" to coordinator-level play calls.

Adds three columns:

1. `formation`  - derived from FTN charting (works for ALL seasons):
   QB alignment (under center / shotgun / pistol) + how many players are
   in the backfield. e.g. "shotgun_1back", "under_center_2back", "empty".

2. `personnel`  - the standard NFL personnel code parsed from the
   participation data: "11" = 1 RB + 1 TE (3 WR),
   "12" = 1 RB + 2 TE, "21" = 2 RB + 1 TE, etc.

3. `play_concept` (the new model target) - ~10 classes:
     Passes: screen_pass, pa_short_pass, pa_deep_pass, short_pass, deep_pass
             ("pa_" = play action)
     Runs:   inside_run, off_tackle_run, outside_run, qb_sneak
     Other:  trick_play

   `play_concept_detailed` appends direction (left/middle/right) where
   known, e.g. "deep_pass_right", "inside_run_left" (~28 classes).

Plays where the true call is unknowable stay NaN and are excluded from
model training:
  - QB scrambles (a called pass that turned into a run - we can't know
    the called depth/direction)
  - Sacks and throwaways with no recorded pass depth
"""

import re

import numpy as np
import pandas as pd


def _flag(df, col):
    """
    Return an FTN boolean column as a clean True/False series.

    After the left-merge, plays without FTN data have NaN in these
    columns - treat those as False. If the column doesn't exist at all
    (no FTN files downloaded), return all-False so the logic still runs.
    """
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def build_formation(df):
    """
    Build a formation label from FTN's QB alignment + backfield count.

    Unlike the participation data (2023 only), FTN covers every season,
    so this works on all our data.
    """
    qb_loc = df.get("qb_location", pd.Series(np.nan, index=df.index))
    qb_map = {"U": "under_center", "S": "shotgun", "P": "pistol"}
    base = qb_loc.map(qb_map)  # anything else (NaN, "0") stays NaN

    backfield = df.get("n_offense_backfield", pd.Series(np.nan, index=df.index))

    formation = pd.Series(np.nan, index=df.index, dtype="object")
    known = base.notna() & backfield.notna()

    # Empty backfield is its own well-known formation name.
    formation[known & (backfield == 0)] = "empty"
    formation[known & (backfield == 1)] = base + "_1back"
    formation[known & (backfield >= 2)] = base + "_2back"
    return formation


def build_personnel(df):
    """
    Parse the participation data's personnel string into the standard
    2-digit code coaches use.

    "1 RB, 1 TE, 3 WR" -> "11"   (1 back, 1 tight end)
    "2 RB, 1 TE, 2 WR" -> "21"
    The 2024+ files list every position on the field (including defensive
    ones on special teams); the regex pulls out just the RB and TE counts,
    so those parse fine too.
    """
    if "offense_personnel" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="object")

    def parse(s):
        if not isinstance(s, str):
            return np.nan
        rb = re.search(r"(\d+)\s*RB", s)
        te = re.search(r"(\d+)\s*TE", s)
        if rb is None or te is None:
            return np.nan
        return f"{rb.group(1)}{te.group(1)}"

    return df["offense_personnel"].map(parse)


def build_play_concept(df):
    """
    Build the multi-class play-concept target. See module docstring for
    the class list. Returns (play_concept, play_concept_detailed).
    """
    concept = pd.Series(np.nan, index=df.index, dtype="object")

    is_pass = df["play_type"] == "pass"
    is_run = df["play_type"] == "run"
    deep = df["pass_length"] == "deep"
    short = df["pass_length"] == "short"
    pa = _flag(df, "is_play_action")
    screen = _flag(df, "is_screen_pass")
    scramble = df.get("qb_scramble", pd.Series(0, index=df.index)).fillna(0) == 1

    # --- Runs (most specific first) -----------------------------------------
    # A scramble is a called PASS that broke down - we can't know the real
    # call, so it stays NaN and is excluded from training.
    concept[is_run & (df["run_gap"] == "end")] = "outside_run"
    concept[is_run & (df["run_gap"] == "tackle")] = "off_tackle_run"
    concept[is_run & (df["run_gap"] == "guard")] = "inside_run"
    concept[is_run & (df["run_location"] == "middle")] = "inside_run"
    concept[is_run & _flag(df, "is_qb_sneak")] = "qb_sneak"
    concept[is_run & scramble] = np.nan

    # --- Passes --------------------------------------------------------------
    # Order matters: later assignments overwrite earlier ones, so we go
    # from generic to specific. Sacks/throwaways with no recorded depth
    # stay NaN unless FTN tells us it was a screen.
    concept[is_pass & short] = "short_pass"
    concept[is_pass & deep] = "deep_pass"
    concept[is_pass & short & pa] = "pa_short_pass"
    concept[is_pass & deep & pa] = "pa_deep_pass"
    concept[is_pass & screen] = "screen_pass"

    # --- Trick plays override everything ---------------------------------------
    concept[_flag(df, "is_trick_play")] = "trick_play"

    # --- Detailed version: append direction where we know it -------------------
    direction = df["pass_location"].where(is_pass, df["run_location"])
    detailed = concept.copy()
    has_dir = concept.notna() & direction.notna() & (concept != "qb_sneak")
    detailed[has_dir] = concept[has_dir] + "_" + direction[has_dir]

    return concept, detailed


def add_play_concepts(df):
    """
    Add formation, personnel, and play-concept columns to the cleaned
    dataframe. Prints a summary of the new target classes.
    """
    print("=" * 60)
    print("BUILDING PLAY CONCEPTS (formation / personnel / concept)")
    print("=" * 60)

    df = df.copy()
    df["formation"] = build_formation(df)
    df["personnel"] = build_personnel(df)
    df["play_concept"], df["play_concept_detailed"] = build_play_concept(df)

    n = len(df)
    print(f"formation    : {df['formation'].notna().sum():,} of {n:,} plays labeled")
    print(f"personnel    : {df['personnel'].notna().sum():,} of {n:,} plays labeled")
    print(f"play_concept : {df['play_concept'].notna().sum():,} of {n:,} plays labeled")

    print("\nplay_concept class counts:")
    print(df["play_concept"].value_counts().to_string())
    print(f"\nplay_concept_detailed has {df['play_concept_detailed'].nunique()} classes")
    print("=" * 60)
    return df


if __name__ == "__main__":
    # Smoke test: python -m src.data.concepts
    from src.data.cleaner import clean_data
    from src.data.loader import load_data

    df = add_play_concepts(clean_data(load_data()))
