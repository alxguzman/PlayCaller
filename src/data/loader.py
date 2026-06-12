"""
loader.py
---------
Loads NFL play-by-play and participation data from CSV files and merges
them into a single dataframe.

Data sources (downloaded from nflverse GitHub releases):
  - data/raw/play_by_play_YYYY.csv (or .csv.gz - pandas reads both)
  - data/raw/participation_YYYY.csv (or .csv.gz)

The participation files add formation/personnel info (e.g. "11 personnel",
shotgun formation, defenders in the box). The NFL stopped publishing this
data after the 2023 season, so the loader treats it as OPTIONAL:
plays without participation data simply get NaN in those columns.
"""

import glob
import os

import pandas as pd

# Project root = two levels up from this file (src/data/loader.py -> project/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def _find_files(pattern):
    """
    Find files in data/raw matching a pattern, accepting both .csv and .csv.gz.

    pandas can read gzip-compressed CSVs directly, so we don't need to
    decompress anything ourselves.
    """
    files = sorted(
        glob.glob(os.path.join(RAW_DIR, pattern + ".csv"))
        + glob.glob(os.path.join(RAW_DIR, pattern + ".csv.gz"))
    )
    return files


def load_play_by_play():
    """
    Load every play_by_play_YYYY file found in data/raw and stack them
    into one dataframe (one row per play).

    Returns
    -------
    pd.DataFrame
        All seasons of play-by-play data combined.
    """
    files = _find_files("play_by_play_*")
    if not files:
        raise FileNotFoundError(
            f"No play_by_play files found in {RAW_DIR}. "
            "Download them from nflverse and place them there."
        )

    frames = []
    for f in files:
        print(f"Loading {os.path.basename(f)} ...")
        # low_memory=False reads the whole file at once so pandas can infer
        # column types correctly (these files have 350+ columns).
        frames.append(pd.read_csv(f, low_memory=False))

    pbp = pd.concat(frames, ignore_index=True)
    return pbp


def load_participation():
    """
    Load every participation_YYYY file found in data/raw.

    Returns
    -------
    pd.DataFrame or None
        Combined participation data, or None if no files are present.
    """
    files = _find_files("participation_*")
    if not files:
        print(
            "NOTE: no participation files found in data/raw - "
            "formation/personnel columns will be empty. Download from "
            "https://github.com/nflverse/nflverse-data/releases/tag/pbp_participation"
        )
        return None

    frames = []
    for f in files:
        print(f"Loading {os.path.basename(f)} ...")
        frames.append(pd.read_csv(f, low_memory=False))

    part = pd.concat(frames, ignore_index=True)

    # The participation files call the game id column "nflverse_game_id";
    # rename it to "game_id" so it matches the play-by-play data.
    if "nflverse_game_id" in part.columns and "game_id" not in part.columns:
        part = part.rename(columns={"nflverse_game_id": "game_id"})

    # Only keep the columns we actually merge in - keeps memory usage down.
    keep = [
        "game_id",
        "play_id",
        "offense_formation",
        "offense_personnel",
        "defenders_in_box",
        "defense_personnel",
        "number_of_pass_rushers",
        # Bonus charting columns in the newer participation files:
        "route",                  # route run by the targeted receiver
        "defense_man_zone_type",  # man vs zone coverage
        "defense_coverage_type",  # Cover-1, Cover-2, Cover-3...
        "was_pressure",           # was the QB pressured?
        "time_to_throw",
    ]
    keep = [c for c in keep if c in part.columns]
    part = part[keep]

    # A handful of plays appear twice in the participation data;
    # drop duplicates so the merge doesn't create extra rows.
    part = part.drop_duplicates(subset=["game_id", "play_id"])
    return part


def load_ftn():
    """
    Load every ftn_charting_YYYY file found in data/raw.

    FTN charts every play by hand: QB alignment (under center / shotgun /
    pistol), how many players in the backfield, play-action, screens, RPOs,
    motion, blitzers... This is what lets us build play CONCEPTS instead of
    just run/pass. Free from nflverse, 2022 to present.

    Returns
    -------
    pd.DataFrame or None
        Combined FTN data keyed by (game_id, play_id), or None if no
        files are present.
    """
    files = _find_files("ftn_charting_*")
    if not files:
        print(
            "NOTE: no ftn_charting files found in data/raw - "
            "play-concept columns (play action, screens, RPO...) will be empty."
        )
        return None

    frames = []
    for f in files:
        print(f"Loading {os.path.basename(f)} ...")
        frames.append(pd.read_csv(f, low_memory=False))

    ftn = pd.concat(frames, ignore_index=True)

    # FTN uses nflverse_game_id / nflverse_play_id; rename to match pbp.
    ftn = ftn.rename(
        columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"}
    )

    keep = [
        "game_id",
        "play_id",
        "qb_location",          # U = under center, S = shotgun, P = pistol
        "n_offense_backfield",  # how many players lined up in the backfield
        "n_defense_box",        # defenders in the box (FTN's count)
        "is_motion",
        "is_play_action",
        "is_screen_pass",
        "is_rpo",
        "is_trick_play",
        "is_qb_sneak",
        "n_blitzers",
        "n_pass_rushers",
    ]
    keep = [c for c in keep if c in ftn.columns]
    ftn = ftn[keep]
    ftn = ftn.drop_duplicates(subset=["game_id", "play_id"])
    return ftn


def load_data():
    """
    Load play-by-play + participation data and merge them into one dataframe.

    The merge is a LEFT join on (game_id, play_id): every play is kept,
    and plays without participation data get NaN in those columns.

    Returns
    -------
    pd.DataFrame
        The combined dataframe, one row per play.
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    pbp = load_play_by_play()
    part = load_participation()
    ftn = load_ftn()

    if part is not None:
        before = len(pbp)
        pbp = pbp.merge(part, on=["game_id", "play_id"], how="left")
        assert len(pbp) == before, "Merge created duplicate rows!"
        matched = pbp["offense_formation"].notna().sum()
        print(f"Participation data matched on {matched:,} of {before:,} plays.")

    if ftn is not None:
        before = len(pbp)
        pbp = pbp.merge(ftn, on=["game_id", "play_id"], how="left")
        assert len(pbp) == before, "FTN merge created duplicate rows!"
        matched = pbp["qb_location"].notna().sum()
        print(f"FTN charting matched on {matched:,} of {before:,} plays.")

    # --- Summary printout -------------------------------------------------
    mem_mb = pbp.memory_usage(deep=True).sum() / 1024**2
    print("-" * 60)
    print(f"Combined shape : {pbp.shape[0]:,} rows x {pbp.shape[1]} columns")
    print(f"Memory usage   : {mem_mb:,.0f} MB")
    print("Plays per season:")
    print(pbp["season"].value_counts().sort_index().to_string())
    print("=" * 60)

    return pbp


if __name__ == "__main__":
    # Allows running "python -m src.data.loader" as a quick smoke test.
    df = load_data()
