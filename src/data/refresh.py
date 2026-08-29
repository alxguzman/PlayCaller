"""
refresh.py
----------
Downloads the latest nflverse data files into data/raw and rebuilds
data/processed/pbp_features.csv.

nflverse republishes play_by_play_YYYY.csv.gz nightly during the season,
so running this weekly (Tuesday, after Monday Night Football lands) keeps
the dataset current. The weekly routine is:

    python -m src.data.refresh
    python -m src.model.train
    python -m src.model.evaluate

Missing files are fine: FTN/participation for a new season may not exist
yet, and play_by_play_2026 won't exist until week 1 is played. The script
notes what's missing and moves on - the loader already treats those files
as optional.

Run with:  python -m src.data.refresh  [--skip-rebuild]
"""

import argparse
import os
import urllib.error
import urllib.request

from src.data.loader import RAW_DIR

# Edit this list once per year when a new season starts.
SEASONS = [2023, 2024, 2025, 2026]

# (release tag, remote filename, local filename, required?)
# Local names differ from remote for participation: nflverse publishes
# "pbp_participation_YYYY.csv" but loader.py globs "participation_*".
ASSETS = [
    ("pbp", "play_by_play_{y}.csv.gz", "play_by_play_{y}.csv.gz", True),
    ("ftn_charting", "ftn_charting_{y}.csv", "ftn_charting_{y}.csv", False),
    ("pbp_participation", "pbp_participation_{y}.csv", "participation_{y}.csv", False),
]

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/{tag}/{fname}"


def download(url, dest):
    """
    Download url to dest. Returns True on success, False if the file
    doesn't exist on the server (HTTP 404).

    Writes to dest + ".part" first, then renames - so a dropped
    connection never corrupts or deletes a file we already have.
    """
    part = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
    except urllib.error.HTTPError as e:
        if os.path.exists(part):
            os.remove(part)
        if e.code == 404:
            return False
        raise
    except Exception:
        if os.path.exists(part):
            os.remove(part)
        raise
    os.replace(part, dest)
    return True


def refresh_raw_files():
    """Download every configured asset for every configured season."""
    os.makedirs(RAW_DIR, exist_ok=True)
    print("=" * 60)
    print("REFRESHING RAW DATA (nflverse)")
    print("=" * 60)

    for year in SEASONS:
        for tag, remote_tpl, local_tpl, required in ASSETS:
            remote = remote_tpl.format(y=year)
            local = local_tpl.format(y=year)
            url = BASE_URL.format(tag=tag, fname=remote)
            dest = os.path.join(RAW_DIR, local)

            ok = download(url, dest)
            if ok:
                size_mb = os.path.getsize(dest) / 1024**2
                print(f"  {local:<32} downloaded ({size_mb:,.1f} MB)")
            elif required:
                # Expected for the newest season before week 1 is played.
                print(f"  {local:<32} NOT AVAILABLE yet on nflverse - skipped")
            else:
                print(f"  {local:<32} not published (optional) - skipped")


def rebuild_features():
    """Rerun the full feature pipeline (same as python -m src.data.features)."""
    from src.data.cleaner import clean_data
    from src.data.concepts import add_play_concepts
    from src.data.features import add_team_features
    from src.data.loader import load_data

    add_team_features(add_play_concepts(clean_data(load_data())))


def main():
    parser = argparse.ArgumentParser(description="Refresh nflverse data files.")
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="only download files; don't rebuild data/processed/pbp_features.csv",
    )
    args = parser.parse_args()

    refresh_raw_files()

    if args.skip_rebuild:
        print("Skipping feature rebuild (--skip-rebuild).")
    else:
        rebuild_features()
        print("\nNext steps:  python -m src.model.train")
        print("             python -m src.model.evaluate")


if __name__ == "__main__":
    main()
