"""Sync nflverse's release API, rebuild features, and optionally retrain.

Run ``python -m src.data.refresh --update-model`` for a complete update.
The API checks on startup and each Tuesday at 09:00 Pacific while running.
"""

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

from src.data.loader import PROJECT_ROOT, RAW_DIR

API_URL = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}"
STATUS_PATH = Path(RAW_DIR) / "refresh_status.json"
PIPELINE_VERSION = "2-missing-context-training"
ASSETS = [
    ("pbp", "play_by_play_{y}.csv.gz", "play_by_play_{y}.csv.gz"),
    ("ftn_charting", "ftn_charting_{y}.csv", "ftn_charting_{y}.csv"),
    ("pbp_participation", "pbp_participation_{y}.csv", "participation_{y}.csv"),
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def requested_seasons(today=None):
    # January/February games belong to the previous calendar year's season.
    today = today or datetime.now(timezone.utc).date()
    season = today.year - (today.month < 3)
    return list(range(2023, season + 1))


def read_status():
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_status(status):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    part = STATUS_PATH.with_suffix(".tmp")
    part.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(part, STATUS_PATH)


@contextmanager
def refresh_lock():
    """OS lock releases even if a refresh process crashes or is terminated."""
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(RAW_DIR) / "refresh.lock", "a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def release_assets(tag):
    headers = {"User-Agent": "PlayCaller", "Accept": "application/vnd.github+json"}
    # Hosted CI shares public IPs. Authenticate metadata requests to avoid
    # the low anonymous GitHub API limit; never pass this to asset downloads.
    token = os.environ.get("NFLVERSE_GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        API_URL.format(tag=tag),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    return {asset["name"]: asset for asset in release["assets"]}


def download(url, dest, expected_size=None):
    """Replace a cached asset only after its complete download succeeds."""
    part = str(dest) + ".part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "PlayCaller"})
        with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
            while chunk := response.read(1024 * 256):
                out.write(chunk)
        if expected_size is not None and os.path.getsize(part) != expected_size:
            raise IOError(f"Incomplete download: {Path(dest).name}")
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)


def refresh_raw_files(status=None):
    """Discover published assets and download only changed versions."""
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    status = status if status is not None else read_status()
    cached = status.setdefault("assets", {})
    seasons = requested_seasons()
    published = []
    for tag, remote_template, local_template in ASSETS:
        assets = release_assets(tag)
        for year in seasons:
            remote = remote_template.format(y=year)
            asset = assets.get(remote)
            if asset is None:
                if tag == "pbp" and year < seasons[-1]:
                    raise RuntimeError(f"Required play-by-play season {year} is missing from nflverse.")
                print(f"{remote}: not published yet; skipped.", flush=True)
                continue
            local = local_template.format(y=year)
            dest = Path(RAW_DIR) / local
            version = {key: asset[key] for key in ("id", "updated_at", "size")}
            if cached.get(local) != version or not dest.exists() or dest.stat().st_size != asset["size"]:
                download(asset["browser_download_url"], dest, asset["size"])
                cached[local] = version
                write_status(status)
                print(f"{local}: downloaded.", flush=True)
            else:
                print(f"{local}: unchanged.", flush=True)
            if tag == "pbp":
                published.append(year)
    if not published:
        raise RuntimeError("nflverse returned no usable play-by-play seasons.")
    status.update(
        source="nflverse GitHub Releases API",
        available_seasons=published,
        requested_season=seasons[-1],
        latest_published_season=max(published),
        checked_at=utc_now(),
    )
    return status


def rebuild_features():
    from src.data.cleaner import clean_data
    from src.data.concepts import add_play_concepts
    from src.data.features import add_team_features
    from src.data.loader import load_data

    return add_team_features(add_play_concepts(clean_data(load_data())))


def refresh(skip_rebuild=False, update_model=False, max_age_hours=0):
    with refresh_lock():
        status = read_status()
        processed = Path(PROJECT_ROOT) / "data/processed/pbp_features.csv"
        model = Path(PROJECT_ROOT) / "model/play_success_model.joblib"
        if (max_age_hours and status.get("completed_at") and not status.get("error")
                and status.get("pipeline_version") == PIPELINE_VERSION):
            age = datetime.now(timezone.utc) - datetime.fromisoformat(status["completed_at"])
            features_ready = skip_rebuild or (
                processed.exists() and status.get("features_revision") == status.get("revision")
            )
            model_ready = not update_model or (
                model.exists() and status.get("model_revision") == status.get("revision")
            )
            if age.total_seconds() < max_age_hours * 3600 and features_ready and model_ready:
                return
        status.update(started_at=utc_now(), refreshing=True, error=None)
        write_status(status)
        try:
            status = refresh_raw_files(status)
            revision = hashlib.sha256(json.dumps(
                {"assets": status["assets"], "pipeline_version": PIPELINE_VERSION}, sort_keys=True,
            ).encode()).hexdigest()
            status["revision"] = revision
            status["pipeline_version"] = PIPELINE_VERSION
            if not skip_rebuild:
                if status.get("features_revision") != revision or not processed.exists():
                    df = rebuild_features()
                    latest = df[df["season"] == df["season"].max()]
                    status.update(
                        features_revision=revision,
                        latest_game_date=str(df["game_date"].max()),
                        latest_week=int(latest["week"].max()),
                        play_count=len(df),
                    )
                    write_status(status)
            if update_model:
                from src.model.train import MODEL_PATH, train
                from src.model.season_log import main as build_scorecard

                if status.get("model_revision") != revision or not Path(MODEL_PATH).exists():
                    artifact = train()
                    build_scorecard(artifact)
                    status.update(model_revision=revision, model_updated_at=utc_now())
            status["completed_at"] = utc_now()
        except Exception as exc:
            status["error"] = str(exc)
            raise
        finally:
            status["refreshing"] = False
            write_status(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rebuild", action="store_true", help="only sync raw API data")
    parser.add_argument("--update-model", action="store_true", help="also retrain and rebuild the season scorecard")
    parser.add_argument("--max-age-hours", type=float, default=0, help="skip a recently completed update")
    args = parser.parse_args()
    if args.skip_rebuild and args.update_model:
        parser.error("--update-model requires the feature rebuild")
    refresh(**vars(args))


if __name__ == "__main__":
    main()
