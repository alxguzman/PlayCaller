"""Run data/model updates in a child process without blocking API requests."""

import logging
from datetime import datetime, time, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys
import threading
from zoneinfo import ZoneInfo

from src.data.loader import PROJECT_ROOT, RAW_DIR

logger = logging.getLogger(__name__)


def refresh_enabled():
    # Existing Render services may not be managed by render.yaml. Keep the
    # hosted API serving the shipped artifact unless explicitly opted in.
    setting = os.environ.get("NFL_DATA_REFRESH_ENABLED")
    if setting is not None:
        return setting.strip().lower() in {"1", "true", "yes", "on"}
    # Preserve existing Render/dashboard opt-outs. Positive legacy values
    # enable the Tuesday schedule; they no longer specify an hourly interval.
    legacy = os.environ.get("NFL_DATA_REFRESH_HOURS")
    if legacy is not None:
        return float(legacy) > 0
    return os.environ.get("RENDER", "").lower() != "true"


SCHEDULE_TIMEZONE = "America/Los_Angeles"


def next_refresh_at(now=None):
    """Next Tuesday at 09:00 Pacific, preserving local time across DST."""
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo(SCHEDULE_TIMEZONE))
    day = local.date() + timedelta(days=(1 - local.weekday()) % 7)
    scheduled = datetime.combine(day, time(9), tzinfo=local.tzinfo)
    if scheduled <= local:
        scheduled += timedelta(days=7)
    return scheduled


def seconds_until_refresh(now=None):
    now = now or datetime.now(timezone.utc)
    return (next_refresh_at(now).astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds()


class DataRefreshWorker:
    def __init__(self, on_update):
        self.on_update = on_update
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if refresh_enabled():
            logger.warning("NFL background training enabled (startup and Tuesday 09:00 Pacific); requires more than 1.4 GB RAM")
            self.thread = threading.Thread(target=self.run, name="nfl-data-refresh", daemon=True)
            self.thread.start()
        else:
            logger.info("NFL background training disabled; serving the deployed model artifact")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)

    def run(self):
        while not self.stop_event.is_set():
            try:
                Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
                log_path = Path(RAW_DIR) / "refresh.log"
                with log_path.open("a", encoding="utf-8") as log:
                    command = [
                        sys.executable, "-u", "-m", "src.data.refresh",
                        "--update-model",
                    ]
                    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                    with subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=log, **options) as process:
                        while process.poll() is None:
                            if self.stop_event.wait(1):
                                process.terminate()
                                try:
                                    process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                return
                        if process.returncode:
                            logger.warning("NFL refresh failed; keeping the loaded model. See %s", log_path)
                        else:
                            self.on_update()
            except Exception:
                logger.exception("NFL refresh failed; keeping the loaded model")
            if self.stop_event.wait(seconds_until_refresh()):
                return
