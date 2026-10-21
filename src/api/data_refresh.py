"""Run data/model updates in a child process without blocking API requests."""

import logging
import os
from pathlib import Path
import subprocess
import sys
import threading

from src.data.loader import PROJECT_ROOT, RAW_DIR

logger = logging.getLogger(__name__)


def refresh_hours():
    # Existing Render services may not be managed by render.yaml. Keep the
    # hosted API serving the shipped artifact unless explicitly opted in.
    default = "0" if os.environ.get("RENDER", "").lower() == "true" else "6"
    return max(0, float(os.environ.get("NFL_DATA_REFRESH_HOURS", default)))


class DataRefreshWorker:
    def __init__(self, on_update):
        self.on_update = on_update
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        hours = refresh_hours()
        if hours > 0:
            logger.warning("NFL background training enabled (every %s hours); requires more than 1.4 GB RAM", hours)
            self.thread = threading.Thread(target=self.run, name="nfl-data-refresh", daemon=True)
            self.thread.start()
        else:
            logger.info("NFL background training disabled; serving the deployed model artifact")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)

    def run(self):
        hours = refresh_hours()
        while not self.stop_event.is_set():
            try:
                Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
                log_path = Path(RAW_DIR) / "refresh.log"
                with log_path.open("a", encoding="utf-8") as log:
                    command = [
                        sys.executable, "-u", "-m", "src.data.refresh",
                        "--update-model", "--max-age-hours", str(hours),
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
            if self.stop_event.wait(hours * 3600):
                return
