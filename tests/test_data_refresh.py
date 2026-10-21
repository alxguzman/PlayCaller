import io
from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data import loader, refresh
from src.model.train import split_by_season


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.patches = [
            patch.object(refresh, "RAW_DIR", str(self.root)),
            patch.object(refresh, "STATUS_PATH", self.root / "status.json"),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(self.directory.cleanup)
        for p in self.patches:
            self.addCleanup(p.stop)

    def test_season_rollover_and_postseason(self):
        self.assertEqual(refresh.requested_seasons(date(2027, 1, 15))[-1], 2026)
        self.assertEqual(refresh.requested_seasons(date(2027, 9, 15))[-1], 2027)

    def test_compressed_cache_does_not_double_count_csv(self):
        (self.root / "play_by_play_2026.csv").touch()
        (self.root / "play_by_play_2026.csv.gz").touch()
        with patch.object(loader, "RAW_DIR", str(self.root)):
            files = loader._find_files("play_by_play_*")
        self.assertEqual(files, [str(self.root / "play_by_play_2026.csv.gz")])

    def test_truncated_download_preserves_last_good_file(self):
        dest = self.root / "data.csv"
        dest.write_bytes(b"old data")
        with patch.object(refresh.urllib.request, "urlopen", return_value=io.BytesIO(b"short")):
            with self.assertRaises(OSError):
                refresh.download("https://example.com/data.csv", dest, 100)
        self.assertEqual(dest.read_bytes(), b"old data")
        self.assertFalse(Path(str(dest) + ".part").exists())

    def test_changed_assets_download_and_unchanged_assets_are_cached(self):
        asset = {
            "name": "play_by_play_2026.csv.gz", "id": 1,
            "updated_at": "2026-10-21T14:35:32Z", "size": 3,
            "browser_download_url": "https://example.com/pbp",
        }
        def download(url, dest, size):
            Path(dest).write_bytes(b"csv")
        with (
            patch.object(refresh, "ASSETS", [refresh.ASSETS[0]]),
            patch.object(refresh, "requested_seasons", return_value=[2026]),
            patch.object(refresh, "release_assets", return_value={asset["name"]: asset}),
            patch.object(refresh, "download", side_effect=download) as fetch,
        ):
            status = refresh.refresh_raw_files({})
            refresh.refresh_raw_files(status)
            self.assertEqual(fetch.call_count, 1)
            asset["updated_at"] = "2026-10-22T12:00:00Z"
            refresh.refresh_raw_files(status)
            self.assertEqual(fetch.call_count, 2)

    def test_unpublished_new_season_falls_back_to_available_season(self):
        asset = {"name": "play_by_play_2025.csv.gz", "id": 1, "updated_at": "now", "size": 0,
                 "browser_download_url": "https://example.com/pbp"}
        with (
            patch.object(refresh, "ASSETS", [refresh.ASSETS[0]]),
            patch.object(refresh, "requested_seasons", return_value=[2025, 2026]),
            patch.object(refresh, "release_assets", return_value={asset["name"]: asset}),
            patch.object(refresh, "download", side_effect=lambda url, dest, size: Path(dest).touch()),
        ):
            status = refresh.refresh_raw_files({})
        self.assertEqual(status["latest_published_season"], 2025)
        self.assertEqual(status["requested_season"], 2026)

    def test_failed_refresh_reports_error_without_marking_success(self):
        with patch.object(refresh, "refresh_raw_files", side_effect=OSError("network unavailable")):
            with self.assertRaises(OSError):
                refresh.refresh()
        status = refresh.read_status()
        self.assertFalse(status["refreshing"])
        self.assertEqual(status["error"], "network unavailable")
        self.assertNotIn("completed_at", status)

    def test_os_lock_prevents_concurrent_updates(self):
        with refresh.refresh_lock():
            with self.assertRaises(OSError):
                with refresh.refresh_lock():
                    self.fail("Second refresh acquired the lock")

    def test_newest_season_is_held_out_and_future_rows_never_train(self):
        meta = pd.DataFrame({"season": [2023, 2024, 2025, 2025, 2026, 2027], "week": [1, 18, 1, 18, 1, 1]})
        X = pd.DataFrame({"x": range(len(meta))})
        y = pd.Series([0] * len(meta))
        train, _, val, _, test, _ = split_by_season(X, y, meta, test_season=2026)
        self.assertEqual(list(train.index), [0, 1, 2])
        self.assertEqual(list(val.index), [3])
        self.assertEqual(list(test.index), [4])
        *_, test, _ = split_by_season(X, y, meta)
        self.assertEqual(list(test.index), [5])


class ApiStatusTests(unittest.TestCase):
    def test_background_worker_reloads_only_after_success(self):
        from src.api.data_refresh import DataRefreshWorker
        for returncode in (0, 1):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as directory:
                callback = MagicMock()
                worker = DataRefreshWorker(callback)
                process = MagicMock()
                process.poll.return_value = returncode
                process.returncode = returncode
                worker.stop_event.wait = MagicMock(return_value=True)
                with (
                    patch("src.api.data_refresh.RAW_DIR", directory),
                    patch("src.api.data_refresh.subprocess.Popen") as popen,
                ):
                    popen.return_value.__enter__.return_value = process
                    worker.run()
                    command = popen.call_args.args[0]
                    self.assertIn("--update-model", command)
                    self.assertNotIn("--max-age-hours", command)
                self.assertEqual(callback.call_count, 1 if returncode == 0 else 0)

    def test_api_stays_available_and_reports_loaded_model_freshness(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        artifact = {"trained_at": "2026-10-21", "data_through": "2026-10-19",
                    "test_season": 2026, "data_week": 7}
        with (
            patch.dict("os.environ", {"NFL_DATA_REFRESH_ENABLED": "0"}),
            patch("src.api.main.load_artifact", return_value=artifact),
            patch("src.api.main.read_status", return_value={"error": "Source unavailable"}),
            TestClient(app) as client,
        ):
            self.assertTrue(client.get("/healthz").json()["model_loaded"])
            status = client.get("/data/status").json()
            self.assertEqual(status["model_season"], 2026)
            self.assertEqual(status["error"], "Source unavailable")
            self.assertFalse(status["refresh_enabled"])
            self.assertIsNone(status["next_scheduled_refresh_at"])


if __name__ == "__main__":
    unittest.main()
