from collections import OrderedDict
import io
import unittest
from unittest.mock import MagicMock, patch

from src.api.data_refresh import DataRefreshWorker, refresh_hours
from src.live import espn


class HostedRefreshTests(unittest.TestCase):
    def test_render_without_blueprint_does_not_start_training(self):
        with patch.dict("os.environ", {"RENDER": "true"}, clear=True):
            self.assertEqual(refresh_hours(), 0)
            with patch("src.api.data_refresh.threading.Thread") as thread:
                worker = DataRefreshWorker(MagicMock())
                worker.start()
                worker.stop()
                thread.assert_not_called()

    def test_local_default_still_refreshes(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(refresh_hours(), 6)

    def test_explicit_setting_wins(self):
        for value in ("0", "12"):
            with patch.dict("os.environ", {"RENDER": "true", "NFL_DATA_REFRESH_HOURS": value}, clear=True):
                self.assertEqual(refresh_hours(), float(value))


class LiveCacheTests(unittest.TestCase):
    def test_cache_is_bounded_and_keeps_recently_used_games(self):
        with (
            patch.object(espn, "_cache", OrderedDict()) as cache,
            patch.object(espn, "_CACHE_MAX", 2),
            patch.object(espn.time, "monotonic", return_value=0),
            patch.object(espn.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(b'{"plays": []}')) as fetch,
        ):
            for game in ("1", "2", "1", "3"):
                espn.get_game_summary(game)
            self.assertEqual(fetch.call_count, 3)
            self.assertEqual(len(cache), 2)
            self.assertIn(espn.SUMMARY_URL.format(event_id="1"), cache)
            self.assertNotIn(espn.SUMMARY_URL.format(event_id="2"), cache)

    def test_expired_games_are_released_and_refetched(self):
        with (
            patch.object(espn, "_cache", OrderedDict()) as cache,
            patch.object(espn.time, "monotonic", return_value=0) as clock,
            patch.object(espn.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(b'{}')) as fetch,
        ):
            espn.get_game_summary("1")
            espn.get_game_summary("2")
            clock.return_value = 9
            espn.get_game_summary("1")
            self.assertEqual(fetch.call_count, 3)
            self.assertEqual(list(cache), [espn.SUMMARY_URL.format(event_id="1")])


if __name__ == "__main__":
    unittest.main()
