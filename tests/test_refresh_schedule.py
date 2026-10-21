from datetime import datetime, timezone
import unittest
from zoneinfo import ZoneInfo

from src.api.data_refresh import next_refresh_at, seconds_until_refresh


class TuesdayScheduleTests(unittest.TestCase):
    def test_monday_schedules_tuesday_morning(self):
        now = datetime(2026, 10, 26, 20, tzinfo=ZoneInfo("America/Los_Angeles"))
        self.assertEqual(next_refresh_at(now).isoformat(), "2026-10-27T09:00:00-07:00")

    def test_tuesday_before_and_after_scheduled_time(self):
        zone = ZoneInfo("America/Los_Angeles")
        before = datetime(2026, 10, 27, 8, 59, tzinfo=zone)
        at = datetime(2026, 10, 27, 9, tzinfo=zone)
        self.assertEqual(seconds_until_refresh(before), 60)
        self.assertEqual(next_refresh_at(at).isoformat(), "2026-11-03T09:00:00-08:00")

    def test_dst_transitions_preserve_nine_am_local(self):
        zone = ZoneInfo("America/Los_Angeles")
        for now, hours in [(datetime(2026, 3, 3, 9, tzinfo=zone), 167),
                           (datetime(2026, 10, 27, 9, tzinfo=zone), 169)]:
            with self.subTest(now=now):
                self.assertEqual(next_refresh_at(now).hour, 9)
                self.assertEqual(seconds_until_refresh(now), hours * 3600)

    def test_utc_server_uses_pacific_schedule(self):
        now = datetime(2026, 10, 27, 15, tzinfo=timezone.utc)
        self.assertEqual(seconds_until_refresh(now), 3600)


if __name__ == "__main__":
    unittest.main()
