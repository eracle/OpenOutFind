from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

from openoutfind.core.business_time import business_days_between

# Reference week: Mon 2026-03-16 … Sun 2026-03-22.
MON = datetime(2026, 3, 16, 10, 0, tzinfo=dt_timezone.utc)
FRI = datetime(2026, 3, 20, 10, 0, tzinfo=dt_timezone.utc)
SAT = datetime(2026, 3, 21, 10, 0, tzinfo=dt_timezone.utc)
SUN = datetime(2026, 3, 22, 10, 0, tzinfo=dt_timezone.utc)
NEXT_MON = datetime(2026, 3, 23, 10, 0, tzinfo=dt_timezone.utc)


class TestBusinessDaysBetween:
    def test_consecutive_weekdays(self):
        assert business_days_between(MON, MON.replace(day=17)) == 1

    def test_weekend_does_not_count(self):
        assert business_days_between(FRI, SAT) == 0
        assert business_days_between(FRI, SUN) == 0

    def test_friday_to_monday_is_one(self):
        assert business_days_between(FRI, NEXT_MON) == 1

    def test_spans_a_full_week(self):
        # Mon → Mon a week later: 5 working days, not 7.
        assert business_days_between(MON, MON.replace(day=23)) == 5

    def test_same_day_is_zero(self):
        assert business_days_between(MON, MON.replace(hour=23)) == 0

    def test_never_negative(self):
        assert business_days_between(NEXT_MON, FRI) == 0
