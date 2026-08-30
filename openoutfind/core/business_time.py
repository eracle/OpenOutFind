# openoutfind/core/business_time.py
"""Working-day arithmetic — how old a thread really is.

A thread that sat untouched over a weekend has not really gone quiet, and the
outreach agent is told the gap in working days so it reads a Friday reply answered
on Monday as one day old rather than three.

Measuring is all that is left here. ``add_business_hours`` used to schedule the
*next* touch, back when an unanswered thread earned a follow-up on a countdown;
nobody is chased any more, so there is no gap to schedule and the only remaining
question is how much working time has passed.

Public holidays are deliberately not modelled: they are per-country, per-year data
we do not carry, and this figure is coarse enough that a missed holiday costs one
slightly-off number in a prompt.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

SATURDAY = 5


def is_business_day(day: date) -> bool:
    """True for Monday–Friday."""
    return day.weekday() < SATURDAY


def business_days_between(start: datetime, end: datetime) -> int:
    """Whole working days elapsed from ``start`` to ``end`` (weekends excluded).

    Counts the business days in the half-open range ``(start.date(), end.date()]``,
    so Friday → Monday is 1 and Friday → Sunday is 0. Never negative.
    """
    return _business_days_in_range(start.date() + timedelta(days=1),
                                   end.date() + timedelta(days=1))


# ── Internals ─────────────────────────────────────────────────────


def _business_days_in_range(start: date, end: date) -> int:
    """Number of Monday–Friday days in the half-open range ``[start, end)``."""
    days = (end - start).days
    if days <= 0:
        return 0
    full_weeks, remainder = divmod(days, 7)
    count = full_weeks * 5
    for offset in range(remainder):
        if is_business_day(start + timedelta(days=offset)):
            count += 1
    return count
