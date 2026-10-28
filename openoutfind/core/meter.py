# openoutfind/core/meter.py
"""Running averages of what the run measures, kept in the store.

Each quantity is a ``RunningMean`` row: ``record`` adds one event's value to its sum and
one to its count, ``mean`` divides them. No decay and no window — the mean of every event
so far — except where the caller says a quantity starts over (``reset``): the pool's fit
rate does, each time a search adds leads, because a new page makes it a new pool.

The names are the module's constants, so a typo is an ``AttributeError`` and not a
second row nobody reads.
"""
from __future__ import annotations

# Wall time of one Lead Finder page, polling included.
SEARCH_SECONDS = "search_seconds"
# Wall time of one verdict: a model's call, or for a calling agent the time from the
# candidate being handed over to the verdict coming back.
VERDICT_SECONDS_MODEL = "verdict_seconds_model"
VERDICT_SECONDS_AGENT = "verdict_seconds_agent"
# 1 for a fit, 0 for a no-fit, over the verdicts since a search last added leads.
POOL_FIT = "pool_fit"


def record(name: str, value: float) -> None:
    """Add one event to ``name``'s running mean."""
    from django.db.models import F

    from openoutfind.core.models import RunningMean

    RunningMean.objects.get_or_create(name=name)
    RunningMean.objects.filter(name=name).update(total=F("total") + value, count=F("count") + 1)


def totals(name: str) -> tuple[float, int]:
    """``(sum, count)`` for ``name`` — ``(0.0, 0)`` before its first event."""
    from openoutfind.core.models import RunningMean

    row = RunningMean.objects.filter(name=name).first()
    return (row.total, row.count) if row is not None else (0.0, 0)


def mean(name: str, default: float) -> float:
    """``name``'s running mean, or ``default`` until it has an event."""
    total, count = totals(name)
    return total / count if count else default


def reset(name: str) -> None:
    """Start ``name`` over: its next mean is of the events after this call."""
    from openoutfind.core.models import RunningMean

    RunningMean.objects.filter(name=name).update(total=0.0, count=0)
