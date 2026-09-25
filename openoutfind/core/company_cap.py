"""``OPENOUTFIND_MAX_PER_COMPANY``: at most N qualified leads from one company.

Once the walk finds a team whose titles match the seed, it drains it — eight of one
sample's ten came from one employer. A list of a hundred people at a dozen companies is a
dozen prospects, so a campaign may cap how many of its fits one company supplies.

A company is **full** once it holds N qualified leads (a deal that is not ``FAILED`` on a
lead that is not disqualified — what the export counts). A full company is read at two
points and nowhere else:

* **discovery** — a row whose company is full is not accepted, so it never becomes a Lead;
* **the qualification scan** — a lead already stored for a full company is never put up
  for a verdict. It stays in the pool; it is only not judged.

Nothing is capped at export: with both points closed a company cannot reach N + 1. Which
N qualify is the qualifier's ranking, the same as without a cap. A search that only
returns full companies stops yielding new leads, so the walk moves on by itself.

Companies are matched on ``Company.key`` (the domain, or the name when there is none). A
lead with no company — none named, or a placeholder employer — never counts toward a cap.

The cap belongs to the campaign, like its product and target: set once, blank for none.
"""
from __future__ import annotations

from openoutfind.core.config import SiteConfig, variable_for
from openoutfind.core.errors import ErrorType, OpenOutFindError


def limit() -> int | None:
    """The campaign's cap, or ``None`` when it has none.

    Raises ``OpenOutFindError(BAD_CONFIG)`` for a value that is not a positive integer:
    a cap of zero would starve the campaign, and a typo read as "no cap" would be the
    silent failure the variable exists to prevent.
    """
    value = SiteConfig.load().max_per_company
    if not value:
        return None
    if not value.isdigit() or int(value) < 1:
        raise OpenOutFindError(
            ErrorType.BAD_CONFIG,
            f"{variable_for('max_per_company')}={value!r} — set a whole number of 1 or more, "
            "or leave it unset for no cap.",
        )
    return int(value)


def full_company_keys() -> frozenset[str]:
    """The ``Company.key`` of every company already holding the cap's worth of fits."""
    from django.db.models import Count

    from openoutfind.crm.models import Deal, DealState

    cap = limit()
    if cap is None:
        return frozenset()
    counts = (
        Deal.objects.filter(lead__disqualified=False, lead__company__isnull=False)
        .exclude(state=DealState.FAILED)
        .values("lead__company__key")
        .annotate(fits=Count("id"))
        .filter(fits__gte=cap)
    )
    return frozenset(row["lead__company__key"] for row in counts)
