# openoutfind/core/pipeline/walk.py
"""The walk — a best-first search, one search at a time.

The initial population is written whole by the LLM (``icp.write_initial_population``), and
the first round fires it whole: nothing is known yet to choose among its searches by.
Every later round is one search, one blocking step of the run:

1. **Read the frontier** (``search_arms``) as the store stands now, after the run has judged
   what the last round brought in: every fired search's next page, every unfired written
   search, and the searches composed from every keyword the store holds (``compose``).
2. **Fire the search whose page brings the most fits at its mean plus one σ**
   (``Frontier.pick``). The worker thread only makes the API call.
3. **Wait for it**, then, in the main thread, record its page and take the page's people
   into the store (``discover.harvest``): one embedding pass, shortest profile first,
   since embedding is most of its time.
4. Hand back to the run, which goes on judging until it asks for the next round
   (``judge_or_search``).

**Nothing is ranked when it lands.** A page is evidence the next round reads, with the
verdicts judged since, and an arm never fired is never cut: it stays on the frontier,
recounted every round.

**Every store is anchored.** The anchors — the ICP written as profiles — are how the ICP
is always present in the round, so they are written with the initial population, and a
store from before them (one seeded with marks, which never ran cold) gets them before its
next round. When none can be written the walk does not start (``Unanchored``).

A search whose page 1 comes back empty is **dead**, and so is every search holding all its
terms. A search at the **bottom** — the page past its count, or page 50, where the
provider's 10,000-row reach ends — no longer pages, but a narrower or sibling search can
still be composed, since it opens a fresh window. A search whose pages
returned everything it counts is **drained**, and every search holding all its terms can
only re-find people the store holds: the frontier never offers one.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from termcolor import colored

from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.core.pipeline import search_arms

logger = logging.getLogger(__name__)

# Where a search's first poll goes until the store has measured its own ready time: the
# probe saw searches ready 1.1–3.6s after the submit.
FIRST_POLL_SECONDS = 1.5


@dataclass(frozen=True)
class Round:
    """What one round did."""

    number: int
    fired: int
    new_leads: int


# ── firing ───────────────────────────────────────────────────────────

def _shape() -> dict:
    """The size band and market code every search rides — the initial population's."""
    from openoutfind.core.models import Search

    first = Search.objects.filter(origin=Search.Origin.WRITTEN).order_by("pk").first()
    if first is None:
        return {}
    return {"headcount_min": first.headcount_min, "headcount_max": first.headcount_max,
            "country_code": first.country_code}


def _fire_all(arms, api_key: str) -> list:
    """Fire every arm at once; return each one's ``Fetched`` or the exception it raised.

    The threads make the API calls and nothing else, so an outage on one search is
    handed back rather than raised in a worker.
    """
    from openoutfind.core import meter
    from openoutfind.core.models import PAGE_SIZE
    from openoutfind.discovery import search
    from openoutfind.enrichment.bettercontact import BetterContactUnavailable

    first_poll = meter.mean(meter.SEARCH_READY_SECONDS, FIRST_POLL_SECONDS)
    shape = _shape()
    headcount = (shape.get("headcount_min", 1), shape.get("headcount_max", 10000))

    def fire(arm):
        try:
            return search(api_key, arm.terms.to_filters(headcount), limit=PAGE_SIZE,
                          offset=(arm.number - 1) * PAGE_SIZE, first_poll_s=first_poll)
        except BetterContactUnavailable as exc:
            return exc

    with ThreadPoolExecutor(max_workers=len(arms)) as executor:
        return list(executor.map(fire, arms))


def _take_in(arm, result, round_number: int):
    """Record one fired arm and its page. Returns ``(page, rows)``, or ``None`` when it
    leaves no page; ``harvest`` takes the whole round's rows in at once.

    A provider refusal (bad key, no credits, rate-limited through the backoff) is
    re-raised: every later search would repeat it. An outage is not evidence about the
    search and leaves no trace. An empty page 1 with a positive count is a transport
    artifact — the burst that once answered a 71-million-lead query with nothing in 0.0s
    — and leaves no trace either; with a count of 0 the search is dead. An empty deeper
    page is recorded: it marks the search's bottom.
    """
    from openoutfind.core import meter
    from openoutfind.core.errors import ErrorType
    from openoutfind.core.models import PAGE_SIZE, Page, Search
    from openoutfind.discovery import query_header, step_line
    from openoutfind.enrichment.bettercontact import BetterContactUnavailable

    if isinstance(result, BetterContactUnavailable):
        if result.error_type != ErrorType.PROVIDER_UNAVAILABLE:
            raise result
        logger.warning("%s", step_line("fetch", f"provider unavailable ({result}) — "
                                                f"{arm} not fired", glyph="⚠", color="red"))
        return None

    fetched, ready = result
    meter.record(meter.SEARCH_READY_SECONDS, ready)
    logger.info("%s", query_header(arm.terms.to_filters(), (arm.number - 1) * PAGE_SIZE))
    found = f" · {fetched.leads_found:,} in the index" if fetched.leads_found else ""
    logger.info("%s", step_line("leadfinder", f"{len(fetched.leads)} leads{found} "
                                              f"· {arm.move}"))

    if arm.search is None:
        arm.search, _ = Search.objects.get_or_create(key=arm.terms.key, defaults={
            "terms": arm.terms.as_json(), "move": arm.move, **_shape()})

    if arm.number == 1 and not fetched.leads:
        if fetched.leads_found:
            logger.warning("%s", step_line("fetch", f"empty page but {fetched.leads_found:,} "
                                                    f"in the index — transport artifact",
                                           glyph="⚠", color="yellow"))
            return None
        arm.search.dead = True
        arm.search.save(update_fields=["dead"])
        logger.debug("%s", step_line("fetch", "nobody matches — the search is dead, and so "
                                              "is any holding all its terms",
                                     glyph="✗", color="yellow"))
        return None

    if arm.number == 1:
        arm.search.leads_found = fetched.leads_found
        arm.search.save(update_fields=["leads_found"])
        if arm.count:
            logger.debug("  N(q): estimated ~%.0f, the provider counts %s  %s", arm.count,
                         fetched.leads_found, arm)

    page = Page.objects.create(search=arm.search, number=arm.number, round=round_number,
                               rows=len(fetched.leads), expected_count=arm.count or None,
                               expected_rate=arm.theta)
    return page, fetched.leads


# ── one round ────────────────────────────────────────────────────────

class Unanchored(OpenOutFindError):
    """The anchors could not be written, and a round covers them: the walk does not
    start without the ICP written as profiles."""

    def __init__(self):
        super().__init__(ErrorType.UNANCHORED,
                         "the ideal profiles could not be written (the model failed or "
                         "returned none) — the walk does not start without them; run again")


def _anchor(site_config) -> None:
    """Write the anchors when the store has none: with the initial population, or once on
    a store from before them. Never tops up a set that exists."""
    from openoutfind.core.pipeline.icp import anchor_leads, ensure_anchors

    if anchor_leads():
        return
    ensure_anchors(site_config)
    if not anchor_leads():
        raise Unanchored()


def run_round(site_config, qualifier) -> Round | None:
    """Read the frontier, fire its search, take the page in. ``None`` when no arm is worth
    firing.

    The first call writes the initial population and the anchors (an LLM call each, or
    the one ``population_pending`` under ``--agent-qualify``) and fires the population
    whole.
    """
    from django.db.models import Max

    from openoutfind.core import meter
    from openoutfind.core.models import Page, Search
    from openoutfind.core.pipeline.discover import harvest
    from openoutfind.core.pipeline.icp import write_initial_population
    from openoutfind.crm.models import Lead

    if not Search.objects.exists():
        write_initial_population(site_config)
    _anchor(site_config)

    started = time.monotonic()
    before = Lead.objects.filter(synthetic=False, marked=False).count()
    last = Page.objects.aggregate(Max("round"))["round__max"]
    number = 0 if last is None else last + 1

    arms = (search_arms.written_arms() if last is None
            else search_arms.next_round(qualifier, number))
    if not arms:
        return None
    results = _fire_all(arms, site_config.bettercontact_api_key)

    pages = [taken for arm, result in zip(arms, results)
             if (taken := _take_in(arm, result, number)) and taken[1]]
    if pages:
        ingest_started = time.monotonic()
        created = harvest(pages, pages[0][0].search.country_code)
        if created:
            meter.record(meter.INGEST_SECONDS_PER_LEAD,
                         (time.monotonic() - ingest_started) / created)

    new_leads = Lead.objects.filter(synthetic=False, marked=False).count() - before
    meter.record(meter.ROUND_SECONDS, time.monotonic() - started)
    meter.record(meter.ROUND_LEADS, new_leads)
    if new_leads:
        # New leads make a new pool, so how often its verdicts fit starts over.
        meter.reset(meter.POOL_FIT)

    return _log(number, arms, len(pages), new_leads)


def _log(number: int, arms, fired: int, new_leads: int) -> Round:
    logger.info("%s", colored(
        f"■ round {number}: {fired} of {len(arms)} search(es) fired · "
        f"+{new_leads} new lead(s)", "blue"))
    return Round(number, fired, new_leads)
