# openoutfind/core/pipeline/judge_or_search.py
"""Judge the best lead waiting, or search for more — whichever buys a fit for less.

Exploit asks this when no lead clears the bar (``calibration.current_threshold``). Each
route's cost per fit is a cost over a chance:

    judge   = c_verdict / p_pool
    search  = (c_search / leads_per_search + c_verdict) / p_page

and the run takes the cheaper one. Every term is measured on this store's run:

- ``c_verdict`` and ``c_search`` (``calibration.verdict_cost``/``search_cost``): the
  running mean of their wall time (``core/meter.py``) priced at ``run_hour_dollars``,
  plus a verdict's tokens at the model's rates, or an agent's fixed cost;
- ``leads_per_search``: the new leads a page has added, counted from the walk;
- ``p_page``: the walk's own estimate for the best page it could fire next
  (``select.best_estimate``) — how its own people, or its parent's, look to the GP;
- ``p_pool``: how often the verdicts since a search last added leads were fits, a
  running mean that starts over with each new page, smoothed toward ``p_page`` with the
  walk's weight of two — so a fresh page is judged before its first verdict is in, and
  one no-fit does not send the run back to searching.

Solved for the pool's rate, judging pays while ``p_pool ≥ p_page · c_verdict /
(c_verdict + c_search / leads_per_search)``: the bar moves with the measured costs instead
of being set. A search is cheap next to an agent's verdict, so under ``--agent-qualify``
that is close to *judge while the pool fits as often as a fresh page would*.

This replaced a fixed floor on BALD (``min_bald_gain``, 0.04 nats) that judged any lead
the model was unsure of before searching. Off-industry profiles are the ones it is most
unsure of, so on CaleafTech's batch 5 it judged soap makers, perfume brands and an SBA
lender — 42 verdicts, no fit — while 6 queries fired in the whole store.
"""
from __future__ import annotations

import logging
import math

from termcolor import colored

logger = logging.getLogger(__name__)

# The pseudo-verdicts ``p_page`` lends the pool's rate: the walk's own smoothing weight
# (``select``: ``P̂ = (a + 2·P̂(parent)) / (a + b + 2)``).
POOL_PRIOR_WEIGHT = 2


def pool_fit_rate(page_rate: float) -> float:
    """How often the pool's verdicts fit, smoothed toward ``page_rate``."""
    from openoutfind.core import meter

    fits, verdicts = meter.totals(meter.POOL_FIT)
    return (fits + POOL_PRIOR_WEIGHT * page_rate) / (verdicts + POOL_PRIOR_WEIGHT)


def cost_per_fit(cost: float, rate: float) -> float:
    """Dollars per fit for a step costing ``cost`` that fits at ``rate`` — inf at 0."""
    return cost / rate if rate > 0 else math.inf


def judging_pays(qualifier) -> bool:
    """Whether judging the best lead waiting is cheaper per fit than searching first.

    Logs both costs, so a run says why it searched or judged.
    """
    from openoutfind.core import meter
    from openoutfind.core.ml import calibration
    from openoutfind.core.pipeline import select

    scores = select.HarvestScores.load(qualifier, select.LabelStore.load().base_rate)
    page_rate = select.best_estimate(scores)
    pool_rate = pool_fit_rate(page_rate)
    verdict = calibration.verdict_cost()
    leads = calibration.leads_per_search()

    judge = cost_per_fit(verdict, pool_rate)
    search = (cost_per_fit(calibration.search_cost() / leads + verdict, page_rate)
              if leads > 0 else math.inf)
    pays = judge <= search

    fits, verdicts = meter.totals(meter.POOL_FIT)
    logger.info("  %s judge %s/fit (%d of %d fit since the last new page) vs search %s/fit "
                "(best next page P̂ %.2f) → %s",
                colored("·", "cyan", attrs=["bold"]), _dollars(judge), int(fits), verdicts,
                _dollars(search), page_rate,
                colored("judge" if pays else "search", "cyan", attrs=["bold"]))
    return pays


def _dollars(value: float) -> str:
    return "∞" if math.isinf(value) else f"${value:.3f}"
