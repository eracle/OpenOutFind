# openoutfind/core/pipeline/judge_or_search.py
"""Judge the best lead waiting, or fire a round of the walk — whichever buys a fit for
less.

Exploit asks this when no lead clears the bar (``calibration.current_threshold``). Both
sides are dollars per fit:

    judge   = c_verdict / rate(best waiting)
    search  = cost(arm) / expected fits(arm)

- **rate** is a fit rate on both sides — the GP's score through the isotonic map fitted on
  the store's verdicts (``store_snapshot.fit_rate_map``) — so a lead waiting and the people a
  search would bring are weighed on one scale, never the raw score against a mean of raw
  scores;
- **the arm** is the one worth the most at the mean belief (``search_arms.best_expected``),
  its cost its ready and ingest time and a verdict on each person it brings: the inverse
  of the value the walk ranks arms on, so the pick and the decision read one number;
- ``c_verdict`` (``calibration.verdict_cost``) is a verdict's measured time at
  ``run_hour_dollars`` plus its LLM bill, or an agent's fixed cost.

A search is cheap next to a verdict, so this is close to *judge while the best lead
waiting fits as often as the people the next search would bring*, whoever the judge is.
Before the first round there is nothing to weigh, and the pool is judged. **After a round
that added leads, the pool is judged at least once before the next**: an arm is valued on
the GP as the run's verdicts leave it, and with no verdict on the new people nothing has
tested what the round brought. On batch 5 a GP scoring everyone alike put the best waiting
lead at the base rate round after round, and six rounds fired with no verdict between them.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from termcolor import colored

logger = logging.getLogger(__name__)


def cost_per_fit(cost: float, rate: float) -> float:
    """Dollars per fit for a step costing ``cost`` that fits at ``rate`` — inf at 0."""
    return cost / rate if rate > 0 else math.inf


def best_waiting(qualifier, candidates, fit_rate) -> float:
    """The fit rate of the best lead in ``candidates`` — 0.0 when the GP cannot score."""
    if not candidates:
        return 0.0
    X = np.array([c.embedding_array for c in candidates], dtype=np.float64)
    probs = qualifier.predict_probs(X)
    return float(np.max(fit_rate(probs))) if probs is not None else 0.0


def decide(qualifier, candidates) -> bool:
    """Whether to judge the best lead waiting rather than search, on dollars per fit. Logs
    both costs, so a run says why it searched or judged."""
    from openoutfind.core import meter
    from openoutfind.core.ml import calibration
    from openoutfind.core.models import Page
    from openoutfind.core.pipeline import search_arms, store_snapshot

    from openoutfind.core.discovery_source import current

    if current() is not None:
        # The selected paid source is one bounded page, not a free frontier arm.
        # Exhaust useful local qualification work before asking it for anything.
        return True

    waiting = best_waiting(qualifier, candidates, store_snapshot.fit_rate_map())
    if not Page.objects.exists():
        return True
    fits, verdicts = meter.totals(meter.POOL_FIT)
    if not verdicts:
        logger.info("  %s the last round's people have no verdict yet (best waiting %.2f) → %s",
                    colored("·", "cyan", attrs=["bold"]), waiting,
                    colored("judge", "cyan", attrs=["bold"]))
        return True
    arm = search_arms.best_expected(qualifier)

    judge = cost_per_fit(calibration.verdict_cost(), waiting)
    search = cost_per_fit(arm.cost, arm.expected_fits) if arm else math.inf
    pays = judge <= search

    beyond = f"{arm.rate:.2f} × ~{arm.new:.0f} new" if arm else "no search worth firing"
    logger.info("  %s judge %s/fit (best waiting %.2f; %d of %d fit since the last new "
                "leads) vs search %s/fit (%s) → %s",
                colored("·", "cyan", attrs=["bold"]), _dollars(judge), waiting,
                int(fits), verdicts, _dollars(search), beyond,
                colored("judge" if pays else "search", "cyan", attrs=["bold"]))
    return pays


def _dollars(value: float) -> str:
    return "∞" if math.isinf(value) else f"${value:.3f}"
