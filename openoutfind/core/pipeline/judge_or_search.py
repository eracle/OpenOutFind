# openoutfind/core/pipeline/judge_or_search.py
"""Judge the best lead waiting, or fire a round of the walk — whichever buys a fit for
less.

Exploit asks this when no lead clears the bar (``calibration.current_threshold``). Both
sides are a fit rate over a cost, in dollars per fit:

    judge   = c_verdict / rate(best waiting)
    search  = (c_arm + c_verdict · new) / (new · rate(arm))      for the best arm

- **rate** is a fit rate on both sides — the GP's score through the isotonic map fitted on
  the store's verdicts (``frontier.fit_rate_map``) — so a lead waiting and the people a
  search would bring are weighed on one scale, never the raw score against a mean of raw
  scores;
- **the arm** is the frontier's best (``frontier.plan``): a search whose new people are
  expected to fit more often than the best lead waiting, at ``new`` of them for
  ``c_arm`` (its ready time and their ingest time);
- ``c_verdict`` (``calibration.verdict_cost``) is a verdict's measured time at
  ``run_hour_dollars`` plus its LLM bill, or an agent's fixed cost.

A search is cheap next to a verdict, so this is close to *judge while the best lead
waiting fits as often as the people the best search would bring*. Before the first round
there is nothing to weigh, and the pool is judged. The best waiting lead's rate is handed
on as the **floor** a round's arms must beat (``walk.run_round``).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from termcolor import colored

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decision:
    """Whether to judge, and the fit rate a search's people must beat to be worth one."""

    judge: bool
    floor: float


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


def decide(qualifier, candidates) -> Decision:
    """Judge the best lead waiting, or search, on dollars per fit. Logs both costs, so a
    run says why it searched or judged."""
    from openoutfind.core import meter
    from openoutfind.core.ml import calibration
    from openoutfind.core.models import Page
    from openoutfind.core.pipeline import frontier

    floor = best_waiting(qualifier, candidates, frontier.fit_rate_map())
    if not Page.objects.exists():
        return Decision(True, floor)
    batch = frontier.plan(qualifier, floor)
    verdict = calibration.verdict_cost()

    judge = cost_per_fit(verdict, floor)
    best = min(batch, key=lambda arm: arm.cost_per_fit(verdict), default=None)
    search = best.cost_per_fit(verdict) if best is not None else math.inf
    pays = judge <= search

    fits, verdicts = meter.totals(meter.POOL_FIT)
    beyond = (f"{len(batch)} search(es) would beat it, the best {best} at "
              f"{best.rate:.2f} × ~{best.new:.0f} new" if best else "no search beats it")
    logger.info("  %s judge %s/fit (best waiting %.2f; %d of %d fit since the last new "
                "leads) vs search %s/fit (%s) → %s",
                colored("·", "cyan", attrs=["bold"]), _dollars(judge), floor,
                int(fits), verdicts, _dollars(search), beyond,
                colored("judge" if pays else "search", "cyan", attrs=["bold"]))
    return Decision(pays, floor)


def _dollars(value: float) -> str:
    return "∞" if math.isinf(value) else f"${value:.3f}"
