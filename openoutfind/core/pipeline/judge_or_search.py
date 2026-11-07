# openoutfind/core/pipeline/judge_or_search.py
"""Judge the best lead waiting, or fire a round of the walk — whichever buys a fit for
less.

Exploit asks this when no lead clears the bar (``calibration.current_threshold``). Both
sides are a fit rate over a cost, in dollars per fit:

    judge   = c_verdict / rate(best waiting)
    search  = (c_round + c_verdict · new) / (new · rate(round))  for the round it would fire

- **rate** is a fit rate on both sides — the GP's score through the isotonic map fitted on
  the store's verdicts (``frontier.fit_rate_map``) — so a lead waiting and the people a
  search would bring are weighed on one scale, never the raw score against a mean of raw
  scores;
- **the round** is the search ``frontier.next_round`` would fire next: its ``new`` people,
  its ``rate``, and ``c_round`` its ready and ingest time;
- ``c_verdict`` (``calibration.verdict_cost``) is a verdict's measured time at
  ``run_hour_dollars`` plus its LLM bill, or an agent's fixed cost.

A search is cheap next to a verdict, so this is close to *judge while the best lead
waiting fits as often as the people the next round would bring*, whoever the judge is.
Before the first round
there is nothing to weigh, and the pool is judged. **After a round that added leads, the
pool is judged at least once before the next**: an arm is valued on the GP as the run's
verdicts leave it, and with no verdict on the new people nothing has tested what the
round brought. On batch 5 a GP scoring everyone alike put the best waiting lead at the
base rate round after round, and six rounds fired with no verdict between them. The best
waiting lead's rate is handed on as the **floor** a round's arms must beat
(``walk.run_round``).
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


def round_rate(batch) -> float:
    """The fit rate of the people ``batch`` would bring, each arm weighing by its new
    people — 0.0 when it would bring nobody new."""
    new = sum(arm.new for arm in batch)
    return sum(arm.new * arm.rate for arm in batch) / new if new else 0.0


def round_cost_per_fit(batch, verdict: float) -> float:
    """Dollars per fit to fire ``batch`` and judge every new person it brings."""
    new = sum(arm.new for arm in batch)
    if not new:
        return math.inf
    return cost_per_fit(sum(arm.cost for arm in batch) / new + verdict, round_rate(batch))


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
    from django.db.models import Max

    from openoutfind.core import meter
    from openoutfind.core.ml import calibration
    from openoutfind.core.models import Page
    from openoutfind.core.pipeline import frontier

    floor = best_waiting(qualifier, candidates, frontier.fit_rate_map())
    if not Page.objects.exists():
        return Decision(True, floor)
    fits, verdicts = meter.totals(meter.POOL_FIT)
    if not verdicts:
        logger.info("  %s the last round's people have no verdict yet (best waiting %.2f) → %s",
                    colored("·", "cyan", attrs=["bold"]), floor,
                    colored("judge", "cyan", attrs=["bold"]))
        return Decision(True, floor)
    number = Page.objects.aggregate(Max("round"))["round__max"] + 1
    batch = frontier.next_round(qualifier, number, floor)
    verdict = calibration.verdict_cost()

    judge = cost_per_fit(verdict, floor)
    search = round_cost_per_fit(batch, verdict)
    pays = judge <= search

    new = sum(arm.new for arm in batch)
    beyond = (f"a round of {len(batch)} at {round_rate(batch):.2f} × ~{new:.0f} new"
              if new else "no round brings anyone")
    logger.info("  %s judge %s/fit (best waiting %.2f; %d of %d fit since the last new "
                "leads) vs search %s/fit (%s) → %s",
                colored("·", "cyan", attrs=["bold"]), _dollars(judge), floor,
                int(fits), verdicts, _dollars(search), beyond,
                colored("judge" if pays else "search", "cyan", attrs=["bold"]))
    return Decision(pays, floor)


def _dollars(value: float) -> str:
    return "∞" if math.isinf(value) else f"${value:.3f}"
