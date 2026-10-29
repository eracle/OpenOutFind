# openoutfind/core/pipeline/judge_or_search.py
"""Judge the best lead waiting, or run a generation of the walk — whichever buys a fit
for less.

Exploit asks this when no lead clears the bar (``calibration.current_threshold``). Each
route's cost per fit is a cost over a chance:

    judge   = c_verdict / p_pool
    search  = (c_generation / leads_per_generation + c_verdict) / p_generation

and the run takes the cheaper one. The costs are measured on this store's run:

- ``c_verdict`` and ``c_generation`` (``calibration.verdict_cost``/``search_cost``): the
  running mean of their wall time (``core/meter.py``) priced at ``run_hour_dollars``,
  plus a verdict's tokens at the model's rates, or an agent's fixed cost;
- ``leads_per_generation``: the new leads a generation added, from the store's lead
  count before and after it, as a running mean (``calibration.leads_per_search``).

**The two chances are compared by rank, on the GP's one scale.** ``p_generation`` is the
mean fitness of the last generation's children (``walk.last_children_fitness``) — what a
generation brings, not its best survivor, whose people are already in the pool — and
``p_pool`` is the GP's score for the best lead waiting. The GP orders people but is not
calibrated (0.15 where the truth was 0.28), so neither is read as a probability and
neither is mixed with a measured fit rate: the two are only ever weighed against each
other. Before the first generation there is nothing to weigh, and the pool is judged.

A search is cheap next to an agent's verdict, so under ``--agent-qualify`` this is close
to *judge while the best lead waiting scores like the people a generation brings*.
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


def best_waiting(qualifier, candidates) -> float:
    """The GP's score for the best lead in ``candidates`` — 0.0 when it cannot score."""
    if not candidates:
        return 0.0
    X = np.array([c.embedding_array for c in candidates], dtype=np.float64)
    probs = qualifier.predict_probs(X)
    return float(np.max(probs)) if probs is not None else 0.0


def judging_pays(qualifier, candidates) -> bool:
    """Whether judging the best lead waiting is cheaper per fit than a generation first.

    Logs both costs, so a run says why it searched or judged.
    """
    from openoutfind.core import meter
    from openoutfind.core.ml import calibration
    from openoutfind.core.pipeline.walk import last_children_fitness

    generation_rate = last_children_fitness(qualifier)
    if generation_rate is None:
        return True
    pool_rate = best_waiting(qualifier, candidates)
    verdict = calibration.verdict_cost()
    leads = calibration.leads_per_search()

    judge = cost_per_fit(verdict, pool_rate)
    search = (cost_per_fit(calibration.search_cost() / leads + verdict, generation_rate)
              if leads > 0 else math.inf)
    pays = judge <= search

    fits, verdicts = meter.totals(meter.POOL_FIT)
    logger.info("  %s judge %s/fit (best waiting %.2f; %d of %d fit since the last new "
                "leads) vs search %s/fit (a generation's children %.2f) → %s",
                colored("·", "cyan", attrs=["bold"]), _dollars(judge), pool_rate,
                int(fits), verdicts, _dollars(search), generation_rate,
                colored("judge" if pays else "search", "cyan", attrs=["bold"]))
    return pays


def _dollars(value: float) -> str:
    return "∞" if math.isinf(value) else f"${value:.3f}"
