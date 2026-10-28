# openoutfind/core/pipeline/top_up.py
"""Filling the pipeline — the one step whose queue is a campaign, not a deal.

Everything else the cycle does is found by querying deals. This cannot be: a lead
nobody has discovered yet has no row to be found. So the bottom of the hierarchy
asks a campaign instead — *do you have enough people waiting?* — and if not, spends
**one** unit of work: label a lead, discover a page, or (in the cold phase) both.

This was the `find_email` handler's hidden second job. Given one queue token it ran
``pools.find_candidate``, a ``while True`` that discovered, LLM-qualified and
promoted until it produced a candidate — anywhere between zero and dozens of LLM
calls behind a single row. Here the loop *is* the cycle: one move per call, and the
caller decides whether to come back.

**The move is the qualifier's own acquisition strategy, unchanged.** This is the
crown jewel and it is ported verbatim:

- **cold phase** (``is_cold`` — invented anchor positives still padding the positive
  class) — do **both** moves every pass: one query in, one label out. Ranking still
  leans on the anchors' *guess* at the ICP, so no observed signal says a label is
  worth more than a page, and a rule that picked one would be a preference dressed
  as a policy. Discovery's return is ignored, so a saturated pool or a provider
  outage still leaves a lead to label; only an empty pool stalls.
- **explore** (``neg ≤ pos``) — label the most *informative* lead (max BALD), with no
  confidence gate: a low-confidence lead is exactly the label that teaches the GP
  most, so filtering by confidence would throw away the point of exploring.
- **calibrate** (fewer than √N calibration verdicts, N the unjudged pool) — judge one
  lead drawn at random from the thinnest score band, ahead of explore and exploit, and
  once √N are in, measure the bar exploit reads. See ``core/ml/calibration.py``.
- **exploit** (``neg > pos``, calibration verdicts left out) — qualify the strongest lead
  clearing ``calibration.current_threshold()`` (``min_gp_confidence`` until a bar is
  measured), the one whose qualification will buy an email rather than
  park at QUALIFIED. When none clears it, **judge or search, whichever is cheaper per
  fit** on this run's measured costs (``judge_or_search``): judging takes the best lead
  by P from the whole pool, searching fires the walk's next page. The bar itself never
  moves here — a pass that discovers leaves the class balance untouched and the frontier
  one node further on, and when the frontier is drained the pool is judged to its end.

There used to be a second path here: the freemium promo campaign ran none of the
above, because its leads were already in the account and a downloaded kit model
ranked them. That campaign is gone, so every campaign now takes the one path.
"""
from __future__ import annotations

import logging

import numpy as np
from termcolor import colored

from openoutfind.core.ml import calibration
from openoutfind.core.ml.qualifier import BayesianQualifier
from openoutfind.core.pipeline import vocabulary
from openoutfind.core.pipeline.discover import discover
from openoutfind.core.pipeline.icp import ANCHOR_COUNT
from openoutfind.core.pipeline.judge_or_search import judging_pays
from openoutfind.core.pipeline.qualify import fetch_qualification_candidates, run_qualification

logger = logging.getLogger(__name__)


def top_up(site_config, qualifier: BayesianQualifier) -> bool:
    """Spend one unit of work filling the pipeline. Returns whether it did.

    Label a lead, discover leads, or (cold) both. False means there is nothing left to
    do: nothing worth labelling and nothing left to discover.

    ``qualifier`` is the caller's, not one built here: the cycle's scoring row may
    already have fitted the model in the same action, and the fit is the expensive
    part. See ``cycle._one_model_per_action``.
    """
    # Whatever last pass accepted reaches the frontier here. Discovery refreshes too, but
    # a run that only ever labels would otherwise never fold its own acceptances in — and
    # the walk's ranking is counted from exactly those profiles. Guarded by an accepted
    # count, so a pass that changed nothing costs one `COUNT`.
    vocabulary.refresh()

    if qualifier.is_cold:
        logger.info("  %s cold phase — %d/%d real positive(s), exploiting the anchors' guess",
                   colored("·", "cyan", attrs=["bold"]),
                   qualifier.n_real_positives, ANCHOR_COUNT)
        discover(site_config, qualifier)
        candidates = fetch_qualification_candidates()
        if not candidates:
            return False
        return run_qualification(site_config, qualifier, candidates=candidates) is not None

    candidates = fetch_qualification_candidates()

    # Calibrate — ahead of both, because what exploit can judge depends on the bar this
    # measures. It waits on neither the balance nor the bar, so it cannot stall the way
    # exploit did at an unmeasured 0.7. See `core/ml/calibration.py`.
    pool = calibration.calibration_pool(candidates)
    if calibration.should_calibrate(pool):
        lead = calibration.next_lead(qualifier, pool)
        if lead is not None:
            judged = run_qualification(site_config, qualifier, candidates=[lead],
                                       preselected=True)
            if judged is not None and judged != lead.profile_url:
                # Under --agent-qualify a candidate an earlier run left pending is
                # answered first, whoever asks. Its verdict is not a calibration verdict;
                # the draw stays open for the next pass.
                logger.info("  %s answered a candidate left pending earlier — the "
                            "calibration draw is next", colored("·", "cyan", attrs=["bold"]))
            qualifier.exclude_from_balance(*calibration.verdict_counts())
            return judged is not None
    elif calibration.needs_measuring():
        calibration.measure(qualifier, pool)

    mode = qualifier.acquisition_mode()

    if mode == "exploit (p)":
        consumable = _consumable_candidates(qualifier, candidates)
        if consumable:
            return run_qualification(site_config, qualifier, candidates=consumable) is not None
        # **Nothing clearing the bar is not evidence the pool is spent**, nor that a
        # search is the better move: it is a choice between two costs, measured on this
        # run (`judge_or_search`). Judging takes the best lead by P from the whole pool.
        # Falling straight to discovery here cost a live install 14h33m with 3 real
        # positives and a pool that topped out at P=0.37, so the rule judges a fresh
        # pool before its first verdict is in.
        if candidates and judging_pays(qualifier):
            return run_qualification(site_config, qualifier, candidates=candidates) is not None
        # A fired page is the work, however much of it we had already seen. Reading
        # *new leads* here is what stopped a live run with 100 rows in hand: the page
        # was all familiar profiles, so discovery reported nothing and the whole job
        # ended `goal_unreached` with the frontier untouched below it.
        if discover(site_config, qualifier):
            return True
        # Nothing left to search, so the pool is all there is.
        return bool(candidates) and run_qualification(
            site_config, qualifier, candidates=candidates) is not None

    # Explore — label the most informative lead we have. The GP is fitted here, so it
    # ranks the pool and there is a best lead to pick; an empty pool is the one case
    # with no lead to label, so page one in first.
    if not candidates:
        if not discover(site_config, qualifier):
            return False
        candidates = fetch_qualification_candidates()
        if not candidates:
            return True  # every profile on that page was already ours — still a move
    return run_qualification(site_config, qualifier, candidates=candidates) is not None


def _consumable_candidates(qualifier: BayesianQualifier, candidates: list) -> list:
    """The candidates clearing the spend gate — the ones a qualification can convert.

    Empty means exploit has nothing to convert (so it should widen instead): either
    the GP is unfitted or no lead reaches ``calibration.current_threshold()``, the same
    bar the promote gate uses.
    """
    if not candidates:
        return []

    X = np.array([c.embedding_array for c in candidates], dtype=np.float64)
    probs = qualifier.predict_probs(X)
    if probs is None:
        return []
    threshold = calibration.current_threshold()
    return [c for c, p in zip(candidates, probs) if p >= threshold]
