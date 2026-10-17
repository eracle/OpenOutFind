# openoutfind/core/ml/calibration.py
"""The calibrate arm: the bar a lead must clear, derived from costs instead of chosen.

``min_gp_confidence`` is one bar read in two places — exploit judges only a lead
clearing it, and the promote gate buys an address only for one. Keeping one bar for
both is right. Leaving it a constant nobody measured was not: a 500-lead order held at
0.7 judged 71 people in 17h24m with 21,134 in its pool, because 2 of them cleared it,
while out of fold the leads scoring 0.4–0.7 fit 54–77% of the time.

**What the bar decides.** Every lead is judged before it ships, so the bar does not set
the list's quality; it sets where the run's spend goes — its LLM bill and its machine
hours. Each unit of work either judges the best lead waiting — a fit with probability p,
for one verdict — or searches for more, for a search plus the verdicts its new leads are
worth. The bar is the lowest score where judging is the cheaper way to a fit
(``openoutlearn.calibration``):

- **the fit rate at each score**, p(s): an isotonic regression of calibration verdicts
  on their out-of-fold scores. That takes verdicts on leads the model did not choose —
  so past the cold phase, while the store holds fewer than **√N calibration verdicts**
  (N = the unjudged pool), each ``top_up`` pass judges one lead drawn at random from the
  score band holding the fewest, ahead of explore and exploit;
- **the break-even rate**: what a dollar of searching buys at its best — a search adds
  the leads a page adds on this store, those worth it are judged — times a verdict's
  cost. Costs are in dollars: run time priced at ``run_hour_dollars`` plus what the step
  is billed — nothing for a Lead Finder page, the LLM's bill for a verdict. Times and
  bills are settings; leads per search is counted from the walk's own pages.

Once √N are in, ``measure`` stores the result as a ``CalibrationBar``, read by both
places through ``current_threshold``.

Choices worth knowing before changing anything:

- **√N of the pool as it stands, every pass.** Discovery growing the pool raises the
  target by itself, so there is no redraw rule and no sample rounds.
- **Scores are out of fold, on the current model** — not the score at draw time. Early
  draws were scored by a GP many verdicts older, and the calibration verdicts themselves
  (mostly low-band rejections) move the scores the bar is compared against.
- **No weights.** Stratifying on the score does not bias the fit rate *at* a score, and
  the rule only ever asks about one score at a time.
- **The verdict cost depends on who judges**: a model in seconds, at its own per-token
  rate (``MODEL_PRICES``: Haiku 5.5 ~$0.0004 a verdict, Opus 5.5 ~$0.015), a calling agent
  under ``--agent-qualify`` in tens of seconds and a few cents — so the same store gets a
  higher bar under a dearer judge, because each verdict costs it more.
- **The arm switches itself on** only when √N reaches bands ×
  ``calibration_min_band_verdicts``, so a small order's pool never spends verdicts on a
  sample too thin to draw a curve from.

Calibration verdicts train the GP and count in the walk's ``a``/``b`` like any other,
but not in the mode balance (``BayesianQualifier.balance_counts``).
"""
from __future__ import annotations

import bisect
import logging
import math
import random
from collections import Counter

import numpy as np
from termcolor import colored

from openoutlearn.calibration import break_even_rate, fit_rate_curve, lowest_bar

from openoutfind.core.conf import CAMPAIGN_CONFIG

logger = logging.getLogger(__name__)

# 0.1 wide up to 0.7, then 0.7–1.0 as one: the pool above 0.7 is a handful of leads.
BAND_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)
N_BANDS = len(BAND_EDGES) - 1

# k for the out-of-fold scores.
FOLDS = 5

_RNG = random.Random()


def band_of(p: float) -> int:
    """The band a score falls in, 0 to ``N_BANDS - 1``."""
    return min(max(bisect.bisect_right(BAND_EDGES, p) - 1, 0), N_BANDS - 1)


def _band_label(band: int) -> str:
    return f"{BAND_EDGES[band]:.1f}–{BAND_EDGES[band + 1]:.1f}"


def _dot() -> str:
    return colored("·", "cyan", attrs=["bold"])


# ── Size ──

def target_size(pool_size: int) -> int:
    """How many calibration verdicts a pool this size wants: √N."""
    return math.isqrt(pool_size)


def is_switched_on(pool_size: int) -> bool:
    """Whether √N is large enough to give every band its minimum."""
    return target_size(pool_size) >= N_BANDS * CAMPAIGN_CONFIG["calibration_min_band_verdicts"]


def calibration_pool(candidates: list) -> list:
    """The candidates a calibration draw may land on: the ones a verdict can be had for.

    ``run_qualification`` skips a lead with no profile text, and a draw it skips would
    be an open draw forever.
    """
    return [c for c in candidates if c.profile_text]


# ── Verdicts ──

def _labels() -> dict[int, int]:
    """Calibration verdicts, as ``lead_id → label`` — the same reading as
    ``Lead.get_labeled_rows``: a rejection is 0, any deal not FAILED is 1."""
    from openoutfind.core.models import CalibrationDraw
    from openoutfind.crm.models import Deal, DealState, Outcome

    labels: dict[int, int] = {}
    deals = Deal.objects.filter(
        lead_id__in=CalibrationDraw.objects.values("lead_id"),
    ).values_list("lead_id", "state", "outcome")
    for lead_id, state, outcome in deals:
        if state != DealState.FAILED:
            labels[lead_id] = 1
        elif outcome == Outcome.WRONG_FIT:
            labels[lead_id] = 0
    return labels


def verdict_counts() -> tuple[int, int]:
    """``(n_neg, n_pos)`` among the calibration verdicts."""
    labels = _labels()
    n_pos = sum(labels.values())
    return len(labels) - n_pos, n_pos


def should_calibrate(pool: list) -> bool:
    """Whether this pass belongs to the calibrate arm: switched on and short of √N."""
    if not is_switched_on(len(pool)):
        return False
    return len(_labels()) < target_size(len(pool))


# ── The arm ──

def next_lead(qualifier, pool: list, rng: random.Random = _RNG):
    """The lead the calibrate arm judges next, or None when it cannot draw.

    A draw still waiting on its verdict comes first — an LLM error or an
    ``--agent-qualify`` stop between the draw and the verdict must not let the arm draw
    again and tilt its bands. Otherwise a new draw from the thinnest band.
    """
    from openoutfind.core.models import CalibrationDraw

    drawn = {d.lead_id: d for d in CalibrationDraw.objects.all()}
    for lead in pool:
        if lead.pk in drawn:
            logger.info("  %s %s — verdict %d of %d, resuming the draw from P %s",
                        _dot(), colored("calibrate", "cyan", attrs=["bold"]),
                        len(_labels()) + 1, target_size(len(pool)),
                        _band_label(drawn[lead.pk].band))
            return lead
    return _draw(qualifier, pool, rng)


def _draw(qualifier, pool: list, rng: random.Random):
    """Score the pool, take the band with the fewest calibration verdicts, draw one
    lead uniformly from it, and record the draw."""
    from openoutfind.core.models import CalibrationDraw

    if not pool:
        return None
    probs = qualifier.predict_probs(np.array([c.embedding_array for c in pool], dtype=np.float64))
    if probs is None:
        return None

    bands = [band_of(p) for p in probs]
    sizes = Counter(bands)
    judged = Counter(
        CalibrationDraw.objects.filter(lead_id__in=_labels()).values_list("band", flat=True)
    )
    band = min(sizes, key=lambda b: (judged[b], b))
    index = rng.choice([i for i, b in enumerate(bands) if b == band])
    lead = pool[index]
    CalibrationDraw.objects.create(
        lead=lead, band=band, p_at_draw=float(probs[index]), band_size=sizes[band],
    )
    logger.info("  %s %s — verdict %d of %d, drawn at random from P %s (%d leads in it)",
                _dot(), colored("calibrate", "cyan", attrs=["bold"]),
                sum(judged.values()) + 1, target_size(len(pool)),
                _band_label(band), sizes[band])
    return lead


# ── The costs ──

def leads_per_search() -> float:
    """New leads a search has added on this store: leads discovered over pages fetched.

    A page is counted where the walk left a trace of it: every 100 offset a node
    advanced, plus the empty page that retired each dead or drained node.
    """
    from openoutfind.core.models import QueryNode
    from openoutfind.core.pipeline.select import DISCOVERY_PAGE_SIZE
    from openoutfind.crm.models import Lead

    nodes = QueryNode.objects.all()
    pages = sum(n.next_offset for n in nodes) / DISCOVERY_PAGE_SIZE + nodes.filter(
        state__in=[QueryNode.State.DEAD, QueryNode.State.DRAINED]).count()
    if not pages:
        return 0.0
    return Lead.objects.filter(synthetic=False).count() / pages


def _dollars(seconds: float, billed: float) -> float:
    """A step's cost: its run time at ``run_hour_dollars``, plus what it is billed."""
    return seconds / 3600 * CAMPAIGN_CONFIG["run_hour_dollars"] + billed


def search_cost() -> float:
    """What one search costs, in dollars: its time — Lead Finder bills nothing."""
    return _dollars(CAMPAIGN_CONFIG["search_seconds"], 0.0)


def verdict_cost() -> float:
    """What one verdict costs this run, in dollars: a model's call, or a calling agent's
    turn — its time plus the LLM's bill for it."""
    from openoutfind.core import agent_qualify
    from openoutfind.core.config import SiteConfig

    if agent_qualify.active():
        return _dollars(CAMPAIGN_CONFIG["verdict_seconds_agent"],
                        CAMPAIGN_CONFIG["verdict_dollars_agent"])
    return _dollars(CAMPAIGN_CONFIG["verdict_seconds_model"],
                    model_verdict_bill(SiteConfig.load().ai_model))


def model_verdict_bill(ai_model: str) -> float:
    """What ``ai_model`` is billed for one verdict: its rates times a verdict's tokens.

    ``verdict_dollars_model`` for a model ``MODEL_PRICES`` does not list.
    """
    from openoutfind.core.conf import MODEL_PRICES

    name = ai_model.rpartition(":")[2]
    family = max((key for key in MODEL_PRICES if name.startswith(key)), key=len, default=None)
    if family is None:
        return CAMPAIGN_CONFIG["verdict_dollars_model"]
    input_rate, output_rate = MODEL_PRICES[family]
    return (CAMPAIGN_CONFIG["verdict_input_tokens"] * input_rate
            + CAMPAIGN_CONFIG["verdict_output_tokens"] * output_rate) / 1_000_000


# ── The bar ──

def needs_measuring() -> bool:
    """Whether there are calibration verdicts the latest bar was not measured on.

    Only once there are enough to give every band its minimum — fewer means the arm is
    still filling, or never switched on.
    """
    from openoutfind.core.models import CalibrationBar

    n = len(_labels())
    if n < N_BANDS * CAMPAIGN_CONFIG["calibration_min_band_verdicts"]:
        return False
    latest = CalibrationBar.objects.order_by("-pk").first()
    return latest is None or latest.n_verdicts != n


def measure(qualifier, pool: list):
    """Measure the fit-rate curve, derive the bar from the costs, and store it.

    The scores are out of fold on a model over today's labels, so no calibration lead
    is scored by a fit that saw its own verdict. Returns the stored row.
    """
    from openoutfind.core.ml.qualifier import build_qualifier
    from openoutfind.core.models import CalibrationBar, CalibrationDraw
    from openoutfind.crm.models import Lead

    ids, X, y = Lead.get_labeled_rows()
    drawn = set(CalibrationDraw.objects.filter(lead_id__in=ids).values_list("lead_id", flat=True))
    index = [i for i, lead_id in enumerate(ids) if lead_id in drawn]

    logger.info("  %s measuring the fit rate on %d calibration verdicts (%d fits, one per fold)",
                _dot(), len(index), FOLDS)
    oof = build_qualifier(X, y).out_of_fold_probs(index, k=FOLDS)
    pool_scores = _scores(qualifier, pool)
    if oof is None or pool_scores is None:
        # Kept as a result, like a curve no score clears: measuring again on the same
        # verdicts would fail the same way, five fits a pass.
        logger.info("  %s no model to measure with — keeping %.2f", _dot(),
                    CAMPAIGN_CONFIG["min_gp_confidence"])
        return CalibrationBar.objects.create(threshold=None, break_even=0.0,
                                             n_verdicts=len(index))

    curve = fit_rate_curve(oof, y[index])
    pool_rates = curve.predict(pool_scores) if len(pool_scores) else np.empty(0)
    costs = (leads_per_search(), search_cost(), verdict_cost())
    break_even = break_even_rate(pool_rates, *costs)
    threshold = lowest_bar(curve, break_even)

    bar = CalibrationBar.objects.create(threshold=threshold, break_even=break_even,
                                        n_verdicts=len(index))
    _log_bar(bar, curve, pool_scores, pool_rates, costs)
    return bar


def _scores(qualifier, pool: list) -> np.ndarray | None:
    if not pool:
        return np.empty(0)
    return qualifier.predict_probs(np.array([c.embedding_array for c in pool], dtype=np.float64))


def _log_bar(bar, curve, pool_scores, pool_rates, costs):
    leads, search, verdict = costs
    logger.info("  %s a search adds %.1f leads for $%.4f, a verdict costs $%.4f — "
                "judging pays above a %.1f%% fit rate",
                _dot(), leads, search, verdict, bar.break_even * 100)
    if bar.threshold is None:
        logger.info("  %s no score fits that often — keeping %.2f", _dot(),
                    CAMPAIGN_CONFIG["min_gp_confidence"])
        return
    opened = pool_scores >= bar.threshold
    logger.info("  %s bar measured at %s (fits %.0f%% there) — opens %d of %d waiting, "
                "~%.0f fits expected",
                _dot(), colored(f"{bar.threshold:.2f}", "cyan", attrs=["bold"]),
                curve.predict([bar.threshold])[0] * 100,
                int(opened.sum()), len(pool_scores), float(pool_rates[opened].sum()))


def current_threshold() -> float:
    """The bar exploit and the promote gate read: the latest measured one, or
    ``min_gp_confidence`` until there is one."""
    from openoutfind.core.models import CalibrationBar

    latest = CalibrationBar.objects.order_by("-pk").first()
    if latest is None or latest.threshold is None:
        return CAMPAIGN_CONFIG["min_gp_confidence"]
    return latest.threshold
