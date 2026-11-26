# tests/core/pipeline/test_judge_or_search.py
"""Judge the pool or fire a round, on costs measured from the run
(``core/pipeline/judge_or_search.py``). The running means it reads are
``tests/core/test_meter.py``; how the frontier values its arms is ``test_search_arms.py``.
"""
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core import meter
from openoutfind.core.pipeline import judge_or_search
from openoutfind.crm.models import Lead


# ── The rule ──

def _arm(rate, new=20.0, cost=None):
    """An arm whose cost is, unless given, a verdict at $0.05 on each person it brings."""
    from openoutfind.core.pipeline.search_arms import Arm
    from openoutfind.core.pipeline.terms import Terms

    return Arm(Terms.written(title="production", tag="coffee"), 1, "swap", new=new,
               rate=rate, cost=0.0001 + 0.05 * new if cost is None else cost)


def _decide(*, waiting, arm, verdict=0.05, fired=True, judged=True):
    """``decide`` with a verdict at ``verdict`` dollars, the best lead waiting at a fit
    rate of ``waiting`` and the frontier's best arm at ``arm``; ``judged``, a verdict
    given since the last round's new leads."""
    from openoutfind.core.models import Page, Search

    if fired and not Page.objects.exists():
        search = Search.objects.create(key="k", terms={"title": ["founder"]})
        Page.objects.create(search=search, number=1, round=0, rows=3)
    if judged and not meter.totals(meter.POOL_FIT)[1]:
        meter.record(meter.POOL_FIT, 0)
    plan = patch("openoutfind.core.pipeline.search_arms.best_expected", return_value=arm)
    with (
        plan as planned,
        patch("openoutfind.core.pipeline.store_snapshot.fit_rate_map", return_value=lambda p: p),
        patch.object(judge_or_search, "best_waiting", return_value=waiting),
        patch("openoutfind.core.ml.calibration.verdict_cost", return_value=verdict),
    ):
        decision = judge_or_search.decide(None, [object()])
    return decision, planned


@pytest.mark.django_db
class TestTheRule:
    def test_before_the_first_round_the_pool_is_judged(self):
        decision, plan = _decide(waiting=0.1, arm=_arm(0.5), fired=False)
        assert decision is True
        plan.assert_not_called()

    def test_a_round_that_added_leads_is_judged_before_the_next(self, caplog):
        """A GP scoring everyone alike would fire round after round with no verdict between."""
        meter.record(meter.POOL_FIT, 1)
        meter.reset(meter.POOL_FIT)  # what a round adding leads does
        with caplog.at_level("INFO"):
            decision, plan = _decide(waiting=0.12, arm=_arm(0.5), judged=False)

        assert decision is True
        plan.assert_not_called()
        assert "no verdict yet" in caplog.text

    def test_one_verdict_after_a_round_lets_the_rule_weigh_again(self):
        decision, _ = _decide(waiting=0.12, arm=_arm(0.5))
        assert decision is False

    def test_a_best_lead_below_what_a_search_would_bring_is_left_for_a_search(self):
        """The pool's best is off-industry, and a search's people fit more."""
        decision, _ = _decide(waiting=0.15, arm=_arm(0.3))
        assert decision is False

    def test_a_best_lead_fitting_like_the_best_search_is_judged(self):
        decision, _ = _decide(waiting=0.3, arm=_arm(0.3))
        assert decision is True

    def test_the_search_side_pays_a_verdict_for_each_person_it_brings(self):
        """Without that bill a page would look cheaper than judging at the same fit rate."""
        unbilled, _ = _decide(waiting=0.3, arm=_arm(0.3, cost=0.0001))
        billed, _ = _decide(waiting=0.3, arm=_arm(0.3))

        assert (unbilled, billed) == (False, True)

    def test_with_no_search_left_to_fire_the_pool_is_judged(self):
        decision, _ = _decide(waiting=0.1, arm=None)
        assert decision is True

    def test_with_no_search_and_no_lead_worth_a_verdict_the_pool_is_still_judged(self):
        decision, _ = _decide(waiting=0.0, arm=None)
        assert decision is True

    def test_a_dearer_search_keeps_the_run_on_the_pool_longer(self):
        cheap, _ = _decide(waiting=0.25, arm=_arm(0.3))
        dear, _ = _decide(waiting=0.25, arm=_arm(0.3, new=2.0, cost=1.0))

        assert (cheap, dear) == (False, True)

    def test_the_decision_is_logged_with_both_costs(self, caplog):
        meter.record(meter.POOL_FIT, 0)
        meter.record(meter.POOL_FIT, 0)
        with caplog.at_level("INFO"):
            _decide(waiting=0.1, arm=_arm(0.3))

        assert "judge $" in caplog.text
        assert "0 of 2 fit" in caplog.text
        assert "0.30 × ~20 new" in caplog.text
        assert "→" in caplog.text and "search" in caplog.text


class TestArithmetic:
    def test_a_zero_rate_costs_without_bound(self):
        assert judge_or_search.cost_per_fit(0.05, 0.0) == math.inf
        assert judge_or_search.cost_per_fit(0.05, 0.25) == pytest.approx(0.2)

    def test_the_best_waiting_lead_is_the_highest_fit_rate(self):
        class Fixed:
            def predict_probs(self, X):
                return np.array([0.1, 0.4, 0.2])

        leads = [Lead(embedding=np.ones(384, dtype=np.float32).tobytes()) for _ in range(3)]
        halved = lambda p: np.asarray(p) / 2  # noqa: E731
        assert judge_or_search.best_waiting(Fixed(), leads, halved) == pytest.approx(0.2)
        assert judge_or_search.best_waiting(Fixed(), [], halved) == 0.0
