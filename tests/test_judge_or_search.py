# tests/test_judge_or_search.py
"""Judge the pool or fire a round, on costs measured from the run.

Covers the running means (``core/meter.py``), where the run takes its verdict
measurements, and the rule itself (``core/pipeline/judge_or_search.py``). Where a round
measures itself is in ``tests/test_walk.py``, and how the frontier values its arms in
``tests/test_frontier.py``.
"""
from __future__ import annotations

import math
from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pytest
from django.utils import timezone

from openoutfind.core import agent_qualify, meter
from openoutfind.core.ml.qualifier import BayesianQualifier
from openoutfind.core.pipeline import judge_or_search
from openoutfind.core.pipeline.qualify import run_qualification
from openoutfind.crm.models import Lead, PendingQualification


# ── The running means ──

@pytest.mark.django_db
class TestMeter:
    def test_the_mean_is_the_sum_over_the_count(self):
        for value in (10, 20, 60):
            meter.record(meter.ROUND_SECONDS, value)

        assert meter.totals(meter.ROUND_SECONDS) == (90.0, 3)
        assert meter.mean(meter.ROUND_SECONDS, default=70) == pytest.approx(30)

    def test_the_default_stands_in_until_the_first_event(self):
        assert meter.totals(meter.ROUND_SECONDS) == (0.0, 0)
        assert meter.mean(meter.ROUND_SECONDS, default=70) == 70

    def test_a_reset_starts_the_mean_over(self):
        meter.record(meter.POOL_FIT, 1)
        meter.reset(meter.POOL_FIT)
        meter.record(meter.POOL_FIT, 0)

        assert meter.totals(meter.POOL_FIT) == (0.0, 1)

    def test_resetting_a_quantity_never_measured_is_harmless(self):
        meter.reset(meter.POOL_FIT)
        assert meter.totals(meter.POOL_FIT) == (0.0, 0)


# ── Where the run measures a verdict ──

@pytest.fixture
def _agent():
    yield
    agent_qualify._active.set(False)
    agent_qualify._verdict.set(None)


def _candidate(url="https://www.linkedin.com/in/alice/"):
    return Lead.objects.create(profile_url=url, profile_text="extraction lead",
                               embedding=np.ones(384, dtype=np.float32).tobytes())


@pytest.mark.django_db
class TestVerdictMeasurements:
    def test_a_model_verdict_is_timed_and_counted_toward_the_pool(self, site_config):
        lead = _candidate()
        with (
            patch("openoutfind.core.ml.qualifier.qualify_with_llm", return_value=(1, "fits")),
            patch("openoutfind.core.pipeline.brief.ensure_rules", return_value="rules"),
        ):
            run_qualification(site_config, BayesianQualifier(seed=42), candidates=[lead])

        assert meter.totals(meter.VERDICT_SECONDS_MODEL)[1] == 1
        assert meter.totals(meter.POOL_FIT) == (1.0, 1)

    def test_an_agent_verdict_takes_from_the_hand_over_to_the_answer(self, site_config, _agent):
        lead = _candidate()
        pending = PendingQualification.objects.create(lead=lead)
        PendingQualification.objects.filter(pk=pending.pk).update(
            created=timezone.now() - timedelta(seconds=45))
        agent_qualify.enable(agent_qualify.Verdict(fit=False, reason="a supplier"))

        run_qualification(site_config, BayesianQualifier(seed=42))

        total, count = meter.totals(meter.VERDICT_SECONDS_AGENT)
        assert count == 1
        assert 45 <= total < 60
        assert meter.totals(meter.POOL_FIT) == (0.0, 1)

    def test_a_session_left_waiting_is_not_a_verdicts_time(self, site_config, _agent):
        """Batch 5's first measured agent verdict had waited 1h46m on a stopped session."""
        lead = _candidate()
        pending = PendingQualification.objects.create(lead=lead)
        PendingQualification.objects.filter(pk=pending.pk).update(
            created=timezone.now() - timedelta(hours=2))
        agent_qualify.enable(agent_qualify.Verdict(fit=False, reason="a supplier"))

        run_qualification(site_config, BayesianQualifier(seed=42))

        assert meter.totals(meter.VERDICT_SECONDS_AGENT) == (0.0, 0)
        assert meter.totals(meter.POOL_FIT) == (0.0, 1)


# ── The rule ──

def _arm(rate, new=20.0, cost=0.0001):
    from openoutfind.core.pipeline.frontier import Arm
    from openoutfind.core.pipeline.query import Terms

    return Arm(Terms.written(title="production", tag="cannabis"), 1, "swap",
               new=new, rate=rate, cost=cost)


def _decide(*, waiting, batch, verdict=0.05, fired=True, judged=True):
    """``decide`` with a verdict at ``verdict`` dollars, the best lead waiting at a fit
    rate of ``waiting`` and the frontier's batch at ``batch``; ``judged``, a verdict
    given since the last round's new leads."""
    from openoutfind.core.models import Page, Search

    if fired and not Page.objects.exists():
        search = Search.objects.create(key="k", terms={"title": ["founder"]})
        Page.objects.create(search=search, number=1, round=0, rows=3)
    if judged and not meter.totals(meter.POOL_FIT)[1]:
        meter.record(meter.POOL_FIT, 0)
    plan = patch("openoutfind.core.pipeline.frontier.arms_beating", return_value=batch)
    with (
        plan as planned,
        patch("openoutfind.core.pipeline.frontier.fit_rate_map", return_value=lambda p: p),
        patch.object(judge_or_search, "best_waiting", return_value=waiting),
        patch("openoutfind.core.ml.calibration.verdict_cost", return_value=verdict),
    ):
        decision = judge_or_search.decide(None, [object()])
    return decision, planned


@pytest.mark.django_db
class TestTheRule:
    def test_before_the_first_round_the_pool_is_judged(self):
        decision, plan = _decide(waiting=0.1, batch=[_arm(0.5)], fired=False)
        assert decision.judge is True
        plan.assert_not_called()

    def test_a_round_that_added_leads_is_judged_before_the_next(self, caplog):
        """Batch 5: a GP scoring everyone alike fired six rounds with no verdict between."""
        meter.record(meter.POOL_FIT, 1)
        meter.reset(meter.POOL_FIT)  # what a round adding leads does
        with caplog.at_level("INFO"):
            decision, plan = _decide(waiting=0.12, batch=[_arm(0.5)], judged=False)

        assert decision == judge_or_search.Decision(True, 0.12)
        plan.assert_not_called()
        assert "no verdict yet" in caplog.text

    def test_one_verdict_after_a_round_lets_the_rule_weigh_again(self):
        decision, _ = _decide(waiting=0.12, batch=[_arm(0.5)])
        assert decision.judge is False

    def test_a_best_lead_below_what_a_search_would_bring_is_left_for_a_search(self):
        """The batch-5 shape: the pool's best is off-industry, a search's people fit more."""
        decision, _ = _decide(waiting=0.15, batch=[_arm(0.3)])
        assert decision == judge_or_search.Decision(False, 0.15)

    def test_the_batch_is_asked_for_people_beating_the_best_waiting(self):
        _, plan = _decide(waiting=0.15, batch=[_arm(0.3)])
        assert plan.call_args.args[1] == 0.15

    def test_a_best_lead_fitting_like_the_best_search_is_judged(self):
        decision, _ = _decide(waiting=0.3, batch=[_arm(0.3)])
        assert decision.judge is True

    def test_with_no_search_worth_firing_the_pool_is_judged(self):
        decision, _ = _decide(waiting=0.1, batch=[])
        assert decision.judge is True

    def test_a_dearer_search_keeps_the_run_on_the_pool_longer(self):
        cheap, _ = _decide(waiting=0.25, batch=[_arm(0.3, cost=0.0001)])
        dear, _ = _decide(waiting=0.25, batch=[_arm(0.3, new=2.0, cost=1.0)])

        assert (cheap.judge, dear.judge) == (False, True)

    def test_the_decision_is_logged_with_both_costs(self, caplog):
        meter.record(meter.POOL_FIT, 0)
        meter.record(meter.POOL_FIT, 0)
        with caplog.at_level("INFO"):
            _decide(waiting=0.1, batch=[_arm(0.3)])

        assert "judge $" in caplog.text
        assert "0 of 2 fit" in caplog.text
        assert "1 search(es) would beat it" in caplog.text
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
