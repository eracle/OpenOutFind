# tests/test_judge_or_search.py
"""Judge the pool or run a generation, on costs measured from the run.

Covers the running means (``core/meter.py``), where the run takes its verdict
measurements, and the rule itself (``core/pipeline/judge_or_search.py``). Where a
generation measures itself is in ``tests/test_walk.py``.
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
            meter.record(meter.GENERATION_SECONDS, value)

        assert meter.totals(meter.GENERATION_SECONDS) == (90.0, 3)
        assert meter.mean(meter.GENERATION_SECONDS, default=70) == pytest.approx(30)

    def test_the_default_stands_in_until_the_first_event(self):
        assert meter.totals(meter.GENERATION_SECONDS) == (0.0, 0)
        assert meter.mean(meter.GENERATION_SECONDS, default=70) == 70

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

def _decide(*, generation, waiting, verdict=0.05, search=0.0004, leads=50.0):
    """``judging_pays`` with the costs set, a generation's children at ``generation`` and
    the best lead waiting at ``waiting`` — both on the GP's scale."""
    with (
        patch("openoutfind.core.pipeline.walk.last_children_fitness", return_value=generation),
        patch.object(judge_or_search, "best_waiting", return_value=waiting),
        patch("openoutfind.core.ml.calibration.verdict_cost", return_value=verdict),
        patch("openoutfind.core.ml.calibration.search_cost", return_value=search),
        patch("openoutfind.core.ml.calibration.leads_per_search", return_value=leads),
    ):
        return judge_or_search.judging_pays(None, [object()])


@pytest.mark.django_db
class TestTheRule:
    def test_before_the_first_generation_the_pool_is_judged(self):
        assert _decide(generation=None, waiting=0.1) is True

    def test_a_best_lead_below_what_a_generation_brings_is_left_for_a_search(self):
        """The batch-5 shape: the pool's best is off-industry, a fresh generation's
        people score higher."""
        assert _decide(generation=0.3, waiting=0.15) is False

    def test_a_best_lead_scoring_like_a_generations_children_is_judged(self):
        assert _decide(generation=0.3, waiting=0.3) is True

    def test_with_no_generation_yet_measured_there_is_no_search_rate_to_beat(self):
        assert _decide(generation=0.3, waiting=0.1, leads=0.0) is True

    def test_a_generation_that_brought_nobody_scoring_leaves_the_pool_judged(self):
        assert _decide(generation=0.0, waiting=0.1) is True

    def test_a_dearer_search_keeps_the_run_on_the_pool_longer(self):
        cheap = _decide(generation=0.3, waiting=0.2, search=0.0004)
        dear = _decide(generation=0.3, waiting=0.2, search=1.0, leads=5.0)

        assert (cheap, dear) == (False, True)

    def test_the_decision_is_logged_with_both_costs(self, caplog):
        meter.record(meter.POOL_FIT, 0)
        meter.record(meter.POOL_FIT, 0)
        with caplog.at_level("INFO"):
            _decide(generation=0.3, waiting=0.1)

        assert "judge $" in caplog.text
        assert "0 of 2 fit" in caplog.text
        assert "→" in caplog.text and "search" in caplog.text


class TestArithmetic:
    def test_a_zero_rate_costs_without_bound(self):
        assert judge_or_search.cost_per_fit(0.05, 0.0) == math.inf
        assert judge_or_search.cost_per_fit(0.05, 0.25) == pytest.approx(0.2)

    def test_the_best_waiting_lead_is_the_highest_gp_score(self):
        class Fixed:
            def predict_probs(self, X):
                return np.array([0.1, 0.4, 0.2])

        leads = [Lead(embedding=np.ones(384, dtype=np.float32).tobytes()) for _ in range(3)]
        assert judge_or_search.best_waiting(Fixed(), leads) == pytest.approx(0.4)
        assert judge_or_search.best_waiting(Fixed(), []) == 0.0
