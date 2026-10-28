# tests/test_judge_or_search.py
"""Judge the pool or search for more, on costs measured from the run.

Covers the running means (``core/meter.py``), where the run takes its measurements
(a page fetched, a verdict given, a new page resetting the pool), the walk's estimate of
its next page, and the rule itself (``core/pipeline/judge_or_search.py``).
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
from openoutfind.core.models import Keyword, QueryNode
from openoutfind.core.pipeline import judge_or_search, select
from openoutfind.core.pipeline.discover import discover
from openoutfind.core.pipeline.qualify import run_qualification
from openoutfind.crm.models import Deal, DealState, Lead, Outcome, PendingQualification
from openoutfind.discovery import Page


# ── The running means ──

@pytest.mark.django_db
class TestMeter:
    def test_the_mean_is_the_sum_over_the_count(self):
        for value in (10, 20, 60):
            meter.record(meter.SEARCH_SECONDS, value)

        assert meter.totals(meter.SEARCH_SECONDS) == (90.0, 3)
        assert meter.mean(meter.SEARCH_SECONDS, default=70) == pytest.approx(30)

    def test_the_default_stands_in_until_the_first_event(self):
        assert meter.totals(meter.SEARCH_SECONDS) == (0.0, 0)
        assert meter.mean(meter.SEARCH_SECONDS, default=70) == 70

    def test_a_reset_starts_the_mean_over(self):
        meter.record(meter.POOL_FIT, 1)
        meter.reset(meter.POOL_FIT)
        meter.record(meter.POOL_FIT, 0)

        assert meter.totals(meter.POOL_FIT) == (0.0, 1)

    def test_resetting_a_quantity_never_measured_is_harmless(self):
        meter.reset(meter.POOL_FIT)
        assert meter.totals(meter.POOL_FIT) == (0.0, 0)


# ── Where the run measures ──

@pytest.fixture
def _finder(db, configure):
    import os

    from openoutfind.core.config import SiteConfig, variable_for

    configure(bettercontact_api_key="k")
    os.environ[variable_for("product_docs")] = "p"
    os.environ[variable_for("campaign_target")] = "t"
    return SiteConfig.load()


def _node(pairs, **kw):
    node = QueryNode.objects.create(token_key=select.token_key(pairs), **kw)
    node.keywords.set(Keyword.rows_for(pairs))
    return node


def _row(url):
    return {"contact_linkedin_profile_url": url, "contact_job_title": "founder",
            "contact_headline": "cannabis extraction", "contact_location_country": "united states"}


class TestSearchMeasurements:
    def test_a_page_is_timed(self, _finder):
        _node([("lead_job_title", "founder")])
        with patch("openoutfind.discovery.search",return_value=Page([_row("https://x/1")], 10)):
            assert discover(_finder) is True

        assert meter.totals(meter.SEARCH_SECONDS)[1] == 1

    def test_an_outage_is_not_a_search(self, _finder):
        from openoutfind.core.errors import ErrorType
        from openoutfind.enrichment.bettercontact import BetterContactUnavailable

        _node([("lead_job_title", "founder")])
        outage = BetterContactUnavailable("down", error_type=ErrorType.PROVIDER_UNAVAILABLE)
        with patch("openoutfind.discovery.search",side_effect=outage):
            assert discover(_finder) is False

        assert meter.totals(meter.SEARCH_SECONDS) == (0.0, 0)

    def test_new_leads_start_the_pools_fit_rate_over(self, _finder):
        meter.record(meter.POOL_FIT, 0)
        _node([("lead_job_title", "founder")])
        with patch("openoutfind.discovery.search",return_value=Page([_row("https://x/1")], 10)):
            discover(_finder)

        assert meter.totals(meter.POOL_FIT) == (0.0, 0)

    def test_a_page_of_known_profiles_leaves_the_pool_as_it_was(self, _finder):
        Lead.objects.create(profile_url="https://x/1", profile_text="known")
        meter.record(meter.POOL_FIT, 0)
        _node([("lead_job_title", "founder")])
        with patch("openoutfind.discovery.search",return_value=Page([_row("https://x/1")], 10)):
            discover(_finder)

        assert meter.totals(meter.POOL_FIT) == (0.0, 1)


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


# ── The walk's next page ──

def _found(node, qualified):
    lead = Lead.objects.create(profile_url=f"https://x/l{Lead.objects.count()}",
                               profile_text="someone", discovered_by=node)
    Deal.objects.create(lead=lead, state=DealState.QUALIFIED if qualified else DealState.FAILED,
                        outcome="" if qualified else Outcome.WRONG_FIT)


def _scores():
    return select.HarvestScores.load(None, select.LabelStore.load().base_rate)


@pytest.mark.django_db
class TestBestEstimate:
    def test_nothing_to_fire_is_worth_nothing(self):
        _node([("lead_job_title", "a")], state=QueryNode.State.DEAD)
        assert select.best_estimate(_scores()) == 0.0

    def test_it_is_the_best_node_on_the_frontier(self):
        good = _node([("lead_job_title", "good")], state=QueryNode.State.FIRED, next_offset=100)
        bad = _node([("lead_job_title", "bad")], state=QueryNode.State.FIRED, next_offset=100)
        for _ in range(5):
            _found(good, qualified=True)
            _found(bad, qualified=False)
        scores = _scores()

        assert select.best_estimate(scores) == pytest.approx(select.estimate(good, scores))


# ── The rule ──

def _decide(*, page_rate, pool, verdict=0.05, search=0.0004, leads=50.0):
    """``judging_pays`` with the costs and the page estimate set, after ``pool`` verdicts
    (1 a fit, 0 a no-fit) since the last new page."""
    for label in pool:
        meter.record(meter.POOL_FIT, label)
    with (
        patch.object(select, "best_estimate", return_value=page_rate),
        patch("openoutfind.core.ml.calibration.verdict_cost", return_value=verdict),
        patch("openoutfind.core.ml.calibration.search_cost", return_value=search),
        patch("openoutfind.core.ml.calibration.leads_per_search", return_value=leads),
    ):
        return judge_or_search.judging_pays(None)


@pytest.mark.django_db
class TestTheRule:
    def test_a_fresh_pool_is_judged_before_its_first_verdict(self):
        assert _decide(page_rate=0.3, pool=[]) is True

    def test_a_pool_that_keeps_missing_is_left_for_a_search(self):
        """The batch-5 shape: no fit in a run of verdicts, a page the walk rates at 0.3."""
        assert _decide(page_rate=0.3, pool=[0, 0, 0]) is False

    def test_one_miss_does_not_send_the_run_searching(self):
        """The smoothing toward the page's rate: one no-fit after a cheap-enough search
        still leaves judging ahead when searching is dear."""
        assert _decide(page_rate=0.3, pool=[0], search=5.0, leads=10.0) is True

    def test_a_pool_fitting_as_often_as_a_page_keeps_being_judged(self):
        assert _decide(page_rate=0.3, pool=[1, 0, 0, 1, 0]) is True

    def test_with_no_search_yet_there_is_no_search_rate_to_beat(self):
        assert _decide(page_rate=0.3, pool=[0, 0, 0], leads=0.0) is True

    def test_with_nothing_left_to_fire_the_pool_is_judged(self):
        assert _decide(page_rate=0.0, pool=[0, 0, 0]) is True

    def test_a_dearer_search_keeps_the_run_on_the_pool_longer(self):
        cheap = _decide(page_rate=0.3, pool=[0, 1, 0, 0], search=0.0004)
        meter.reset(meter.POOL_FIT)
        dear = _decide(page_rate=0.3, pool=[0, 1, 0, 0], search=1.0, leads=5.0)

        assert (cheap, dear) == (False, True)

    def test_the_decision_is_logged_with_both_costs(self, caplog):
        with caplog.at_level("INFO"):
            _decide(page_rate=0.3, pool=[0, 0])

        assert "judge $" in caplog.text
        assert "0 of 2 fit" in caplog.text
        assert "→" in caplog.text and "search" in caplog.text


class TestArithmetic:
    def test_the_pool_rate_leans_on_the_page_with_the_walks_weight(self, db):
        meter.record(meter.POOL_FIT, 0)
        assert judge_or_search.pool_fit_rate(0.3) == pytest.approx(0.6 / 3)

    def test_a_zero_rate_costs_without_bound(self):
        assert judge_or_search.cost_per_fit(0.05, 0.0) == math.inf
        assert judge_or_search.cost_per_fit(0.05, 0.25) == pytest.approx(0.2)
