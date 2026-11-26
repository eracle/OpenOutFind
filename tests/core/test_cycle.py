# tests/core/test_cycle.py
"""The cycle: the hierarchy, the ``not_before`` gate, and no free step gated on a paid one.

The test that matters most is ``test_a_stalled_lookup_gates_only_its_own_row`` — it
is the 2026-08-05 incident written down. A deal's timestamp can gate that deal and
nothing else, which is the whole reason the task queue is gone.
"""
from datetime import timedelta
from unittest.mock import patch

import pytest
from django.utils import timezone

from openoutfind.core import cycle
from openoutfind.crm.models import DealState
from tests.factories import DealFactory, LeadFactory


def _deal(site_config, state, **kwargs):
    lead_kwargs = {"email": kwargs.pop("email", None)}
    return DealFactory(
        lead=LeadFactory(**lead_kwargs), state=state, **kwargs)


@pytest.fixture
def steps():
    """Every step stubbed, so a test asserts *which* one the cycle chose."""
    with patch("openoutfind.enrichment.lookup.check_lookup",
               return_value=None) as check, \
            patch("openoutfind.core.pipeline.ready_pool.promote_to_ready",
                  return_value=1) as score, \
            patch("openoutfind.core.ml.qualifier.qualifier_for",
                  return_value=object()), \
            patch("openoutfind.enrichment.lookup.buy_address",
                  return_value=None) as buy, \
            patch("openoutfind.core.pipeline.top_up.top_up",
                  return_value=False) as fill:
        yield {"check": check, "score": score, "buy": buy, "top_up": fill}


def _called(steps):
    return {name for name, mock in steps.items() if mock.called}


# ── The hierarchy ─────────────────────────────────────────────────


@pytest.mark.django_db
class TestPriority:
    def test_an_in_flight_lookup_outranks_everything(self, site_config, steps):
        _deal(site_config, DealState.FINDING_EMAIL, lookup_request_id="req1")
        _deal(site_config, DealState.QUALIFIED)

        assert cycle.run_one_action(site_config) is True
        assert _called(steps) == {"check"}

    def test_a_lookup_with_no_job_handle_is_reclaimed_not_stranded(self, site_config, steps):
        """Measured on a live install: two deals sat at FINDING_EMAIL with an empty
        ``request_id`` for 206 hours — the poll row skipped them and no other row
        claims that state."""
        deal = _deal(site_config, DealState.FINDING_EMAIL, lookup_request_id="")

        assert cycle.run_one_action(site_config) is True
        deal.refresh_from_db()
        assert deal.state == DealState.READY_TO_FIND_EMAIL
        assert "check" not in _called(steps)

    def test_ranking_the_pool_outranks_buying_an_address(self, site_config, steps):
        """Rank first: the gate that decides who is worth a credit runs before the
        credit is spent."""
        _deal(site_config, DealState.QUALIFIED)
        _deal(site_config, DealState.READY_TO_FIND_EMAIL)

        with patch("openoutfind.enrichment.bettercontact.is_configured",
                   return_value=True):
            cycle.run_one_action(site_config)
        assert _called(steps) == {"score"}

    def test_buying_an_address_outranks_finding_more_leads(self, site_config, steps):
        _deal(site_config, DealState.READY_TO_FIND_EMAIL)

        with patch("openoutfind.enrichment.bettercontact.is_configured",
                   return_value=True):
            cycle.run_one_action(site_config, buy_addresses=True)
        assert _called(steps) == {"buy"}

    def test_topping_up_is_the_last_resort(self, site_config, steps):
        """And it is ungated: discovery and qualification wait on no paid step. A gate
        on send headroom once made a mailbox-less install produce nothing, silently."""
        cycle.run_one_action(site_config)
        assert _called(steps) == {"top_up"}

    def test_an_idle_campaign_with_nothing_to_do_says_so(self, site_config, steps):
        assert cycle.run_one_action(site_config) is False


# ── the paid row ──────────────────────────────────────────────────


@pytest.mark.django_db
class TestBuyingIsOffUnlessAskedFor:
    """``buy_addresses=False`` is the default, and it reaches the paid row.

    A run counting *leads* would otherwise reach the buy row for anything an earlier
    run left past the confidence gate — which is what it used to do, before `--emails`
    made the spend opt-in.
    """

    def test_the_paid_row_is_skipped_and_the_free_rows_still_run(self, site_config, steps):
        """Withholding permission to spend must not turn off the work that costs
        nothing, or the default would just mean "do less"."""
        _deal(site_config, DealState.READY_TO_FIND_EMAIL)

        with patch("openoutfind.enrichment.bettercontact.is_configured",
                   return_value=True):
            cycle.run_one_action(site_config, buy_addresses=False)

        assert _called(steps) == {"top_up"}

    def test_the_row_is_reached_without_a_provider_key(self, site_config, steps):
        """**The row is not a gate on the spend.** It used to decline unless a key was
        configured, which also switched off the free sources inside ``buy_address`` —
        an address in hand and the hub cache — exactly when a free hit was worth most.
        The key is now checked on the paid leg alone, so the row still runs.
        """
        _deal(site_config, DealState.READY_TO_FIND_EMAIL, email="known@corp.com")

        with patch("openoutfind.enrichment.bettercontact.is_configured",
                   return_value=False):
            cycle.run_one_action(site_config, buy_addresses=True)

        assert "buy" in _called(steps)

    def test_an_lookup_already_paid_for_is_still_collected(self, site_config, steps):
        """Abandoning an in-flight lookup would waste a credit already committed
        rather than save one, so the poll row is not a paid row."""
        _deal(site_config, DealState.FINDING_EMAIL, lookup_request_id="req1")

        assert cycle.run_one_action(site_config, buy_addresses=False) is True
        assert _called(steps) == {"check"}


# ── not_before ────────────────────────────────────────────────────


@pytest.mark.django_db
class TestNotBefore:
    def test_a_deal_told_to_wait_is_not_served(self, site_config, steps):
        _deal(site_config, DealState.FINDING_EMAIL, lookup_request_id="req1",
              not_before=timezone.now() + timedelta(hours=1))

        cycle.run_one_action(site_config)
        assert "check" not in _called(steps)

    def test_a_deal_whose_wait_has_elapsed_is_served(self, site_config, steps):
        _deal(site_config, DealState.FINDING_EMAIL, lookup_request_id="req1",
              not_before=timezone.now() - timedelta(seconds=1))

        cycle.run_one_action(site_config)
        assert "check" in _called(steps)

    def test_a_stalled_lookup_gates_only_its_own_row(self, site_config, steps):
        """**The 2026-08-05 incident.** Two lookups had backed off 45 hours; they
        were the only rows in the queue, so the daemon slept 34 hours while 55 ready
        deals waited. A timestamp now gates its own row and nothing else — the work
        below it in the hierarchy runs regardless."""
        for i in range(2):
            _deal(site_config, DealState.FINDING_EMAIL, lookup_request_id=f"req{i}",
                  not_before=timezone.now() + timedelta(hours=45))
        _deal(site_config, DealState.QUALIFIED)

        assert cycle.run_one_action(site_config) is True
        assert _called(steps) == {"score"}


# ── Failure handling ──────────────────────────────────────────────


@pytest.mark.django_db
class TestFailures:
    def test_an_ordinary_failure_leaves_the_row_untouched(self, site_config):
        """The cycle's try/except is a bug backstop: log, skip, keep going."""
        deal = _deal(site_config, DealState.FINDING_EMAIL, lookup_request_id="req1")

        with patch("openoutfind.enrichment.lookup.check_lookup",
                   side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                cycle.run_one_action(site_config)

        deal.refresh_from_db()
        assert deal.state == DealState.FINDING_EMAIL
        assert deal.not_before is None


# ── Scoring is not repeated for nothing ───────────────────────────


@pytest.mark.django_db
class TestScoringIsSkippedWhenNothingMoved:
    """Fitting the GP dominates the cost of using it (~1.1s at 300 labels, against
    a 5s cycle), and scoring the same pool with the same labels cannot promote
    anybody — so an unchanged site_config must not rebuild the model at all."""

    def _score_twice(self, site_config, between=None):
        _deal(site_config, DealState.QUALIFIED)
        with patch("openoutfind.core.ml.qualifier.qualifier_for",
                   return_value=object()) as build, \
                patch("openoutfind.core.pipeline.ready_pool.promote_to_ready",
                      return_value=0):
            cycle._score_qualified(site_config, cycle._one_model_per_action(site_config))
            if between:
                between(site_config)
            cycle._score_qualified(site_config, cycle._one_model_per_action(site_config))
        return build

    def test_an_unchanged_pool_is_not_rescored(self, site_config):
        assert self._score_twice(site_config).call_count == 1

    def test_a_new_lead_reopens_scoring(self, site_config):
        build = self._score_twice(
            site_config, between=lambda c: _deal(c, DealState.QUALIFIED))
        assert build.call_count == 2

    def test_a_new_verdict_reopens_scoring(self, site_config):
        """A label the GP has not seen changes what it would say."""
        build = self._score_twice(
            site_config, between=lambda c: _deal(c, DealState.FAILED))
        assert build.call_count == 2

    def test_an_empty_pool_never_builds_the_model(self, site_config):
        with patch("openoutfind.core.ml.qualifier.qualifier_for") as build:
            assert cycle._score_qualified(
                site_config, cycle._one_model_per_action(site_config)) is False
        build.assert_not_called()

    def test_one_action_fits_the_model_at_most_once(self, site_config):
        """Scoring and the top-up both need the GP, and the fit is what costs — so an
        action that falls through the promote gate into the top-up must not build a
        second model over the very same labels."""
        _deal(site_config, DealState.QUALIFIED)

        with patch("openoutfind.core.ml.qualifier.qualifier_for",
                   return_value=object()) as build, \
                patch("openoutfind.core.pipeline.ready_pool.promote_to_ready",
                      return_value=0), \
                patch("openoutfind.core.pipeline.top_up.top_up",
                      return_value=True) as fill:
            assert cycle.run_one_action(site_config) is True

        assert build.call_count == 1
        assert fill.call_args.args[1] is build.return_value


# ── An action is one action, and idleness is an answer ────────────


@pytest.mark.django_db
class TestOneActionAtATime:
    """What the bounded job depends on: `run_one_action` does at most one thing and
    reports honestly whether it did. `False` is the job's terminal condition, so a row
    that claims to have acted when it merely retried would be an endless run."""

    def test_nothing_to_do_says_what_it_is_waiting_on(self, site_config, steps, caplog):
        """*Nothing may be reported as an empty result*: a job that stops short has to
        be able to say whether the index is drained or an address is on order.

        The line itself sits at DEBUG, because the operator meets the same summary once,
        as the job's `goal_unreached` detail — see `tests/core/test_job.py`. Printing it here
        too would read as two different findings.
        """
        _deal(site_config, DealState.FINDING_EMAIL, lookup_request_id="req1",
              not_before=timezone.now() + timedelta(hours=1))

        with caplog.at_level("DEBUG"):
            assert cycle.run_one_action(site_config) is False

        idle = [r.getMessage() for r in caplog.records if "nothing to do" in r.getMessage()]
        assert len(idle) == 1 and "address on order" in idle[0]

    def test_a_keyless_run_says_discovery_stopped_too_not_just_the_lookup(self, site_config):
        """One key does two jobs, so losing it has two consequences and both get named.

        Naming only the address half sent the operator after the wrong thing: without a
        key the walk finds nobody at all, which is the larger of the two.
        """
        with patch("openoutfind.enrichment.bettercontact.is_configured", return_value=False):
            summary = cycle.pipeline_summary(site_config)

        assert "no new discovery" in summary and "free address sources only" in summary

    def test_ranked_leads_and_no_emails_flag_names_that_gate(self, site_config):
        """The gate most likely to be holding on a bare `find` is the one it never
        mentioned: spending is opt-in, and an operator expecting addresses has to be
        told which flag turns it on."""
        _deal(site_config, DealState.READY_TO_FIND_EMAIL)

        with patch("openoutfind.enrichment.bettercontact.is_configured", return_value=True):
            summary = cycle.pipeline_summary(site_config, buy_addresses=False)

        assert "--emails" in summary
