# tests/test_provider.py
"""The finder seam: which vendor an install resolves with, and how a lookup lands.

Two properties matter here. **Selection** — a key is all it takes. And
**interchangeability** — a sync provider and an async one must leave the deal in the
same states with the same contribution, differing only in whether it passes through
FINDING_EMAIL on the way. Upstream ships only an async finder, so the sync one is a
stub standing in for whatever a fork adds.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openoutfind.crm.models import DealState
from openoutfind.enrichment import bettercontact, provider
from openoutfind.enrichment.lookup import buy_address
from openoutfind.enrichment.provider import Lookup, PollOutcome
from tests.factories import DealFactory, LeadFactory


@pytest.fixture
def config(db, configure):
    return configure


def _key(configure, bc=""):
    configure(bettercontact_api_key=bc)


def _ready_to_find(site_config):
    return DealFactory(
        lead=LeadFactory(email=None),
        state=DealState.READY_TO_FIND_EMAIL,
    )


def _sync_finder(outcome: PollOutcome):
    """A finder that answers in the same call, the way a match-style API does."""
    return SimpleNamespace(
        NAME="syncfinder",
        is_configured=lambda: True,
        start=MagicMock(return_value=Lookup(outcome=outcome)),
    )


# ── selection ─────────────────────────────────────────────────────


class TestActive:

    def test_no_key_means_no_finder(self, config):
        _key(config)
        assert provider.active() is None
        assert provider.configured() == []

    def test_the_key_selects_bettercontact_with_no_setting(self, config):
        _key(config, bc="secret")
        assert provider.active() is bettercontact
        assert provider.configured() == [bettercontact]

    def test_a_handle_is_polled_by_the_vendor_that_minted_it(self):
        assert provider.by_name("bettercontact") is bettercontact
        assert provider.by_name("") is None


# ── interchangeability ────────────────────────────────────────────


class TestEitherTransportResolves:

    def test_a_sync_finder_resolves_without_entering_finding_email(self, config, site_config):
        deal = _ready_to_find(site_config)
        finder = _sync_finder(PollOutcome(running=False, email="alice@acme.com"))

        with patch.object(provider, "active", return_value=finder), \
             patch("openoutfind.contacts.service.contribute"):
            assert buy_address(deal) == DealState.RESOLVED

        deal.lead.refresh_from_db()
        assert deal.lead.email == "alice@acme.com"
        assert deal.lookup_request_id == ""

    def test_an_async_finder_parks_on_its_handle(self, config, site_config):
        _key(config, bc="secret")
        deal = _ready_to_find(site_config)

        with patch.object(bettercontact, "start", return_value=Lookup(request_id="req1")):
            assert buy_address(deal) == DealState.FINDING_EMAIL

        assert deal.lookup_request_id == "req1"
        assert deal.lookup_provider == "bettercontact"

    def test_a_sync_miss_is_the_same_terminal_as_an_async_one(self, config, site_config):
        deal = _ready_to_find(site_config)
        finder = _sync_finder(PollOutcome(running=False))

        with patch.object(provider, "active", return_value=finder):
            assert buy_address(deal) == DealState.NO_EMAIL_FOUND

    def test_the_contribution_is_stamped_with_the_finder_that_paid(self, config, site_config):
        deal = _ready_to_find(site_config)
        finder = _sync_finder(PollOutcome(running=False, email="alice@acme.com"))

        with patch.object(provider, "active", return_value=finder), \
             patch("openoutfind.contacts.service.contribute") as contribute:
            buy_address(deal)

        assert contribute.call_args.args[2] == "syncfinder"

    def test_the_free_sources_still_come_first(self, config, site_config):
        """The hub cache must not spend a credit, whichever vendor is configured."""
        _key(config, bc="secret")
        deal = _ready_to_find(site_config)

        with patch("openoutfind.contacts.service.resolve", return_value="free@acme.com"), \
             patch.object(bettercontact, "start") as start:
            assert buy_address(deal) == DealState.RESOLVED

        start.assert_not_called()
