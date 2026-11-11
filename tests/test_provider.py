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
from openoutfind.enrichment import bettercontact, enrow, fullenrich, leadmagic, provider
from openoutfind.enrichment.lookup import buy_address
from openoutfind.enrichment.provider import Lookup, PollOutcome, ProviderUnavailable
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


def _finder(name: str, outcome=None, cost=None, fails: bool = False):
    """A configured finder with an optional price tag, able to refuse the call.

    Price is left off by default so a test can prove the unpriced case sorts last
    rather than winning on a fluke. Refusal is a side effect, not a return value,
    because that is how a real adapter reports it cannot run.
    """
    module = SimpleNamespace(
        NAME=name,
        is_configured=lambda: True,
        start=MagicMock(return_value=Lookup(outcome=outcome)),
    )
    if cost is not None:
        module.UNIT_COST_USD = cost
    if fails:
        module.start.side_effect = ProviderUnavailable(f"{name} is down")
    return module


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


# the chain steps over a refusal, and only a refusal


class TestTheChainStepsOverARefusal:
    def test_a_provider_that_cannot_run_hands_the_lookup_to_the_next(self, config, site_config):
        """Availability outranks price: a vendor that cannot run does not get paid."""
        deal = _ready_to_find(site_config)
        down = _finder("down", fails=True)
        up = _finder("up", outcome=PollOutcome(running=False, email="alice@acme.com"))

        with patch.object(provider, "_providers", return_value=(down, up)), \
             patch("openoutfind.contacts.service.contribute"):
            assert buy_address(deal) == DealState.RESOLVED

        assert deal.lead.email == "alice@acme.com"
        up.start.assert_called_once()

    def test_a_terminal_miss_does_not_pay_a_second_vendor(self, config, site_config):
        """A miss is an answer. Paying a second vendor to hear it again is waste."""
        deal = _ready_to_find(site_config)
        first = _finder("first", outcome=PollOutcome(running=False))
        second = _finder("second", outcome=PollOutcome(running=False, email="late@acme.com"))

        with patch.object(provider, "_providers", return_value=(first, second)):
            assert buy_address(deal) == DealState.NO_EMAIL_FOUND

        second.start.assert_not_called()


# the chain is price ordered, and unpriced sorts last


class TestTheChainIsPriceOrdered:
    def test_the_cheapest_configured_provider_leads(self):
        cheap = _finder("cheap", cost=0.05)
        dear = _finder("dear", cost=0.09)

        with patch.object(provider, "_providers", return_value=(dear, cheap)):
            assert [module.NAME for module in provider.chain()] == ["cheap", "dear"]
            assert provider.active() is cheap

    def test_a_provider_with_no_published_price_sorts_last(self):
        priced = _finder("priced", cost=0.2)
        unpriced = _finder("unpriced")

        with patch.object(provider, "_providers", return_value=(unpriced, priced)):
            assert [module.NAME for module in provider.chain()] == ["priced", "unpriced"]

    def test_an_unconfigured_provider_is_not_in_the_chain(self):
        keyed = _finder("keyed", cost=0.1)
        blank = SimpleNamespace(
            NAME="blank",
            is_configured=lambda: False,
            start=MagicMock(),
        )

        with patch.object(provider, "_providers", return_value=(keyed, blank)):
            assert [module.NAME for module in provider.chain()] == ["keyed"]
            assert provider.active() is keyed


# the shipped roster, with no stand-ins


class TestTheRealRoster:
    """The real modules, not doubles: the order the pricing actually justifies.

    A mocked chain proves the sort. This proves the roster, so a registered finder
    whose price or key wiring drifted fails here instead of quietly overspending.
    """

    def test_the_shipped_finders_run_cheapest_first(self, config):
        config(
            bettercontact_api_key="bc",
            enrow_api_key="enrow",
            fullenrich_api_key="fr",
            leadmagic_api_key="lm",
        )

        assert [module.NAME for module in provider.chain()] == [
            "enrow",
            "leadmagic",
            "fullenrich",
            "bettercontact",
        ]

    def test_only_the_keyed_finders_are_in_the_chain(self, config):
        config(enrow_api_key="enrow")

        assert [module.NAME for module in provider.chain()] == ["enrow"]
        assert provider.active() is enrow


# a stored handle names the vendor that minted it


class TestByName:
    def test_a_fullenrich_handle_is_polled_by_fullenrich(self):
        assert provider.by_name("fullenrich") is fullenrich

    def test_an_enrow_handle_is_polled_by_enrow(self):
        assert provider.by_name("enrow") is enrow

    def test_a_leadmagic_handle_is_polled_by_leadmagic(self):
        assert provider.by_name("leadmagic") is leadmagic

    def test_an_unknown_handle_yields_nothing(self):
        assert provider.by_name("no-such-finder") is None
