# tests/test_status.py
"""``status`` — the verb an agent asks instead of tailing a log.

Two things carry the weight here. **Nothing may be reported as an empty result**: a
key that was rejected and a run that has simply found nothing yet must never render
the same, which is why the balance reports *why* it is unknown instead of falling
back to zero. And **the next action is arithmetic, not adjectives** — it names counts
and a URL an agent can relay, and it never asks for money before value exists.
"""
import json
from unittest.mock import patch

import pytest

from openoutfind.core import status as status_module
from openoutfind.core.errors import ErrorType
from openoutfind.core.management.commands.status import render
from openoutfind.crm.models import DealState
from openoutfind.enrichment import bettercontact
from tests.factories import DealFactory, LeadFactory


@pytest.fixture
def configured():
    """Onboarding complete, so the tests below are about the pipeline, not setup."""
    with patch("openoutfind.core.onboarding.missing_env_keys", return_value={}):
        yield


@pytest.fixture
def balance():
    """Control the provider balance without a network call."""
    def _set(value=None, error=None, error_type=ErrorType.PROVIDER_UNAVAILABLE):
        if error is not None:
            from openoutfind.enrichment.bettercontact import BetterContactUnavailable
            return patch("openoutfind.enrichment.bettercontact.credit_balance",
                         side_effect=BetterContactUnavailable(error, error_type))
        return patch("openoutfind.enrichment.bettercontact.credit_balance", return_value=value)
    return _set


@pytest.fixture
def has_key():
    with patch("openoutfind.enrichment.bettercontact.is_configured", return_value=True):
        yield


@pytest.fixture
def hub():
    """Control the hub give-to-get balance without a network call."""
    def _set(balance=None, known=False):
        return patch("openoutfind.contacts.service.hub_balance",
                     return_value={"balance": balance, "known": known})
    return _set


# ── the counts ───────────────────────────────────────────────────

@pytest.mark.django_db
def test_counts_the_deliverable_the_way_the_export_writes_it(site_config, configured, has_key, balance):
    """``exportable`` must agree with the CSV's rows, or the number is a different number."""
    DealFactory(lead=LeadFactory(), state=DealState.RESOLVED, reason="fits")
    DealFactory(lead=LeadFactory(), state=DealState.QUALIFIED, reason="fits")
    DealFactory(lead=LeadFactory(), state=DealState.FAILED, reason="no fit")

    with balance(value=40):
        document = status_module.build_status()

    from openoutfind.core.export import lead_records
    assert document["totals"]["exportable"] == sum(1 for _ in lead_records()) == 2
    assert document["totals"]["rejected"] == 1


@pytest.mark.django_db
def test_exportable_separates_the_rows_that_carry_an_address(
    site_config, configured, has_key, balance
):
    """An exportable row is not necessarily a mailable one, and the count says so."""
    DealFactory(lead=LeadFactory(email="ada@acme.com"),
                state=DealState.RESOLVED, reason="fits")
    DealFactory(lead=LeadFactory(email=""),
                state=DealState.QUALIFIED, reason="fits")

    with balance(value=40):
        totals = status_module.build_status()["totals"]

    assert totals["exportable"] == 2
    assert totals["exportable_with_email"] == 1
    assert totals["exportable_without_email"] == 1


# ── the balance, and the difference between unknown and zero ─────

@pytest.mark.django_db
def test_a_rejected_key_is_not_a_balance_of_zero(site_config, configured, has_key, balance):
    with balance(error="BetterContact rejected the API key (401)",
                 error_type=ErrorType.PROVIDER_AUTH):
        document = status_module.build_status()

    assert document["credits"]["balance"] is None
    assert document["credits"]["error"] == ErrorType.PROVIDER_AUTH
    assert any(item["type"] == ErrorType.PROVIDER_AUTH for item in document["blocked"])


@pytest.mark.django_db
def test_an_unreachable_provider_is_its_own_answer(site_config, configured, has_key, balance):
    with balance(error="BetterContact unreachable: timed out"):
        document = status_module.build_status()

    assert document["credits"]["error"] == ErrorType.PROVIDER_UNAVAILABLE


@pytest.mark.django_db
def test_no_key_reports_no_credential(site_config, configured):
    document = status_module.build_status()

    assert document["credits"]["error"] == ErrorType.NO_CREDENTIAL
    assert any(item["type"] == ErrorType.NO_CREDENTIAL for item in document["blocked"])


@pytest.mark.django_db
def test_zero_credits_with_leads_waiting_is_blocked(site_config, configured, has_key, balance):
    DealFactory(lead=LeadFactory(), state=DealState.READY_TO_FIND_EMAIL)

    with balance(value=0):
        document = status_module.build_status()

    blocked = [item for item in document["blocked"]
               if item["type"] == ErrorType.PROVIDER_OUT_OF_CREDITS]
    assert blocked and "1 ranked lead(s) waiting, 0 credits left" in blocked[0]["message"]


# ── the next action ──────────────────────────────────────────────

@pytest.mark.django_db
def test_next_action_is_onboarding_when_setup_is_incomplete(site_config):
    document = status_module.build_status()

    action = document["next_action"]
    assert action["type"] == "finish_onboarding"
    assert "OPENOUTFIND_BETTERCONTACT_API_KEY" in action["variables"]


@pytest.mark.django_db
def test_nothing_is_asked_of_a_run_that_has_qualified_nobody(site_config, configured, has_key, balance):
    """Never before value: an empty pipeline at zero credits is asked for no money — it
    is told to go and find some leads, which costs nothing."""
    with balance(value=0):
        document = status_module.build_status()

    assert document["next_action"]["type"] == "find_leads"


@pytest.mark.django_db
def test_printing_the_rows_is_the_next_action_once_credits_are_not_the_blocker(
    site_config, configured, has_key, balance
):
    """The leads exist; getting them out costs nothing and spends nothing."""
    DealFactory(lead=LeadFactory(), state=DealState.RESOLVED, reason="fits")

    with balance(value=40):
        action = status_module.build_status()["next_action"]

    assert action["type"] == "print_leads"
    assert action["leads"] == 1
    assert action["command"] == "outfind find 0 > leads.csv"


@pytest.mark.django_db
def test_credit_ask_carries_the_count_and_the_attributed_url(site_config, configured, has_key, balance):
    DealFactory(lead=LeadFactory(), state=DealState.READY_TO_FIND_EMAIL)

    with balance(value=0):
        action = status_module.build_status()["next_action"]

    assert action["type"] == "add_credits"
    assert action["leads"] == 1
    # Attribution is won at signup, so every path we show goes through the one URL
    # that applies it — never a hand-written link.
    assert action["url"] == bettercontact.SIGNUP_URL


@pytest.mark.django_db
def test_a_campaign_with_nothing_yet_is_told_to_go_and_find_some(
    site_config, configured, has_key, balance
):
    with balance(value=40):
        action = status_module.build_status()["next_action"]

    assert action["type"] == "find_leads"


# ── the hub balance — a different number than the provider's own ──

@pytest.mark.django_db
def test_hub_balance_is_its_own_key_not_folded_into_credits(site_config, configured, has_key, balance, hub):
    with balance(value=40), hub(balance=3, known=True):
        document = status_module.build_status()

    assert document["hub"] == {"balance": 3, "known": True}
    assert document["credits"]["balance"] == 40  # unaffected — a different service


@pytest.mark.django_db
def test_unknown_hub_balance_is_not_reported_as_zero(site_config, configured, has_key, balance, hub):
    with balance(value=40), hub(balance=None, known=False):
        document = status_module.build_status()

    assert document["hub"] == {"balance": None, "known": False}


# ── rendering ────────────────────────────────────────────────────

@pytest.mark.django_db
def test_json_is_one_object_and_nothing_else(site_config, configured, has_key, balance, hub, capsys):
    from django.core.management import call_command

    with balance(value=40), hub(balance=0, known=True):
        call_command("status", "--json")

    document = json.loads(capsys.readouterr().out)  # would raise on any stray line
    assert set(document) == {
        "onboarding", "totals", "credits", "hub", "blocked", "next_action",
    }


@pytest.mark.django_db
def test_human_summary_reports_the_balance_and_the_next_action(site_config, configured, has_key, balance, hub):
    DealFactory(lead=LeadFactory(), state=DealState.RESOLVED, reason="fits")

    with balance(value=38), hub(balance=0, known=True):
        text = render(status_module.build_status())

    assert "Credits: 38 left." in text
    assert "1 exportable" in text
    assert "Next: 1 qualified lead(s) ready." in text


@pytest.mark.django_db
def test_human_summary_distinguishes_the_hub_balance_from_bettercontact_credits(
    site_config, configured, has_key, balance, hub
):
    """The give-to-get counter must never be shown as, or beside, the ``Credits:``
    line — that is BetterContact's own prepaid balance, a different service."""
    with balance(value=40), hub(balance=2, known=True):
        text = render(status_module.build_status())

    assert "Hub store: 2 free read(s)" in text
    assert "Credits: 40 left." in text


@pytest.mark.django_db
def test_human_summary_names_a_permanent_zero_hub_balance(site_config, configured, has_key, balance, hub):
    with balance(value=40), hub(balance=0, known=True):
        text = render(status_module.build_status())

    assert "Hub store: no balance — contribute an address to earn a read." in text


@pytest.mark.django_db
def test_human_summary_names_an_unknown_hub_balance(site_config, configured, has_key, balance, hub):
    with balance(value=40), hub(balance=None, known=False):
        text = render(status_module.build_status())

    assert "Hub store: no balance on record" in text
