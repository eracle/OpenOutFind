# tests/contacts/test_service.py
"""Contacts store client — mock at the HTTP boundary (``service.requests``).

Two best-effort calls: ``resolve`` (ask the hub before paying BetterContact) and
``contribute`` (give back what we find, non-EU only, registering on first use).
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from openoutfind.contacts import service
from tests.factories import LeadFactory


def _resp(status_code=200, body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body or {}
    resp.raise_for_status.side_effect = (
        None if status_code < 400 else requests.HTTPError(str(status_code))
    )
    return resp


def _config(token="tok", url="", country_code="us"):
    """This run's contacts configuration — the environment, which is all there is.

    ``country_code`` is the operator's jurisdiction (the give-back gate); the default is
    non-EEA so ``contribute`` proceeds, and the EEA test overrides it.
    """
    from openoutfind.core.config import SiteConfig, variable_for

    for field, value in (("contacts_api_token", token), ("contacts_api_url", url),
                         ("country_code", country_code)):
        os.environ[variable_for(field)] = value
    return SiteConfig.load()


@pytest.fixture(autouse=True)
def _no_token_held_over(configure):
    """A token minted in one test must not identify the next one."""
    service._minted_token = None
    _config()
    yield
    service._minted_token = None


@pytest.fixture(autouse=True)
def _operator(db):
    """The register path stamps the operator's email; give it one to find."""
    from tests.factories import UserFactory

    return UserFactory(username="me", email="me@x.com")


# ── resolve ──────────────────────────────────────────────────────────


@pytest.mark.django_db
class TestResolve:
    def test_no_token_returns_none_without_a_call(self):
        _config(token="")
        lead = LeadFactory(profile_url="jane-doe")
        with patch.object(service.requests, "get") as get:
            assert service.resolve(lead) is None
        get.assert_not_called()

    def test_hit_returns_email(self):
        _config()
        lead = LeadFactory(profile_url="jane-doe")
        body = {"public_identifier": "jane-doe", "emails": ["jane@acme.com"]}
        with patch.object(service.requests, "get", return_value=_resp(200, body)):
            assert service.resolve(lead) == "jane@acme.com"

    def test_hit_with_multiple_emails_takes_first(self):
        _config()
        lead = LeadFactory(profile_url="jane-doe")
        body = {"public_identifier": "jane-doe", "emails": ["jane@acme.com", "j@personal.com"]}
        with patch.object(service.requests, "get", return_value=_resp(200, body)):
            assert service.resolve(lead) == "jane@acme.com"

    def test_hit_with_empty_emails_returns_none(self):
        _config()
        lead = LeadFactory(profile_url="jane-doe")
        with patch.object(service.requests, "get", return_value=_resp(200, {"emails": []})):
            assert service.resolve(lead) is None

    def test_miss_returns_none(self):
        _config()
        lead = LeadFactory()
        with patch.object(service.requests, "get", return_value=_resp(404, {})):
            assert service.resolve(lead) is None

    def test_outage_returns_none(self):
        _config()
        lead = LeadFactory()
        with patch.object(
            service.requests, "get", side_effect=requests.ConnectionError("boom"),
        ):
            assert service.resolve(lead) is None

    def test_zero_balance_miss_logs_as_no_balance_not_as_a_store_miss(self, caplog):
        """A permanent zero must not read like a miss — the client can tell the two
        zeros apart on the wire (``credits``) and must say which it is."""
        _config()
        lead = LeadFactory(profile_url="jane-doe")
        with patch.object(service.requests, "get", return_value=_resp(404, {"credits": 0})):
            with caplog.at_level("INFO"):
                assert service.resolve(lead) is None
        assert any("no balance" in r.message for r in caplog.records)
        assert not any("no stored email" in r.message for r in caplog.records)

    def test_positive_balance_miss_logs_as_a_store_miss(self, caplog):
        _config()
        lead = LeadFactory(profile_url="jane-doe")
        with patch.object(service.requests, "get", return_value=_resp(404, {"credits": 3})):
            with caplog.at_level("INFO"):
                assert service.resolve(lead) is None
        assert any("no stored email" in r.message for r in caplog.records)
        assert not any("no balance" in r.message for r in caplog.records)


# ── contribute ───────────────────────────────────────────────────────


@pytest.mark.django_db
class TestContribute:
    def test_empty_emails_is_a_noop(self):
        _config()
        lead = LeadFactory(country_code="in")
        with patch.object(service.requests, "post") as post:
            service.contribute(lead, [], service.ORIGIN_BETTERCONTACT)
        post.assert_not_called()

    def test_eea_lead_is_skipped_client_side(self):
        _config()
        lead = LeadFactory(country_code="de")
        with patch.object(service.requests, "post") as post:
            service.contribute(lead, ["jane@acme.com"], service.ORIGIN_BETTERCONTACT)
        post.assert_not_called()

    def test_unknown_country_is_skipped(self):
        _config()
        lead = LeadFactory(country_code="")
        with patch.object(service.requests, "post") as post:
            service.contribute(lead, ["jane@acme.com"], service.ORIGIN_BETTERCONTACT)
        post.assert_not_called()

    def test_with_token_posts_the_record(self):
        _config(token="tok")
        lead = LeadFactory(profile_url="jane-doe", country_code="in")
        with patch.object(
            service.requests, "post", return_value=_resp(200, {"accepted": 1, "credits": 7}),
        ) as post:
            # the empty string is filtered out
            service.contribute(lead, ["jane@acme.com", ""], service.ORIGIN_PROFILE_INFO)
        url, kwargs = post.call_args.args[0], post.call_args.kwargs
        assert url.endswith("/api/v2/contribute/")
        assert kwargs["headers"]["Authorization"] == "Bearer tok"
        # The build fields ride along on every record (see TestBuildReporting);
        # this asserts the payload proper.
        record = {k: v for k, v in kwargs["json"].items() if not k.startswith("client_")}
        assert record == {
            "public_identifier": "jane-doe",
            "country_code": "in",
            "emails": ["jane@acme.com"],
            "origin": "profile_info",
        }

    def test_a_run_with_no_token_registers_first_and_then_contributes(self):
        """Identity is minted record-less, and the record follows under it.

        The fold — a register carrying the contribution — is the compatibility path for a
        hub that still demands one, and it only runs when the plain register failed.
        """
        _config(token="")
        lead = LeadFactory(profile_url="jane-doe", country_code="br")
        with patch.object(
            service.requests, "post", return_value=_resp(200, {"token": "NEW", "credits": 1}),
        ) as post:
            service.contribute(lead, ["jane@acme.com"], service.ORIGIN_BETTERCONTACT)

        registered, contributed = post.call_args_list
        assert registered.args[0].endswith("/api/v2/register/")
        assert registered.kwargs["json"]["operator_email"] == "me@x.com"
        assert "public_identifier" not in registered.kwargs["json"]

        assert contributed.args[0].endswith("/api/v2/contribute/")
        assert contributed.kwargs["json"]["origin"] == "bettercontact"
        assert contributed.kwargs["headers"]["Authorization"] == "Bearer NEW"
        assert service._minted_token == "NEW"

    def test_outage_is_swallowed_and_no_token_stored(self):
        _config(token="")
        lead = LeadFactory(country_code="in")
        with patch.object(
            service.requests, "post", side_effect=requests.ConnectionError("boom"),
        ):
            # must not raise
            service.contribute(lead, ["jane@acme.com"], service.ORIGIN_BETTERCONTACT)
        assert not service._minted_token

    def test_eea_operator_contributes_nothing(self):
        """An operator inside the EEA/UK/CH does not give back (jurisdiction gate)."""
        _config(token="tok", country_code="de")
        lead = LeadFactory(country_code="in")
        with patch.object(service.requests, "post") as post:
            service.contribute(lead, ["jane@acme.com"], service.ORIGIN_BETTERCONTACT)
        post.assert_not_called()

    def test_cached_embedding_rides_along(self):
        _config(token="tok")
        lead = LeadFactory(profile_url="jane-doe", country_code="in")
        lead.embedding_array = np.arange(384, dtype=np.float32)
        with patch.object(
            service.requests, "post", return_value=_resp(200, {"accepted": 1, "credits": 7}),
        ) as post:
            service.contribute(lead, ["jane@acme.com"], service.ORIGIN_BETTERCONTACT)
        assert post.call_args.kwargs["json"]["embedding"] == list(range(384))

    def test_uncached_embedding_is_omitted(self):
        _config(token="tok")
        lead = LeadFactory(country_code="in")  # no embedding cached
        with patch.object(
            service.requests, "post", return_value=_resp(200, {"accepted": 1, "credits": 7}),
        ) as post:
            service.contribute(lead, ["jane@acme.com"], service.ORIGIN_BETTERCONTACT)
        assert "embedding" not in post.call_args.kwargs["json"]


# ── identity, minted at onboarding ───────────────────────────────────


@pytest.mark.django_db
class TestRegisterOperator:
    """Identity is not entitlement.

    The token says *which install this is*; the balance says what it may read. They
    used to be one act — a token existed only as a side effect of a first
    contribution — so an install that cannot contribute had no identity at all.
    """

    def test_it_mints_from_the_email_alone_with_no_record(self):
        _config(token="")
        with patch.object(
            service.requests, "post", return_value=_resp(200, {"token": "NEW"}),
        ) as post:
            assert service.register_operator() is True

        url, kwargs = post.call_args.args[0], post.call_args.kwargs
        assert url.endswith("/api/v2/register/")
        assert kwargs["json"]["operator_email"] == "me@x.com"
        # No record rides along — that is the whole point of the standalone mint.
        assert "public_identifier" not in kwargs["json"]
        assert "emails" not in kwargs["json"]
        assert service._minted_token == "NEW"

    def test_it_names_the_build_it_is_running(self):
        """An install that never contributes reports its version here or nowhere."""
        _config(token="")
        with patch.object(
            service.requests, "post", return_value=_resp(200, {"token": "NEW"}),
        ) as post, patch.object(service.version, "commit_sha", return_value="abc123"), \
                patch.object(service.version, "is_dirty", return_value=False):
            assert service.register_operator() is True

        assert post.call_args.kwargs["json"]["client_sha"] == "abc123"
        assert post.call_args.kwargs["json"]["client_dirty"] is False

    def test_an_eea_operator_still_gets_a_token(self):
        """The jurisdiction rule governs *contributing records*, a different act.

        Minting was gated on it only because the two were the same call. An install
        that can never contribute must still be addressable, or it is invisible to
        the hub for its whole life.
        """
        _config(token="", country_code="de")
        with patch.object(
            service.requests, "post", return_value=_resp(200, {"token": "NEW"}),
        ):
            assert service.register_operator() is True

        assert service._minted_token == "NEW"

    def test_an_install_that_already_has_one_asks_for_nothing(self):
        _config(token="tok")
        with patch.object(service.requests, "post") as post:
            assert service.register_operator() is True

        post.assert_not_called()

    def test_a_hub_outage_is_a_no_op_the_next_run_retries(self):
        _config(token="")
        with patch.object(
            service.requests, "post", side_effect=requests.ConnectionError("boom"),
        ):
            assert service.register_operator() is False  # must not raise

        assert not service._minted_token

    def test_a_hub_that_still_demands_a_record_leaves_the_token_unset(self):
        """The compatibility case: a hub predating the record-less register answers
        400, and the first contribution mints the old way instead."""
        _config(token="")
        with patch.object(service.requests, "post", return_value=_resp(400)):
            assert service.register_operator() is False

        assert not service._minted_token


# ── which build sent it ──────────────────────────────────────────────


@pytest.mark.django_db
class TestBuildReporting:
    """The client names its build; the hub decides what that name means."""

    def test_contribute_sends_the_commit_sha_and_dirty_flag(self):
        _config()
        lead = LeadFactory(profile_url="jane-doe", country_code="us")
        with patch.object(service.version, "commit_sha", return_value="a" * 40), \
             patch.object(service.version, "is_dirty", return_value=True), \
             patch.object(service.requests, "post", return_value=_resp(body={"credits": 1})) as post:
            service.contribute(lead, ["jane@acme.com"], service.ORIGIN_BETTERCONTACT)
        body = post.call_args.kwargs["json"]
        assert body["client_sha"] == "a" * 40
        assert body["client_dirty"] is True

    def test_undeterminable_dirtiness_is_omitted_not_sent_as_false(self):
        _config()
        lead = LeadFactory(profile_url="jane-doe", country_code="us")
        with patch.object(service.version, "commit_sha", return_value="a" * 40), \
             patch.object(service.version, "is_dirty", return_value=None), \
             patch.object(service.requests, "post", return_value=_resp(body={"credits": 1})) as post:
            service.contribute(lead, ["jane@acme.com"], service.ORIGIN_BETTERCONTACT)
        assert "client_dirty" not in post.call_args.kwargs["json"]

    def test_every_call_carries_the_version_user_agent(self):
        """Including resolve, which never reaches a stored row."""
        _config()
        lead = LeadFactory(profile_url="jane-doe")
        with patch.object(service.version, "version_string", return_value="2026.08.07+gabc1234"), \
             patch.object(service.requests, "get", return_value=_resp(body={"emails": []})) as get:
            service.resolve(lead)
        assert get.call_args.kwargs["headers"]["User-Agent"] == "OpenOutFind/2026.08.07+gabc1234"

    def test_register_carries_the_build_of_the_first_contribution(self):
        _config(token="")
        lead = LeadFactory(profile_url="jane-doe", country_code="us")
        with patch.object(service.version, "commit_sha", return_value="b" * 40), \
             patch.object(service.requests, "post",
                          return_value=_resp(body={"token": "t", "credits": 1})) as post:
            service.contribute(lead, ["jane@acme.com"], service.ORIGIN_BETTERCONTACT)
        assert post.call_args.kwargs["json"]["client_sha"] == "b" * 40


# ── the give-to-get balance, for status ───────────────────────────────


@pytest.mark.django_db
class TestHubBalance:
    """Read-back for ``status`` — a different number than the provider's own
    credits, read without spending the one it reports."""

    def test_no_token_is_unknown_without_a_call(self):
        _config(token="")
        with patch.object(service.requests, "post") as post:
            assert service.hub_balance() == {"balance": None, "known": False}
        post.assert_not_called()

    def test_known_balance_reuses_the_existing_token(self):
        _config(token="tok")
        with patch.object(
            service.requests, "post", return_value=_resp(200, {"token": "tok", "credits": 4}),
        ) as post:
            assert service.hub_balance() == {"balance": 4, "known": True}
        url, kwargs = post.call_args.args[0], post.call_args.kwargs
        assert url.endswith("/api/v2/register/")
        assert kwargs["headers"]["Authorization"] == "Bearer tok"
        assert "public_identifier" not in kwargs["json"]

    def test_zero_balance_is_known_not_unknown(self):
        _config(token="tok")
        with patch.object(
            service.requests, "post", return_value=_resp(200, {"token": "tok", "credits": 0}),
        ):
            assert service.hub_balance() == {"balance": 0, "known": True}

    def test_outage_is_unknown(self):
        _config(token="tok")
        with patch.object(
            service.requests, "post", side_effect=requests.ConnectionError("boom"),
        ):
            assert service.hub_balance() == {"balance": None, "known": False}
