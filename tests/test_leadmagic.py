"""LeadMagic resolves an address inside the ``start`` call.

The other two adapters speak a two-leg handshake because their APIs run a job. LeadMagic's
finder answers in the same response, so ``start`` is the whole lookup and every outcome is
already finished: there is no ``poll_once`` here to be left holding a handle.

The refusals matter more than usual in this file. A provider that cannot run must raise, so
``lookup.py`` steps over it to the next finder; a provider that does run is authoritative,
miss included. A missing key, an unreachable service, an empty wallet and a throttle are all
"could not run", never "this lead has no address".
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from openoutfind.core.errors import ErrorType
from openoutfind.enrichment import leadmagic
from openoutfind.enrichment.leadmagic import LeadMagicUnavailable

URL = "https://www.linkedin.com/in/alice/"


@pytest.fixture
def keyed(configure):
    return configure(leadmagic_api_key="secret")


@pytest.fixture
def unkeyed(configure):
    return configure(leadmagic_api_key="")


@pytest.fixture(autouse=True)
def _fresh_request_window():
    """The request budget is one per process, so the tests share its window.

    Emptying the window per test keeps the real ``take`` in every path and stops a later
    test inheriting a full minute of sleep from an earlier one.
    """
    leadmagic._BUDGET._sent.clear()


def _response(body, error=None, status_code=200, headers=None):
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status.side_effect = error
    resp.status_code = status_code
    resp.headers = headers or {}
    return resp


def _fake_session(post=None, get=None):
    """A ``requests.Session`` stand-in usable as a context manager.

    ``_request`` goes through ``session.request(method, url, ...)``, the one place the
    status refusals are typed, so the fake dispatches by method and the tests mock the same
    boundary the real session presents.
    """
    session = MagicMock()
    session.__enter__.return_value = session
    session.post = post or MagicMock()
    session.get = get or MagicMock()

    def request(method, url, **kwargs):
        handler = session.post if method.upper() == "POST" else session.get
        return handler(url, **kwargs)

    session.request = MagicMock(side_effect=request)
    return session


def _patch_session(post=None, get=None):
    return patch.object(leadmagic, "_session", return_value=_fake_session(post, get))


def _hit(email="alice@acme.com", **fields):
    return _response({"status": "valid", "email": email, **fields})


# leadmagic.start

class TestStart:
    def test_start_is_the_whole_lookup(self, keyed):
        """The answer is already terminated, so the deal never enters FINDING_EMAIL."""
        post = MagicMock(return_value=_hit())
        with _patch_session(post=post):
            lookup = leadmagic.start(URL)

        assert lookup.outcome is not None
        assert not lookup.pending
        assert lookup.outcome.hit
        assert lookup.outcome.email == "alice@acme.com"

    def test_start_carries_no_request_id(self, keyed):
        """Nothing is submitted to be collected later, so there is no handle to record."""
        with _patch_session(post=MagicMock(return_value=_hit())):
            lookup = leadmagic.start(URL)

        assert lookup.request_id == ""

    def test_a_miss_is_also_finished(self, keyed):
        post = MagicMock(return_value=_response({"status": None}))
        with _patch_session(post=post):
            lookup = leadmagic.start(URL)

        assert not lookup.pending
        assert lookup.outcome.miss

    def test_there_is_no_poll_once(self):
        """A synchronous finder must not offer a poll, or the seam would call one."""
        assert not hasattr(leadmagic, "poll_once")

    def test_sends_only_the_profile_url(self, keyed):
        """The API takes a name and a company and resolves better with them, but the query
        stays as narrow as the other two adapters': the less of a lead's record leaves for
        a third party, the better.
        """
        post = MagicMock(return_value=_hit())
        with _patch_session(post=post):
            leadmagic.start(URL)

        assert post.call_args.kwargs["json"] == {"profile_url": URL}

    def test_no_key_is_unavailable(self, unkeyed):
        post = MagicMock()
        with _patch_session(post=post), pytest.raises(LeadMagicUnavailable):
            leadmagic.start(URL)
        post.assert_not_called()

    def test_http_error_is_unavailable(self, keyed):
        post = MagicMock(
            return_value=_response({}, error=requests.HTTPError("500"), status_code=500)
        )
        with _patch_session(post=post), pytest.raises(LeadMagicUnavailable):
            leadmagic.start(URL)

    def test_network_error_is_unavailable(self, keyed):
        post = MagicMock(side_effect=requests.ConnectionError("boom"))
        with _patch_session(post=post), pytest.raises(LeadMagicUnavailable):
            leadmagic.start(URL)


# leadmagic._outcome, driven through find_email

class TestOutcome:
    def test_valid_status_is_a_hit(self, keyed):
        with _patch_session(post=MagicMock(return_value=_hit())):
            outcome = leadmagic.find_email(URL)
        assert outcome.hit and outcome.email == "alice@acme.com"

    def test_a_hit_carries_the_names_from_the_same_response(self, keyed):
        post = MagicMock(
            return_value=_hit("elon@tesla.com", first_name="Elon", last_name="Musk")
        )
        with _patch_session(post=post):
            outcome = leadmagic.find_email(URL)
        assert outcome.hit
        assert (outcome.first_name, outcome.last_name) == ("Elon", "Musk")

    def test_names_are_trimmed(self, keyed):
        post = MagicMock(return_value=_hit(first_name="  El on ", last_name=" Musk\n"))
        with _patch_session(post=post):
            outcome = leadmagic.find_email(URL)
        assert (outcome.first_name, outcome.last_name) == ("El on", "Musk")

    def test_blank_names_are_absent(self, keyed):
        post = MagicMock(return_value=_hit(first_name="   ", last_name=""))
        with _patch_session(post=post):
            outcome = leadmagic.find_email(URL)
        assert outcome.first_name is None and outcome.last_name is None

    def test_a_hit_without_names_is_still_a_hit(self, keyed):
        with _patch_session(post=MagicMock(return_value=_hit())):
            outcome = leadmagic.find_email(URL)
        assert outcome.hit
        assert outcome.first_name is None and outcome.last_name is None

    def test_null_status_is_a_miss(self, keyed):
        """``null`` is LeadMagic finding nothing, and it bills nothing for it."""
        post = MagicMock(return_value=_response({"status": None, "credits_consumed": 0}))
        with _patch_session(post=post):
            outcome = leadmagic.find_email(URL)
        assert outcome.miss and not outcome.hit

    def test_a_missing_status_is_a_miss(self, keyed):
        post = MagicMock(return_value=_response({"email": "alice@acme.com"}))
        with _patch_session(post=post):
            outcome = leadmagic.find_email(URL)
        assert outcome.miss

    def test_an_empty_email_is_a_miss(self, keyed):
        post = MagicMock(return_value=_response({"status": "valid", "email": ""}))
        with _patch_session(post=post):
            outcome = leadmagic.find_email(URL)
        assert outcome.miss

    def test_a_non_string_email_is_a_miss(self, keyed):
        post = MagicMock(return_value=_response({"status": "valid", "email": 42}))
        with _patch_session(post=post):
            outcome = leadmagic.find_email(URL)
        assert outcome.miss

    def test_a_domain_mismatch_is_a_free_miss(self, keyed):
        """LeadMagic returns the unrelated-domain address as a not-found with a flag, so
        it reads as a miss without a special case here.
        """
        post = MagicMock(
            return_value=_response(
                {
                    "status": None,
                    "email": "alice@elsewhere.com",
                    "domain_mismatch": True,
                    "credits_consumed": 0,
                }
            )
        )
        with _patch_session(post=post):
            outcome = leadmagic.find_email(URL)
        assert outcome.miss and not outcome.hit


# the typed refusals the HTTP boundary owes the caller

class TestHttpRefusals:
    def test_401_is_auth(self, keyed):
        post = MagicMock(return_value=_response({}, status_code=401))
        with _patch_session(post=post), pytest.raises(LeadMagicUnavailable) as caught:
            leadmagic.start(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_AUTH

    def test_402_is_out_of_credits(self, keyed):
        """LeadMagic documents 402 as insufficient credits, not a generic failure."""
        post = MagicMock(return_value=_response({}, status_code=402))
        with _patch_session(post=post), pytest.raises(LeadMagicUnavailable) as caught:
            leadmagic.start(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_OUT_OF_CREDITS

    def test_exhausted_backoff_is_rate_limited(self, keyed):
        post = MagicMock(side_effect=requests.exceptions.RetryError("too many"))
        with _patch_session(post=post), pytest.raises(LeadMagicUnavailable) as caught:
            leadmagic.start(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_RATE_LIMITED

    def test_a_server_error_is_unavailable_not_a_miss(self, keyed):
        """A 500 body carries no ``status``, so without the raise it would read as a miss
        and tell the operator a reachable person has no address.
        """
        post = MagicMock(
            return_value=_response({}, error=requests.HTTPError("500"), status_code=500)
        )
        with _patch_session(post=post), pytest.raises(LeadMagicUnavailable) as caught:
            leadmagic.start(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_UNAVAILABLE


# leadmagic.credit_balance

class TestCreditBalance:
    def test_a_float_balance_is_floored(self, keyed):
        """The provider answers ``{"credits": 15432.50}``, a float, and a fraction of a
        credit buys nothing.
        """
        get = MagicMock(return_value=_response({"credits": 15432.50}))
        with _patch_session(get=get):
            assert leadmagic.credit_balance() == 15432

    def test_a_missing_count_is_a_refusal(self, keyed):
        get = MagicMock(return_value=_response({}))
        with _patch_session(get=get), pytest.raises(LeadMagicUnavailable):
            leadmagic.credit_balance()

    def test_a_negative_balance_is_a_refusal(self, keyed):
        get = MagicMock(return_value=_response({"credits": -1}))
        with _patch_session(get=get), pytest.raises(LeadMagicUnavailable):
            leadmagic.credit_balance()

    def test_no_key_is_unavailable(self, unkeyed):
        get = MagicMock()
        with _patch_session(get=get), pytest.raises(LeadMagicUnavailable):
            leadmagic.credit_balance()
        get.assert_not_called()

    def test_an_unreachable_service_is_unavailable(self, keyed):
        get = MagicMock(side_effect=requests.ConnectionError("boom"))
        with _patch_session(get=get), pytest.raises(LeadMagicUnavailable):
            leadmagic.credit_balance()


# the per-process request budget

class TestRequestBudget:
    def test_a_finder_call_takes_a_place(self, keyed):
        session = _fake_session(post=MagicMock(return_value=_hit()))
        with patch.object(leadmagic._BUDGET, "take") as take:
            leadmagic._request(session, "POST", leadmagic._FINDER_URL, json={})
        take.assert_called_once()

    def test_the_balance_read_does_not_take_a_place(self):
        """The vendor documents the balance as free and outside the limiter, the same
        exemption BetterContact gives its own account read.
        """
        session = _fake_session(get=MagicMock(return_value=_response({"credits": 1})))
        with patch.object(leadmagic._BUDGET, "take") as take:
            leadmagic._request(session, "GET", leadmagic._CREDITS_URL)
        take.assert_not_called()


# leadmagic.is_configured

class TestIsConfigured:
    def test_false_when_key_blank(self, unkeyed):
        assert leadmagic.is_configured() is False

    def test_true_when_key_set(self, keyed):
        assert leadmagic.is_configured() is True


class TestSignupUrl:
    def test_is_the_plain_signup_page_with_no_invented_attribution(self):
        """LeadMagic runs no affiliate redirect of its own, so a query string here would
        be a link claiming an attribution nobody granted.
        """
        assert leadmagic.SIGNUP_URL == "https://leadmagic.io"
        assert "?" not in leadmagic.SIGNUP_URL
