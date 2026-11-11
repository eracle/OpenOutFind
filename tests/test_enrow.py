# tests/test_enrow.py
"""Enrow slice, mocked at the HTTP boundary (`enrow._session`).

The same two-leg async handshake as FullEnrich: ``start`` fires one single search and
returns Enrow's ``id``; ``poll_once`` checks it once and reports running / hit / miss. A
running search answers 202, so the status code is the cheap signal and the qualification
is the confirmation.

The refusals carry more weight here than elsewhere. Enrow's LinkedIn lookup reports
``lookup_failed`` as retryable and free, and the API answers an unknown id with a 500
rather than a 404, so both must defer the lookup rather than settle it as a miss: losing a
reachable person to a provider hiccup is worse than one more poll.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from openoutfind.core.errors import ErrorType
from openoutfind.enrichment import enrow
from openoutfind.enrichment.enrow import EnrowUnavailable

URL = "https://www.linkedin.com/in/alice/"


@pytest.fixture
def keyed(configure):
    return configure(enrow_api_key="secret")


@pytest.fixture
def unkeyed(configure):
    return configure(enrow_api_key="")


@pytest.fixture(autouse=True)
def _fresh_request_window():
    """The request budget is one per process, so the tests share its window.

    Emptying the window per test keeps the real ``take`` in every path and stops a later
    test inheriting a full minute of sleep from an earlier one.
    """
    enrow._BUDGET._sent.clear()


def _response(body, error=None, status_code=200, headers=None):
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status.side_effect = error
    resp.status_code = status_code
    resp.headers = headers or {}
    return resp


def _unreadable():
    """A 200 whose body is not JSON, which is a refusal and never a miss."""
    resp = _response(None)
    resp.json.side_effect = ValueError("not JSON")
    return resp


def _fake_session(post=None, get=None):
    """A ``requests.Session`` stand-in usable as a context manager.

    ``_request`` goes through ``session.request(method, url, ...)``, the one place the
    status refusals are typed, so the fake dispatches by method and the tests mock the
    same boundary the real session presents.
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
    return patch.object(enrow, "_session", return_value=_fake_session(post, get))


def _hit(email="alice@acme.com", **fields):
    return _response({"qualification": "valid", "email": email, **fields})


# enrow.start

class TestStart:
    def test_returns_enrows_own_handle(self, keyed):
        post = MagicMock(
            return_value=_response(
                {"message": "Single search operating", "id": "e1", "credits_used": 1}
            )
        )
        with _patch_session(post=post):
            lookup = enrow.start(URL)

        assert lookup.request_id == "e1"
        assert lookup.pending

    def test_sends_only_the_profile_url(self, keyed):
        """The API takes a name and a company and resolves better with them, but the
        query stays as narrow as the other adapters': the less of a lead's record leaves
        for a third party, the better.
        """
        post = MagicMock(return_value=_response({"id": "e1"}))
        with _patch_session(post=post):
            enrow.start(URL)

        assert post.call_args.kwargs["json"] == {"linkedin_url": URL}

    def test_no_key_is_unavailable(self, unkeyed):
        post = MagicMock()
        with _patch_session(post=post), pytest.raises(EnrowUnavailable):
            enrow.start(URL)
        post.assert_not_called()

    def test_a_response_with_no_id_is_unavailable(self, keyed):
        post = MagicMock(return_value=_response({"message": "queued"}))
        with _patch_session(post=post), pytest.raises(EnrowUnavailable):
            enrow.start(URL)

    def test_network_error_is_unavailable(self, keyed):
        post = MagicMock(side_effect=requests.ConnectionError("boom"))
        with _patch_session(post=post), pytest.raises(EnrowUnavailable):
            enrow.start(URL)


# enrow.poll_once

class TestPollOnce:
    def test_a_running_search_is_202(self, keyed):
        """The status code is the signal, even when the body looks finished."""
        get = MagicMock(
            return_value=_response(_hit(), status_code=202)
        )
        with _patch_session(get=get):
            outcome = enrow.poll_once("e1")

        assert outcome.running
        assert not outcome.hit

    def test_the_handle_is_read_by_id(self, keyed):
        get = MagicMock(return_value=_response({"qualification": "ongoing"}))
        with _patch_session(get=get):
            enrow.poll_once("e1")

        assert get.call_args.args[0] == f"{enrow._SINGLE_URL}?id=e1"

    def test_a_valid_qualification_with_an_address_is_a_hit(self, keyed):
        with _patch_session(get=MagicMock(return_value=_hit())):
            outcome = enrow.poll_once("e1")

        assert outcome.hit
        assert outcome.email == "alice@acme.com"

    def test_the_address_carries_the_name_enrow_resolved(self, keyed):
        body = _hit(info={"firstname": "Alice", "lastname": "Anders"})
        with _patch_session(get=MagicMock(return_value=body)):
            outcome = enrow.poll_once("e1")

        assert outcome.first_name == "Alice"
        assert outcome.last_name == "Anders"

    def test_a_valid_qualification_with_no_address_is_still_running(self, keyed):
        """A half-filled document is not a lead, and settling it would drop the person."""
        with _patch_session(get=MagicMock(return_value=_response({"qualification": "valid"}))):
            assert enrow.poll_once("e1").running

    def test_an_invalid_qualification_is_a_miss(self, keyed):
        with _patch_session(get=MagicMock(return_value=_response({"qualification": "invalid"}))):
            outcome = enrow.poll_once("e1")

        assert outcome.miss
        assert not outcome.running

    def test_ongoing_is_running(self, keyed):
        with _patch_session(get=MagicMock(return_value=_response({"qualification": "ongoing"}))):
            assert enrow.poll_once("e1").running

    def test_an_unrecognised_qualification_reads_as_running(self, keyed):
        """Recording it as a miss would settle the deal as NO_EMAIL_FOUND and never retry."""
        with _patch_session(get=MagicMock(return_value=_response({"qualification": "queued"}))):
            assert enrow.poll_once("e1").running

    def test_a_non_json_body_is_unavailable(self, keyed):
        with _patch_session(get=MagicMock(return_value=_unreadable())), pytest.raises(
            EnrowUnavailable
        ):
            enrow.poll_once("e1")

    def test_no_key_is_unavailable(self, unkeyed):
        get = MagicMock()
        with _patch_session(get=get), pytest.raises(EnrowUnavailable):
            enrow.poll_once("e1")
        get.assert_not_called()

    def test_the_search_is_checked_exactly_once(self, keyed):
        get = MagicMock(return_value=_response({"qualification": "ongoing"}))
        with _patch_session(get=get):
            enrow.poll_once("e1")

        get.assert_called_once()


# a transient LinkedIn status defers, even alongside a verdict

class TestTransientLinkedInStatus:
    def test_lookup_failed_defers_even_when_the_qualification_says_invalid(self, keyed):
        """The status is checked first: ``lookup_failed`` costs nothing and is retryable,
        so it must outrank a qualification that would otherwise settle the lookup.
        """
        body = _response({"qualification": "invalid", "linkedin_status": "lookup_failed"})
        with _patch_session(get=MagicMock(return_value=body)), pytest.raises(EnrowUnavailable):
            enrow.poll_once("e1")

    def test_a_settled_status_falls_through_to_the_miss(self, keyed):
        body = _response({"qualification": "invalid", "linkedin_status": "email_not_found"})
        with _patch_session(get=MagicMock(return_value=body)):
            outcome = enrow.poll_once("e1")

        assert outcome.miss


# enrow._request, the typed refusals

class TestHttpRefusals:
    def test_401_is_auth(self, keyed):
        post = MagicMock(return_value=_response({}, status_code=401))
        with _patch_session(post=post), pytest.raises(EnrowUnavailable) as caught:
            enrow.start(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_AUTH

    def test_402_is_out_of_credits(self, keyed):
        """A single search that finds no credits answers 402, not a generic failure."""
        post = MagicMock(return_value=_response({}, status_code=402))
        with _patch_session(post=post), pytest.raises(EnrowUnavailable) as caught:
            enrow.start(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_OUT_OF_CREDITS

    def test_a_server_error_on_a_search_is_unavailable(self, keyed):
        post = MagicMock(
            return_value=_response({}, error=requests.HTTPError("500"), status_code=500)
        )
        with _patch_session(post=post), pytest.raises(EnrowUnavailable) as caught:
            enrow.start(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_UNAVAILABLE

    def test_an_unknown_id_on_poll_is_unavailable_not_a_miss(self, keyed):
        """The API answers an expired or unknown id with a 500, indistinguishable from a
        transient server error, so the lookup defers rather than recording a miss.
        """
        get = MagicMock(
            return_value=_response({}, error=requests.HTTPError("500"), status_code=500)
        )
        with _patch_session(get=get), pytest.raises(EnrowUnavailable) as caught:
            enrow.poll_once("stale")
        assert caught.value.error_type == ErrorType.PROVIDER_UNAVAILABLE

    def test_exhausted_backoff_is_rate_limited(self, keyed):
        post = MagicMock(side_effect=requests.exceptions.RetryError("too many"))
        with _patch_session(post=post), pytest.raises(EnrowUnavailable) as caught:
            enrow.start(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_RATE_LIMITED


# enrow.credit_balance

class TestCreditBalance:
    def test_reads_the_whole_credit_count(self, keyed):
        """The account document carries the balance beside the webhooks."""
        get = MagicMock(return_value=_response({"credits": 36000, "webhooks": []}))
        with _patch_session(get=get):
            assert enrow.credit_balance() == 36000

    def test_a_float_balance_is_floored(self, keyed):
        """A fraction of a credit buys nothing, and rounding up would claim a balance that
        cannot pay for the next lookup.
        """
        get = MagicMock(return_value=_response({"credits": 15432.50}))
        with _patch_session(get=get):
            assert enrow.credit_balance() == 15432

    def test_a_missing_count_is_a_refusal(self, keyed):
        get = MagicMock(return_value=_response({}))
        with _patch_session(get=get), pytest.raises(EnrowUnavailable):
            enrow.credit_balance()

    def test_a_non_numeric_count_is_a_refusal(self, keyed):
        get = MagicMock(return_value=_response({"credits": "plenty"}))
        with _patch_session(get=get), pytest.raises(EnrowUnavailable):
            enrow.credit_balance()

    def test_a_negative_balance_is_a_refusal(self, keyed):
        get = MagicMock(return_value=_response({"credits": -1}))
        with _patch_session(get=get), pytest.raises(EnrowUnavailable):
            enrow.credit_balance()

    def test_no_key_is_unavailable(self, unkeyed):
        get = MagicMock()
        with _patch_session(get=get), pytest.raises(EnrowUnavailable):
            enrow.credit_balance()
        get.assert_not_called()

    def test_an_unreachable_service_is_unavailable(self, keyed):
        get = MagicMock(side_effect=requests.ConnectionError("boom"))
        with _patch_session(get=get), pytest.raises(EnrowUnavailable):
            enrow.credit_balance()


# the per-process request budget

class TestRequestBudget:
    def test_a_search_takes_a_place(self, keyed):
        session = _fake_session(post=MagicMock(return_value=_response({"id": "e1"})))
        with patch.object(enrow._BUDGET, "take") as take:
            enrow._request(session, "POST", enrow._SINGLE_URL, json={})
        take.assert_called_once()

    def test_a_poll_takes_a_place(self, keyed):
        session = _fake_session(get=MagicMock(return_value=_response({})))
        with patch.object(enrow._BUDGET, "take") as take:
            enrow._request(session, "GET", f"{enrow._SINGLE_URL}?id=e1")
        take.assert_called_once()

    def test_the_balance_read_does_not_take_a_place(self):
        """Enrow documents the account read as free and outside its limiter, the same
        exemption BetterContact gives its own account read.
        """
        session = _fake_session(get=MagicMock(return_value=_response({"credits": 1})))
        with patch.object(enrow._BUDGET, "take") as take:
            enrow._request(session, "GET", enrow._ACCOUNT_URL)
        take.assert_not_called()


# enrow's published price and setup

class TestIsConfigured:
    def test_false_when_key_blank(self, unkeyed):
        assert enrow.is_configured() is False

    def test_true_when_key_set(self, keyed):
        assert enrow.is_configured() is True


class TestSignupUrl:
    def test_is_the_plain_signup_page_with_no_invented_attribution(self):
        """Enrow runs no affiliate redirect of its own, so a query string here would be a
        link claiming an attribution nobody granted.
        """
        assert enrow.SIGNUP_URL == "https://enrow.io"
        assert "?" not in enrow.SIGNUP_URL


class TestThePublishedPrice:
    def test_the_entry_plan_is_cheaper_than_the_other_three(self):
        """The cheapest-first pick depends on this, so the anchor is pinned here."""
        from openoutfind.enrichment import bettercontact, fullenrich, leadmagic

        assert enrow.UNIT_COST_USD < min(
            bettercontact.UNIT_COST_USD, fullenrich.UNIT_COST_USD, leadmagic.UNIT_COST_USD
        )
