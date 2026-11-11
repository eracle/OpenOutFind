# tests/test_fullenrich.py
"""FullEnrich slice, mocked at the HTTP boundary (`fullenrich._session`).

The second paid finder is the same two-leg async handshake as BetterContact:
``submit`` fires one bulk job and returns FullEnrich's ``enrichment_id``;
``poll_once`` checks that job exactly once and reports running / hit / miss. A
missing key or an unreachable service raises FullEnrichUnavailable rather than a
bare error, and a finished job that says the wallet was empty or the client was
throttled raises too, since neither is ever an unreachable lead.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from openoutfind.core.errors import ErrorType
from openoutfind.enrichment import fullenrich
from openoutfind.enrichment.fullenrich import FullEnrichUnavailable

URL = "https://www.linkedin.com/in/alice/"


@pytest.fixture
def keyed(db, configure):
    return configure(fullenrich_api_key="secret")


@pytest.fixture
def unkeyed(db, configure):
    return configure(fullenrich_api_key="")


@pytest.fixture(autouse=True)
def _fresh_request_window():
    """The request budget is one per process, so the tests share its window.

    ``RequestBudget.take`` waits for real once the minute is full, which turned
    the fifteenth filer in this file into a real minute of sleep. Emptying the
    window per test keeps the real ``take`` in every path and the suite quick.
    """
    fullenrich._BUDGET._sent.clear()


def _response(body, error=None, status_code=200, headers=None):
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status.side_effect = error
    resp.status_code = status_code
    resp.headers = headers or {}
    return resp


def _fake_session(post=None, get=None):
    """A requests.Session stand-in usable as a context manager.

    ``_request`` goes through ``session.request(method, url, ...)``, the one place
    the status refusals are typed, so the fake dispatches by method and the tests
    mock the same boundary the real session presents.
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
    return patch.object(fullenrich, "_session", return_value=_fake_session(post, get))


def _finished(contact_info, input_=None):
    """One FINISHED status document carrying a single record."""
    row = {"contact_info": contact_info}
    if input_ is not None:
        row["input"] = input_
    return _response({"status": "FINISHED", "data": [row]})


# fullenrich.submit

class TestSubmit:
    def test_returns_enrichment_id(self, keyed):
        post = MagicMock(return_value=_response({"enrichment_id": "e1"}))
        with _patch_session(post):
            assert fullenrich.submit(URL) == "e1"

    def test_sends_only_the_profile_url(self, keyed):
        """The API takes a name and a company and resolves better with them, but
        the query stays as narrow as BetterContact's: the less of a lead's record
        leaves for a third party, the better.
        """
        post = MagicMock(return_value=_response({"enrichment_id": "e1"}))
        with _patch_session(post):
            fullenrich.submit(URL)

        row = post.call_args.kwargs["json"]["data"][0]
        assert row["professional_network_url"] == URL
        assert row["enrich_fields"] == ["contact.work_emails"]
        assert set(row) == {"professional_network_url", "enrich_fields"}

    def test_no_key_is_unavailable(self, unkeyed):
        post = MagicMock()
        with _patch_session(post), pytest.raises(FullEnrichUnavailable):
            fullenrich.submit(URL)
        post.assert_not_called()

    def test_missing_enrichment_id_is_unavailable(self, keyed):
        post = MagicMock(return_value=_response({}))
        with _patch_session(post), pytest.raises(FullEnrichUnavailable):
            fullenrich.submit(URL)

    def test_http_error_is_unavailable(self, keyed):
        post = MagicMock(
            return_value=_response({}, error=requests.HTTPError("500"), status_code=500)
        )
        with _patch_session(post), pytest.raises(FullEnrichUnavailable):
            fullenrich.submit(URL)

    def test_network_error_is_unavailable(self, keyed):
        post = MagicMock(side_effect=requests.ConnectionError("boom"))
        with _patch_session(post), pytest.raises(FullEnrichUnavailable):
            fullenrich.submit(URL)


# fullenrich.poll_once

class TestPollOnce:
    def test_created_is_running(self, keyed):
        get = MagicMock(return_value=_response({"status": "CREATED"}))
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.running and not outcome.hit and not outcome.miss

    def test_in_progress_is_running_and_polls_once(self, keyed):
        get = MagicMock(return_value=_response({"status": "IN_PROGRESS"}))
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.running
        get.assert_called_once()  # a single poll, no retry loop

    def test_polls_the_jobs_own_url(self, keyed):
        get = MagicMock(return_value=_response({"status": "CREATED"}))
        with _patch_session(get=get):
            fullenrich.poll_once("e1")
        assert get.call_args.args[0].endswith("/e1")

    def test_finished_deliverable_is_a_hit(self, keyed):
        get = MagicMock(
            return_value=_finished(
                {"work_emails": [{"email": "alice@acme.com", "status": "DELIVERABLE"}]}
            )
        )
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.hit and outcome.email == "alice@acme.com"

    def test_high_probability_is_a_hit(self, keyed):
        get = MagicMock(
            return_value=_finished(
                {"work_emails": [{"email": "alice@acme.com", "status": "HIGH_PROBABILITY"}]}
            )
        )
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.hit

    def test_catch_all_is_a_miss(self, keyed):
        """CATCH_ALL only says the domain accepts everything, a domain-level fact
        with no claim about the mailbox, so it is not a usable status here.
        """
        get = MagicMock(
            return_value=_finished(
                {"work_emails": [{"email": "alice@acme.com", "status": "CATCH_ALL"}]}
            )
        )
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.miss and not outcome.hit

    def test_a_later_usable_entry_wins_when_the_first_is_not(self, keyed):
        get = MagicMock(
            return_value=_finished(
                {
                    "work_emails": [
                        {"email": "alice@acme.com", "status": "UNVERIFIED"},
                        {"email": "a.smith@acme.com", "status": "DELIVERABLE"},
                    ]
                }
            )
        )
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.email == "a.smith@acme.com"

    def test_a_bare_most_probable_email_is_a_hit(self, keyed):
        """With no status beside it there is nothing to check it against, so
        discarding it would spend a credit and throw the answer away.
        """
        get = MagicMock(
            return_value=_finished({"most_probable_work_email": "alice@acme.com"})
        )
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.hit and outcome.email == "alice@acme.com"

    def test_a_hit_carries_the_identity_the_job_echoed(self, keyed):
        get = MagicMock(
            return_value=_finished(
                {"work_emails": [{"email": "elon@tesla.com", "status": "DELIVERABLE"}]},
                input_={"first_name": "Elon", "last_name": "Musk"},
            )
        )
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.hit
        assert (outcome.first_name, outcome.last_name) == ("Elon", "Musk")

    def test_a_hit_without_identity_fields_is_still_a_hit(self, keyed):
        get = MagicMock(
            return_value=_finished(
                {"work_emails": [{"email": "alice@acme.com", "status": "DELIVERABLE"}]}
            )
        )
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.hit
        assert outcome.first_name is None and outcome.last_name is None

    def test_finished_with_no_usable_email_is_a_miss(self, keyed):
        get = MagicMock(return_value=_finished({"work_emails": []}))
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.miss and not outcome.hit

    def test_finished_with_no_rows_is_a_miss(self, keyed):
        get = MagicMock(return_value=_response({"status": "FINISHED", "data": []}))
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.miss

    def test_canceled_is_a_miss(self, keyed):
        get = MagicMock(return_value=_response({"status": "CANCELED"}))
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.miss and not outcome.running

    def test_unknown_reads_as_still_running(self, keyed):
        """UNKNOWN is FullEnrich's own catch-all, so a transient server-side
        answer cannot be recorded as a lead with no address.
        """
        get = MagicMock(return_value=_response({"status": "UNKNOWN"}))
        with _patch_session(get=get):
            outcome = fullenrich.poll_once("e1")
        assert outcome.running and not outcome.miss

    def test_rate_limited_job_is_typed(self, keyed):
        get = MagicMock(return_value=_response({"status": "RATE_LIMIT"}))
        with _patch_session(get=get), pytest.raises(FullEnrichUnavailable) as caught:
            fullenrich.poll_once("e1")
        assert caught.value.error_type == ErrorType.PROVIDER_RATE_LIMITED

    def test_out_of_credits_job_is_typed(self, keyed):
        get = MagicMock(return_value=_response({"status": "CREDITS_INSUFFICIENT"}))
        with _patch_session(get=get), pytest.raises(FullEnrichUnavailable) as caught:
            fullenrich.poll_once("e1")
        assert caught.value.error_type == ErrorType.PROVIDER_OUT_OF_CREDITS

    def test_no_key_is_unavailable(self, unkeyed):
        get = MagicMock()
        with _patch_session(get=get), pytest.raises(FullEnrichUnavailable):
            fullenrich.poll_once("e1")
        get.assert_not_called()

    def test_transport_error_is_unavailable(self, keyed):
        get = MagicMock(
            return_value=_response({}, error=requests.HTTPError("500"), status_code=500)
        )
        with _patch_session(get=get), pytest.raises(FullEnrichUnavailable):
            fullenrich.poll_once("e1")


# the typed refusals the HTTP boundary owes the caller

class TestHttpRefusals:
    def test_401_is_auth(self, keyed):
        post = MagicMock(return_value=_response({}, status_code=401))
        with _patch_session(post), pytest.raises(FullEnrichUnavailable) as caught:
            fullenrich.submit(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_AUTH

    def test_402_is_out_of_credits(self, keyed):
        post = MagicMock(return_value=_response({}, status_code=402))
        with _patch_session(post), pytest.raises(FullEnrichUnavailable) as caught:
            fullenrich.submit(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_OUT_OF_CREDITS

    def test_exhausted_backoff_is_rate_limited(self, keyed):
        post = MagicMock(side_effect=requests.exceptions.RetryError("too many"))
        with _patch_session(post), pytest.raises(FullEnrichUnavailable) as caught:
            fullenrich.submit(URL)
        assert caught.value.error_type == ErrorType.PROVIDER_RATE_LIMITED


# fullenrich.credit_balance

class TestCreditBalance:
    def test_raises_rather_than_reporting_a_balance_of_zero(self):
        """FullEnrich exposes no balance endpoint, so a zero read here would send
        an operator to top up an account that is already funded.
        """
        with patch.object(fullenrich, "_session") as session, pytest.raises(
            FullEnrichUnavailable
        ):
            fullenrich.credit_balance()
        session.assert_not_called()


# the per-process request budget

class TestRequestBudget:
    def test_every_request_takes_a_place(self):
        session = _fake_session(get=MagicMock(return_value=_response({})))
        with patch.object(fullenrich._BUDGET, "take") as take:
            fullenrich._request(session, "GET", "https://app.fullenrich.com/api/v2/x")
        take.assert_called_once()


# fullenrich.is_configured

class TestIsConfigured:
    def test_false_when_key_blank(self, unkeyed):
        assert fullenrich.is_configured() is False

    def test_true_when_key_set(self, keyed):
        assert fullenrich.is_configured() is True


class TestSignupUrl:
    def test_is_the_plain_signup_page_with_no_invented_attribution(self):
        """FullEnrich runs no affiliate redirect of its own, so a query string here
        would be a link claiming an attribution nobody granted.
        """
        assert fullenrich.SIGNUP_URL == "https://fullenrich.com"
        assert "?" not in fullenrich.SIGNUP_URL
