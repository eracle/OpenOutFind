# tests/enrichment/test_bettercontact.py
"""BetterContact slice — mock at the HTTP boundary (`bettercontact._session`).

The paid finder is a two-leg async handshake: ``submit`` fires a job and returns
its ``request_id``; ``poll_once`` checks that job exactly once and reports
running / hit / miss. A missing key or an unreachable service raises
BetterContactUnavailable rather than a bare error.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from openoutfind.core.errors import ErrorType
from openoutfind.enrichment import bettercontact
from openoutfind.enrichment.bettercontact import (
    BetterContactQuery,
    BetterContactUnavailable,
)

QUERY = BetterContactQuery(linkedin_url="https://www.linkedin.com/in/alice/")


@pytest.fixture
def keyed(db, configure):
    return configure(bettercontact_api_key="secret")


@pytest.fixture
def unkeyed(db, configure):
    return configure(bettercontact_api_key="")


def _response(body, error=None, status_code=200, headers=None):
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status.side_effect = error
    resp.status_code = status_code
    resp.headers = headers or {}
    return resp


def _fake_session(post=None, get=None):
    """A requests.Session stand-in usable as a context manager.

    Every call now goes through ``session.request(method, url, …)`` — the one place
    the status refusals are typed — so the fake dispatches by method and the tests
    keep mocking the same boundary they always did.
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
    return patch.object(bettercontact, "_session", return_value=_fake_session(post, get))


def _terminal(email, status):
    return _response({
        "status": "terminated",
        "data": [{"contact_email_address": email, "contact_email_address_status": status}],
    })


# ── bettercontact.submit ──────────────────────────────────────────────

class TestSubmit:
    def test_returns_request_id(self, keyed):
        post = MagicMock(return_value=_response({"id": "req1"}))
        with _patch_session(post):
            assert bettercontact.submit(QUERY) == "req1"

    def test_no_key_is_unavailable(self, unkeyed):
        post = MagicMock()
        with _patch_session(post), pytest.raises(BetterContactUnavailable):
            bettercontact.submit(QUERY)
        post.assert_not_called()

    def test_missing_request_id_is_unavailable(self, keyed):
        post = MagicMock(return_value=_response({}))  # no "id"/"request_id"
        with _patch_session(post), pytest.raises(BetterContactUnavailable):
            bettercontact.submit(QUERY)

    def test_http_error_is_unavailable(self, keyed):
        post = MagicMock(return_value=_response({}, error=requests.HTTPError("403")))
        with _patch_session(post), pytest.raises(BetterContactUnavailable):
            bettercontact.submit(QUERY)

    def test_network_error_is_unavailable(self, keyed):
        post = MagicMock(side_effect=requests.ConnectionError("boom"))
        with _patch_session(post), pytest.raises(BetterContactUnavailable):
            bettercontact.submit(QUERY)


# ── bettercontact.poll_once ───────────────────────────────────────────

class TestPollOnce:
    def test_running(self, keyed):
        get = MagicMock(return_value=_response({"status": "in progress"}))
        with _patch_session(get=get):
            outcome = bettercontact.poll_once("req1")
        assert outcome.running and not outcome.hit and not outcome.miss
        get.assert_called_once()  # a single poll, no retry loop

    def test_hit(self, keyed):
        get = MagicMock(return_value=_terminal("alice@acme.com", "valid"))
        with _patch_session(get=get):
            outcome = bettercontact.poll_once("req1")
        assert outcome.hit and outcome.email == "alice@acme.com"

    def test_a_hit_carries_the_identity_the_waterfall_resolved(self, keyed):
        """The provider derives the contact from the URL and echoes back who it is.

        Same call, same credit — so first/last name come from the provider and nothing
        in this codebase ever splits a full name into parts.
        """
        get = MagicMock(return_value=_response({
            "status": "terminated",
            "data": [{
                "contact_email_address": "elon@tesla.com",
                "contact_email_address_status": "deliverable",
                "contact_first_name": "Elon",
                "contact_last_name": "Musk",
            }],
        }))
        with _patch_session(get=get):
            outcome = bettercontact.poll_once("req1")

        assert outcome.hit
        assert (outcome.first_name, outcome.last_name) == ("Elon", "Musk")

    def test_a_hit_without_name_fields_is_still_a_hit(self, keyed):
        get = MagicMock(return_value=_terminal("alice@acme.com", "valid"))
        with _patch_session(get=get):
            outcome = bettercontact.poll_once("req1")

        assert outcome.hit
        assert outcome.first_name is None and outcome.last_name is None

    def test_terminal_no_usable_email_is_miss(self, keyed):
        get = MagicMock(return_value=_terminal(None, "not_found"))
        with _patch_session(get=get):
            outcome = bettercontact.poll_once("req1")
        assert outcome.miss and not outcome.hit

    def test_no_key_is_unavailable(self, unkeyed):
        get = MagicMock()
        with _patch_session(get=get), pytest.raises(BetterContactUnavailable):
            bettercontact.poll_once("req1")
        get.assert_not_called()

    def test_transport_error_is_unavailable(self, keyed):
        get = MagicMock(return_value=_response({}, error=requests.HTTPError("500")))
        with _patch_session(get=get), pytest.raises(BetterContactUnavailable):
            bettercontact.poll_once("req1")


# ── bettercontact.search_and_poll ─────────────────────────────────────

class TestSearchAndPoll:
    def test_the_first_poll_waits_the_ready_time_and_the_rest_one_second(self):
        get = MagicMock(side_effect=[_response({"status": "not_started"}),
                                     _response({"status": "terminated", "leads": [1]})])
        post = MagicMock(return_value=_response({"request_id": "r1"}))
        with _patch_session(post=post, get=get), \
                patch.object(bettercontact.time, "sleep") as sleep:
            body, ready = bettercontact.search_and_poll("k", "https://lf", {}, first_poll_s=1.7)

        assert body["leads"] == [1]
        assert [c.args[0] for c in sleep.call_args_list] == [1.7, 1]
        assert ready >= 0

    def test_a_transport_failure_is_unavailable(self):
        post = MagicMock(return_value=_response({}, error=requests.HTTPError("500")))
        with _patch_session(post=post), pytest.raises(BetterContactUnavailable):
            bettercontact.search_and_poll("k", "https://lf", {}, first_poll_s=0)


class TestRequestBudget:
    def _budget(self, limit=2):
        clock = {"now": 0.0}

        def sleep(seconds):
            clock["now"] += seconds

        budget = bettercontact.RequestBudget(limit, 60.0, clock=lambda: clock["now"],
                                             sleep=sleep)
        return budget, clock

    def test_requests_within_the_limit_never_wait(self):
        budget, clock = self._budget()
        budget.take()
        budget.take()
        assert clock["now"] == 0.0

    def test_one_over_the_limit_waits_for_the_oldest_to_leave_the_window(self):
        budget, clock = self._budget()
        budget.take()
        clock["now"] = 10.0
        budget.take()
        budget.take()
        assert clock["now"] == pytest.approx(60.0)

    def test_every_request_but_the_account_read_takes_a_place(self):
        session = _fake_session(get=MagicMock(return_value=_response({})))
        with patch.object(bettercontact._BUDGET, "take") as take:
            bettercontact._request(session, "GET", bettercontact._ACCOUNT_URL)
            bettercontact._request(session, "GET", "https://lf/r1")
        assert take.call_count == 1


# ── the provider's refusals are three different things ────────────────

class TestRefusals:
    @pytest.mark.parametrize("status_code,expected", [
        (401, ErrorType.PROVIDER_AUTH),
        (402, ErrorType.PROVIDER_OUT_OF_CREDITS),
    ])
    def test_each_refusal_carries_its_own_type(self, keyed, status_code, expected):
        get = MagicMock(return_value=_response({}, status_code=status_code))
        with _patch_session(get=get), pytest.raises(BetterContactUnavailable) as exc:
            bettercontact.credit_balance()

        assert exc.value.error_type == expected

    def test_an_exhausted_429_backoff_is_reported_as_rate_limited(self, keyed):
        get = MagicMock(side_effect=requests.exceptions.RetryError("too many 429s"))
        with _patch_session(get=get), pytest.raises(BetterContactUnavailable) as exc:
            bettercontact.credit_balance()

        assert exc.value.error_type == ErrorType.PROVIDER_RATE_LIMITED

    def test_the_session_backs_off_on_429_and_only_on_429(self):
        """Their docs warn that a client which keeps firing can get the account blocked;
        retrying a 401 or a 402 would just be noise — those are final answers."""
        retry = bettercontact._session("k").get_adapter("https://x").max_retries

        assert retry.status_forcelist == (429,)
        assert retry.total == bettercontact._RATE_LIMIT_ATTEMPTS
        assert retry.backoff_factor >= 5          # seconds, doubling
        assert retry.respect_retry_after_header   # the provider's own number wins


# ── bettercontact.credit_balance ──────────────────────────────────────

class TestCreditBalance:
    @pytest.mark.parametrize("sent,expected", [
        ("520.0", 520),   # what the provider really returns — a string holding a float
        (520, 520),
        ("0", 0),         # an empty wallet is a balance, not a failure to read one
        ("519.7", 519),   # floored: a fraction of a credit buys nothing
    ])
    def test_the_balance_is_read_however_the_provider_spells_it(self, keyed, sent, expected):
        """`isinstance(credits, int)` rejected `'520.0'`, so the balance was never readable
        and the run's `add_credits` ask could never fire."""
        get = MagicMock(return_value=_response({"credits_left": sent}))
        with _patch_session(get=get):
            assert bettercontact.credit_balance() == expected

    @pytest.mark.parametrize("body", [
        {}, {"credits_left": None}, {"credits_left": "many"}, {"credits_left": "-5"},
    ])
    def test_an_unreadable_balance_is_still_an_error(self, keyed, body):
        get = MagicMock(return_value=_response(body))
        with _patch_session(get=get), pytest.raises(BetterContactUnavailable):
            bettercontact.credit_balance()


# ── bettercontact.is_configured ───────────────────────────────────────

class TestIsConfigured:
    def test_false_when_key_blank(self, unkeyed):
        assert bettercontact.is_configured() is False

    def test_true_when_key_set(self, keyed):
        assert bettercontact.is_configured() is True


class TestSignupUrl:
    """Attribution is won at signup and cannot be repaired afterwards, so the
    parameter is guarded rather than trusted to whoever writes the next call site.
    """

    def test_survives_a_terminal_that_linkifies_only_part_of_a_url(self):
        """No query string: a terminal that stops at the `?` would otherwise hand the
        reader a bare domain, and an unattributed signup cannot be repaired.
        """
        assert "?" not in bettercontact.SIGNUP_URL
        assert bettercontact.SIGNUP_URL.startswith("https://openoutreach.app/go/")

    def test_nothing_we_ship_writes_the_signup_url_without_it(self):
        """The one path to an account is the constant. A bare literal — in code or in
        the docs a reader follows — is an unattributed signup waiting to happen.
        """
        package = Path(bettercontact.__file__).parent.parent
        root = package.parent
        prose = [path for path in (root / "README.md", *root.glob("docs/**/*.md")) if path.exists()]
        offenders = [
            path.relative_to(root)
            for path in (*package.rglob("*.py"), *prose)
            for line in path.read_text().splitlines()
            # `://bettercontact.rocks` is the signup host as a URL — it does not match
            # the API host (`app.bettercontact.rocks`) or a prose mention of the name.
            if "://bettercontact.rocks" in line and "fpr=openoutreach" not in line
        ]
        assert not offenders, f"signup URL without the affiliate parameter: {offenders}"
