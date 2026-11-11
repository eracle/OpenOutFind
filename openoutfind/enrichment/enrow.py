"""Enrow: the fourth paid finder, and a waterfall behind the same seam as the rest.

Same two-leg handshake as FullEnrich: ``start`` fires one single search and hands back
Enrow's own ``id``; ``poll_once`` checks it once and returns a ``PollOutcome``. Nothing
here blocks on a poll: the collect step owns the wait, on the deal's own ``not_before``,
so a run parked at FINDING_EMAIL costs this process nothing.

Why a fourth adapter: the seam's job is to pick the finder that is cheapest on the
operator's plan (``provider.py``), and the waterfalls quote different prices per credit at
different tiers. Enrow is the cheapest of the four at entry, so a keyed install picks it
first and the others stay as the fallback chain.

Discovery is **not** here. Lead Finder is BetterContact's free index and stays on
BetterContact's key, so an Enrow-only install can resolve an address but cannot search,
by decision (see ``REQUIRED_DISCOVERY_FIELDS`` in ``core/config.py``).
"""

from __future__ import annotations

import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from openoutfind.core.errors import ErrorType
from openoutfind.enrichment.bettercontact import RequestBudget
from openoutfind.enrichment.provider import Lookup, PollOutcome, ProviderUnavailable

logger = logging.getLogger(__name__)

NAME = "enrow"

# The one path to an account. Enrow runs no affiliate redirect of its own, so this is
# their plain signup page rather than a parameterised one: inventing a tracking query
# here would be a link that claims an attribution nobody granted.
SIGNUP_URL = "https://enrow.io"

# One credit on the entry plan ($17/month for 1,000 credits), which is what an operator
# starting today pays: the per-credit price falls on larger plans, so this is an anchor
# for the cheapest-first pick and not a claim about volume. Kept beside the adapter, as
# the others' is, so the selector reads it from the provider rather than a table that
# goes stale without the vendor.
UNIT_COST_USD = 0.017

_SINGLE_URL = "https://api.enrow.io/email/find/single"
_ACCOUNT_URL = "https://api.enrow.io/account/info"

# One HTTP call's ceiling, not a job's. A single search walks Enrow's own waterfall for
# a while, so the gap between polls belongs to the collect step's cadence rather than to
# a constant here.
_HTTP_TIMEOUT_S = 30

# Enrow documents 10 requests per second on every POST endpoint, for every plan, and
# asks for backoff on a 429. This is a courtesy fifth of that figure, per process, so
# several concurrent runs stay well inside anything they would call abuse. It is a share
# of the key, not a measurement of the account, and their GET endpoints are not limited
# at all, so the poll leg could climb higher safely if a run ever needed it.
REQUESTS_PER_MINUTE = 120
_BUDGET_WINDOW_S = 60.0
_RATE_LIMIT_ATTEMPTS = 5

# The one verdict that means an address was found. Enrow keeps the vocabulary to three
# values, so anything else is either still running or a failure we must not record.
_FOUND = "valid"
_NOT_FOUND = "invalid"
_RUNNING = "ongoing"

# A LinkedIn search that failed on Enrow's side costs nothing and is explicitly
# retryable, so it must defer the lookup rather than settle it as a miss.
_TRANSIENT_LINKEDIN_STATUSES = frozenset({"lookup_failed"})

_RETRY = Retry(
    total=_RATE_LIMIT_ATTEMPTS,
    status_forcelist=(429,),
    allowed_methods=frozenset({"GET", "POST"}),
    backoff_factor=5,
    backoff_max=120,
    respect_retry_after_header=True,
)

_BUDGET = RequestBudget(REQUESTS_PER_MINUTE, _BUDGET_WINDOW_S, label=NAME)


class EnrowUnavailable(ProviderUnavailable):
    """Enrow could not answer, as distinct from answering that there is no address.

    The seam catches ``ProviderUnavailable`` and never names a vendor, so this exists
    for tests and readers rather than for callers to branch on. A 5xx and a transient
    LinkedIn status both read as this: losing a real lead to a provider hiccup is worse
    than deferring its lookup by one backoff step.
    """

    def __init__(
        self, message: str, error_type: ErrorType = ErrorType.PROVIDER_UNAVAILABLE
    ) -> None:
        super().__init__(message, error_type)


def is_configured() -> bool:
    """True when the Enrow paid finder is configured (an API key is set)."""
    from openoutfind.core.config import SiteConfig

    return bool(SiteConfig.load().enrow_api_key)


def credit_balance() -> int:
    """The credits left on the account, as Enrow counts them.

    Like LeadMagic and unlike FullEnrich, Enrow has a balance endpoint,
    ``GET /account/info``, so this is a real number rather than a raised refusal and the
    run can warn before a lead fails instead of after a 402. The response is
    ``{"credits": 36000, "webhooks": [...]}``, so it goes through the same
    floor-don't-round reading LeadMagic's balance does: a fraction of a credit buys
    nothing, and rounding it up would report a balance that cannot pay for the next
    lookup. The read is free per Enrow's own docs, so it sits outside the budget.
    """
    api_key = _require_key()
    with _session(api_key) as session:
        try:
            body = _request(session, "GET", _ACCOUNT_URL).json()
        except (requests.RequestException, TimeoutError) as exc:
            raise EnrowUnavailable(f"Enrow unreachable: {exc}") from exc
    return _as_credit_count(body)


def start(profile_url: str) -> Lookup:
    """Fire one single search and hand back Enrow's handle for it.

    **Only the profile URL is sent.** The API accepts a name and a company too, and
    resolves better with them, but the query stays as narrow as the other adapters' for
    the same reason: the less of a lead's record leaves for a third party, the better.
    Their docs treat ``linkedin_url`` as sufficient on its own.
    """
    api_key = _require_key()
    with _session(api_key) as session:
        try:
            body = _request(
                session, "POST", _SINGLE_URL, json={"linkedin_url": profile_url}
            ).json()
        except (requests.RequestException, TimeoutError) as exc:
            raise EnrowUnavailable(f"Enrow unreachable: {exc}") from exc
    request_id = _clean(body.get("id"))
    if not request_id:
        raise EnrowUnavailable("Enrow returned no search id")
    return Lookup(request_id=request_id)


def poll_once(request_id: str) -> PollOutcome:
    """Check one search once. A completed search is a hit or a miss; else still running."""
    api_key = _require_key()
    url = f"{_SINGLE_URL}?id={request_id}"
    with _session(api_key) as session:
        try:
            response = _request(session, "GET", url)
        except (requests.RequestException, TimeoutError) as exc:
            raise EnrowUnavailable(f"Enrow unreachable: {exc}") from exc
        # A running search answers 202 with the same body a completed one carries, so the
        # status code is the cheapest signal and the qualification is the confirmation.
        if response.status_code == 202:
            return PollOutcome(running=True)
        try:
            body = response.json()
        except ValueError as exc:
            raise EnrowUnavailable("Enrow returned a response that is not JSON") from exc
    return _outcome(body)


def _outcome(body: dict) -> PollOutcome:
    """Turn one search document into a hit, a miss, or a still-running outcome.

    Only ``valid`` with an address is a hit and only ``invalid`` is a miss. Anything
    else, including a status this adapter does not recognise, reads as still running: an
    unrecognised answer must never be recorded as a lead with no address, because the
    deal would then settle as NO_EMAIL_FOUND and never be retried.
    """
    linkedin_status = _clean(body.get("linkedin_status"))
    if linkedin_status in _TRANSIENT_LINKEDIN_STATUSES:
        raise EnrowUnavailable(
            f"Enrow could not read the LinkedIn profile yet ({linkedin_status})"
        )

    qualification = _clean(body.get("qualification"))
    if qualification == _RUNNING:
        return PollOutcome(running=True)
    if qualification == _FOUND:
        email = _clean(body.get("email"))
        if not email:
            return PollOutcome(running=True)
        info = body.get("info") or {}
        return PollOutcome(
            running=False,
            email=email,
            first_name=_clean(info.get("firstname")),
            last_name=_clean(info.get("lastname")),
        )
    if qualification == _NOT_FOUND:
        return PollOutcome(running=False)
    return PollOutcome(running=True)


def _clean(value) -> str | None:
    """A stripped string, or None when the field is absent or blank."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_credit_count(body: dict) -> int:
    """Read ``credits`` as a whole number, flooring any fraction.

    Enrow answers with an integer today, but the same reader BetterContact and LeadMagic
    use keeps one shape for all three, and a float in a JSON number is the kind of thing
    an API adds without warning.
    """
    raw = body.get("credits")
    try:
        credits = int(float(raw))
    except (TypeError, ValueError):
        raise EnrowUnavailable(f"Enrow returned no credit count: {body!r}") from None
    if credits < 0:
        raise EnrowUnavailable(f"Enrow returned a negative balance: {body!r}")
    return credits


def _require_key() -> str:
    from openoutfind.core.config import SiteConfig

    api_key = SiteConfig.load().enrow_api_key
    if not api_key:
        raise EnrowUnavailable("no Enrow API key configured")
    return api_key


def _session(api_key: str) -> requests.Session:
    """A session with Enrow's key and 429 backoff mounted on it.

    Enrow authenticates with ``x-api-key`` on every request and asks for exponential
    backoff on a 429, so the retry lives in the transport where it belongs rather than in
    a hand-rolled loop here.
    """
    session = requests.Session()
    session.headers.update({"x-api-key": api_key, "Accept": "application/json"})
    adapter = HTTPAdapter(max_retries=_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _request(session: requests.Session, method: str, url: str, **kwargs):
    """One HTTP call, with the budget spent and errors typed at the boundary."""
    kwargs.setdefault("timeout", _HTTP_TIMEOUT_S)
    # The account read is free per Enrow's docs and outside their limiter, so it does
    # not spend the budget the way a search does.
    if url != _ACCOUNT_URL:
        _BUDGET.take()
    try:
        response = session.request(method, url, **kwargs)
    except requests.exceptions.RetryError as exc:
        raise EnrowUnavailable(
            "Enrow rate limit not cleared after retries",
            ErrorType.PROVIDER_RATE_LIMITED,
        ) from exc
    if response.status_code == 401:
        raise EnrowUnavailable("Enrow rejected the API key", ErrorType.PROVIDER_AUTH)
    if response.status_code == 402:
        raise EnrowUnavailable(
            "Enrow reports insufficient credits", ErrorType.PROVIDER_OUT_OF_CREDITS
        )
    # The API never answers 404: an unknown or expired id comes back as a 500 on the
    # single endpoints, which is indistinguishable from a transient server error. Both
    # defer the lookup rather than settle it as a miss.
    if response.status_code >= 500:
        raise EnrowUnavailable(
            f"Enrow returned {response.status_code}", ErrorType.PROVIDER_UNAVAILABLE
        )
    response.raise_for_status()
    return response
