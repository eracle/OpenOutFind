"""FullEnrich: the paid finder's second waterfall, behind the same seam as BetterContact.

Same two-leg handshake: ``start`` fires one bulk job and hands back FullEnrich's
``enrichment_id``; ``poll_once`` checks it once and returns a ``PollOutcome``. Nothing
here blocks on a poll: the collect step owns the wait, on the deal's own ``not_before``,
so a run parked at FINDING_EMAIL costs this process nothing.

Why a second adapter at all: the seam's job is to pick the finder that is cheapest on
the operator's plan (``provider.py``), and the two waterfalls quote different prices per
credit at different tiers: FullEnrich is cheaper at entry and BetterContact is cheaper
at the top, so which one wins flips with volume. That is a pricing decision, not an
architecture change, which is the point of the seam.

Discovery is **not** here. Lead Finder is BetterContact's free index and stays on
BetterContact's key, so a FullEnrich-only install can resolve an address but cannot
search, by decision (see ``REQUIRED_DISCOVERY_FIELDS`` in ``core/config.py``).
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

NAME = "fullenrich"

# The one path to an account. FullEnrich runs no affiliate redirect of its own yet, so
# this is their plain signup page rather than a parameterised one: inventing a tracking
# query here would be a link that claims an attribution nobody granted.
SIGNUP_URL = "https://fullenrich.com"

# One credit on the entry plan ($55/month for 1,000 credits), which is what an operator
# starting today pays: the per-credit price falls on larger plans, so this is an anchor
# for the cheapest-first pick and not a claim about volume. Kept beside the adapter, as
# BetterContact's is, so the selector reads it from the provider rather than a table
# that goes stale without the vendor.
UNIT_COST_USD = 0.055

_BULK_URL = "https://app.fullenrich.com/api/v2/contact/enrich/bulk"

# One HTTP call's ceiling, not a job's. A bulk job walks FullEnrich's own waterfall for
# minutes, so the gap between polls belongs to the collect step's cadence rather than
# to a constant here.
_HTTP_TIMEOUT_S = 30

# FullEnrich publishes no per-minute figure on the API surface we can verify (their
# rate limit shows up as a job status, `RATE_LIMIT`), so this is our own courtesy
# budget rather than a quoted limit: the same 15/minute BetterContact's key lives
# under, per process, so six concurrent runs stay well inside anything either vendor
# would call abuse. It is a share of the key, not a measurement of the account.
REQUESTS_PER_MINUTE = 15
_BUDGET_WINDOW_S = 60.0
_RATE_LIMIT_ATTEMPTS = 5

# The statuses that actually mean "this address is one you can mail". FullEnrich's
# `CATCH_ALL` is **not** here on purpose: it only says the domain accepts everything,
# which is a domain-level fact with no claim about the mailbox, whereas BetterContact's
# `catch_all_safe` is their assertion that a specific mailbox was safely verified. The
# two providers' usable sets therefore are not interchangeable, and treating them as
# if they were would import unverified addresses into someone's sender reputation.
_USABLE_STATUSES = frozenset({"DELIVERABLE", "HIGH_PROBABILITY"})

_RUNNING_STATUSES = frozenset({"CREATED", "IN_PROGRESS"})

_RETRY = Retry(
    total=_RATE_LIMIT_ATTEMPTS,
    status_forcelist=(429,),
    allowed_methods=frozenset({"GET", "POST"}),
    backoff_factor=5,
    backoff_max=120,
    respect_retry_after_header=True,
)

# The limiter is provider-agnostic and already tested where BetterContact defines it;
# FullEnrich gets its own instance because it is a different key on a different account,
# and labels its log line with its own name.
_BUDGET = RequestBudget(REQUESTS_PER_MINUTE, _BUDGET_WINDOW_S, label="fullenrich")


class FullEnrichUnavailable(ProviderUnavailable):
    """FullEnrich could not run: no API key configured, or the service was unreachable.

    Distinct from a genuine miss (it ran, found no usable work email), exactly as
    ``BetterContactUnavailable`` is. Carries a stable ``error_type`` from
    ``core.errors.ErrorType``; the default is ``provider_unavailable`` and the HTTP and
    job-status layers narrow it to auth, out-of-credits or rate-limited.

    ``lookup.py`` catches the base class and never names a vendor, so this name exists
    for the tests and for a reader, not for a caller to branch on.
    """


def start(profile_url: str) -> Lookup:
    """Fire one lookup job and hand back its handle: the async half of the interface.

    Never terminates here: a bulk job runs FullEnrich's waterfall, so the deal parks at
    FINDING_EMAIL on the ``enrichment_id`` and ``poll_once`` finishes it later.

    **Only the profile URL is sent.** The API accepts a name and a company too, and
    resolves better with them, but the query stays as narrow as BetterContact's for the
    same reason: the less of a lead's record leaves for a third party, the better. Their
    docs list ``professional_network_url`` as sufficient on its own.
    """
    return Lookup(request_id=submit(profile_url))


def is_configured() -> bool:
    """True when the FullEnrich paid finder is configured (an API key is set)."""
    from openoutfind.core.config import SiteConfig

    return bool(SiteConfig.load().fullenrich_api_key)


def credit_balance() -> int:
    """FullEnrich exposes no balance endpoint, so this is not a balance of zero.

    BetterContact answers ``GET /account`` with ``credits_left``; FullEnrich's documented
    surface reports only what a finished job cost (``cost.credits`` on the enrichment
    itself), and no credits, account or team read exists to ask (checked 2026-11-11
    against their OpenAPI document). The seam's contract is *raise rather than guess*, so
    the run reports why it cannot see a balance instead of reading it as zero and sending
    the operator to top up an account that is already funded.
    """
    raise FullEnrichUnavailable(
        "FullEnrich does not expose a credit balance through its API; "
        "check the account dashboard"
    )


def submit(profile_url: str) -> str:
    """One bulk job for one profile URL; returns FullEnrich's own job id."""
    api_key = _require_key()
    with _session(api_key) as session:
        try:
            payload = _request(
                session, "POST", _BULK_URL, json=_enrich_body(profile_url)
            ).json()
        except (requests.RequestException, TimeoutError) as exc:
            raise FullEnrichUnavailable(f"FullEnrich unreachable: {exc}") from exc
    enrichment_id = payload.get("enrichment_id")
    if not enrichment_id:
        raise FullEnrichUnavailable("FullEnrich returned no enrichment id")
    return enrichment_id


def poll_once(enrichment_id: str) -> PollOutcome:
    """Ask once. No wait, no retry loop: the collect step decides when to ask again.

    FullEnrich runs a bulk job in minutes, so the gap between asks is the engine's
    cadence rather than anything here. Returns ``running`` for a job FullEnrich has not
    finished, a hit when it has and the address is one it vouched for, and a miss when
    it has and it is not. A finished job that says the wallet was empty or the client
    was throttled is a refusal, not a result, and raises: neither may ever be recorded
    as an unreachable lead.
    """
    api_key = _require_key()
    url = f"{_BULK_URL}/{enrichment_id}"
    with _session(api_key) as session:
        try:
            body = _request(session, "GET", url).json()
        except (requests.RequestException, TimeoutError) as exc:
            raise FullEnrichUnavailable(f"FullEnrich unreachable: {exc}") from exc
    return _outcome(body)


def _enrich_body(profile_url: str) -> dict:
    """The smallest bulk body that asks for a work email.

    ``enrich_fields`` is required and closed: asking for nothing parses to
    ``error.enrichment.enrich_fields.empty``, and only ``contact.work_emails`` is wanted,
    since a personal address costs three credits and a phone ten for a field this tool
    does not send to. The job name is a label that makes the account's own log readable.
    """
    return {
        "name": "outfind",
        "data": [
            {
                "professional_network_url": profile_url,
                "enrich_fields": ["contact.work_emails"],
            }
        ],
    }


def _outcome(body: dict) -> PollOutcome:
    """Turn one status document into the seam's three answers."""
    status = body.get("status")

    if status in _RUNNING_STATUSES:
        return PollOutcome(running=True)
    if status == "RATE_LIMIT":
        raise FullEnrichUnavailable(
            "FullEnrich rate-limited the job", ErrorType.PROVIDER_RATE_LIMITED
        )
    if status == "CREDITS_INSUFFICIENT":
        raise FullEnrichUnavailable(
            "FullEnrich account is out of credits", ErrorType.PROVIDER_OUT_OF_CREDITS
        )
    if status == "CANCELED":
        return PollOutcome(running=False)
    if status != "FINISHED":
        # `UNKNOWN` is FullEnrich's own catch-all, and an unrecognised value is the same
        # kind of answer: the job exists but its state was not resolved. Read as still
        # running, as BetterContact's non-terminated states are, so a transient
        # server-side answer cannot be recorded as a lead with no address.
        return PollOutcome(running=True)

    row = (body.get("data") or [{}])[0]
    email = _work_email(row.get("contact_info") or {})
    if not email:
        return PollOutcome(running=False)
    given = row.get("input") or {}
    return PollOutcome(
        running=False,
        email=email,
        first_name=given.get("first_name") or None,
        last_name=given.get("last_name") or None,
    )


def _work_email(contact: dict) -> str:
    """The first work address FullEnrich vouched for, or an empty string.

    The list is the evidence: each entry carries its own status. ``most_probable_work_email``
    is read as a fallback because the documented response shape lets it be either the
    address itself or the same object, and a finished job that returned an address only
    there is still an answer we paid for.
    """
    for candidate in (
        *(contact.get("work_emails") or ()),
        contact.get("most_probable_work_email"),
    ):
        address = _usable(candidate)
        if address:
            return address
    return ""


def _usable(candidate: object) -> str:
    """The address when the entry vouched for it. A bare string carries no status.

    A bare ``most_probable_work_email`` is the one exception: it is FullEnrich's own
    most-probable address for a finished job, and with no status beside it there is
    nothing to check it against, so discarding it would spend a credit and throw the
    answer away.
    """
    if isinstance(candidate, dict):
        address = candidate.get("email")
        if address and candidate.get("status") in _USABLE_STATUSES:
            return address
        return ""
    return candidate if isinstance(candidate, str) else ""


def _require_key() -> str:
    from openoutfind.core.config import SiteConfig

    api_key = SiteConfig.load().fullenrich_api_key
    if not api_key:
        raise FullEnrichUnavailable("no FullEnrich API key configured")
    return api_key


def _session(api_key: str) -> requests.Session:
    """A session that backs off on 429 before any of our code sees the response."""
    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    )
    adapter = HTTPAdapter(max_retries=_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _request(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    """One HTTP call, with the answers FullEnrich actually gives.

    The 429 backoff lives in the transport adapter, not here: a hand-rolled loop would be
    a worse copy of what ``requests`` and ``urllib3`` already do. What is left is naming
    the refusals a reader has to tell apart: a bad key and an empty wallet are final
    answers, not things to retry.

    Any other error status is raised as ``HTTPError`` and surfaces as unavailable at the
    seam. Without that, a 500 body carries no ``status``, ``_outcome`` reads it as a job
    still running, and the lead polls a broken job forever.
    """
    kwargs.setdefault("timeout", _HTTP_TIMEOUT_S)
    _BUDGET.take()
    try:
        resp = session.request(method, url, **kwargs)
    except requests.exceptions.RetryError as exc:
        raise FullEnrichUnavailable(
            f"FullEnrich rate-limited this client through {_RATE_LIMIT_ATTEMPTS} "
            "backed-off attempts",
            ErrorType.PROVIDER_RATE_LIMITED,
        ) from exc
    if resp.status_code == 401:
        raise FullEnrichUnavailable(
            "FullEnrich rejected the API key", ErrorType.PROVIDER_AUTH
        )
    if resp.status_code == 402:
        raise FullEnrichUnavailable(
            "FullEnrich account is out of credits", ErrorType.PROVIDER_OUT_OF_CREDITS
        )
    resp.raise_for_status()
    return resp
