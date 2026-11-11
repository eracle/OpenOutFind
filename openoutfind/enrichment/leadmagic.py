"""LeadMagic: the paid finder that answers in a single call.

The other two adapters speak a two-leg handshake because their APIs run a job: ``start``
returns a handle and ``poll_once`` finishes it later. LeadMagic's email finder is
synchronous, so ``start`` is the whole lookup and returns a terminated ``PollOutcome``.
The seam already allows that (``Lookup.outcome`` rather than ``Lookup.request_id``), so
there is no ``poll_once`` here at all: a hit lands on RESOLVED without the deal ever
entering FINDING_EMAIL, and nothing ever parks on a handle that a vendor could strand.

Why a third adapter: the seam's job is to pick the finder that is cheapest on the
operator's plan (``provider.py``), and every vendor quotes a different price per credit
at a different tier. LeadMagic answers in one call and undercuts both waterfalls at
entry ($99 buys 5,000 credits), though Enrow's entry plan is cheaper again. That is a
pricing decision, not an architecture change, which is the point of the seam.

Discovery is **not** here either. Lead Finder is BetterContact's free index and stays on
BetterContact's key, so a LeadMagic-only install can resolve an address but cannot
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

NAME = "leadmagic"

# The one path to an account. LeadMagic runs no affiliate redirect of its own, so this is
# their plain signup page rather than a parameterised one: inventing a tracking query here
# would be a link that claims an attribution nobody granted.
SIGNUP_URL = "https://leadmagic.io"

# One credit on the entry plan ($99/month for 5,000 credits), which is what an operator
# starting today pays: the per-credit price falls on larger plans, so this is an anchor
# for the cheapest-first pick and not a claim about volume. Kept beside the adapter, as
# the other two are, so the selector reads it from the provider rather than a table that
# goes stale without the vendor.
UNIT_COST_USD = 0.0198

_FINDER_URL = "https://api.leadmagic.io/v1/people/email-finder"

# The balance read, and the one call the request budget does not count. LeadMagic
# documents it as free and outside the rate limiter, which is what lets ``status``
# ask for it without spending the operator's allowance.
_CREDITS_URL = "https://api.leadmagic.io/v1/credits"

# One HTTP call's ceiling. A sync finder answers or refuses inside a single request, so
# there is no job clock to hand out.
_HTTP_TIMEOUT_S = 30

# LeadMagic publishes a per-plan rate limit (300 requests/minute on the entry plan) but
# that allowance is per organisation and shared by every key and credit bank, so a process
# taking the whole of it would starve the operator's other tools. This is our courtesy
# fifth of the published figure, per process, and it is deliberately not presented as a
# vendor number: the limiter is a share of the key, not a measurement of the account.
REQUESTS_PER_MINUTE = 60
_BUDGET_WINDOW_S = 60.0
_RATE_LIMIT_ATTEMPTS = 5

# LeadMagic's own word for an address its pipeline verified. Anything else (including
# ``null``) is a miss, and the vendor bills a miss at zero credits, so recording it as a
# miss costs the operator nothing.
_VALID_STATUS = "valid"

_RETRY = Retry(
    total=_RATE_LIMIT_ATTEMPTS,
    status_forcelist=(429,),
    allowed_methods=frozenset({"GET", "POST"}),
    backoff_factor=5,
    backoff_max=120,
    respect_retry_after_header=True,
)

# The limiter is provider-agnostic and already tested where BetterContact defines it;
# LeadMagic gets its own instance because it is a different key on a different account,
# and labels its log line with its own name.
_BUDGET = RequestBudget(REQUESTS_PER_MINUTE, _BUDGET_WINDOW_S, label="leadmagic")


class LeadMagicUnavailable(ProviderUnavailable):
    """LeadMagic could not run: no API key configured, or the service was unreachable.

    Distinct from a genuine miss (it ran, found no verified work email), exactly as
    ``BetterContactUnavailable`` is. Carries a stable ``error_type`` from
    ``core.errors.ErrorType``; the default is ``provider_unavailable`` and the HTTP layer
    narrows it to auth, out-of-credits or rate-limited.

    ``lookup.py`` catches the base class and never names a vendor, so this name exists
    for the tests and for a reader, not for a caller to branch on.
    """


def start(profile_url: str) -> Lookup:
    """Resolve one address and hand back a finished answer: the synchronous half.

    This is the whole lookup. LeadMagic's finder returns the result in the same response,
    so the deal moves straight to RESOLVED or NO_EMAIL_FOUND and never occupies a poll
    slot, which also means it can never be left holding a handle the vendor forgot.

    **Only the profile URL is sent.** The API accepts a name and a company too, and
    resolves better with them, but the query stays as narrow as the other two adapters'
    for the same reason: the less of a lead's record leaves for a third party, the
    better. Their docs list ``profile_url`` as the strongest single input, sufficient on
    its own without either name or domain.
    """
    return Lookup(outcome=find_email(profile_url))


def is_configured() -> bool:
    """True when the LeadMagic paid finder is configured (an API key is set)."""
    from openoutfind.core.config import SiteConfig

    return bool(SiteConfig.load().leadmagic_api_key)


def credit_balance() -> int:
    """The credits left across the whole organisation, as LeadMagic counts them.

    Unlike FullEnrich, LeadMagic has a balance endpoint, ``GET /v1/credits``, so this is
    a real number rather than a raised refusal and the run can warn before a lead fails
    instead of after a 402. The response is ``{"credits": 15432.50}``, a float, so it goes
    through the same floor-don't-round reading BetterContact's balance does: a fraction
    of a credit buys nothing, and rounding it up would report a balance that cannot pay
    for the next lookup.
    """
    api_key = _require_key()
    with _session(api_key) as session:
        try:
            body = _request(session, "GET", _CREDITS_URL).json()
        except (requests.RequestException, TimeoutError) as exc:
            raise LeadMagicUnavailable(f"LeadMagic unreachable: {exc}") from exc
    return _as_credit_count(body)


def find_email(profile_url: str) -> PollOutcome:
    """One synchronous lookup for one profile URL, as a terminated outcome."""
    api_key = _require_key()
    with _session(api_key) as session:
        try:
            body = _request(
                session, "POST", _FINDER_URL, json={"profile_url": profile_url}
            ).json()
        except (requests.RequestException, TimeoutError) as exc:
            raise LeadMagicUnavailable(f"LeadMagic unreachable: {exc}") from exc
    return _outcome(body)


def _outcome(body: dict) -> PollOutcome:
    """Turn one finder document into a hit or a miss. Nothing here is still running.

    ``status`` is LeadMagic's own verdict on the address: ``valid`` means the pipeline
    verified the work email, and ``null`` means it found nothing and billed nothing. A
    domain-mismatched address arrives as a free not-found with ``domain_mismatch: true``
    unless the operator opts into accepting it, so it reads as a miss here without a
    special case.

    The name parts ride the same response the email does, which is what lets a hit carry
    them into the export with no second call and no second credit.
    """
    if body.get("status") != _VALID_STATUS:
        return PollOutcome(running=False)
    email = body.get("email")
    if not isinstance(email, str) or not email:
        return PollOutcome(running=False)
    return PollOutcome(
        running=False,
        email=email,
        first_name=_clean(body.get("first_name")),
        last_name=_clean(body.get("last_name")),
    )


def _clean(value: object) -> str | None:
    """A name part as a non-empty string, or ``None`` when the vendor said nothing."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_credit_count(body: dict) -> int:
    """The credit count as an int, refusing anything that is not one.

    The provider answers with a JSON number holding a float (``15432.50``), so an int
    check alone would reject a real answer and report ``provider_unavailable`` against a
    200 that carried the number. The floor is deliberate: a fraction of a credit buys
    nothing, and a negative count is a broken answer rather than an empty wallet, so it
    is refused instead of clamped to zero.
    """
    raw = body.get("credits")
    try:
        credits = int(float(raw))
    except (TypeError, ValueError):
        raise LeadMagicUnavailable(
            f"LeadMagic returned no credit count: {body!r}"
        ) from None
    if credits < 0:
        raise LeadMagicUnavailable(
            f"LeadMagic returned a negative balance: {body!r}"
        )
    return credits


def _require_key() -> str:
    from openoutfind.core.config import SiteConfig

    api_key = SiteConfig.load().leadmagic_api_key
    if not api_key:
        raise LeadMagicUnavailable("no LeadMagic API key configured")
    return api_key


def _session(api_key: str) -> requests.Session:
    """A session that backs off on 429 before any of our code sees the response.

    No browser User-Agent is spoofed here, unlike BetterContact's Cloudflare front: a
    probe of this API with no key answered 401 with and without a Chrome UA, so there is
    no gate to work around and pretending to be a browser would only be a lie in the log.
    """
    session = requests.Session()
    session.headers.update(
        {"X-API-Key": api_key, "Accept": "application/json"}
    )
    adapter = HTTPAdapter(max_retries=_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _request(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    """One HTTP call, with the answers LeadMagic actually gives.

    The 429 backoff lives in the transport adapter, not here: a hand-rolled loop would be
    a worse copy of what ``requests`` and ``urllib3`` already do. What is left is naming
    the refusals a reader has to tell apart: a bad key and an empty wallet are final
    answers, not things to retry. LeadMagic documents 402 as insufficient credits and
    reuses it for a spent budget, so it maps to out-of-credits rather than to a generic
    failure the caller might retry.

    Any other error status is raised as ``HTTPError`` and surfaces as unavailable at the
    seam. Without that, a 500 body carries no ``status``, ``_outcome`` reads it as a miss,
    and the operator is told a reachable person has no address.

    The balance read skips the request budget because the vendor documents it as free and
    outside the limiter, the same exemption BetterContact gives its own account read.
    """
    kwargs.setdefault("timeout", _HTTP_TIMEOUT_S)
    if url != _CREDITS_URL:
        _BUDGET.take()
    try:
        resp = session.request(method, url, **kwargs)
    except requests.exceptions.RetryError as exc:
        raise LeadMagicUnavailable(
            f"LeadMagic rate-limited this client through {_RATE_LIMIT_ATTEMPTS} "
            "backed-off attempts",
            ErrorType.PROVIDER_RATE_LIMITED,
        ) from exc
    if resp.status_code == 401:
        raise LeadMagicUnavailable(
            "LeadMagic rejected the API key", ErrorType.PROVIDER_AUTH
        )
    if resp.status_code == 402:
        raise LeadMagicUnavailable(
            "LeadMagic account is out of credits", ErrorType.PROVIDER_OUT_OF_CREDITS
        )
    resp.raise_for_status()
    return resp
