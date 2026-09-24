# openoutfind/enrichment/provider.py
"""The email-finder seam: one interface, one configured provider, two transports.

``lookup.py`` drives *a* provider, never a named one. Which module answers is decided
here by ``active()`` — today that is BetterContact whenever its key is set, the only
finder upstream ships.

**The interface spans sync and async, because finders differ.** BetterContact is a
waterfall that takes seconds to minutes: ``start`` fires a job and returns a handle, the
deal parks at FINDING_EMAIL, and ``poll_once`` checks it later. A match-style finder
answers in the same HTTP call. Forcing either into the other's shape would cost
something real — a fake handle would invent a poll that resolves instantly and a state
the deal passes through in microseconds, while blocking on a waterfall would hold a run
open for minutes per lead. So ``start`` returns a ``Lookup`` that is *either* a finished
``PollOutcome`` or a handle to poll, and the caller branches once on which it got.

A provider module is a plain module — no classes to register, no plugin table. It
implements:

    NAME            str, the wire value for the hub's contribution origin
    SIGNUP_URL      str, the one attributed path to an account
    is_configured() bool — is there a key
    credit_balance() int — raises ProviderUnavailable if it cannot be read
    start(url)      Lookup — resolved outright, or a handle
    poll_once(id)   PollOutcome — async providers only; sync ones never see a call

**Adding a provider (including in a fork).** Write the module against the six names
above, add it to ``_providers()`` below and give it a key in ``core/config.py`` — that
is the whole integration; nothing else in the codebase names a vendor. With more than
one key set, the first configured entry in ``_providers()`` wins.

``NAME`` is the part that leaves the machine. ``lookup`` passes it straight to
``contacts.contribute`` as the give-back's ``origin``, and it travels to the hub as
that **string**, never as a number — so pick a short lowercase slug for your vendor
(``"hunter"``, ``"dropcontact"``) and keep it stable, because it is what your
contributions are grouped by forever after. The hub stores it verbatim whether or not
it has ever heard of you, so a fork's own finder stays legible on the analytics side
with no change needed there; only the small-int fast path in the hub's
``Contribution.Origin`` is limited to the vendors upstream ships. Do not reuse another
vendor's ``NAME`` to "fit in" — that silently merges your spend into theirs.
"""
from __future__ import annotations

from dataclasses import dataclass

from openoutfind.core.errors import ErrorType


class ProviderUnavailable(Exception):
    """The finder could not run — no key, or the service was unreachable.

    Distinct from a genuine miss (it ran and found no address), which is a terminal
    answer about the lead rather than about the provider. Carries a stable
    ``error_type`` from ``core.errors.ErrorType`` so a caller can tell *why* without
    matching on the message.
    """

    def __init__(self, message: str, error_type: str = ErrorType.PROVIDER_UNAVAILABLE) -> None:
        self.error_type = error_type
        super().__init__(message)


@dataclass(frozen=True)
class PollOutcome:
    """A lookup's state at one moment: still running, or terminated hit/miss.

    A hit carries the name parts the provider resolved alongside the address. The
    provider returns them in the same response that carries the email — same call, same
    credit — which is why nothing in this codebase splits a full name itself.
    """
    running: bool
    email: str = ""
    first_name: str | None = None
    last_name: str | None = None

    @property
    def hit(self) -> bool:
        return not self.running and bool(self.email)

    @property
    def miss(self) -> bool:
        return not self.running and not self.email


@dataclass(frozen=True)
class Lookup:
    """What ``start`` returned: a finished answer, or a handle to poll.

    Exactly one is set. ``outcome`` means the provider answered synchronously and the
    deal never touches FINDING_EMAIL; ``request_id`` means a job is in flight and the
    deal parks on the handle.
    """
    outcome: PollOutcome | None = None
    request_id: str = ""

    @property
    def pending(self) -> bool:
        return self.outcome is None


def _providers() -> tuple:
    """Every finder upstream ships, in the order ``active`` prefers them."""
    from openoutfind.enrichment import bettercontact

    return (bettercontact,)


def active():
    """The provider module this install resolves with, or ``None`` if none is configured."""
    return next((module for module in _providers() if module.is_configured()), None)


def by_name(name: str):
    """The provider module called *name*, or ``None``.

    Used to poll a job with the vendor that issued its handle rather than with whichever
    vendor is configured *now*.
    """
    return next((module for module in _providers() if module.NAME == name), None)


def configured() -> list:
    """Every provider module with a key set — for ``status``, which reports on all of
    them rather than only the one that would run."""
    return [module for module in _providers() if module.is_configured()]
