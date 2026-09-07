# openoutfind/core/errors.py
"""The error vocabulary — one set of stable strings serving three jobs.

They tell the operator the truth, give an agent something to branch on, and fire the
funnel. Stable means: a value here is part of the CLI's contract, so it may be added
to but not renamed.

The rule these exist to enforce is that **nothing may be reported as an empty
result**. A throttled or unauthorised run that says *"no leads found for your
product"* is the worst failure this product can have, because the reader concludes
the tool does not work and nobody reports it.
"""
from __future__ import annotations


class ErrorType:
    """Stable ``type`` strings for ``error: <type>: <message>``."""

    NO_CREDENTIAL = "no_credential"
    """No BetterContact key configured."""

    PROVIDER_AUTH = "provider_auth"
    """401 — the key is invalid or missing."""

    PROVIDER_OUT_OF_CREDITS = "provider_out_of_credits"
    """402, or a balance of zero — credits exhausted."""

    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    """429 — back off; never retry at speed, their docs say that can block the account."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    """The provider could not be reached at all — distinct from any answer it gave."""

    BAD_CONFIG = "bad_config"
    """A configuration value is set but unusable."""

    ONBOARDING_INCOMPLETE = "onboarding_incomplete"
    """Onboarding cannot finish from what is configured, and no TTY is available."""

    NOT_INITIALIZED = "not_initialized"
    """The database has no schema yet — nothing has run here. Only ``find`` creates it."""

    GOAL_UNREACHED = "goal_unreached"
    """A job stopped short of what was asked for. Its rows are still on stdout.

    Exit 0 means *I got what you asked for* and nothing else, which is the property that
    lets a caller branch without parsing. The message carries what it is short by and
    what it was waiting on — a drained index and three addresses on order are a dead end
    and a reason to run again in an hour, and "7 of 10" alone cannot tell them apart.
    """

    QUALIFY_PENDING = "qualify_pending"
    """``find --agent-qualify`` stopped one candidate short of the LLM call it opted out
    of. Its rows so far are still on stdout; the candidate waiting for a verdict rides
    the error object's own fields (``profile_text``, ``company``, ``job_title``), same
    place an agent already reads a refusal from. Resume with the same command plus
    ``--verdict``/``--reason``.
    """


def format_error(error_type: str, message: str) -> str:
    """The one error line every verb writes: ``error: <type>: <message>``."""
    return f"error: {error_type}: {message}"


class OpenOutFindError(Exception):
    """A failure with a stable ``type``, rendered as the contract's error line.

    Deliberately **not** a ``CommandError``: Django catches that one and prefixes it
    with the exception's class name, which would put noise in front of the line an
    agent parses. ``OpenOutFindCommand`` catches this instead and writes it verbatim.

    ``payload`` carries whatever extra fields a caller needs to *act* on this
    particular error rather than merely report it — today only ``qualify_pending``'s
    candidate. It rides alongside ``type``/``message`` in the ``--json`` object and is
    absent from everything else, including the plain-text line.
    """

    def __init__(self, error_type: str, message: str, payload: dict | None = None) -> None:
        self.error_type = error_type
        self.message = message
        self.payload = payload or {}
        super().__init__(format_error(error_type, message))
