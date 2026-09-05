# openoutfind/core/readiness.py
"""What a run has to be given before it can find anybody.

Four things, and none of them is a preference. This install has to say what it sells and
to whom, or there is no ICP to search or judge against. A model has to be reachable, or
there is nothing to judge with. Discovery has to have a key, because the search itself
runs on one. And somebody has to be running it, under a jurisdiction, having accepted
what the tool does.

**Everything comes from the environment. Nothing is asked.** This program is a library,
a pipe stage and a scripted command as often as it is something a person types at, and
none of those can answer a prompt. So a run that is missing something raises one error
naming every variable that would have satisfied it, rather than blocking on a question
nobody is there to answer. An operator who wants to be asked runs the wizard in
OpenOutreach, which owns the human half and exports these names.

**The model is checked, not stored.** One ping before any work starts is what makes
reading configuration fresh on every run safe: a key rotated out from under a timer fails
here, before a lead is chosen, instead of halfway through a pass with a lead already in
hand. That was the whole objection to reading config from the environment, and it is
answered by checking rather than by remembering.

**Two things are records, not answers.** The operator is a ``User`` row, written once
from the environment and never re-read — a renamed variable must not rename the person a
campaign belongs to. The newsletter subscription is an act, performed once when that row
is created, and only on an explicit yes.
"""
from __future__ import annotations

import logging

from openoutfind.core.config import (
    REQUIRED_DISCOVERY_FIELDS,
    REQUIRED_ICP_FIELDS,
    REQUIRED_LLM_FIELDS,
    ENV_PREFIX,
    SiteConfig,
    missing,
    variable_for,
)
from openoutfind.core.errors import ErrorType, OpenOutFindError

logger = logging.getLogger(__name__)

# The canonical Legal Notice — the single source of truth for what OpenOutFind does with
# the people it finds. Named in the error rather than rendered: this is a command, not a
# page of reflowed Markdown.
LEGAL_NOTICE_URL = "https://github.com/eracle/OpenOutFind/blob/main/LEGAL_NOTICE.md"

OPERATOR_EMAIL = ENV_PREFIX + "OPERATOR_EMAIL"
ACCEPT_LEGAL_NOTICE = ENV_PREFIX + "ACCEPT_LEGAL_NOTICE"
NEWSLETTER = ENV_PREFIX + "NEWSLETTER"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

# The groups a run is checked in, in the order somebody would go and find the values.
# They are how the failure reads, not how it is enforced — every one is variables.
GROUPS = ("campaign", "llm", "bettercontact", "account")


def check_ready() -> None:
    """Verify this run has everything finding needs, or stop naming what would give it.

    One error at the end rather than one per round trip: a run missing three things
    should learn all three from a single failure.
    """
    unsatisfied = [name for names in missing_variables().values() for name in names]
    if unsatisfied:
        raise OpenOutFindError(
            ErrorType.ONBOARDING_INCOMPLETE,
            "not ready to find — set " + ", ".join(unsatisfied) + ".\n"
            f"Optional: {ENV_PREFIX}LLM_API_BASE (required for openai_compatible:*), "
            f"{ENV_PREFIX}APOLLO_API_KEY, {NEWSLETTER}.\n"
            f"{ACCEPT_LEGAL_NOTICE} must be set to 'true' — it records that you accept "
            f"{LEGAL_NOTICE_URL}.",
        )

    _check_llm()
    _ensure_operator()


def missing_variables() -> dict[str, list[str]]:
    """Each unsatisfied group mapped to the variables that would satisfy it.

    The groups are the questions a person is asked, kept because that is how the failure
    reads to somebody who has to go and find four values. An empty dict is a ready run.
    """
    from django.contrib.auth.models import User

    config = SiteConfig.load()
    groups = {
        "campaign": missing(config, REQUIRED_ICP_FIELDS),
        "llm": missing(config, REQUIRED_LLM_FIELDS),
        "bettercontact": missing(config, REQUIRED_DISCOVERY_FIELDS),
        "account": _account_missing(config, User.objects.filter(
            is_active=True, is_staff=True).exclude(email="").exists()),
    }
    return {group: names for group, names in groups.items() if names}


def _account_missing(config: SiteConfig, operator_exists: bool) -> list[str]:
    """What the account group still lacks.

    **Acceptance is never inferred**, and it is asked for on every run whether or not an
    operator row exists: the variable has to say yes, so an install cannot inherit
    somebody else's agreement by inheriting their database. The email is only asked for
    while there is no operator — after that it is a row, and the row is the identity.
    """
    import os

    names = [] if (operator_exists or os.environ.get(OPERATOR_EMAIL, "").strip()) \
        else [OPERATOR_EMAIL]
    if not config.operator_country_code:
        names.append(variable_for("operator_country_code"))
    if not _flag(ACCEPT_LEGAL_NOTICE):
        names.append(ACCEPT_LEGAL_NOTICE)
    return names


# ── the model ─────────────────────────────────────────────────────


def _check_llm() -> None:
    """Confirm the model will answer to the key it was given.

    The ping costs a round trip and runs on every pass, which is the price of not storing
    the answer. It buys what storing never did: the key is known to work before the run
    spends a credit, not after.
    """
    from openoutfind.core.llm import verify_llm_credentials

    config = SiteConfig.load()
    refused = verify_llm_credentials(config.ai_model, config.llm_api_key, config.llm_api_base)
    if refused:
        raise OpenOutFindError(
            ErrorType.BAD_CONFIG, f"{config.ai_model} refused these credentials: {refused}")

    logger.info("judging with %s", config.ai_model)


# ── the operator ──────────────────────────────────────────────────


def _ensure_operator() -> None:
    """Record who runs this install, once.

    Skipped once an operator exists: this is identity, and the Django ``User`` row is
    what the rest of the codebase reads — the contacts-store key, the seller name the
    agents write as, and the newsletter target. Both children share it under one
    registry when OpenOutreach hosts them.
    """
    import os

    from openoutfind.contacts.service import register_operator
    from openoutfind.core.newsletter import subscribe_to_newsletter
    from openoutfind.core.operator import get_active_user

    if get_active_user() is not None:
        return

    email = (os.environ.get(OPERATOR_EMAIL) or "").strip()
    user = _create_operator(email)
    logger.info("running as %s", user.username)

    if _flag(NEWSLETTER):
        subscribe_to_newsletter(email)

    # Identity, not entitlement, and not consent: the hub token names this install so it
    # can hold a balance, be metered and be revoked. Minted here because the email is
    # already in hand, and **regardless of jurisdiction** — the EEA/UK/CH rule governs
    # contributing records, which is a different act. Best-effort: a hub that is down
    # leaves the run without one, and the first contribution mints it the old way.
    register_operator()


def _create_operator(email: str):
    """Create the operator Django ``User`` from their email (the human's own inbox)."""
    from django.contrib.auth.models import User

    handle = email.split("@")[0].lower().replace(".", "_").replace("+", "_")
    user, created = User.objects.get_or_create(
        username=handle,
        defaults={"is_staff": True, "is_active": True, "email": email},
    )
    if created:
        user.set_unusable_password()
        user.save()
    return user


def _flag(variable: str) -> bool:
    """Read a yes/no variable, rejecting anything that is not plainly one or the other.

    A bad value is a different thing from an absent one: absent means *not given*, bad
    means *stop and say so*. Falling through to "missing" would name a variable the
    operator has already set.
    """
    import os

    raw = (os.environ.get(variable) or "").strip().lower()
    if not raw:
        return False
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise OpenOutFindError(
        ErrorType.BAD_CONFIG,
        f"{variable}: expected one of {sorted(_TRUE | _FALSE)}, got {raw!r}")
