# openoutfind/core/readiness.py
"""What a run has to be given before it can find anybody.

Three things, and none of them is a preference. This install has to say what it sells
and to whom, or there is no ICP to search or judge against. A model has to be reachable,
or there is nothing to judge with. And discovery has to have a key, because the search
itself runs on one.

The operator's email and country are not among them. The email gives the install a hub
identity and names the operator row; without it the row is ``operator`` and the hub is
never asked. The country only narrows the contacts-store give-back
(`contacts/service.py:contribute`), and a run that declares none contributes.

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

**The operator is a record, not an answer.** It is a ``User`` row, written once from the
environment — a renamed variable must not rename the person a campaign belongs to.
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
)
from openoutfind.core.errors import ErrorType, OpenOutFindError

logger = logging.getLogger(__name__)

OPERATOR_EMAIL = ENV_PREFIX + "OPERATOR_EMAIL"

# The groups a run is checked in, in the order somebody would go and find the values.
# They are how the failure reads, not how it is enforced — every one is variables.
GROUPS = ("campaign", "llm", "bettercontact")


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
            f"{OPERATOR_EMAIL}, {ENV_PREFIX}OPERATOR_COUNTRY.",
        )

    if not _agent_qualify_active():
        _check_llm()
    _ensure_operator()


def missing_variables() -> dict[str, list[str]]:
    """Each unsatisfied group mapped to the variables that would satisfy it.

    The groups are the questions a person is asked, kept because that is how the failure
    reads to somebody who has to go and find the values. An empty dict is a ready run.
    """
    config = SiteConfig.load()
    groups = {
        "campaign": missing(config, REQUIRED_ICP_FIELDS),
        # `--agent-qualify` never calls AI_MODEL for a verdict — the whole point is a
        # calling agent answering instead of a second, separately-keyed model — so an
        # install running only that way is not asked for a key it will never spend.
        "llm": [] if _agent_qualify_active() else missing(config, REQUIRED_LLM_FIELDS),
        "bettercontact": missing(config, REQUIRED_DISCOVERY_FIELDS),
    }
    return {group: names for group, names in groups.items() if names}


def _agent_qualify_active() -> bool:
    from openoutfind.core import agent_qualify

    return agent_qualify.active()


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

    This is identity, and the Django ``User`` row is what the rest of the codebase reads
    — the contacts-store key and the seller name the agents write as. Both children share
    it under one registry when OpenOutreach hosts them.

    The email is optional. Without one the operator is a row named ``operator`` and the
    install has no hub identity. An email given later fills that blank once; an email
    already on the row is never replaced, since a renamed variable must not rename the
    person a campaign belongs to.
    """
    import os

    from openoutfind.contacts.service import register_operator
    from openoutfind.core.operator import get_active_user

    email =(os.environ.get(OPERATOR_EMAIL) or "").strip()
    user = get_active_user()

    if user is None:
        user = _create_operator(email)
        logger.info("running as %s", user.username)
    elif email and not user.email:
        user.email = email
        user.save(update_fields=["email"])
    else:
        return

    # Identity, not entitlement: the hub token names this install so it can hold a
    # balance, be metered and be revoked. Minted **regardless of jurisdiction** — the
    # EEA/UK/CH rule governs contributing records, which is a different act. Best-effort:
    # a hub that is down leaves the run without one, and the first contribution mints it
    # the old way.
    if user.email:
        register_operator()


def _create_operator(email: str):
    """Create the operator Django ``User``, named from their email when there is one."""
    from django.contrib.auth.models import User

    handle = email.split("@")[0].lower().replace(".", "_").replace("+", "_") or "operator"
    user, created = User.objects.get_or_create(
        username=handle,
        defaults={"is_staff": True, "is_active": True, "email": email},
    )
    if created:
        user.set_unusable_password()
        user.save()
    return user
