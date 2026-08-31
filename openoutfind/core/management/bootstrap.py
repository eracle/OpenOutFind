# openoutfind/core/management/bootstrap.py
"""The three things that must be true before any campaign work can start.

These used to be private methods on the `find` command, which made *getting ready* and
*finding leads* one verb with one exit code. They are shared now because `init` exists to
do exactly this and nothing else — and a phase worth its own verb is a phase worth naming
in one place rather than two.

The order matters and is not arbitrary: there is no schema to onboard into until the
migrations run, no operator to validate until onboarding has made one, and no campaign to
work until the operator owns one.
"""
from __future__ import annotations

import logging
import sys

from django.core.management import call_command

from openoutfind.core.errors import ErrorType, OpenOutFindError

logger = logging.getLogger(__name__)


def ensure_database(stderr) -> None:
    """Migrate to the current schema and make sure the CRM's fixtures exist.

    ``stderr`` is where Django's migration narration goes. That is not a style choice:
    stdout carries the CSV, and a stray "Applying core.0001_initial… OK" in a redirected
    file is exactly what the output contract exists to prevent.
    """
    call_command("migrate", "--no-input", stdout=stderr)

    from openoutfind.core.management.setup_crm import setup_crm
    setup_crm()


def ensure_onboarded() -> None:
    """Environment first, wizard only if a human is there to answer.

    The order is the point: an agent-driven install has no TTY, so the non-interactive
    path is the main path. What the environment cannot satisfy goes to the wizard on a
    TTY, or exits **naming the variables** that would have satisfied it — never a bare
    "onboarding incomplete".
    """
    from openoutfind.core import onboarding

    if not onboarding.missing_keys():
        return

    filled = onboarding.hydrate_from_env()
    if filled:
        logger.info("Configured from the environment: %s.", ", ".join(sorted(filled)))
    if not onboarding.missing_keys():
        return

    if sys.stdin.isatty():
        onboarding.onboard_interactive()
        return

    raise OpenOutFindError(
        ErrorType.ONBOARDING_INCOMPLETE,
        "no TTY, and the environment does not carry everything.\n"
        "Set these and run again:\n"
        f"{onboarding.env_help()}\n"
        "Optional: "
        f"{onboarding.ENV_PREFIX}LLM_API_BASE "
        f"(required for openai_compatible:*), {onboarding.ENV_PREFIX}NEWSLETTER.\n"
        f"{onboarding.ENV_PREFIX}ACCEPT_LEGAL_NOTICE must be set to 'true' — it "
        f"records that you accept {onboarding.LEGAL_NOTICE_URL}.",
    )


def validate_operator() -> None:
    """Fail loudly on the three things a job cannot run without.

    Each exits with a typed line rather than a log record: these are answers to the
    reader, and a program needs to branch on them.
    """
    from openoutfind.core.models import SiteConfig
    from openoutfind.core.operator import campaigns, get_active_user

    if not SiteConfig.load().llm_api_key:
        raise OpenOutFindError(
            ErrorType.ONBOARDING_INCOMPLETE,
            "no LLM API key — set OPENOUTFIND_LLM_API_KEY, or edit Site "
            "Configuration in the Django Admin.",
        )

    if get_active_user() is None:
        raise OpenOutFindError(
            ErrorType.ONBOARDING_INCOMPLETE, "no active operator account.")

    if not campaigns():
        raise OpenOutFindError(
            ErrorType.ONBOARDING_INCOMPLETE, "no campaigns for this operator.")
