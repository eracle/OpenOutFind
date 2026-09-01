# openoutfind/core/status.py
"""The standing state of the database — as data, for a person or a program.

This module builds one dict and the ``status`` command renders it, as a human summary or
as ``--json``. Nothing here prints, and nothing here mutates. One renderer does live here
— ``render_next_action`` — because the end of a `find` run renders the same ask, and the
sentence must have one spelling.

Three things it answers:

  * what is configured, and what is not;
  * what is **blocked**, and why — in the stable vocabulary of ``core/errors.py``,
    because *no leads yet* and *your key was rejected* must never look alike;
  * the counts toward the deliverable, and the credit balance.

**It is smaller than it was, on purpose.** It began as the verb an agent asked *instead of
tailing a log*, because a daemon could not answer for itself: `next_action` existed so a
caller had something to interrogate, and the counts were the only way to tell whether a
background process had achieved anything. With `find` bounded by a goal, the work verb
returns its own result and this one is back to being what it says — what is in the
database right now, for a reader who did not run the job.
"""
from __future__ import annotations

import logging

from openoutfind.core.errors import ErrorType
from openoutfind.enrichment.bettercontact import SIGNUP_URL as DEFAULT_SIGNUP_URL

logger = logging.getLogger(__name__)


def build_status() -> dict:
    """Assemble the whole status document. Reads only; never raises on a dead provider."""
    config = _config_state()
    totals = _pipeline_counts()
    credits = _credits()
    hub = _hub_balance()
    blocked = _blocked(config, credits, totals)

    return {
        "config": config,
        "totals": totals,
        "credits": credits,
        "hub": hub,
        "blocked": blocked,
        "next_action": next_action(config, credits, totals),
    }


# ── configuration ────────────────────────────────────────────────

def _config_state() -> dict:
    """Which groups this run was given, and the variables that would give it the rest."""
    from openoutfind.core.readiness import GROUPS, missing_variables

    missing = missing_variables()
    return {
        "complete": not missing,
        "done": [group for group in GROUPS if group not in missing],
        "missing": missing,
    }


# ── the counts toward the deliverable ────────────────────────────

def _pipeline_counts() -> dict:
    """The install's pipeline counts."""
    from openoutfind.core.export import export_counts
    from openoutfind.crm.models import Deal, DealState

    deals = Deal.objects.all()
    by_state = {
        state: deals.filter(state=state).count()
        for state in (
            DealState.QUALIFIED,
            DealState.READY_TO_FIND_EMAIL,
            DealState.FINDING_EMAIL,
            DealState.RESOLVED,
            DealState.NO_EMAIL_FOUND,
            DealState.FAILED,
        )
    }
    exportable, with_email = export_counts()
    return {
        "leads_seen": deals.count(),
        "qualified": by_state[DealState.QUALIFIED],
        "ranked_for_lookup": by_state[DealState.READY_TO_FIND_EMAIL],
        "lookup_in_flight": by_state[DealState.FINDING_EMAIL],
        "resolved": by_state[DealState.RESOLVED],
        "no_email_found": by_state[DealState.NO_EMAIL_FOUND],
        "rejected": by_state[DealState.FAILED],
        "exportable": exportable,
        "exportable_with_email": with_email,
        "exportable_without_email": exportable - with_email,
    }


# ── the balance ──────────────────────────────────────────────────

def _credits() -> dict:
    """Read the provider balance, reporting *why* it is unknown rather than guessing.

    A balance we could not read is not a balance of zero, and the difference decides
    whether the operator is asked to top up.
    """
    from openoutfind.enrichment import provider

    finder = provider.active()
    if finder is None:
        return {"balance": None, "error": ErrorType.NO_CREDENTIAL}

    try:
        return {"balance": finder.credit_balance(), "provider": finder.NAME, "error": None}
    except provider.ProviderUnavailable as exc:
        logger.debug("Could not read the credit balance: %s", exc)
        return {"balance": None, "error": exc.error_type, "detail": str(exc)}


def _signup_url() -> str:
    """The attributed account link for the configured finder, or the default."""
    from openoutfind.enrichment import provider

    finder = provider.active()
    return finder.SIGNUP_URL if finder else DEFAULT_SIGNUP_URL


def _hub_balance() -> dict:
    """The give-to-get counter — a different number on a different service than
    ``_credits()``, which is the configured finder's own prepaid balance. Showing one while
    calling it the other would be worse than showing neither, so it gets its own key.
    """
    from openoutfind.contacts.service import hub_balance

    return hub_balance()


# ── what is blocked, and why ─────────────────────────────────────

def _blocked(config: dict, credits: dict, totals: dict) -> list[dict]:
    """Everything standing between the current state and more qualified rows."""
    blocked = []

    if not config["complete"]:
        blocked.append({
            "type": ErrorType.ONBOARDING_INCOMPLETE,
            "message": "this run was not given: " + ", ".join(config["missing"]),
        })

    if credits["error"] == ErrorType.NO_CREDENTIAL:
        blocked.append({
            "type": ErrorType.NO_CREDENTIAL,
            "message": "no BetterContact key — discovery and email finding are both off",
        })
    elif credits["error"] == ErrorType.PROVIDER_AUTH:
        blocked.append({
            "type": ErrorType.PROVIDER_AUTH,
            "message": "BetterContact rejected the API key",
        })
    elif credits["error"] == ErrorType.PROVIDER_OUT_OF_CREDITS:
        blocked.append({
            "type": ErrorType.PROVIDER_OUT_OF_CREDITS,
            "message": "BetterContact reports the credits are exhausted",
        })
    elif credits["error"] == ErrorType.PROVIDER_RATE_LIMITED:
        blocked.append({
            "type": ErrorType.PROVIDER_RATE_LIMITED,
            "message": "BetterContact is rate-limiting this client — the run is backing off",
        })
    elif credits["balance"] == 0 and totals["ranked_for_lookup"]:
        blocked.append({
            "type": ErrorType.PROVIDER_OUT_OF_CREDITS,
            "message": (
                f"{totals['ranked_for_lookup']} ranked lead(s) waiting, 0 credits left"
            ),
        })

    return blocked


# ── the next action ──────────────────────────────────────────────

def next_action(config: dict, credits: dict, totals: dict) -> dict:
    """The one thing to do next — arithmetic, not adjectives.

    Ordered by what actually blocks progress, which is why the credit ask sits above the
    rows: a ranked lead is one that *cannot advance* without credits, whereas printing
    what exists costs nothing and is available at any time.

    That ordering does not break the *never before value* rule. Ranked leads are
    qualified leads with written reasons, so ``ranked_for_lookup > 0`` **is** the proof
    that value exists — a first run with nothing qualified yet is asked for nothing, and
    told to go find some.

    **It is smaller than it was**, because the work verb now returns its own result. This
    used to be the only way an agent could learn what a daemon had been doing; with a
    bounded `find` the answer arrives on stdout, and what is left here is the standing
    state of the database.
    """
    if not config["complete"]:
        variables = sorted({v for names in config["missing"].values() for v in names})
        return {
            "type": "configure",
            "message": "This run was not given everything finding needs.",
            "unlocks": "the run can start",
            "variables": variables,
        }

    if credits["balance"] == 0 and totals["ranked_for_lookup"]:
        return {
            "type": "add_credits",
            "message": (
                f"{totals['ranked_for_lookup']} ranked lead(s) waiting, 0 credits left."
            ),
            "unlocks": "a work email address for each of them",
            "leads": totals["ranked_for_lookup"],
            # Top up where the operator actually banks: the ask must point at the
            # finder that ran out, not at whichever vendor shipped first. An install
            # with no finder at all gets the default, which is also the signup path.
            "url": _signup_url(),
        }

    if totals["exportable"]:
        return {
            "type": "print_leads",
            "message": f"{totals['exportable']} qualified lead(s) ready.",
            "unlocks": "a CSV your sequencer imports without column mapping",
            "leads": totals["exportable"],
            "command": "outfind find 0 > leads.csv",
        }

    return {
        "type": "find_leads",
        "message": "No qualified leads yet.",
        "unlocks": "leads with a written reason",
        "command": "outfind find 10",
    }


def render_next_action(action: dict) -> str:
    """The next action as a short block of text — returned, never printed.

    It lives here rather than in the ``status`` command because ``find`` ends with it
    too: a run that stops with ranked leads and no credits has to say so, and the ask is
    already derived above. **The end of a run renders this; it does not recompute it.**
    Two earlier attempts put the balance read and a deal count inside `core/job.py` and
    `enrichment/lookup.py`, which gave the bounded-goal loop an HTTP call to a payment
    provider and made *read once* need module-level mutable state. One spelling, one
    derivation, two callers.

    ``variables`` is left out: ``status`` lists those per step under configuration, which
    is the more useful grouping for a human, and they stay in ``--json`` where an agent
    wants them flat.
    """
    lines = [f"Next: {action['message']}"]
    if action.get("unlocks"):
        lines.append(f"  unlocks: {action['unlocks']}")
    if action.get("command"):
        lines.append(f"  run: {action['command']}")
    if action.get("url"):
        lines.append(f"  go to: {action['url']}")
    return "\n".join(lines)
