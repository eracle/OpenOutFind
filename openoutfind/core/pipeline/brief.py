"""The brief — our reading of the customer's two texts, written once, read by every prompt.

The customer's words (``product_docs``, ``campaign_target``) come in through the
environment and are never rewritten. What we conclude from them used to live nowhere a
model could read: the qualifier got the two paragraphs and a generic checklist, and on
the first 500-lead order a third of its fits broke rules a person would have applied —
hardware suppliers accepted as buyers, edibles-only brands, finance and sales roles the
target never named.

So the reading is one stored row (``core.models.Brief``), rendered beside the customer's
text into the seed, the anchors and every verdict. It is written by ``AI_MODEL`` the first
time a prompt needs it, or handed in with ``find --brief PATH`` — how a stronger model
than the one judging (the agent running an order) writes it once, so the cheap model
judges every lead against rules a good one wrote.
"""
from __future__ import annotations

import logging

import jinja2
from pydantic import BaseModel, Field
from termcolor import colored

from openoutfind.core.conf import PROMPTS_DIR

logger = logging.getLogger(__name__)

# What a brief handed in with ``--brief`` is recorded as having been written by.
GIVEN = "--brief"


class _WrittenBrief(BaseModel):
    """The model's answer: the rules, as the plain-text list a person would write."""

    rules: str = Field(
        description="The judging rules as plain-text bullets under 'A fit is' and "
                    "'Not a fit'.",
    )


def stored():
    """The brief row, or ``None`` when this store has none yet."""
    from openoutfind.core.models import Brief

    return Brief.objects.first()


def adopt(rules: str) -> None:
    """Store ``rules`` as this store's brief, replacing any brief already there.

    Silent: ``find`` echoes the brief in force at the start of every run anyway.
    """
    from openoutfind.core.models import Brief

    rules = rules.strip()
    brief = stored()
    if brief is not None and brief.rules == rules:
        return
    Brief.objects.all().delete()
    Brief.objects.create(rules=rules, written_by=GIVEN)


def ensure_rules(site_config) -> str:
    """The brief's rules, written by ``AI_MODEL`` now if this store has none.

    ``""`` when there is nothing to write them from (no ICP text) or nobody to write them
    (``--agent-qualify``, which calls no model — the agent hands one in with ``--brief``,
    or judges without).
    """
    from openoutfind.core import agent_qualify
    from openoutfind.core.models import Brief

    brief = stored()
    if brief is not None:
        return brief.rules
    if agent_qualify.active() or not (site_config.product_docs or site_config.campaign_target):
        return ""

    rules = _write_with_llm(site_config).strip()
    Brief.objects.create(rules=rules, written_by=site_config.ai_model)
    log_brief_echo()
    return rules


def _write_with_llm(site_config) -> str:
    """Ask ``AI_MODEL`` for the rules."""
    from pydantic_ai import Agent

    from openoutfind.core.llm import get_llm_model, run_agent_sync

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    prompt = env.get_template("write_brief.j2").render(
        product_docs=site_config.product_docs,
        campaign_target=site_config.campaign_target,
    )
    agent = Agent(get_llm_model(), output_type=_WrittenBrief, model_settings={"timeout": 120})
    return run_agent_sync(agent.run(prompt)).output.rules


def log_brief_echo() -> None:
    """Tell the operator the rules this store judges by. No-op without a brief.

    Printed when ``AI_MODEL`` writes one and at the start of every run, beside the ICP
    echo: it is the cheapest place to catch a misreading, before hundreds of verdicts
    are given under it.
    """
    brief = stored()
    if brief is None:
        return
    logger.info("%s %s", colored("Judging by:", "cyan", attrs=["bold"]),
                colored(f"({brief})", "cyan"))
    for line in brief.rules.splitlines():
        if line.strip():
            logger.info("    %s", line.rstrip())
