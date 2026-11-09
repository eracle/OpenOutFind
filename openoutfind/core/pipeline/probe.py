# openoutfind/core/pipeline/probe.py
"""The probe: a few verdicts at random on what each search brought, so its page is measured.

A fired search's next page is valued on its own effect in the fit-rate model (``belief``),
fitted on the people its pages brought — and nothing else judges them. Exploit takes the best-scored
lead, BALD the one the GP is least sure of, calibrate one from the thinnest score band;
none of them reaches a page of strangers far from every verdict, where the GP reads the
base rate whoever they are. On batch 6, founder · ``food`` brought 186 people and not one
was judged, so its σ never shrank and an optimistic pick would have fired it forever.

So past the cold phase, each ``top_up`` pass first takes a person drawn at random from the
latest search whose people hold fewer than ``PROBE_VERDICTS`` verdicts. Random, so the
verdicts are a fair sample of the page; five, so a search with no fits on them falls from
0.22 ± 0.21 to about 0.08 ± 0.09 and stops being worth its next page. The draw is seeded by
the search and its verdict count, so a run stopped between the draw and the verdict
(``--agent-qualify``) draws the same person again.
"""
from __future__ import annotations

import logging
import random

from termcolor import colored

logger = logging.getLogger(__name__)

PROBE_VERDICTS = 5


def next_lead(candidates: list):
    """The person the probe judges next, or None when every fired search is measured or
    has nobody left to judge."""
    from openoutfind.core.pipeline.store_snapshot import fired_searches, judged_leads

    searches = sorted(fired_searches(), key=lambda f: (-f.latest.round, f.search.key))
    if not searches:
        return None
    waiting = {c.pk: c for c in candidates if c.profile_text}
    judged = judged_leads()
    for fired in searches:
        verdicts = len(fired.people & judged)
        unjudged = sorted(pk for pk in fired.people if pk in waiting)
        if verdicts >= PROBE_VERDICTS or not unjudged:
            continue
        rng = random.Random(f"{fired.search.key}:{verdicts}")
        lead = waiting[rng.choice(unjudged)]
        logger.info("  %s %s — %s, verdict %d of %d, drawn at random from the %d it brought",
                    colored("·", "cyan", attrs=["bold"]), colored("probe", "cyan", attrs=["bold"]),
                    fired.terms.as_text(), verdicts + 1, PROBE_VERDICTS, len(fired.people))
        return lead
    return None
