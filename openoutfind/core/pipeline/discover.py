# openoutfind/core/pipeline/discover.py
"""Discovery leg — one generation of the walk into first-touch Leads.

The top of the funnel: the walk's pages become ``Lead`` rows awaiting qualification. Free
(Lead Finder bills nothing) and browserless, so wall-clock is the only budget. A
generation is the unit: 10 searches fired at once (``walk.run_generation``), their pages
taken in together.
"""
from __future__ import annotations

import logging

from termcolor import colored

logger = logging.getLogger(__name__)


def harvest(pages: list[tuple], country_code: str) -> int:
    """Take a generation's fetched pages, ``(page, rows)`` each, into the store. Returns
    how many people were new.

    Rows with no headline are dropped (``discovery.is_live_profile`` — forgotten
    profiles, ~40% of Lead Finder's rows), and so are rows from a company already holding
    the campaign's cap of fits (``core/company_cap.py``). Every other row is written to
    the page–lead table, whether the person was new to the store or not — a person on
    several pages is evidence about each — except a marked lead or an anchor, whose
    verdict came from elsewhere and must not inflate a page.

    The pages' new people are stored in one call, so the whole generation is embedded
    in one pass, shortest profile first (``embed_texts``): embedding is most of a
    generation's time, and a pass padded page by page cost three times as much.

    Each whole page — dropped rows and known profiles included — is then given back to the
    hub (``share_profiles``), best-effort, and its distinct employers too
    (``share_companies``), for the hub's company index — one request per page, so no body grows with the generation.
    """
    from openoutfind.contacts.service import share_companies, share_profiles
    from openoutfind.core.db.leads import store_rows
    from openoutfind.core.models import PageLead

    accepted = {page: _accepted(rows) for page, rows in pages}
    leads, created = store_rows([row for rows in accepted.values() for row in rows],
                                country_code)
    by_url = {lead.profile_url: lead for lead in leads}
    PageLead.objects.bulk_create(
        [PageLead(page=page, lead=lead)
         for page, rows in accepted.items()
         for lead in _leads_of(rows, by_url)
         if not (lead.marked or lead.synthetic)],
        ignore_conflicts=True)

    for _, rows in pages:
        share_profiles(rows, country_code)
        share_companies(rows, country_code)
    return created


def _accepted(rows: list[dict]) -> list[dict]:
    """One page's rows worth storing: a live profile, at a company not at its cap."""
    from openoutfind.core.company_cap import full_company_keys
    from openoutfind.discovery import company_key_for, is_live_profile

    live = [row for row in rows if is_live_profile(row)]
    full = full_company_keys()
    accepted = [row for row in live if company_key_for(row) not in full]
    logger.debug("%d of %d row(s) dropped — no headline, %d — company at the cap",
                 len(rows) - len(live), len(rows), len(live) - len(accepted))
    return accepted


def _leads_of(rows: list[dict], by_url: dict) -> list:
    """The stored leads of one page's rows (a row with no URL was never stored)."""
    urls = dict.fromkeys(row.get("contact_linkedin_profile_url") for row in rows)
    return [by_url[url] for url in urls if url in by_url]


def discover(site_config, qualifier=None) -> bool:
    """Run one generation of the walk. Returns whether one was fired.

    ``False`` means nothing was left to fire — no population to breed from — or a gate is
    shut. A provider *refusal* propagates (``walk._take_in``): a rejected key must never
    be read as a search that matches nobody. ``qualifier`` scores each page's people, which
    is what the population is ranked by; without one, only judged people count.

    Two gates: no finder key means the index can't be reached, and no product or target
    means there is nothing to write the initial population from.
    """
    from openoutfind.core.pipeline.walk import run_generation
    from openoutfind.enrichment import bettercontact

    if not bettercontact.is_configured():
        return False
    if not (site_config.product_docs or site_config.campaign_target):
        return False

    logger.info(colored("▶ discover", "blue", attrs=["bold"]))
    if run_generation(site_config, qualifier) is None:
        logger.info(colored("■ no more people left to find — the walk has nothing left "
                            "to fire", "blue"))
        return False
    return True
