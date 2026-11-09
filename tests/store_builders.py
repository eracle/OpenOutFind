# tests/store_builders.py
"""Builders for a walk's store: people, their companies and verdicts, and fired searches."""
from __future__ import annotations

from openoutfind.core.models import Page, PageLead, Search
from openoutfind.core.pipeline.terms import Terms
from openoutfind.crm.models import Company, Deal, DealState, Lead, Outcome

US = "United States"


def make_lead(title="Purchasing Manager", tags="cannabis, thc vape", state="Ohio",
              company=None, verdict=None, seniority=""):
    """A US person; ``verdict`` 1 or 0 judges them a fit or not."""
    lead = Lead.objects.create(
        profile_url=f"https://x/{Lead.objects.count()}", company=company,
        source_fields={"contact_job_title": title, "company_keywords": tags,
                       "contact_location_state": state,
                       "contact_location_country": "united states",
                       "contact_seniority": seniority})
    if verdict is not None:
        Deal.objects.create(lead=lead, state=DealState.QUALIFIED if verdict else DealState.FAILED,
                            outcome="" if verdict else Outcome.WRONG_FIT)
    return lead


def make_company(name):
    return Company.objects.create(key=name, name=name)


def make_companies(title, verdict, how_many):
    """``how_many`` people titled ``title``, each at a company of their own, judged ``verdict``."""
    for i in range(how_many):
        make_lead(title, company=make_company(f"{title}-{verdict}-{Lead.objects.count()}-{i}"),
                  verdict=verdict)


def fire(terms: Terms, people, found, number=1, origin=Search.Origin.WRITTEN):
    """A search fired to page ``number``, its last page holding ``people``."""
    search, _ = Search.objects.get_or_create(key=terms.key, defaults={
        "terms": terms.as_json(), "origin": origin, "leads_found": found})
    page = Page.objects.create(search=search, number=number, round=0, rows=len(people))
    PageLead.objects.bulk_create([PageLead(page=page, lead=lead) for lead in people])
    return search
