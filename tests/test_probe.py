# tests/test_probe.py
"""The probe: a few random verdicts on what each search brought."""
from __future__ import annotations

import pytest

from openoutfind.core.models import Page, PageLead, Search
from openoutfind.core.pipeline import probe
from openoutfind.core.pipeline.query import Terms
from openoutfind.crm.models import Deal, DealState, Lead, Outcome


def _lead(verdict=None):
    lead = Lead.objects.create(profile_url=f"https://x/{Lead.objects.count()}",
                               profile_text="Founder at Acme",
                               source_fields={"contact_job_title": "Founder"})
    if verdict is not None:
        Deal.objects.create(lead=lead, state=DealState.QUALIFIED if verdict else DealState.FAILED,
                            outcome="" if verdict else Outcome.WRONG_FIT)
    return lead


def _fired(title, people, round_number):
    terms = Terms.written(title=title)
    search = Search.objects.create(key=terms.key, terms=terms.as_json(), leads_found=5000)
    page = Page.objects.create(search=search, number=1, round=round_number, rows=len(people))
    PageLead.objects.bulk_create([PageLead(page=page, lead=lead) for lead in people])
    return search


def _waiting():
    return list(Lead.objects.filter(deal__isnull=True))


@pytest.mark.django_db
class TestTheProbe:
    def test_it_draws_from_what_the_latest_search_brought(self):
        older = [_lead() for _ in range(4)]
        latest = [_lead() for _ in range(4)]
        _fired("owner", older, 1)
        _fired("founder", latest, 2)

        assert probe.next_lead(_waiting()) in latest

    def test_a_search_with_its_verdicts_in_is_measured(self):
        page = [_lead(verdict=0) for _ in range(probe.PROBE_VERDICTS)] + [_lead()]
        _fired("founder", page, 1)

        assert probe.next_lead(_waiting()) is None

    def test_a_measured_search_hands_over_to_the_next_one_short(self):
        measured = [_lead(verdict=1) for _ in range(probe.PROBE_VERDICTS)] + [_lead()]
        short = [_lead(), _lead()]
        _fired("owner", short, 1)
        _fired("founder", measured, 2)

        assert probe.next_lead(_waiting()) in short

    def test_the_same_draw_comes_back_until_its_verdict_is_in(self):
        """An `--agent-qualify` stop between the draw and the verdict draws it again."""
        _fired("founder", [_lead() for _ in range(20)], 1)

        assert probe.next_lead(_waiting()) == probe.next_lead(_waiting())

    def test_a_person_with_no_profile_text_is_never_drawn(self):
        blank = _lead()
        Lead.objects.filter(pk=blank.pk).update(profile_text="")
        _fired("founder", [blank], 1)

        assert probe.next_lead(_waiting()) is None

    def test_with_nothing_fired_there_is_nothing_to_probe(self):
        assert probe.next_lead([object()]) is None
