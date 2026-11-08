# tests/test_walk.py
"""The walk end to end — rounds of searches, with Lead Finder stubbed.

The stub answers each search with a few people whose job title is the search's own, so
every person matches the search that found them and the frontier has something to read.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from openoutfind.core import meter
from openoutfind.core.conf import WALK_CONFIG
from openoutfind.core.config import SiteConfig
from openoutfind.core.errors import ErrorType
from openoutfind.core.models import PAGE_SIZE, Page, PageLead, Search
from openoutfind.core.pipeline import walk
from openoutfind.core.pipeline.discover import discover
from openoutfind.core.pipeline.frontier import Arm
from openoutfind.core.pipeline.icp import (
    ANCHOR_COUNT,
    Anchor,
    InitialPopulation,
    anchor_leads,
)
from openoutfind.core.pipeline.terms import Terms
from openoutfind.crm.models import Deal, DealState, Lead, Outcome
from openoutfind.discovery import Fetched, Page as LeadFinderPage
from openoutfind.enrichment.bettercontact import BetterContactUnavailable

POPULATION = WALK_CONFIG["population"]
TITLES = ["founder", "purchasing", "procurement", "production", "operations", "buyer",
          "extraction", "cultivation", "compliance", "quality", "sourcing", "supply",
          "logistics", "packaging", "formulation"][:POPULATION]


def _population():
    return InitialPopulation(
        searches=[{"title": title, "tag": "cannabis" if i % 2 else ""}
                  for i, title in enumerate(TITLES)],
        country="United States", country_code="us", headcount_min=1, headcount_max=50)


def _row(url, title, company):
    return {"contact_linkedin_profile_url": url, "contact_job_title": title,
            "contact_headline": f"{title} at {company}", "company_name": company,
            "company_keywords": ["cannabis", "thc vape"],
            "contact_location_country": "united states"}


class FakeLeadFinder:
    """Answers a search with ``rows`` people titled like it, from ``found`` in the index,
    each at a company of their own.

    ``dead`` titles count 0; ``artifacts`` come back empty with a positive count;
    ``outages`` raise. Every call is recorded as ``(title, offset)``.
    """

    def __init__(self, found=5000, rows=3, dead=(), artifacts=(), outages=(), error=None):
        self.found, self.rows = found, rows
        self.dead, self.artifacts, self.outages = set(dead), set(artifacts), set(outages)
        self.error = error
        self.calls = []

    def __call__(self, api_key, filters, limit, offset, first_poll_s):
        title = filters["lead_job_title"]["include"][0] if "lead_job_title" in filters else ""
        self.calls.append((title, offset))
        found = (lambda n: n if offset == 0 else None)
        if title in self.outages:
            raise self.error or BetterContactUnavailable(
                "down", error_type=ErrorType.PROVIDER_UNAVAILABLE)
        if title in self.dead:
            return Fetched(LeadFinderPage([], found(0)), 1.0)
        if title in self.artifacts:
            return Fetched(LeadFinderPage([], found(71_000_000)), 1.0)
        slug = title.replace(" ", "-")
        rows = [_row(f"https://linkedin.com/in/{slug}-{offset}-{i}", title,
                     f"{slug}-co-{offset}-{i}")
                for i in range(min(self.rows, max(0, self.found - offset)))]
        return Fetched(LeadFinderPage(rows, found(self.found)), 2.0)


@pytest.fixture
def site(db, configure):
    configure(bettercontact_api_key="k", product_docs="cannabis packaging",
              campaign_target="buyers at cannabis brands")
    return SiteConfig.load()


ANCHORS = [Anchor(f"head of procurement cannabis {i}",
                  {"contact_job_title": "head of procurement",
                   "company_keywords": "cannabis extraction",
                   "contact_location_country": "united states"})
           for i in range(ANCHOR_COUNT)]


def _cold_start(anchors=ANCHORS):
    """The model's two cold-start answers: the written population and the anchors."""
    return (patch("openoutfind.core.pipeline.icp._population_from_llm",
                  return_value=_population()),
            patch("openoutfind.core.pipeline.icp.generate_anchors", return_value=anchors))


def _run(site, finder, anchors=ANCHORS):
    population, anchored = _cold_start(anchors)
    with population, anchored, patch("openoutfind.discovery.search", side_effect=finder):
        return walk.run_round(site, None)


def _judge(title_fits: str):
    """Every stored person titled ``title_fits`` a fit, everyone else a no-fit."""
    for lead in Lead.objects.filter(deal__isnull=True):
        fit = lead.source_fields.get("contact_job_title") == title_fits
        Deal.objects.create(lead=lead, state=DealState.QUALIFIED if fit else DealState.FAILED,
                            outcome="" if fit else Outcome.WRONG_FIT)


class TestTheFirstRound:
    def test_the_written_population_is_fired_whole_at_page_one(self, site):
        finder = FakeLeadFinder()

        fired = _run(site, finder)

        assert fired.number == 0 and fired.fired == POPULATION
        assert sorted(title for title, _ in finder.calls) == sorted(TITLES)
        assert {offset for _, offset in finder.calls} == {0}
        assert set(Page.objects.values_list("round", flat=True)) == {0}

    def test_it_values_nothing(self, site):
        """None of its people judged yet: there is nothing to value it on."""
        with patch.object(walk.frontier, "next_round") as planned:
            _run(site, FakeLeadFinder())

        planned.assert_not_called()

    def test_it_writes_the_anchors_with_the_population(self, site):
        """A store seeded with marks starts with fits, so it never ran cold — batch 5
        had none."""
        mark = Lead.objects.create(profile_url="https://linkedin.com/in/mark", marked=True)
        Deal.objects.create(lead=mark, state=DealState.QUALIFIED)

        _run(site, FakeLeadFinder())

        assert [lead.profile_text for lead in anchor_leads()] == [a.profile for a in ANCHORS]

    def test_without_anchors_it_fires_nothing(self, site):
        finder = FakeLeadFinder()

        with pytest.raises(walk.Unanchored):
            _run(site, finder, anchors=[])

        assert finder.calls == [] and not Page.objects.exists()

    def test_a_store_with_searches_and_no_anchors_writes_them_before_its_next_round(self, site):
        _run(site, FakeLeadFinder())
        Lead.objects.filter(synthetic=True).delete()
        _judge("founder")

        _run(site, FakeLeadFinder())

        assert len(anchor_leads()) == ANCHOR_COUNT
        assert Page.objects.filter(round=1).exists()

    def test_a_standing_set_is_never_topped_up(self, site):
        _run(site, FakeLeadFinder())
        Lead.objects.filter(synthetic=True).exclude(pk=anchor_leads()[0].pk).delete()
        _judge("founder")

        _run(site, FakeLeadFinder())

        assert len(anchor_leads()) == 1

    def test_every_search_rides_the_customers_shape(self, site):
        _run(site, FakeLeadFinder())

        assert set(Search.objects.values_list("headcount_min", "headcount_max",
                                              "country_code")) == {(1, 50, "us")}

    def test_it_is_measured(self, site):
        _run(site, FakeLeadFinder())

        assert meter.totals(meter.ROUND_SECONDS)[1] == 1
        assert meter.totals(meter.ROUND_LEADS) == (3.0 * POPULATION, 1)
        assert meter.totals(meter.SEARCH_READY_SECONDS) == (2.0 * POPULATION, POPULATION)
        assert meter.totals(meter.INGEST_SECONDS_PER_LEAD)[1] == 1


class TestTakingPagesIn:
    def test_a_written_page_holds_no_expectation(self, site):
        _run(site, FakeLeadFinder())

        assert not Page.objects.filter(expected_count__isnull=False).exists()
        assert not Page.objects.filter(expected_rate__isnull=False).exists()

    def test_a_page_the_walk_chose_keeps_what_it_expected(self, site):
        terms = Terms.written(title="founder")
        arm = Arm(terms, 1, "swap", count=6.0, theta=0.8)
        _run(site, FakeLeadFinder())
        fetched = Fetched(LeadFinderPage([{"contact_job_title": "founder"}], 33), 1.0)

        page, _ = walk._take_in(arm, fetched, 3)

        page.refresh_from_db()
        assert (page.expected_count, page.expected_rate) == (6.0, 0.8)
        assert page.search.leads_found == 33

    def test_a_round_the_walk_chose_keeps_its_expectation_on_the_page(self, site):
        _run(site, FakeLeadFinder())
        _judge("founder")

        _run(site, FakeLeadFinder())

        chosen = Page.objects.get(round=1)
        assert chosen.expected_count > 0
        assert 0 < chosen.expected_rate < 1

    def test_a_page_records_every_person_it_returned(self, site):
        """Known people too: a page that returns them is evidence about its search."""
        known = Lead.objects.create(profile_url="https://linkedin.com/in/founder-0-0",
                                    source_fields={"contact_job_title": "founder"})

        _run(site, FakeLeadFinder())

        page = Page.objects.get(search__terms__title=["founder"])
        assert page.leads.count() == 3
        assert known in page.leads.all()

    def test_a_round_is_embedded_in_one_call(self, site):
        """Embedding is most of a round's time; one pass sorts every page's people."""
        from openoutfind.core.ml import embeddings

        _run(site, FakeLeadFinder())

        assert embeddings.embed_texts.call_count == 1
        assert len(embeddings.embed_texts.call_args.args[0]) == 3 * POPULATION

    def test_a_round_given_to_the_hub_embeds_no_text_twice(self, db, configure):
        """A stored person's vector goes out as it is; a dropped row is embedded once."""
        from openoutfind.core.ml import embeddings

        configure(bettercontact_api_key="k", product_docs="cannabis packaging",
                  campaign_target="buyers at cannabis brands", contacts_api_token="tok")
        finder = FakeLeadFinder()
        shared = _row("https://linkedin.com/in/shared", "founder", "Shared")
        dropped = {"contact_linkedin_profile_url": "https://linkedin.com/in/forgotten",
                   "contact_job_title": "former buyer"}

        def both(api_key, filters, limit, offset, first_poll_s):
            page, ready = finder(api_key, filters, limit, offset, first_poll_s)
            return Fetched(LeadFinderPage(page.leads + [shared, dropped], page.leads_found), ready)

        with patch("openoutfind.contacts.service._send") as send:
            _run(SiteConfig.load(), both)

        embedded = [text for call in embeddings.embed_texts.call_args_list
                    for text in call.args[0]]
        assert len(embedded) == len(set(embedded)) == 3 * POPULATION + 2
        sent = [record for call in send.call_args_list for record in call.args[1]["profiles"]]
        assert len(sent) == (3 + 2) * POPULATION
        assert all(len(record["embedding"]) == 384 for record in sent)

    def test_a_person_on_two_pages_is_stored_once_and_on_both(self, site):
        finder = FakeLeadFinder()
        shared = _row("https://linkedin.com/in/shared", "founder", "Shared")

        def both(api_key, filters, limit, offset, first_poll_s):
            page, ready = finder(api_key, filters, limit, offset, first_poll_s)
            return Fetched(LeadFinderPage(page.leads + [shared], page.leads_found), ready)

        _run(site, both)

        lead = Lead.objects.get(profile_url="https://linkedin.com/in/shared")
        assert PageLead.objects.filter(lead=lead).count() == POPULATION

    def test_a_marked_lead_is_on_no_page(self, site):
        Lead.objects.create(profile_url="https://linkedin.com/in/founder-0-0", marked=True)

        _run(site, FakeLeadFinder())

        assert not PageLead.objects.filter(lead__marked=True).exists()

    def test_an_empty_first_page_with_no_count_kills_the_search(self, site):
        _run(site, FakeLeadFinder(dead={"founder"}))

        search = Search.objects.get(terms__title=["founder"])
        assert search.dead and not search.pages.exists()

    def test_an_empty_first_page_with_a_count_is_an_artifact_and_leaves_no_trace(self, site):
        _run(site, FakeLeadFinder(artifacts={"founder"}))

        search = Search.objects.get(terms__title=["founder"])
        assert not search.dead and not search.pages.exists()

    def test_an_outage_leaves_the_search_on_the_frontier(self, site):
        _run(site, FakeLeadFinder(outages={"founder"}))

        assert any(arm.terms.title == {"founder"} and arm.number == 1
                   for arm in walk.frontier.Frontier(None).arms())

    @pytest.mark.parametrize("error_type", [ErrorType.PROVIDER_AUTH,
                                            ErrorType.PROVIDER_RATE_LIMITED])
    def test_a_refusal_stops_the_run_instead_of_reading_as_nobody(self, site, error_type):
        refusal = BetterContactUnavailable("no", error_type=error_type)
        with pytest.raises(BetterContactUnavailable):
            _run(site, FakeLeadFinder(outages={"founder"}, error=refusal))


class TestTheNextRound:
    def test_it_fires_one_search(self, site):
        _run(site, FakeLeadFinder())
        _judge("founder")
        finder = FakeLeadFinder()

        fired = _run(site, finder)

        assert fired.number == 1
        assert len(finder.calls) == 1
        assert Page.objects.filter(round=1).count() == 1

    def test_the_first_search_fired_holds_the_fits(self, site):
        """Five founders judged fits outnumber the three anchors."""
        _run(site, FakeLeadFinder(rows=5))
        _judge("founder")
        finder = FakeLeadFinder()

        _run(site, finder)

        (title, _), = finder.calls
        assert "founder" in title.split()

    def test_a_drained_search_is_neither_paged_nor_narrowed(self, site):
        """Its first page held everyone the provider counts: anything holding all its
        terms can only re-find people the store already has."""
        _run(site, FakeLeadFinder(found=3))
        _judge("founder")
        finder = FakeLeadFinder(found=3)

        _run(site, finder)

        assert all(offset == 0 for _, offset in finder.calls)
        founder = Terms.written(title="founder")
        assert not any(search.query.holds_all_of(founder)
                       for search in Search.objects.filter(origin=Search.Origin.DRAWN))

    def test_no_arm_repeats_a_fired_search_or_holds_a_dead_ones_terms(self, site):
        _run(site, FakeLeadFinder(dead={"purchasing"}))
        _judge("founder")
        _run(site, FakeLeadFinder())

        dead = Terms.written(title="purchasing")
        for search in Search.objects.filter(origin=Search.Origin.DRAWN):
            assert not search.query.holds_all_of(dead)
        assert Search.objects.filter(pages__number=1).count() == Page.objects.filter(
            number=1).count()

    def test_nothing_is_fired_when_no_arm_is_expected_to_bring_a_fit(self, site):
        _run(site, FakeLeadFinder())
        _judge("founder")
        finder = FakeLeadFinder()

        with patch.object(walk.frontier, "FITS_WORTH_A_SEARCH", 1e9):
            assert _run(site, finder) is None
        assert finder.calls == []

    def test_a_walk_with_nothing_left_to_fire_says_so(self, site):
        _run(site, FakeLeadFinder(dead=set(TITLES)))

        assert _run(site, FakeLeadFinder()) is None


class TestThePool:
    def test_new_leads_start_the_pools_fit_rate_over(self, site):
        meter.record(meter.POOL_FIT, 0)
        _run(site, FakeLeadFinder())

        assert meter.totals(meter.POOL_FIT) == (0.0, 0)

    def test_a_round_of_known_people_leaves_the_pool_as_it_was(self, site):
        _run(site, FakeLeadFinder())
        _judge("founder")
        meter.record(meter.POOL_FIT, 0)
        _run(site, FakeLeadFinder(rows=0))

        assert meter.totals(meter.POOL_FIT) == (0.0, 1)


class TestDiscover:
    def test_no_finder_key_is_a_no_op(self, db, configure):
        configure(product_docs="p", campaign_target="t")
        with patch.object(walk, "run_round") as fired:
            assert discover(SiteConfig.load()) is False
        fired.assert_not_called()

    def test_no_icp_text_is_a_no_op(self, db, configure):
        configure(bettercontact_api_key="k")
        with patch.object(walk, "run_round") as fired:
            assert discover(SiteConfig.load()) is False
        fired.assert_not_called()

    def test_a_fired_round_is_work(self, site):
        population, anchored = _cold_start()
        with population, anchored, patch("openoutfind.discovery.search",
                                         side_effect=FakeLeadFinder()):
            assert discover(site) is True
