# tests/test_frontier.py
"""The frontier — every search the walk could fire next, valued from the store."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core.models import PAGE_SIZE, Page, PageLead, Search
from openoutfind.core.pipeline import frontier
from openoutfind.core.pipeline.frontier import Arm, Frontier, People
from openoutfind.core.pipeline.query import Terms
from openoutfind.crm.models import Company, Deal, DealState, Lead, Outcome

US = "United States"


def _lead(title="Purchasing Manager", tags="cannabis, thc vape", state="Ohio",
          company=None, verdict=None, seniority=""):
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


def _fired(terms: Terms, people, found, number=1, origin=Search.Origin.WRITTEN):
    """A search fired to page ``number``, its last page holding ``people``."""
    search, _ = Search.objects.get_or_create(key=terms.key, defaults={
        "terms": terms.as_json(), "origin": origin, "leads_found": found})
    page = Page.objects.create(search=search, number=number, round=0, rows=len(people))
    PageLead.objects.bulk_create([PageLead(page=page, lead=lead) for lead in people])
    return search


def _company(name):
    return Company.objects.create(key=name, name=name)


@pytest.mark.django_db
class TestPeople:
    def test_match_is_what_terms_matches_says_over_everyone(self):
        _lead("Purchasing Manager", "cannabis edibles, thc vape", "Ohio")
        _lead("Head of Procurement", "cannabis extraction", "California", seniority="head")
        _lead("Co-Founder", "e-commerce, food & beverage", "Ohio")
        _lead("Production Manager", "vapes", "Texas")
        searches = [
            Terms.written(title="manager"),
            Terms.written(title="purchasing", tag="cannabis"),
            Terms.written(title="procurement head", tag="cannabis extraction"),
            Terms.written(title="cofounder", tag="food beverage"),
            Terms.written(title="founder", tag="e commerce", place=f"Ohio, {US}"),
            Terms.written(title="manager", place=f"Texas, {US}"),
            Terms.written(title="procurement", seniority="head", place=US),
            Terms.written(title="manager", tag="edibles cannabis"),
        ]
        people = People.load({})
        leads = list(Lead.objects.all())

        for terms in searches:
            assert people.match(terms) == {
                lead.pk for lead in leads if terms.matches(lead.source_fields)}, terms

    def test_a_region_counts_each_company_once(self):
        acme = _company("acme")
        fits = [_lead(company=acme, verdict=1) for _ in range(5)]
        miss = _lead(company=_company("other"), verdict=0)
        people = People.load({lead.pk: float(v) for lead, v in
                              [(f, 1) for f in fits] + [(miss, 0)]})

        with patch.dict(frontier.WALK_CONFIG, {"prior_companies": 0}):
            assert people.rate_of({lead.pk for lead in fits + [miss]}, 0.0) == 0.5

    def test_a_region_is_shrunk_toward_the_store(self):
        fit = _lead(company=_company("acme"), verdict=1)
        people = People.load({fit.pk: 1.0})

        with patch.dict(frontier.WALK_CONFIG, {"prior_companies": 3}):
            assert people.rate_of({fit.pk}, 0.2) == pytest.approx((1.0 + 3 * 0.2) / 4)
            assert people.rate_of(set(), 0.2) == pytest.approx(0.2)

    def test_people_with_no_company_but_one_tag_list_are_one_company(self):
        """The provider's "Self employed" row staples one list onto hundreds."""
        placeholder = "finacial services, private investigations, dive master"
        alone = [_lead(tags=placeholder) for _ in range(4)]
        fit = _lead(company=_company("acme"))
        people = People.load({**{lead.pk: 0.4 for lead in alone}, fit.pk: 1.0})

        with patch.dict(frontier.WALK_CONFIG, {"prior_companies": 0}):
            assert people.rate_of({lead.pk for lead in alone + [fit]}, 0.0) == 0.7


@pytest.mark.django_db
class TestTheArms:
    def test_a_drained_search_offers_neither_its_next_page_nor_a_stricter_search(self):
        """Batch 5 spent all 5 stricter slots of a round re-finding people this way."""
        people = [_lead(verdict=1) for _ in range(3)]
        drained = Terms.written(title="purchasing", tag="cannabis")
        _fired(drained, people, found=3)

        arms = Frontier(None).arms()

        assert not any(arm.terms.holds_all_of(drained) for arm in arms)

    def test_a_search_with_more_to_page_offers_both(self):
        people = [_lead(verdict=1) for _ in range(3)]
        roomy = Terms.written(title="purchasing", tag="cannabis")
        _fired(roomy, people, found=5000)

        arms = Frontier(None).arms()

        assert any(arm.move == "deepen" and arm.number == 2 for arm in arms)
        assert any(arm.number == 1 and arm.terms != roomy and arm.terms.holds_all_of(roomy)
                   for arm in arms)

    def test_a_dead_searchs_terms_are_never_an_arm(self):
        _fired(Terms.written(title="purchasing"), [_lead(verdict=1)], found=5000)
        Search.objects.create(key=Terms.written(title="manager").key, dead=True,
                              terms=Terms.written(title="manager").as_json())
        dead = Terms.written(title="manager")

        assert not any(arm.terms.holds_all_of(dead) for arm in Frontier(None).arms())

    def test_a_stricter_search_is_counted_from_the_one_it_narrows(self):
        """Its count is the fired search's count times the share of its people it holds."""
        managers = [_lead("Purchasing Manager") for _ in range(3)]
        others = [_lead("Purchasing Director") for _ in range(5)]
        _fired(Terms.written(title="purchasing"), managers + others, found=8000)

        arm = Frontier(None)._first_page(Terms.written(title="purchasing manager"), "stricter")

        assert arm.size == PAGE_SIZE  # 8000 × 3/8 = 3,000 counted, one page of them
        assert arm.new == pytest.approx(PAGE_SIZE * (1 - 3 / 3000))

    def test_a_search_none_of_whose_people_were_seen_is_counted_as_the_stores_typical(self):
        """Batch 5: a famous-company page of 49,004 `production`s said nothing about
        `production × cannabis` in Maine."""
        _fired(Terms.written(title="production"), [_lead("Production Lead")], found=49_004)
        for title in ("founder", "owner"):
            _fired(Terms.written(title=title, tag="cannabis"),
                   [_lead(title.title()) for _ in range(3)], found=9)

        f = Frontier(None)
        narrow = Terms.written(title="production manager", tag="cannabis", place=US)
        wider = [fired for fired in f.fired if narrow.holds_all_of(fired.terms)]

        assert f._count(narrow, f.people.match(narrow), wider) == 9

    def test_a_search_is_rated_on_the_pages_of_the_search_it_narrows(self):
        """Batch 5: the store's founders were cannabis founders, the tag-less `founder`
        page's were not — that page is the provider's sample of `founder officer`."""
        fit = _company("fit")
        for _ in range(3):
            _lead("Founder Officer", company=fit, verdict=1)
        stranger = _lead("Founder Officer", tags="banking", company=_company("bank"),
                         verdict=0)
        _fired(Terms.written(title="founder"), [stranger], found=474_609)

        arm = Frontier(None)._first_page(Terms.written(title="founder officer"), "stricter")

        # The stranger's 0, shrunk toward the store's 0.5 — the cannabis founders, also
        # matched, would have pulled it to 0.5.
        assert arm.rate == pytest.approx(3 * 0.5 / 4)

    def test_a_page_that_brought_nobody_new_is_not_worth_its_next(self):
        known = [_lead(verdict=1) for _ in range(3)]
        first = Terms.written(title="purchasing")
        _fired(first, known, found=5000)
        _fired(Terms.written(title="manager"), known, found=5000)

        deeper = [arm for arm in Frontier(None).arms() if arm.move == "deepen"]

        by_title = {next(iter(arm.terms.title)): arm.new for arm in deeper}
        assert by_title["purchasing"] > 0 and by_title["manager"] == 0


def _arm(title, tag="cannabis", rate=0.2, new=10.0, number=1, move="swap", sample=()):
    return Arm(Terms.written(title=title, tag=tag), number, move, new=new, rate=rate,
               sample=set(sample))


def _anchor(title="Head of Procurement", tags="cannabis extraction"):
    return Lead.objects.create(synthetic=True, profile_text=title.lower(), source_fields={
        "contact_job_title": title.lower(), "company_keywords": tags,
        "contact_location_country": "united states"})


@pytest.mark.django_db
class TestTheBatch:
    def test_it_takes_the_arm_holding_the_most_fits_first(self):
        for _ in range(3):
            _lead("Founder", verdict=1)
        _lead("Buyer", verdict=1)
        buyer, founder = _arm("buyer"), _arm("founder")

        assert Frontier(None).batch([buyer, founder], 1, slots=1) == [founder]
        assert founder.covers == 3

    def test_an_arm_holding_only_fits_already_covered_is_skipped(self):
        """Batch 5: ten near-copies resting on one fit could fill a round."""
        for _ in range(2):
            _lead("Founder", verdict=1)
        _lead("Buyer", verdict=1)
        founder, copy, buyer = _arm("founder"), _arm("founder", tag="thc vape"), _arm("buyer")

        assert Frontier(None).batch([founder, copy, buyer], 1, slots=2) == [founder, buyer]

    def test_a_no_fit_is_not_promising(self):
        _lead("Founder", verdict=0)
        _lead("Buyer", verdict=1)
        founder, buyer = _arm("founder"), _arm("buyer")

        assert Frontier(None).batch([founder, buyer], 1, slots=1) == [buyer]

    def test_an_anchors_fields_count_as_a_promising_person(self):
        anchor = _anchor()
        f = Frontier(None)

        assert f.holds(_arm("procurement head", tag="cannabis")) == {anchor.pk}
        assert f.holds(_arm("procurement", tag="banking")) == set()

    def test_a_next_page_counts_the_fits_on_its_latest_page(self):
        fit, miss, elsewhere = _lead(verdict=1), _lead(verdict=0), _lead(verdict=1)
        _fired(Terms.written(title="purchasing"), [fit, miss], found=5000)

        f = Frontier(None)
        (deepen,) = [arm for arm in f.arms() if arm.move == "deepen"]

        assert f.holds(deepen) == {fit.pk}
        assert elsewhere.pk in f.fits

    def test_a_tie_goes_to_the_narrower_search(self):
        _lead("Founder", verdict=1)
        wide = _arm("founder", sample=range(50))
        narrow = _arm("founder", tag="thc vape", sample=range(5))

        assert Frontier(None).batch([wide, narrow], 1, slots=1) == [narrow]

    def test_the_slots_left_are_filled_the_same_way_twice(self):
        fit = _lead("Founder", verdict=1)
        arms = [_arm("founder")] + [_arm(f"title{i}") for i in range(30)]

        first = Frontier(None).batch(list(arms), 4, slots=10)
        again = Frontier(None).batch(list(reversed(arms)), 4, slots=10)
        other_round = Frontier(None).batch(list(arms), 5, slots=10)

        assert len(first) == 10 and first[0].terms.title == frozenset({"founder"})
        assert [a.terms.key for a in first] == [a.terms.key for a in again]
        assert [a.terms.key for a in first] != [a.terms.key for a in other_round]
        assert fit.pk in Frontier(None).holds(first[0])

    def test_a_filled_round_holds_nothing_fired_dead_or_drained(self):
        people = [_lead(verdict=1) for _ in range(3)]
        roomy = Terms.written(title="purchasing", tag="cannabis")
        drained = Terms.written(title="manager", tag="cannabis")
        _fired(roomy, people, found=5000)
        _fired(drained, people, found=3)
        dead = Terms.written(title="buyer", tag="cannabis")
        Search.objects.create(key=dead.key, dead=True, terms=dead.as_json())

        f = Frontier(None)
        batch = f.batch(f.arms(), 1)

        assert len(batch) == frontier.WALK_CONFIG["batch"]
        for arm in batch:
            assert not (arm.number == 1 and arm.terms in (roomy, drained))
            assert not arm.terms.holds_all_of(dead)
            assert not arm.terms.holds_all_of(drained)

    def test_a_round_fires_only_while_an_arm_beats_the_floor(self):
        _fired(Terms.written(title="purchasing", tag="cannabis"),
               [_lead(verdict=1) for _ in range(3)], found=5000)

        assert frontier.next_round(None, 1, floor=0.0)
        assert frontier.next_round(None, 1, floor=1.0) == []

    def test_an_arms_sample_is_the_people_it_matches(self):
        people = [_lead("Purchasing Manager") for _ in range(2)]
        _fired(Terms.written(title="purchasing"), people, found=5000)

        f = Frontier(None)
        unfired = f._first_page(Terms.written(title="purchasing manager"), "stricter")
        (deepen,) = [arm for arm in f.arms() if arm.move == "deepen"]

        assert unfired.sample == deepen.sample == {lead.pk for lead in people}


@pytest.mark.django_db
class TestTheFitRate:
    def test_one_class_of_verdicts_leaves_the_score_as_it_is(self):
        _lead(verdict=1)
        assert list(frontier.fit_rate_map()([0.2, 0.7])) == [0.2, 0.7]

    def test_the_score_is_mapped_through_the_verdicts_at_their_out_of_fold_scores(self):
        for verdict in (0, 0, 1, 1):
            lead = _lead(verdict=verdict)
            lead.embedding_array = np.full(384, 0.1, dtype=np.float32)
            lead.save()

        class Folds:
            def out_of_fold_probs(self, indices, k):
                return np.array([0.1, 0.2, 0.8, 0.9])

        with patch("openoutfind.core.ml.qualifier.build_qualifier", return_value=Folds()):
            rate = frontier.fit_rate_map()

        assert rate([0.05])[0] == 0.0 and rate([0.95])[0] == 1.0

    def test_it_is_fitted_once_per_set_of_verdicts(self):
        for verdict in (0, 1):
            lead = _lead(verdict=verdict)
            lead.embedding_array = np.full(384, 0.1, dtype=np.float32)
            lead.save()

        class Folds:
            def out_of_fold_probs(self, indices, k):
                return np.array([0.2, 0.8])

        with patch("openoutfind.core.ml.qualifier.build_qualifier",
                   return_value=Folds()) as build:
            frontier.fit_rate_map()
            frontier.fit_rate_map()
        assert build.call_count == 1
