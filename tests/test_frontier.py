# tests/test_frontier.py
"""The frontier — every search the walk could fire next, valued from the store."""
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core.models import PAGE_SIZE, Page, PageLead, Search
from openoutfind.core.pipeline import belief, frontier
from openoutfind.core.pipeline.frontier import Arm, Frontier, People
from openoutfind.core.pipeline.terms import Terms, split_place
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

    def test_a_row_with_no_state_is_matched_by_a_state_of_its_own_country(self):
        stateless = _lead(state="")
        Lead.objects.filter(pk=stateless.pk).update(source_fields={
            "contact_job_title": "Owner", "company_keywords": "restaurant",
            "contact_location_country": "indonesia"})
        ohio = _lead(state="Ohio")
        people = People.load({})

        assert people.match(Terms.written(place="Bali, Indonesia")) == {stateless.pk}
        assert people.match(Terms.written(place=f"Ohio, {US}")) == {ohio.pk}
        assert people.match(Terms.written(place=f"Texas, {US}")) == set()

    def test_a_state_is_held_by_its_rows_and_its_countrys_rows_with_no_state(self):
        ohio, texas, stateless = _lead(state="Ohio"), _lead(state="Texas"), _lead(state="")
        people = People.load({})

        assert people.holding(("state", "ohio", "united states")) == {ohio.pk, stateless.pk}
        assert texas.pk not in people.holding(("state", "ohio", "united states"))

    def test_a_hyphenated_title_offers_none_of_its_pieces(self):
        """`co` alone reaches co-owners and co-CEOs: 283,068 against 179,644 co-founders."""
        _lead("Co-Founder & CEO", "cannabis edibles, food & beverage", "Ohio")
        _lead("Founder", "cannabis", "Ohio")

        people = People.load({})

        assert people.title_vocabulary() == {"founder", "ceo"}
        assert {("cannabis",), ("cannabis", "edibles"), ("food", "beverage")} <= (
            people.tag_vocabulary())
        assert people.states_in("United States") == {"ohio"}

    def test_the_vocabulary_holds_whole_tags_and_a_phrase_inside_one_still_matches(self):
        """`pre` from `pre-rolls` brought 6,512 founders of pre-seed funds and pre-k schools."""
        held = _lead("Founder", "pre-rolls, health care, cannabis extraction", "Ohio")

        people = People.load({})

        assert people.tag_vocabulary() == {
            ("pre", "rolls"), ("health", "care"), ("cannabis", "extraction")}
        assert people.match(Terms.written(title="founder", tag="cannabis")) == {held.pk}
        assert people.match(Terms.written(title="founder", tag="pre")) == {held.pk}

    def test_a_store_with_nobody_scored_reads_as_neutral(self):
        _lead()
        assert People.load({}).base_rate() == frontier.NEUTRAL_RATE


@pytest.mark.django_db
class TestTheArms:
    def test_a_drained_search_offers_neither_its_next_page_nor_a_stricter_search(self):
        """Batch 5 spent all 5 stricter slots of a round re-finding people this way."""
        people = [_lead(verdict=1) for _ in range(3)]
        drained = Terms.written(title="purchasing", tag="cannabis")
        _fired(drained, people, found=3)

        arms = Frontier(None).arms()

        assert not any(arm.terms.holds_all_of(drained) for arm in arms)

    def test_a_market_written_in_states_never_goes_country_wide(self):
        """The Bali sample: a fired country search's next page and its neighbours are out
        with the state's own widening, so neither the cover nor the fill can leave Bali."""
        people = [_lead(verdict=1) for _ in range(3)]
        _fired(Terms.written(title="owner", tag="cannabis", place="Bali, Indonesia"),
               people, found=500)
        _fired(Terms.written(title="manager", tag="cannabis", place="Indonesia"), people,
               found=500, origin=Search.Origin.DRAWN)

        arms = Frontier(None).arms()

        assert arms and all(split_place(arm.terms.place)[0] for arm in arms)

    def test_a_market_written_partly_on_the_country_keeps_it(self):
        people = [_lead(verdict=1) for _ in range(3)]
        _fired(Terms.written(title="owner", tag="cannabis", place=f"Ohio, {US}"), people,
               found=500)
        _fired(Terms.written(title="manager", tag="cannabis", place=US), people, found=500)

        arms = Frontier(None).arms()

        assert any(not split_place(arm.terms.place)[0] for arm in arms)

    def test_a_search_with_more_to_page_offers_its_next_page_beside_the_composed(self):
        people = [_lead(verdict=1) for _ in range(3)]
        roomy = Terms.written(title="purchasing", tag="cannabis")
        _fired(roomy, people, found=5000)

        arms = Frontier(None).arms()

        assert any(arm.move == "deepen" and arm.number == 2 for arm in arms)
        assert any(arm.move == "drawn" and arm.number == 1 and arm.terms != roomy
                   for arm in arms)

    def test_a_composed_search_holds_a_title_and_a_tag_from_the_store(self):
        _lead("Extraction Lead", tags="cannabis extraction, live resin", verdict=1)
        _fired(Terms.written(title="purchasing", tag="cannabis", place=f"Ohio, {US}"),
               [_lead(verdict=1)], found=5000)

        composed = [arm.terms for arm in Frontier(None).arms() if arm.move == "drawn"]

        assert composed
        vocabulary = {"extraction", "lead", "purchasing", "manager"}
        assert all(t.title and t.title <= vocabulary and t.tag for t in composed)
        assert any(t.tag == ("live", "resin") for t in composed)

    def test_a_store_with_no_written_search_composes_nothing(self):
        _lead(verdict=1)

        assert not any(arm.move == "drawn" for arm in Frontier(None).arms())

    def test_a_dead_searchs_terms_are_never_an_arm(self):
        _fired(Terms.written(title="purchasing"), [_lead(verdict=1)], found=5000)
        Search.objects.create(key=Terms.written(title="manager").key, dead=True,
                              terms=Terms.written(title="manager").as_json())
        dead = Terms.written(title="manager")

        assert not any(arm.terms.holds_all_of(dead) for arm in Frontier(None).arms())

    def test_an_unfired_count_stays_within_what_it_narrows_and_above_the_store(self):
        people = [_lead("Purchasing Manager") for _ in range(3)]
        _fired(Terms.written(title="purchasing"), people, found=8000)
        stricter = Terms.written(title="purchasing manager")

        f = Frontier(None)
        with patch.object(f.beliefs, "count", return_value=1e9):
            high = f._first_page(stricter, "stricter")
        with patch.object(f.beliefs, "count", return_value=1.0):
            low = f._first_page(stricter, "stricter")

        assert high.count == 8000
        assert low.count == pytest.approx(3 * frontier.SEEN_MARGIN)
        assert low.new > 0  # a low estimate does not close the search for good

    def test_a_known_count_replaces_the_estimate(self):
        _fired(Terms.written(title="purchasing"), [_lead()], found=5000)

        f = Frontier(None)
        with patch.object(f.beliefs, "count", return_value=7.0):
            (deepen,) = [f._deepen(fired) for fired in f.fired]

        assert deepen.count == 5000

    def test_a_page_is_valued_at_no_more_than_a_page_however_large_the_search(self):
        _fired(Terms.written(title="purchasing"), [_lead()], found=10_000_000)

        f = Frontier(None)
        (deepen,) = [f._deepen(fired) for fired in f.fired]
        with patch.object(f.beliefs, "count", return_value=10_000_000.0):
            unfired = f._first_page(Terms.written(title="founder"), "swap")

        assert deepen.new <= PAGE_SIZE and unfired.new <= PAGE_SIZE

    def test_a_page_that_brought_nobody_new_is_not_worth_its_next(self):
        known = [_lead(verdict=1) for _ in range(3)]
        first = Terms.written(title="purchasing")
        _fired(first, known, found=5000)
        _fired(Terms.written(title="manager"), known, found=5000)

        deeper = [arm for arm in Frontier(None).arms() if arm.move == "deepen"]

        by_title = {next(iter(arm.terms.title)): arm.new for arm in deeper}
        assert by_title["purchasing"] > 0 and by_title["manager"] == 0


def _arm(title, count=1000.0, new=100.0, seen=0, held_fits=0, rate=0.0):
    terms = Terms.written(title=title)
    return Arm(terms, 1, "swap", keywords=belief.query_keywords(terms), count=count, seen=seen,
               held_fits=held_fits, new=new, rate=rate, cost=1.0)


def _settled(arms):
    f = Frontier(None)
    f._settle(arms)
    return f


def _companies(title, verdict, how_many):
    for i in range(how_many):
        _lead(title, company=_company(f"{title}-{verdict}-{Lead.objects.count()}-{i}"),
              verdict=verdict)


class TestAnArmsValue:
    def test_an_area_whose_fits_the_store_holds_is_worth_nothing_however_large(self):
        arm = _arm("founder", count=100_000.0, seen=5000, held_fits=5000)

        assert arm.density(0.05) == 0.0

    def test_a_search_with_nobody_out_there_is_worth_nothing(self):
        assert _arm("founder", count=50.0, seen=50).density(0.9) == 0.0

    def test_the_rate_is_of_the_people_still_out_there(self):
        """300 in the market at 0.4 is 120 fits: 100 held leaves 20 among the last 100."""
        assert _arm("founder", count=300.0, seen=200, held_fits=100).density(0.4) == (
            pytest.approx(0.2))


@pytest.mark.django_db
class TestThePeopleHeld:
    TERMS = Terms.written(title="founder")

    def test_a_person_counts_as_their_verdict_or_their_fit_rate(self):
        fit, miss = _lead(verdict=1), _lead(verdict=0)
        scored = _lead()
        f = Frontier(None)
        f.people.rates[scored.pk] = 0.25

        arm = f._valued(self.TERMS, 1, "swap", None, 100.0, {fit.pk, miss.pk, scored.pk}, 10.0)

        assert arm.held_fits == pytest.approx(1.25)

    def test_a_person_the_gp_cannot_score_counts_at_the_market_rate(self):
        _lead(verdict=1), _lead(verdict=0)
        unscored = _lead()

        arm = Frontier(None)._valued(self.TERMS, 1, "swap", None, 100.0, {unscored.pk}, 10.0)

        assert arm.held_fits == pytest.approx(0.5)

    def test_unjudged_people_leave_the_density_at_the_markets_rate(self):
        """515: 141 of `president` x `insurance` held, 1 judged a fit, read 1.11 and capped."""
        people = [_lead() for _ in range(20)]
        f = Frontier(None)
        f.people.rates.update({lead.pk: 0.8 for lead in people})

        arm = f._valued(self.TERMS, 1, "swap", None, 100.0, {p.pk for p in people}, 10.0)

        assert arm.density(0.8) == pytest.approx(0.8)


@pytest.mark.django_db
class TestTheNextSearch:
    def test_an_arm_with_nobody_new_is_never_fired_while_another_has_people(self):
        dry, live = _arm("buyer", new=0.0, rate=0.9), _arm("founder", new=50.0, rate=0.5)

        assert frontier._worth_firing([dry, live], None) == [live]

    def test_the_same_round_picks_the_same_search_whatever_the_order(self):
        arms = [_arm(f"title{i}") for i in range(30)]
        f = _settled(arms)

        assert f.pick(list(arms), 4) is f.pick(list(reversed(arms)), 4)

    def test_a_known_good_keyword_beats_an_unknown_one_and_an_unknown_one_a_known_bad(self):
        _companies("Founder", 1, 20)
        _companies("Clerk", 0, 20)
        for i in range(20):
            _lead("Buyer", company=_company(f"unjudged-{i}"))
        good, bad, unknown = _arm("founder"), _arm("clerk"), _arm("buyer")
        f = _settled([good, bad, unknown])

        assert f.pick([good, bad, unknown], 1) is good
        assert f.pick([bad, unknown], 1) is unknown
        assert good.score is not None

    def test_a_page_whose_probe_came_back_without_fits_loses_its_next_page(self):
        """Batch 6: founder · `wholesale` brought 170 people and 0 fits, and the keywords'
        marks still read its next page at 0.35. Its own verdicts are what it is valued on."""
        for i in range(10):
            _lead("Owner", tags="wholesale", company=_company(f"mark{i}"), verdict=1)
        page = [_lead("Founder", tags="wholesale", company=_company(f"w{i}"), verdict=0)
                for i in range(5)]
        fired = Terms.written(title="founder", tag="wholesale")
        _fired(fired, page, found=5000)
        untried = Terms.written(title="owner", tag="wholesale")

        f = Frontier(None)
        (deepen,) = [f._deepen(fired) for fired in f.fired]
        fresh = f._first_page(untried, "drawn")
        f._settle([deepen, fresh])

        assert f.beliefs.search_mean(fired.key) < f.beliefs.market_rate
        assert f.pick([deepen, fresh], 1) is fresh

    def test_a_page_whose_probe_came_back_fits_is_paged_again(self):
        for i in range(10):
            _lead("Owner", tags="wholesale", company=_company(f"mark{i}"), verdict=0)
        page = [_lead("Founder", tags="wholesale", company=_company(f"w{i}"), verdict=1)
                for i in range(5)]
        fired = Terms.written(title="founder", tag="wholesale")
        _fired(fired, page, found=5000)

        f = Frontier(None)
        (deepen,) = [f._deepen(fired) for fired in f.fired]
        fresh = f._first_page(Terms.written(title="owner", tag="wholesale"), "drawn")
        f._settle([deepen, fresh])

        assert f.pick([deepen, fresh], 1) is deepen

    def test_the_next_search_is_nothing_fired_dead_or_drained(self):
        people = [_lead(verdict=1) for _ in range(3)]
        roomy = Terms.written(title="purchasing", tag="cannabis")
        drained = Terms.written(title="manager", tag="cannabis")
        _fired(roomy, people, found=5000)
        _fired(drained, people, found=3)
        dead = Terms.written(title="buyer", tag="cannabis")
        Search.objects.create(key=dead.key, dead=True, terms=dead.as_json())

        f = Frontier(None)
        arm = f.pick(f.arms(), 1)

        assert not (arm.number == 1 and arm.terms in (roomy, drained))
        assert not arm.terms.holds_all_of(dead)
        assert not arm.terms.holds_all_of(drained)

    def test_a_next_page_keeps_the_providers_count_on_any_draw(self):
        people = [_lead("Purchasing Manager") for _ in range(3)]
        _fired(Terms.written(title="purchasing", tag="cannabis"), people, found=5000)
        f = Frontier(None)
        next_page = next(arm for arm in f.arms() if arm.move == "deepen")

        assert f._resized(next_page, f.beliefs.draw_sizes(1)) is next_page

    def test_a_drawn_size_stays_within_the_fired_search_it_narrows_and_above_the_store(self):
        people = [_lead("Purchasing Manager") for _ in range(3)]
        _fired(Terms.written(title="purchasing"), people, found=40)
        _fired(Terms.written(title="founder"), [_lead("Founder")], found=9000,
               origin=Search.Origin.DRAWN)
        f = Frontier(None)
        narrower = Arm(Terms.written(title="purchasing manager"), 1, "stricter", seen=3,
                       keywords=belief.query_keywords(Terms.written(title="purchasing manager")))
        f._settle([narrower])

        counts = [f._resized(narrower, f.beliefs.draw_sizes(n, narrower.keywords)).count
                  for n in range(200)]

        assert max(counts) <= 40
        assert min(counts) >= 3 * frontier.SEEN_MARGIN
        assert len(set(counts)) > 1

    def test_the_pick_returns_the_arm_it_was_given_and_leaves_it_at_the_mean(self):
        arms = [_arm(f"title{i}") for i in range(10)]
        f = _settled(arms)
        counts = [arm.count for arm in arms]

        assert any(f.pick(arms, 1) is arm for arm in arms)
        assert [arm.count for arm in arms] == counts

    def test_a_round_fires_one_search_only_while_one_is_expected_to_bring_a_fit(self):
        _fired(Terms.written(title="purchasing", tag="cannabis"),
               [_lead(verdict=1) for _ in range(3)], found=5000)

        assert len(frontier.next_round(None, 1)) == 1
        with patch.object(frontier, "FITS_WORTH_A_SEARCH", 1e9):
            assert frontier.next_round(None, 1) == []
            assert frontier.best_expected(None) is None

    def test_the_cold_phase_fires_while_an_arm_brings_somebody_new(self):
        """Nothing is known yet to stop on: the card's second failure."""
        class Cold:
            is_cold = True

            def predict_probs(self, X):
                return None

        _fired(Terms.written(title="purchasing", tag="cannabis"), [_lead()], found=5000)

        with (
            patch.object(frontier, "FITS_WORTH_A_SEARCH", 1e9),
            patch("openoutfind.core.ml.embeddings.embed_texts",
                  side_effect=lambda texts: np.zeros((len(texts), 384))),
        ):
            assert len(frontier.next_round(Cold(), 1)) == 1

    def test_the_arm_weighed_against_the_pool_is_the_best_at_the_mean(self):
        for _ in range(3):
            _lead("Founder", verdict=1)
        _fired(Terms.written(title="purchasing", tag="cannabis"),
               [_lead(verdict=1) for _ in range(3)], found=5000)

        f = Frontier(None)
        best = frontier.best_expected(None)

        assert best.expected_fits / best.cost == pytest.approx(
            max(arm.expected_fits / arm.cost for arm in f.arms() if arm.new > 0))


@pytest.mark.django_db
class TestTheBelief:
    def _beliefs(self, rates, judged=(), fired=(), store_rate=0.5):
        return belief.Beliefs(People.load(rates), list(fired), set(judged), store_rate)

    def test_a_judged_fit_a_search_brought_lifts_each_of_its_keywords_alike(self):
        fit = _lead("Founder", tags="cannabis", company=_company("acme"), verdict=1)
        _fired(Terms.written(title="founder", tag="cannabis"), [fit], found=500)
        beliefs = self._beliefs({fit.pk: 1.0}, judged=[fit.pk],
                                fired=frontier.fired_searches())
        title, tag = ("title", "founder"), ("tag", ("cannabis",))

        beliefs.settle({title: 0.5, tag: 0.5})

        lifts = beliefs.lifts()
        assert lifts[title] == pytest.approx(lifts[tag])
        assert lifts[title] > 0.0

    def test_a_person_only_holding_a_keyword_moves_its_prior_and_never_its_strength(self):
        """The store is not the market: a mark is evidence about the customer's corner."""
        fits = [_lead("Founder", company=_company(f"m{i}"), verdict=1) for i in range(30)]
        beliefs = self._beliefs({lead.pk: 1.0 for lead in fits}, judged=[f.pk for f in fits])
        keyword = ("title", "founder")

        beliefs.settle({keyword: 0.5})

        assert beliefs.prior[keyword] == pytest.approx((1.5 + 30) / (3 + 30))
        assert keyword not in beliefs.rates.column

    def test_a_fired_searchs_own_rate_reads_its_pages_people(self):
        page = [_lead("Founder", company=_company(f"c{i}"), verdict=0) for i in range(5)]
        terms = Terms.written(title="founder")
        _fired(terms, page, found=5000)
        beliefs = self._beliefs({lead.pk: 0.0 for lead in page}, judged=[p.pk for p in page],
                                fired=frontier.fired_searches())

        beliefs.settle({("title", "founder"): 0.5})

        assert beliefs.search_mean(terms.key) < beliefs.means()[("title", "founder")]
        assert beliefs.search_mean(terms.key) < 0.2

    def test_the_markets_rate_is_learned_from_what_searches_brought_never_the_store(self):
        """Batch 6 read every unknown keyword at the store's 0.21, its marks' rate; the pages
        it brought held 2 fits in 16 verdicts."""
        page = [_lead("Founder", company=_company(f"c{i}"), verdict=0) for i in range(16)]
        _fired(Terms.written(title="founder"), page, found=5000)
        beliefs = self._beliefs({lead.pk: 0.0 for lead in page}, judged=[p.pk for p in page],
                                fired=frontier.fired_searches(), store_rate=0.21)
        unknown = ("tag", ("food",))

        assert beliefs.market_rate == pytest.approx(0.21)
        beliefs.settle({("title", "founder"): 0.21, unknown: 0.21})

        assert beliefs.market_rate < 0.15
        assert beliefs.means()[unknown] == pytest.approx(beliefs.market_rate)

    def test_an_unjudged_person_counts_by_the_gps_confidence_at_them(self):
        """Batch 6: 780 strangers at confidence ~0.05 read the market's rate whoever they were."""
        sure, unsure = _lead(company=_company("a")), _lead(company=_company("b"))
        people = People.load({sure.pk: 0.8, unsure.pk: 0.8}, {sure.pk: 1.0, unsure.pk: 0.0})

        assert belief.evidence({unsure.pk}, people, set()) == (0.0, 0.0)
        assert belief.evidence({sure.pk}, people, set()) == pytest.approx((0.2, 0.05))

    def test_a_company_adds_once_at_the_mean_of_its_people(self):
        acme = _company("acme")
        leads = [_lead(company=acme, verdict=v) for v in (1, 0, 0, 0)]
        rates = {lead.pk: float(v) for lead, v in zip(leads, (1, 0, 0, 0))}
        people = People.load(rates)

        assert belief.evidence(set(rates), people, set(rates)) == (0.25, 0.75)

    def test_an_unjudged_person_adds_their_fit_rate_at_a_quarter_of_a_verdict(self):
        lead = _lead(company=_company("acme"))
        people = People.load({lead.pk: 0.8})

        assert belief.evidence({lead.pk}, people, set()) == pytest.approx((0.2, 0.05))

    def test_a_companys_verdicts_stand_over_its_scores(self):
        acme = _company("acme")
        judged, scored = _lead(company=acme, verdict=1), _lead(company=acme)
        people = People.load({judged.pk: 1.0, scored.pk: 0.1})

        assert belief.evidence({judged.pk, scored.pk}, people, {judged.pk}) == (1.0, 0.0)

    def test_people_with_no_company_but_one_tag_list_are_one_company(self):
        """The provider's "Self employed" row staples one list onto hundreds."""
        placeholder = "finacial services, private investigations, dive master"
        alone = [_lead(tags=placeholder) for _ in range(4)]
        people = People.load({lead.pk: 0.4 for lead in alone})

        fits, misses = belief.evidence({lead.pk for lead in alone}, people, set())

        assert fits + misses == pytest.approx(belief.SCORE_WEIGHT)

    def test_a_keywords_prior_follows_its_score_and_counts_overtake_it(self):
        keyword = ("title", "clerk")
        empty = self._beliefs({})
        empty.settle({keyword: 0.8})
        assert empty.means()[keyword] == pytest.approx(0.8)

        _companies("Clerk", 0, 30)
        rates = dict.fromkeys(Lead.objects.values_list("pk", flat=True), 0.0)
        beliefs = self._beliefs(rates, judged=rates)
        beliefs.settle({keyword: 0.8})
        assert beliefs.means()[keyword] < 0.1

    def test_a_searchs_keywords_fall_after_a_page_of_low_scored_people_nobody_judged(self):
        """They hold no `purchasing` of their own: the search that brought them counts."""
        page = [_lead("Clerk", company=_company(f"c{i}")) for i in range(8)]
        _fired(Terms.written(title="purchasing"), page, found=5000)
        keyword = ("title", "purchasing")

        beliefs = self._beliefs({lead.pk: 0.05 for lead in page}, fired=frontier.fired_searches())
        beliefs.settle({keyword: 0.5})

        assert beliefs.lifts()[keyword] < 0.0
        assert beliefs.means()[keyword] < 0.5

    def test_a_state_and_its_country_are_one_keyword(self):
        assert belief.query_keywords(Terms.written(title="owner", place=f"Ohio, {US}")) == (
            ("title", "owner"), ("state", "ohio", "united states"))
        assert belief.query_keywords(Terms.written(title="owner", place=US)) == (
            ("title", "owner"), ("country", "united states"))

    def test_a_query_adds_its_keywords_lifts_to_the_markets_log_odds(self):
        beliefs = self._beliefs({})
        first, second = ("title", "a"), ("title", "b")
        lift = math.log(3)  # odds ×3

        assert beliefs.rate([first], {first: 0.0}, alpha=0.0) == pytest.approx(0.5)
        assert beliefs.rate([first, second], {first: lift, second: lift},
                            alpha=0.0) == pytest.approx(0.9)

    def test_a_fired_searchs_own_count_is_recovered(self):
        people = [_lead("Purchasing Manager") for _ in range(4)]
        _fired(Terms.written(title="purchasing manager"), people, found=3000)

        beliefs = self._beliefs({}, fired=frontier.fired_searches())

        assert beliefs.count(belief.query_keywords(
            Terms.written(title="purchasing manager"))) == pytest.approx(3000, rel=0.15)

    def test_with_no_count_the_market_is_its_anchor(self):
        _lead()
        assert self._beliefs({}).sizes.mean().log_market == pytest.approx(
            math.log(belief.MARKET_PEOPLE))

    def _counted(self):
        people = [_lead("Purchasing Manager") for _ in range(4)] + [_lead("Founder")]
        _fired(Terms.written(title="purchasing"), people[:2], found=3000)
        _fired(Terms.written(title="manager"), people[2:4], found=900,
               origin=Search.Origin.DRAWN)
        beliefs = self._beliefs({}, fired=frontier.fired_searches())
        beliefs.settle({("title", "founder"): 0.5, ("title", "purchasing"): 0.5})
        return beliefs

    def test_the_sizes_draw_repeats_on_a_round_and_moves_between_rounds(self):
        beliefs = self._counted()

        assert np.array_equal(beliefs.draw_sizes(3).params, beliefs.draw_sizes(3).params)
        assert not np.array_equal(beliefs.draw_sizes(3).params, beliefs.draw_sizes(4).params)

    def test_the_sizes_draw_centres_on_the_fit(self):
        beliefs = self._counted()
        query = [("title", "purchasing")]

        drawn = [math.log(beliefs.count(query, beliefs.draw_sizes(n))) for n in range(2000)]

        assert np.mean(drawn) == pytest.approx(math.log(beliefs.count(query)), abs=0.1)

    def test_a_large_count_is_held_tighter_than_a_small_one(self):
        """A Poisson count of 2 says little; one of 20,000 says its size to a few percent."""
        def spread(found):
            Search.objects.all().delete()
            _fired(Terms.written(title="purchasing"), [_lead("Purchasing Manager")],
                   found=found)
            beliefs = self._beliefs({}, fired=frontier.fired_searches())
            query = [("title", "purchasing")]
            return np.std([math.log(beliefs.count(query, beliefs.draw_sizes(n)))
                           for n in range(500)])

        assert spread(2) > spread(20_000)

    def test_a_keyword_no_count_held_draws_wider_than_one_a_count_held(self):
        beliefs = self._counted()
        counted, uncounted = ("title", "purchasing"), ("title", "founder")

        def spread(keyword):
            return np.std([beliefs.draw_sizes(n, [keyword]).share(keyword, beliefs.people)
                           for n in range(1000)])

        assert spread(uncounted) > spread(counted)

    def test_a_keyword_the_store_holds_more_reads_larger(self):
        """The store's share is evidence, weighed by γ: only its weight is learned."""
        beliefs = self._counted()

        shares = beliefs.shares([("title", "manager"), ("title", "nobody")])

        assert shares[("title", "manager")] > shares[("title", "nobody")]

    def test_no_keyword_draws_more_than_the_market(self):
        beliefs = self._counted()
        keywords = [("title", "purchasing"), ("title", "founder"), ("tag", ("thc", "vape"))]

        for number in range(200):
            assert max(beliefs.shares(keywords, beliefs.draw_sizes(number, keywords))
                       .values()) <= 0.0

    def test_a_keywords_prior_is_the_gps_score_through_the_fit_rate_map(self):
        class Fixed:
            def predict_probs(self, X):
                return np.array([0.8, 0.2])

        with patch("openoutfind.core.ml.embeddings.embed_texts",
                   side_effect=lambda texts: np.zeros((len(texts), 384))):
            priors = belief.keyword_priors([("title", "a"), ("title", "b")], Fixed(),
                                           lambda p: np.asarray(p) / 2, 0.3)

        assert priors == {("title", "a"): 0.4, ("title", "b"): 0.1}

    def test_a_keyword_is_embedded_once_and_read_back_from_the_store(self):
        embedded = []

        def embed(texts):
            embedded.extend(texts)
            return np.ones((len(texts), 384))

        with patch("openoutfind.core.ml.embeddings.embed_texts", side_effect=embed):
            belief._VECTORS.clear()
            first = belief._vectors(["founder", "cannabis"])
            belief._VECTORS.clear()
            second = belief._vectors(["founder", "cannabis", "owner"])

        assert embedded == ["founder", "cannabis", "owner"]
        assert np.array_equal(first, second[:2])

    def test_a_new_embedding_model_embeds_again(self):
        from openoutfind.core.conf import CAMPAIGN_CONFIG

        with patch("openoutfind.core.ml.embeddings.embed_texts",
                   side_effect=lambda texts: np.ones((len(texts), 384))) as embed:
            belief._VECTORS.clear()
            belief._vectors(["founder"])
            belief._VECTORS.clear()
            with patch.dict(CAMPAIGN_CONFIG, {"embedding_model": "another"}):
                belief._vectors(["founder"])

        assert embed.call_count == 2

    def test_a_gp_that_cannot_score_leaves_the_market_rate(self):
        class Unfitted:
            def predict_probs(self, X):
                return None

        with patch("openoutfind.core.ml.embeddings.embed_texts",
                   side_effect=lambda texts: np.zeros((len(texts), 384))):
            assert belief.keyword_priors([("title", "c")], Unfitted(), None, 0.3) == {
                ("title", "c"): 0.3}
        assert belief.keyword_priors([("title", "c")], None, None, 0.3) == {("title", "c"): 0.3}


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
