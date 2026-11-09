# tests/test_search_arms.py
"""The search arms — every search the walk could fire next, valued from the store."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core.models import PAGE_SIZE, Search
from openoutfind.core.pipeline import belief, search_arms
from openoutfind.core.pipeline.search_arms import Arm, Frontier
from openoutfind.core.pipeline.terms import Terms, split_place
from tests.store_builders import US, fire, make_company, make_lead


def arm_for(title, count=1000.0, new=100.0, seen=0, held_fits=0, rate=0.0):
    terms = Terms.written(title=title)
    return Arm(terms, 1, "swap", keywords=belief.query_keywords(terms), count=count, seen=seen,
               held_fits=held_fits, new=new, rate=rate, cost=1.0)


def rated_frontier(arms):
    frontier = Frontier(None)
    frontier.rate_at_mean(arms)
    return frontier


@pytest.mark.django_db
class TestTheArms:
    def test_a_drained_search_offers_neither_its_next_page_nor_a_stricter_search(self):
        """Batch 5 spent all 5 stricter slots of a round re-finding people this way."""
        people = [make_lead(verdict=1) for _ in range(3)]
        drained = Terms.written(title="purchasing", tag="cannabis")
        fire(drained, people, found=3)

        arms = Frontier(None).arms()

        assert not any(arm.terms.holds_all_of(drained) for arm in arms)

    def test_a_market_written_in_states_never_goes_country_wide(self):
        """The Bali sample: a fired country search's next page and its neighbours are out
        with the state's own widening, so neither the cover nor the fill can leave Bali."""
        people = [make_lead(verdict=1) for _ in range(3)]
        fire(Terms.written(title="owner", tag="cannabis", place="Bali, Indonesia"),
             people, found=500)
        fire(Terms.written(title="manager", tag="cannabis", place="Indonesia"), people,
             found=500, origin=Search.Origin.DRAWN)

        arms = Frontier(None).arms()

        assert arms and all(split_place(arm.terms.place)[0] for arm in arms)

    def test_a_market_written_partly_on_the_country_keeps_it(self):
        people = [make_lead(verdict=1) for _ in range(3)]
        fire(Terms.written(title="owner", tag="cannabis", place=f"Ohio, {US}"), people,
             found=500)
        fire(Terms.written(title="manager", tag="cannabis", place=US), people, found=500)

        arms = Frontier(None).arms()

        assert any(not split_place(arm.terms.place)[0] for arm in arms)

    def test_a_search_with_more_to_page_offers_its_next_page_beside_the_composed(self):
        people = [make_lead(verdict=1) for _ in range(3)]
        roomy = Terms.written(title="purchasing", tag="cannabis")
        fire(roomy, people, found=5000)

        arms = Frontier(None).arms()

        assert any(arm.move == "deepen" and arm.number == 2 for arm in arms)
        assert any(arm.move == "drawn" and arm.number == 1 and arm.terms != roomy
                   for arm in arms)

    def test_a_composed_search_holds_a_title_and_a_tag_from_the_store(self):
        make_lead("Extraction Lead", tags="cannabis extraction, live resin", verdict=1)
        fire(Terms.written(title="purchasing", tag="cannabis", place=f"Ohio, {US}"),
             [make_lead(verdict=1)], found=5000)

        composed = [arm.terms for arm in Frontier(None).arms() if arm.move == "drawn"]

        assert composed
        vocabulary = {"extraction", "lead", "purchasing", "manager"}
        assert all(t.title and t.title <= vocabulary and t.tag for t in composed)
        assert any(t.tag == ("live", "resin") for t in composed)

    def test_a_store_with_no_written_search_composes_nothing(self):
        make_lead(verdict=1)

        assert not any(arm.move == "drawn" for arm in Frontier(None).arms())

    def test_a_dead_searchs_terms_are_never_an_arm(self):
        fire(Terms.written(title="purchasing"), [make_lead(verdict=1)], found=5000)
        Search.objects.create(key=Terms.written(title="manager").key, dead=True,
                              terms=Terms.written(title="manager").as_json())
        dead = Terms.written(title="manager")

        assert not any(arm.terms.holds_all_of(dead) for arm in Frontier(None).arms())

    def test_an_unfired_count_stays_within_what_it_narrows_and_above_the_store(self):
        people = [make_lead("Purchasing Manager") for _ in range(3)]
        fire(Terms.written(title="purchasing"), people, found=8000)
        stricter = Terms.written(title="purchasing manager")

        frontier = Frontier(None)
        with patch.object(frontier.beliefs, "count", return_value=1e9):
            high = frontier.first_page_arm(stricter, "stricter")
        with patch.object(frontier.beliefs, "count", return_value=1.0):
            low = frontier.first_page_arm(stricter, "stricter")

        assert high.count == 8000
        assert low.count == pytest.approx(3 * search_arms.SEEN_MARGIN)
        assert low.new > 0  # a low estimate does not close the search for good

    def test_a_known_count_replaces_the_estimate(self):
        fire(Terms.written(title="purchasing"), [make_lead()], found=5000)

        frontier = Frontier(None)
        with patch.object(frontier.beliefs, "count", return_value=7.0):
            (next_page,) = [frontier.next_page_arm(fired) for fired in frontier.fired]

        assert next_page.count == 5000

    def test_a_page_is_valued_at_no_more_than_a_page_however_large_the_search(self):
        fire(Terms.written(title="purchasing"), [make_lead()], found=10_000_000)

        frontier = Frontier(None)
        (next_page,) = [frontier.next_page_arm(fired) for fired in frontier.fired]
        with patch.object(frontier.beliefs, "count", return_value=10_000_000.0):
            unfired = frontier.first_page_arm(Terms.written(title="founder"), "swap")

        assert next_page.new <= PAGE_SIZE and unfired.new <= PAGE_SIZE

    def test_a_page_that_brought_nobody_new_is_not_worth_its_next(self):
        known = [make_lead(verdict=1) for _ in range(3)]
        fire(Terms.written(title="purchasing"), known, found=5000)
        fire(Terms.written(title="manager"), known, found=5000)

        next_pages = [arm for arm in Frontier(None).arms() if arm.is_next_page]

        by_title = {next(iter(arm.terms.title)): arm.new for arm in next_pages}
        assert by_title["purchasing"] > 0 and by_title["manager"] == 0


class TestAnArmsValue:
    def test_an_area_whose_fits_the_store_holds_is_worth_nothing_however_large(self):
        arm = arm_for("founder", count=100_000.0, seen=5000, held_fits=5000)

        assert arm.density(0.05) == 0.0

    def test_a_search_with_nobody_out_there_is_worth_nothing(self):
        assert arm_for("founder", count=50.0, seen=50).density(0.9) == 0.0

    def test_the_rate_is_of_the_people_still_out_there(self):
        """300 in the market at 0.4 is 120 fits: 100 held leaves 20 among the last 100."""
        assert arm_for("founder", count=300.0, seen=200, held_fits=100).density(0.4) == (
            pytest.approx(0.2))


@pytest.mark.django_db
class TestThePeopleHeld:
    TERMS = Terms.written(title="founder")

    def test_a_person_counts_as_their_verdict_or_their_fit_rate(self):
        fit, miss = make_lead(verdict=1), make_lead(verdict=0)
        scored = make_lead()
        frontier = Frontier(None)
        frontier.people.rates[scored.pk] = 0.25

        arm = frontier.valued_arm(self.TERMS, 1, "swap", None, 100.0,
                                  {fit.pk, miss.pk, scored.pk}, 10.0)

        assert arm.held_fits == pytest.approx(1.25)

    def test_a_person_the_gp_cannot_score_counts_at_the_market_rate(self):
        make_lead(verdict=1), make_lead(verdict=0)
        unscored = make_lead()

        arm = Frontier(None).valued_arm(self.TERMS, 1, "swap", None, 100.0, {unscored.pk}, 10.0)

        assert arm.held_fits == pytest.approx(0.5)

    def test_unjudged_people_leave_the_density_at_the_markets_rate(self):
        """515: 141 of `president` x `insurance` held, 1 judged a fit, read 1.11 and capped."""
        people = [make_lead() for _ in range(20)]
        frontier = Frontier(None)
        frontier.people.rates.update({lead.pk: 0.8 for lead in people})

        arm = frontier.valued_arm(self.TERMS, 1, "swap", None, 100.0, {p.pk for p in people}, 10.0)

        assert arm.density(0.8) == pytest.approx(0.8)


@pytest.mark.django_db
class TestTheNextSearch:
    def test_an_arm_with_nobody_new_is_never_fired_while_another_has_people(self):
        dry, live = arm_for("buyer", new=0.0, rate=0.9), arm_for("founder", new=50.0, rate=0.5)

        assert search_arms.worth_firing([dry, live], None) == [live]

    def test_the_same_round_picks_the_same_search_whatever_the_order(self):
        arms = [arm_for(f"title{i}") for i in range(30)]
        frontier = rated_frontier(arms)

        assert frontier.pick(list(arms), 4) is frontier.pick(list(reversed(arms)), 4)

    def test_a_known_good_keyword_beats_an_unknown_one_and_an_unknown_one_a_known_bad(self):
        """Known from what the searches holding them brought: the store alone moves no lift."""
        fits = [make_lead("Founder", company=make_company(f"f{i}"), verdict=1) for i in range(15)]
        misses = [make_lead("Clerk", company=make_company(f"c{i}"), verdict=0) for i in range(15)]
        fire(Terms.written(title="founder", tag="cannabis"), fits, found=5000)
        fire(Terms.written(title="clerk", tag="cannabis"), misses, found=5000,
             origin=Search.Origin.DRAWN)
        good, bad, unknown = arm_for("founder"), arm_for("clerk"), arm_for("buyer")
        frontier = rated_frontier([good, bad, unknown])

        assert frontier.pick([good, bad, unknown], 1) is good
        assert frontier.pick([bad, unknown], 1) is unknown
        assert good.score is not None

    def test_a_page_whose_probe_came_back_without_fits_loses_its_next_page(self):
        """Batch 6: founder · `wholesale` brought 170 people and 0 fits, and the keywords'
        marks still read its next page at 0.35. Its own verdicts are what it is valued on."""
        for i in range(10):
            make_lead("Owner", tags="wholesale", company=make_company(f"mark{i}"), verdict=1)
        page = [make_lead("Founder", tags="wholesale", company=make_company(f"w{i}"), verdict=0)
                for i in range(5)]
        fired = Terms.written(title="founder", tag="wholesale")
        fire(fired, page, found=5000)
        untried = Terms.written(title="owner", tag="wholesale")

        frontier = Frontier(None)
        (next_page,) = [frontier.next_page_arm(f) for f in frontier.fired]
        fresh = frontier.first_page_arm(untried, "drawn")
        frontier.rate_at_mean([next_page, fresh])

        assert frontier.beliefs.search_mean(fired.key) < frontier.beliefs.market_rate
        assert frontier.pick([next_page, fresh], 1) is fresh

    def test_a_page_whose_probe_came_back_fits_is_paged_again(self):
        for i in range(10):
            make_lead("Owner", tags="wholesale", company=make_company(f"mark{i}"), verdict=0)
        page = [make_lead("Founder", tags="wholesale", company=make_company(f"w{i}"), verdict=1)
                for i in range(5)]
        fire(Terms.written(title="founder", tag="wholesale"), page, found=5000)

        frontier = Frontier(None)
        (next_page,) = [frontier.next_page_arm(f) for f in frontier.fired]
        fresh = frontier.first_page_arm(Terms.written(title="owner", tag="wholesale"), "drawn")
        frontier.rate_at_mean([next_page, fresh])

        assert frontier.pick([next_page, fresh], 1) is next_page

    def test_the_next_search_is_nothing_fired_dead_or_drained(self):
        people = [make_lead(verdict=1) for _ in range(3)]
        roomy = Terms.written(title="purchasing", tag="cannabis")
        drained = Terms.written(title="manager", tag="cannabis")
        fire(roomy, people, found=5000)
        fire(drained, people, found=3)
        dead = Terms.written(title="buyer", tag="cannabis")
        Search.objects.create(key=dead.key, dead=True, terms=dead.as_json())

        frontier = Frontier(None)
        arm = frontier.pick(frontier.arms(), 1)

        assert not (arm.number == 1 and arm.terms in (roomy, drained))
        assert not arm.terms.holds_all_of(dead)
        assert not arm.terms.holds_all_of(drained)

    def test_the_belief_refits_only_when_a_page_or_a_verdict_lands(self):
        """It is rebuilt on every judge-or-search decision, and a fit takes seconds."""
        people = [make_lead(company=make_company(f"c{i}")) for i in range(3)]
        fire(Terms.written(title="purchasing", tag="cannabis"), people, found=5000)

        with patch.object(belief, "sample_posterior", wraps=belief.sample_posterior) as fit:
            Frontier(None), Frontier(None)
            assert fit.call_count == 1
            make_lead(verdict=1)
            Frontier(None)
            assert fit.call_count == 2
            fire(Terms.written(title="purchasing", tag="cannabis"), [make_lead()], found=5000,
                 number=2)
            Frontier(None)
            assert fit.call_count == 3

    def test_a_next_page_keeps_the_providers_count_on_any_draw(self):
        people = [make_lead("Purchasing Manager") for _ in range(3)]
        fire(Terms.written(title="purchasing", tag="cannabis"), people, found=5000)
        frontier = Frontier(None)
        next_page = next(arm for arm in frontier.arms() if arm.is_next_page)

        assert frontier.resized(next_page, frontier.beliefs.draw_sizes(1)) is next_page

    def test_a_drawn_size_stays_within_the_fired_search_it_narrows_and_above_the_store(self):
        people = [make_lead("Purchasing Manager") for _ in range(3)]
        fire(Terms.written(title="purchasing"), people, found=40)
        fire(Terms.written(title="founder"), [make_lead("Founder")], found=9000,
             origin=Search.Origin.DRAWN)
        frontier = Frontier(None)
        narrower = Arm(Terms.written(title="purchasing manager"), 1, "stricter", seen=3,
                       keywords=belief.query_keywords(Terms.written(title="purchasing manager")))
        counts = [frontier.resized(narrower, frontier.beliefs.draw_sizes(n)).count
                  for n in range(200)]

        assert max(counts) <= 40
        assert min(counts) >= 3 * search_arms.SEEN_MARGIN
        assert len(set(counts)) > 1

    def test_the_pick_returns_the_arm_it_was_given_and_leaves_it_at_the_mean(self):
        arms = [arm_for(f"title{i}") for i in range(10)]
        frontier = rated_frontier(arms)
        counts = [arm.count for arm in arms]

        assert any(frontier.pick(arms, 1) is arm for arm in arms)
        assert [arm.count for arm in arms] == counts

    def test_a_round_fires_one_search_only_while_one_is_expected_to_bring_a_fit(self):
        fire(Terms.written(title="purchasing", tag="cannabis"),
             [make_lead(verdict=1) for _ in range(3)], found=5000)

        assert len(search_arms.next_round(None, 1)) == 1
        with patch.object(search_arms, "FITS_WORTH_A_SEARCH", 1e9):
            assert search_arms.next_round(None, 1) == []
            assert search_arms.best_expected(None) is None

    def test_the_cold_phase_fires_while_an_arm_brings_somebody_new(self):
        """Nothing is known yet to stop on: the card's second failure."""
        class Cold:
            is_cold = True

            def predict_probs(self, X):
                return None

        fire(Terms.written(title="purchasing", tag="cannabis"), [make_lead()], found=5000)

        with (
            patch.object(search_arms, "FITS_WORTH_A_SEARCH", 1e9),
            patch("openoutfind.core.ml.embeddings.embed_texts",
                  side_effect=lambda texts: np.zeros((len(texts), 384))),
        ):
            assert len(search_arms.next_round(Cold(), 1)) == 1

    def test_the_arm_weighed_against_the_pool_is_the_best_at_the_mean(self):
        for _ in range(3):
            make_lead("Founder", verdict=1)
        fire(Terms.written(title="purchasing", tag="cannabis"),
             [make_lead(verdict=1) for _ in range(3)], found=5000)

        frontier = Frontier(None)
        best = search_arms.best_expected(None)

        assert best.expected_fits / best.cost == pytest.approx(
            max(arm.expected_fits / arm.cost for arm in frontier.arms() if arm.new > 0))
