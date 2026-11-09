# tests/test_belief.py
"""The walk's belief: the sizes and the fit rate per keyword, at the mean and on a draw."""
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core.models import Search
from openoutfind.core.pipeline import belief
from openoutfind.core.pipeline.store_snapshot import People, fired_searches
from openoutfind.core.pipeline.terms import Terms
from openoutfind.crm.models import Lead
from tests.store_builders import US, fire, make_companies, make_company, make_lead


def make_beliefs(rates, judged=(), fired=(), store_rate=0.5):
    return belief.Beliefs(People.load(rates), list(fired), set(judged), store_rate)


def counted_beliefs():
    """Two counted searches and a founder no search holds, settled."""
    people = [make_lead("Purchasing Manager") for _ in range(4)] + [make_lead("Founder")]
    fire(Terms.written(title="purchasing"), people[:2], found=3000)
    fire(Terms.written(title="manager"), people[2:4], found=900, origin=Search.Origin.DRAWN)
    beliefs = make_beliefs({}, fired=fired_searches())
    beliefs.settle({("title", "founder"): 0.5, ("title", "purchasing"): 0.5})
    return beliefs


@pytest.mark.django_db
class TestTheFitRate:
    def test_a_judged_fit_a_search_brought_lifts_each_of_its_keywords_alike(self):
        fit = make_lead("Founder", tags="cannabis", company=make_company("acme"), verdict=1)
        fire(Terms.written(title="founder", tag="cannabis"), [fit], found=500)
        beliefs = make_beliefs({fit.pk: 1.0}, judged=[fit.pk], fired=fired_searches())
        title, tag = ("title", "founder"), ("tag", ("cannabis",))

        beliefs.settle({title: 0.5, tag: 0.5})

        lifts = beliefs.lifts()
        assert lifts[title] == pytest.approx(lifts[tag])
        assert lifts[title] > 0.0

    def test_a_person_only_holding_a_keyword_moves_its_prior_and_never_its_strength(self):
        """The store is not the market: a mark is evidence about the customer's corner."""
        fits = [make_lead("Founder", company=make_company(f"m{i}"), verdict=1) for i in range(30)]
        beliefs = make_beliefs({lead.pk: 1.0 for lead in fits}, judged=[f.pk for f in fits])
        keyword = ("title", "founder")

        beliefs.settle({keyword: 0.5})

        assert beliefs.prior[keyword] == pytest.approx((1.5 + 30) / (3 + 30))
        assert keyword not in beliefs.rates.column

    def test_a_fired_searchs_own_rate_reads_its_pages_people(self):
        page = [make_lead("Founder", company=make_company(f"c{i}"), verdict=0) for i in range(5)]
        terms = Terms.written(title="founder")
        fire(terms, page, found=5000)
        beliefs = make_beliefs({lead.pk: 0.0 for lead in page}, judged=[p.pk for p in page],
                               fired=fired_searches())

        beliefs.settle({("title", "founder"): 0.5})

        assert beliefs.search_mean(terms.key) < beliefs.means()[("title", "founder")]
        assert beliefs.search_mean(terms.key) < 0.2

    def test_the_markets_rate_is_learned_from_what_searches_brought_never_the_store(self):
        """Batch 6 read every unknown keyword at the store's 0.21, its marks' rate; the pages
        it brought held 2 fits in 16 verdicts."""
        page = [make_lead("Founder", company=make_company(f"c{i}"), verdict=0) for i in range(16)]
        fire(Terms.written(title="founder"), page, found=5000)
        beliefs = make_beliefs({lead.pk: 0.0 for lead in page}, judged=[p.pk for p in page],
                               fired=fired_searches(), store_rate=0.21)
        unknown = ("tag", ("food",))

        assert beliefs.market_rate == pytest.approx(0.21)
        beliefs.settle({("title", "founder"): 0.21, unknown: 0.21})

        assert beliefs.market_rate < 0.15
        assert beliefs.means()[unknown] == pytest.approx(beliefs.market_rate)

    def test_a_keywords_prior_follows_its_score_and_counts_overtake_it(self):
        keyword = ("title", "clerk")
        empty = make_beliefs({})
        empty.settle({keyword: 0.8})
        assert empty.means()[keyword] == pytest.approx(0.8)

        make_companies("Clerk", 0, 30)
        rates = dict.fromkeys(Lead.objects.values_list("pk", flat=True), 0.0)
        beliefs = make_beliefs(rates, judged=rates)
        beliefs.settle({keyword: 0.8})
        assert beliefs.means()[keyword] < 0.1

    def test_a_searchs_keywords_fall_after_a_page_of_low_scored_people_nobody_judged(self):
        """They hold no `purchasing` of their own: the search that brought them counts."""
        page = [make_lead("Clerk", company=make_company(f"c{i}")) for i in range(8)]
        fire(Terms.written(title="purchasing"), page, found=5000)
        keyword = ("title", "purchasing")

        beliefs = make_beliefs({lead.pk: 0.05 for lead in page}, fired=fired_searches())
        beliefs.settle({keyword: 0.5})

        assert beliefs.lifts()[keyword] < 0.0
        assert beliefs.means()[keyword] < 0.5

    def test_a_query_adds_its_keywords_lifts_to_the_markets_log_odds(self):
        beliefs = make_beliefs({})
        first, second = ("title", "a"), ("title", "b")
        lift = math.log(3)  # odds ×3

        assert beliefs.rate([first], {first: 0.0}, alpha=0.0) == pytest.approx(0.5)
        assert beliefs.rate([first, second], {first: lift, second: lift},
                            alpha=0.0) == pytest.approx(0.9)

    def test_a_draw_repeats_on_its_seed_and_reads_a_fired_searchs_own_rate(self):
        beliefs = counted_beliefs()
        keywords = {("title", "purchasing"), ("title", "founder")}
        key = Terms.written(title="purchasing").key

        first, again = beliefs.draw((1, 0), keywords), beliefs.draw((1, 0), keywords)

        assert first.lifts == again.lifts and first.alpha == again.alpha
        assert beliefs.theta_on(first, (), key) == first.searches[key]
        assert np.array_equal(first.sizes.params, beliefs.draw_sizes((1, 0)).params)


@pytest.mark.django_db
class TestTheEvidence:
    def test_an_unjudged_person_counts_by_the_gps_confidence_at_them(self):
        """Batch 6: 780 strangers at confidence ~0.05 read the market's rate whoever they were."""
        sure, unsure = make_lead(company=make_company("a")), make_lead(company=make_company("b"))
        people = People.load({sure.pk: 0.8, unsure.pk: 0.8}, {sure.pk: 1.0, unsure.pk: 0.0})

        assert belief.evidence({unsure.pk}, people, set()) == (0.0, 0.0)
        assert belief.evidence({sure.pk}, people, set()) == pytest.approx((0.2, 0.05))

    def test_a_company_adds_once_at_the_mean_of_its_people(self):
        acme = make_company("acme")
        leads = [make_lead(company=acme, verdict=v) for v in (1, 0, 0, 0)]
        rates = {lead.pk: float(v) for lead, v in zip(leads, (1, 0, 0, 0))}
        people = People.load(rates)

        assert belief.evidence(set(rates), people, set(rates)) == (0.25, 0.75)

    def test_an_unjudged_person_adds_their_fit_rate_at_a_quarter_of_a_verdict(self):
        lead = make_lead(company=make_company("acme"))
        people = People.load({lead.pk: 0.8})

        assert belief.evidence({lead.pk}, people, set()) == pytest.approx((0.2, 0.05))

    def test_a_companys_verdicts_stand_over_its_scores(self):
        acme = make_company("acme")
        judged, scored = make_lead(company=acme, verdict=1), make_lead(company=acme)
        people = People.load({judged.pk: 1.0, scored.pk: 0.1})

        assert belief.evidence({judged.pk, scored.pk}, people, {judged.pk}) == (1.0, 0.0)

    def test_people_with_no_company_but_one_tag_list_are_one_company(self):
        """The provider's "Self employed" row staples one list onto hundreds."""
        placeholder = "finacial services, private investigations, dive master"
        alone = [make_lead(tags=placeholder) for _ in range(4)]
        people = People.load({lead.pk: 0.4 for lead in alone})

        fits, misses = belief.evidence({lead.pk for lead in alone}, people, set())

        assert fits + misses == pytest.approx(belief.SCORE_WEIGHT)

    def test_a_state_and_its_country_are_one_keyword(self):
        assert belief.query_keywords(Terms.written(title="owner", place=f"Ohio, {US}")) == (
            ("title", "owner"), ("state", "ohio", "united states"))
        assert belief.query_keywords(Terms.written(title="owner", place=US)) == (
            ("title", "owner"), ("country", "united states"))


@pytest.mark.django_db
class TestTheSizes:
    def test_a_fired_searchs_own_count_is_recovered(self):
        people = [make_lead("Purchasing Manager") for _ in range(4)]
        fire(Terms.written(title="purchasing manager"), people, found=3000)

        beliefs = make_beliefs({}, fired=fired_searches())

        assert beliefs.count(belief.query_keywords(
            Terms.written(title="purchasing manager"))) == pytest.approx(3000, rel=0.15)

    def test_with_no_count_the_market_is_its_anchor(self):
        make_lead()
        assert make_beliefs({}).sizes.mean().log_market == pytest.approx(
            math.log(belief.MARKET_PEOPLE))

    def test_the_sizes_draw_repeats_on_a_round_and_moves_between_rounds(self):
        beliefs = counted_beliefs()

        assert np.array_equal(beliefs.draw_sizes(3).params, beliefs.draw_sizes(3).params)
        assert not np.array_equal(beliefs.draw_sizes(3).params, beliefs.draw_sizes(4).params)

    def test_the_sizes_draw_centres_on_the_fit(self):
        beliefs = counted_beliefs()
        query = [("title", "purchasing")]

        drawn = [math.log(beliefs.count(query, beliefs.draw_sizes(n))) for n in range(2000)]

        assert np.mean(drawn) == pytest.approx(math.log(beliefs.count(query)), abs=0.1)

    def test_a_large_count_is_held_tighter_than_a_small_one(self):
        """A Poisson count of 2 says little; one of 20,000 says its size to a few percent."""
        def spread(found):
            Search.objects.all().delete()
            fire(Terms.written(title="purchasing"), [make_lead("Purchasing Manager")],
                 found=found)
            beliefs = make_beliefs({}, fired=fired_searches())
            query = [("title", "purchasing")]
            return np.std([math.log(beliefs.count(query, beliefs.draw_sizes(n)))
                           for n in range(500)])

        assert spread(2) > spread(20_000)

    def test_a_keyword_no_count_held_draws_wider_than_one_a_count_held(self):
        beliefs = counted_beliefs()
        counted, uncounted = ("title", "purchasing"), ("title", "founder")

        def spread(keyword):
            return np.std([beliefs.draw_sizes(n, [keyword]).share(keyword, beliefs.people)
                           for n in range(1000)])

        assert spread(uncounted) > spread(counted)

    def test_a_keyword_the_store_holds_more_reads_larger(self):
        """The store's share is evidence, weighed by γ: only its weight is learned."""
        beliefs = counted_beliefs()

        shares = beliefs.shares([("title", "manager"), ("title", "nobody")])

        assert shares[("title", "manager")] > shares[("title", "nobody")]

    def test_no_keyword_draws_more_than_the_market(self):
        beliefs = counted_beliefs()
        keywords = [("title", "purchasing"), ("title", "founder"), ("tag", ("thc", "vape"))]

        for number in range(200):
            assert max(beliefs.shares(keywords, beliefs.draw_sizes(number, keywords))
                       .values()) <= 0.0


@pytest.mark.django_db
class TestThePrior:
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
            belief.KEYWORD_VECTORS.clear()
            first = belief.keyword_vectors(["founder", "cannabis"])
            belief.KEYWORD_VECTORS.clear()
            second = belief.keyword_vectors(["founder", "cannabis", "owner"])

        assert embedded == ["founder", "cannabis", "owner"]
        assert np.array_equal(first, second[:2])

    def test_a_new_embedding_model_embeds_again(self):
        from openoutfind.core.conf import CAMPAIGN_CONFIG

        with patch("openoutfind.core.ml.embeddings.embed_texts",
                   side_effect=lambda texts: np.ones((len(texts), 384))) as embed:
            belief.KEYWORD_VECTORS.clear()
            belief.keyword_vectors(["founder"])
            belief.KEYWORD_VECTORS.clear()
            with patch.dict(CAMPAIGN_CONFIG, {"embedding_model": "another"}):
                belief.keyword_vectors(["founder"])

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
