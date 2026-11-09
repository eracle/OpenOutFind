# tests/test_belief.py
"""The walk's belief: one posterior over the sizes and the fit rate, read at the mean and on
a draw."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from openoutfind.core.models import Search
from openoutfind.core.pipeline import belief
from openoutfind.core.pipeline.store_snapshot import People, fired_searches
from openoutfind.core.pipeline.terms import Terms
from tests.store_builders import US, fire, make_company, make_lead

BATCH_6 = Path(__file__).parent / "fixtures" / "batch6_belief.json"
FIELD = {"title": "title", "tag": "tag", "seniority": "seniority", "state": "place",
         "country": "place"}


def batch_6_inputs(searches: int | None = None) -> belief.Inputs:
    """Batch 6's first ``searches`` fired searches as the model reads them."""
    raw = json.loads(BATCH_6.read_text())
    fired = raw["searches"][:searches]
    keywords = tuple(sorted({k for s in fired for k in s["keywords"]}))
    column = {k: j for j, k in enumerate(keywords)}
    member = np.zeros((len(fired), len(keywords)))
    fields = np.zeros((len(fired), len(belief.FIELDS)))
    for i, s in enumerate(fired):
        for keyword in s["keywords"]:
            member[i, column[keyword]] = 1.0
            fields[i, belief.FIELDS.index(FIELD[keyword.split("|")[0]])] += 1.0
    return belief.Inputs(
        keywords=keywords, searches=tuple(f"s{i}" for i in range(len(fired))), member=member,
        fields=fields, store_share=np.array([raw["store_log_share"][k] for k in keywords]),
        counted=np.arange(len(fired)), counts=np.array([s["found"] for s in fired], float),
        fits=np.array([s["fits"] for s in fired]), misses=np.array([s["misses"] for s in fired]),
        market_log_odds=belief.log_odds(raw["store_rate"]))


def make_beliefs(rates, judged=(), fired=(), store_rate=0.5):
    return belief.Beliefs(People.load(rates), list(fired), set(judged), store_rate)


def counted_beliefs():
    """Two counted searches, and a founder no search holds."""
    people = [make_lead("Purchasing Manager") for _ in range(4)] + [make_lead("Founder")]
    fire(Terms.written(title="purchasing"), people[:2], found=3000)
    fire(Terms.written(title="manager"), people[2:4], found=900, origin=Search.Origin.DRAWN)
    return make_beliefs({}, fired=fired_searches())


class TestTheSampler:
    def test_batch_6_reads_its_market_rate(self):
        """The Newton fit this replaced stopped at α = +17.86 on these 23 searches, and read
        every arm at θ = 1."""
        posterior = belief.sample_posterior(batch_6_inputs())

        assert 0.005 < belief.sigmoid(posterior.alpha).mean() < 0.1

    def test_three_fired_searches_fit(self):
        posterior = belief.sample_posterior(batch_6_inputs(3))

        assert posterior.draws == belief.CHAINS * belief.DRAWS
        assert np.isfinite(posterior.alpha).all()

    def test_with_no_fired_search_the_belief_is_the_prior(self):
        posterior = belief.sample_posterior(batch_6_inputs(0))

        assert posterior.log_market.mean() == pytest.approx(math.log(belief.MARKET_PEOPLE),
                                                            abs=0.2)
        assert posterior.alpha.std() == pytest.approx(belief.MARKET_RATE_SPREAD, abs=0.1)


@pytest.mark.django_db
class TestTheFitRate:
    def test_a_keyword_whose_page_brought_fits_lifts_above_one_whose_page_did_not(self):
        fits = [make_lead("Founder", company=make_company(f"f{i}"), verdict=1) for i in range(10)]
        misses = [make_lead("Clerk", company=make_company(f"c{i}"), verdict=0) for i in range(10)]
        fire(Terms.written(title="founder"), fits, found=5000)
        fire(Terms.written(title="clerk"), misses, found=5000, origin=Search.Origin.DRAWN)
        rates = {lead.pk: 1.0 for lead in fits} | {lead.pk: 0.0 for lead in misses}

        beliefs = make_beliefs(rates, judged=rates, fired=fired_searches())

        lifts = beliefs.lifts()
        assert lifts[("title", "founder")] > 0.0 > lifts[("title", "clerk")]

    def test_the_markets_rate_is_learned_from_what_searches_brought_never_the_store(self):
        """Batch 6 read every unknown keyword at the store's 0.21, its marks' rate; the pages
        it brought held 2 fits in 16 verdicts."""
        page = [make_lead("Founder", company=make_company(f"c{i}"), verdict=0) for i in range(16)]
        terms = Terms.written(title="founder")
        fire(terms, page, found=5000)

        beliefs = make_beliefs({lead.pk: 0.0 for lead in page}, judged=[p.pk for p in page],
                               fired=fired_searches(), store_rate=0.21)

        assert beliefs.market_rate < 0.15
        assert beliefs.search_mean(terms.key) < 0.15

    def test_a_keyword_no_search_held_reads_the_markets_rate(self):
        beliefs = counted_beliefs()
        unknown = ("tag", ("food",))

        assert beliefs.lifts()[unknown] == 0.0
        assert beliefs.rate([unknown], beliefs.lifts()) == pytest.approx(beliefs.market_rate)

    def test_a_query_adds_its_keywords_lifts_to_the_markets_log_odds(self):
        beliefs = make_beliefs({})
        first, second = ("title", "a"), ("title", "b")
        lift = math.log(3)  # odds ×3

        assert beliefs.rate([first], {first: 0.0}, alpha=0.0) == pytest.approx(0.5)
        assert beliefs.rate([first, second], {first: lift, second: lift},
                            alpha=0.0) == pytest.approx(0.9)


@pytest.mark.django_db
class TestTheDraw:
    def test_a_seed_names_one_posterior_draw_for_both_halves(self):
        beliefs = counted_beliefs()
        keywords = {("title", "purchasing"), ("title", "founder")}
        key = Terms.written(title="purchasing").key

        first, again = beliefs.draw((1, 0), keywords), beliefs.draw((1, 0), keywords)
        sizes = beliefs.draw_sizes((1, 0))

        assert first.lifts == again.lifts and first.alpha == again.alpha
        index = beliefs.draw_index((1, 0))
        assert first.alpha == beliefs.posterior.alpha[index]
        assert sizes.log_market == first.sizes.log_market == beliefs.posterior.log_market[index]
        assert sizes.pairs(("title", "founder")) == first.sizes.pairs(("title", "founder"))
        assert beliefs.theta_on(first, (), key) == first.searches[key]

    def test_the_draws_move_between_seeds(self):
        beliefs = counted_beliefs()

        assert len({beliefs.draw_sizes(n).log_market for n in range(20)}) > 1

    def test_a_keyword_no_search_held_draws_its_pair_from_the_draws_own_spread(self):
        beliefs = counted_beliefs()
        unknown = ("title", "founder")
        fitted = ("title", "purchasing")

        pairs = np.array([beliefs.draw_sizes(n).pairs(unknown) for n in range(1000)])

        chol = beliefs.posterior.chol
        assert abs(pairs[:, 1].mean()) < 0.15
        assert pairs[:, 1].std() == pytest.approx(
            np.sqrt((chol[:, 1, 0] ** 2 + chol[:, 1, 1] ** 2).mean()), rel=0.15)
        index = beliefs.draw_index(7)
        column = beliefs.posterior.column[fitted]
        assert beliefs.draw_sizes(7).pairs(fitted) == (
            beliefs.posterior.size[index, column], beliefs.posterior.lift[index, column])


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
        """Within a factor of 2: one count leaves its dispersion, φ, little known."""
        people = [make_lead("Purchasing Manager") for _ in range(4)]
        fire(Terms.written(title="purchasing manager"), people, found=3000)

        beliefs = make_beliefs({}, fired=fired_searches())

        count = beliefs.count(belief.query_keywords(Terms.written(title="purchasing manager")))
        assert 1500 < count < 6000

    def test_a_large_count_is_held_tighter_than_a_small_one(self):
        """A count of 2 says little; one of 20,000 says its size to a few percent."""
        def spread(found):
            Search.objects.all().delete()
            belief.FITTED = None
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
            return np.std([beliefs.draw_sizes(n).share(keyword, beliefs.people)
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
            assert max(beliefs.shares(keywords, beliefs.draw_sizes(number)).values()) <= 0.0
