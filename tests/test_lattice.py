# tests/test_lattice.py
"""The lattice's neighbours — each within one field's type, made from the store's terms."""
from collections import Counter
from unittest.mock import patch

import pytest

from openoutfind.core.pipeline import lattice
from openoutfind.core.pipeline.lattice import TermPools
from openoutfind.core.pipeline.query import Terms
from openoutfind.discovery import LEAD_SENIORITIES

US = "United States"


def _pools(titles=("manager",), tags=(("cannabis", "edibles"),), states=("Ohio",)):
    return TermPools(Counter({t: 1.0 for t in titles}), Counter({t: 1.0 for t in tags}),
                     Counter({s: 1.0 for s in states}), tuple(LEAD_SENIORITIES), "united states")


def _on(move, field, terms, pools=None):
    """One move on one named field."""
    return lattice._ONE_FIELD[move](terms, field, pools or _pools())


class TestStricter:
    def test_title_adds_a_word(self):
        assert _on("stricter", "title", Terms.written(title="production")) == [
            Terms.written(title="production manager")]

    def test_a_tag_extends_to_a_real_tag_holding_it(self):
        """`cannabis` → `cannabis edibles`, a phrase a company's tags hold."""
        (child,) = _on("stricter", "tag", Terms.written(title="purchasing", tag="cannabis"))
        assert child.tag == ("cannabis", "edibles")

    def test_a_tag_no_real_tag_extends_has_no_stricter_neighbour(self):
        pools = _pools(tags=(("vapes",),))
        assert _on("stricter", "tag", Terms.written(title="p", tag="cannabis"), pools) == []

    def test_no_tag_gains_one(self):
        (child,) = _on("stricter", "tag", Terms.written(title="purchasing"))
        assert child.tag == ("cannabis", "edibles")

    def test_a_country_narrows_to_a_state_in_it(self):
        (child,) = _on("stricter", "place", Terms.written(title="founder", place=US))
        assert child.place == "Ohio, United States"

    def test_every_seniority_is_added_only_when_there_is_none(self):
        assert len(_on("stricter", "seniority", Terms.written(title="f"))) == len(LEAD_SENIORITIES)
        assert _on("stricter", "seniority", Terms.written(title="f", seniority="vp")) == []

    def test_each_field_offers_only_its_heaviest_terms(self):
        pools = _pools(titles=[f"word{i}" for i in range(30)])
        with patch.dict(lattice.WALK_CONFIG, {"terms_per_field": 4}):
            assert len(_on("stricter", "title", Terms.written(title="f"), pools)) == 4


class TestWider:
    def test_title_drops_each_word(self):
        children = _on("wider", "title", Terms.written(title="production manager"))
        assert {child.title for child in children} == {frozenset({"production"}),
                                                        frozenset({"manager"})}

    def test_a_state_widens_to_its_country(self):
        (child,) = _on("wider", "place", Terms.written(title="f", place="Ohio, United States"))
        assert child.place == US

    def test_a_tag_shortens_from_either_end(self):
        children = _on("wider", "tag", Terms.written(title="p", tag="cannabis edibles"))
        assert {child.tag for child in children} == {("cannabis",), ("edibles",)}

    def test_the_tag_is_never_dropped(self):
        """A tag-less search is the market's head: batch 5's brought 334 strangers."""
        assert _on("wider", "tag", Terms.written(title="p", tag="cannabis")) == []
        for child, _ in lattice.neighbours(Terms.written(title="p", tag="cannabis"), _pools()):
            assert child.tag

    def test_a_tagless_search_has_only_neighbours_that_add_a_tag(self):
        children = lattice.neighbours(Terms.written(title="production", place=US), _pools())
        assert children == [(Terms.written(title="production", tag="cannabis edibles",
                                           place=US), "stricter")]

    def test_an_exchange_never_leaves_the_tag_behind(self):
        a = Terms.written(title="production")
        b = Terms.written(title="purchasing", tag="cannabis")
        assert lattice.exchange(a, b) == [Terms.written(title="production", tag="cannabis")]

    def test_the_last_title_word_never_goes_while_a_tag_is_held(self):
        terms = Terms.written(title="purchasing", tag="cannabis")
        assert all(child.is_legal for child, _ in lattice.neighbours(terms, _pools()))


class TestSwap:
    def test_title_replaces_a_word(self):
        assert _on("swap", "title", Terms.written(title="production")) == [
            Terms.written(title="manager")]

    def test_another_state(self):
        pools = _pools(states=("Ohio", "Texas"))
        (child,) = _on("swap", "place", Terms.written(title="f", place="Ohio, United States"),
                       pools)
        assert child.place == "Texas, United States"

    def test_every_other_seniority(self):
        children = _on("swap", "seniority", Terms.written(title="f", seniority="vp"))
        assert len(children) == len(LEAD_SENIORITIES) - 1
        assert all(child.seniority not in ("", "vp") for child in children)


class TestExchange:
    def test_two_searches_exchange_a_fields_terms(self):
        """`production × vapes` and `purchasing × cannabis` give the two other pairs."""
        a = Terms.written(title="production", tag="vapes")
        b = Terms.written(title="purchasing", tag="cannabis")

        assert set(lattice.exchange(a, b)) == {
            Terms.written(title="production", tag="cannabis"),
            Terms.written(title="purchasing", tag="vapes")}

    def test_identical_searches_give_nothing(self):
        a = Terms.written(title="production")
        assert lattice.exchange(a, a) == []

    def test_two_fields_exchanged_give_each_result_once(self):
        a = Terms.written(title="production", tag="vapes", place=US)
        b = Terms.written(title="purchasing", tag="cannabis", place=US)
        assert len(lattice.exchange(a, b)) == 2

    def test_every_pair_is_exchanged(self):
        searches = [Terms.written(title=title, tag=tag) for title, tag in
                    (("founder", "cannabis"), ("buyer", "vapes"), ("owner", "edibles"))]
        assert {move for _, move in lattice.exchanges(searches)} == {"exchange"}
        assert len(lattice.exchanges(searches)) == 6


class TestNeighbours:
    def test_every_neighbour_is_legal_and_differs_from_its_search(self):
        terms = Terms.written(title="purchasing", tag="cannabis", place=US)
        for child, move in lattice.neighbours(terms, _pools()):
            assert child.is_legal and child != terms
            assert move in lattice.MOVES


@pytest.mark.django_db
class TestPools:
    def test_terms_weigh_what_the_people_holding_them_look_like(self):
        from openoutfind.crm.models import Lead

        fit = Lead.objects.create(profile_url="https://x/1", source_fields={
            "contact_job_title": "Purchasing Manager", "company_keywords": "thc vape",
            "contact_location_state": "Ohio", "contact_location_country": "United States"})
        miss = Lead.objects.create(profile_url="https://x/2", source_fields={
            "contact_job_title": "Chocolatier", "company_keywords": "chocolate",
            "contact_location_state": "Ontario", "contact_location_country": "Canada"})

        pools = TermPools.load({fit.pk: 1.0, miss.pk: 0.0}, {},
                               [Terms.written(title="buyer", tag="cannabis")], "united states")

        assert pools.titles["purchasing"] > pools.titles["chocolatier"] > 0
        assert pools.tags[("thc", "vape")] == 1.0
        assert pools.titles["buyer"] == 1.0 and pools.tags[("cannabis",)] == 1.0
        assert "Ontario" not in pools.states  # outside the market
        assert pools.states["Ohio"] == 1.0

    def test_a_company_weighs_once_however_many_people_it_has(self):
        """The provider's "Self employed" row: one tag list stapled onto 289 people."""
        from openoutfind.crm.models import Lead

        placeholder = [Lead.objects.create(profile_url=f"https://x/p{i}", source_fields={
            "contact_job_title": "Freelancer", "company_keywords": "dive master"})
            for i in range(50)]
        fit = Lead.objects.create(profile_url="https://x/fit", source_fields={
            "contact_job_title": "Buyer", "company_keywords": "thc vape"})
        company = {**{lead.pk: "self employed" for lead in placeholder}, fit.pk: "acme"}

        pools = TermPools.load({**{lead.pk: 0.3 for lead in placeholder}, fit.pk: 1.0},
                               company, [], "united states")

        assert pools.tags[("dive", "master")] == pytest.approx(0.3)
        assert pools.tags[("thc", "vape")] == 1.0
