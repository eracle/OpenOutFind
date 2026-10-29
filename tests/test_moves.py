# tests/test_moves.py
"""The walk's moves — each within one field's type, drawing terms from the store."""
import random
from collections import Counter

import pytest

from openoutfind.core.pipeline import moves
from openoutfind.core.pipeline.moves import TermPools
from openoutfind.core.pipeline.query import Terms
from openoutfind.discovery import LEAD_SENIORITIES

US = "United States"


def _pools(titles=("manager",), tags=(("cannabis", "edibles"),), states=("Ohio",)):
    return TermPools(Counter({t: 1.0 for t in titles}), Counter({t: 1.0 for t in tags}),
                     Counter({s: 1.0 for s in states}), tuple(LEAD_SENIORITIES), "united states")


def _on(move, field, terms, pools=None, seed=0):
    """One move on one named field."""
    return moves._ONE_FIELD[move](terms, field, pools or _pools(), random.Random(seed))


class TestStricter:
    def test_title_adds_a_word(self):
        child = _on("stricter", "title", Terms.written(title="production"))
        assert child.title == {"production", "manager"}

    def test_a_tag_extends_to_a_real_tag_holding_it(self):
        """`cannabis` → `cannabis edibles`, a phrase a company's tags hold."""
        child = _on("stricter", "tag", Terms.written(title="purchasing", tag="cannabis"))
        assert child.tag == ("cannabis", "edibles")

    def test_a_tag_no_real_tag_extends_has_no_stricter_child(self):
        pools = _pools(tags=(("vapes",),))
        assert _on("stricter", "tag", Terms.written(title="p", tag="cannabis"), pools) is None

    def test_no_tag_gains_one(self):
        child = _on("stricter", "tag", Terms.written(title="purchasing"))
        assert child.tag == ("cannabis", "edibles")

    def test_a_country_narrows_to_a_state_in_it(self):
        child = _on("stricter", "place", Terms.written(title="founder", place=US))
        assert child.place == "Ohio, United States"

    def test_a_seniority_is_added_only_when_there_is_none(self):
        assert _on("stricter", "seniority", Terms.written(title="f")).seniority
        assert _on("stricter", "seniority", Terms.written(title="f", seniority="vp")) is None


class TestWider:
    def test_title_drops_a_word(self):
        child = _on("wider", "title", Terms.written(title="production manager"))
        assert len(child.title) == 1

    def test_a_state_widens_to_its_country(self):
        child = _on("wider", "place", Terms.written(title="f", place="Ohio, United States"))
        assert child.place == US

    def test_a_tag_shortens_or_goes(self):
        terms = Terms.written(title="p", tag="cannabis edibles")
        children = {_on("wider", "tag", terms, seed=seed).tag for seed in range(20)}
        assert children <= {(), ("cannabis",), ("edibles",)}
        assert len(children) > 1

    def test_the_last_title_word_never_goes_while_a_tag_is_held(self):
        terms = Terms.written(title="purchasing", tag="cannabis")
        for seed in range(10):
            child = moves.mutate("wider", terms, _pools(), random.Random(seed))
            assert child is None or child.is_legal


class TestSwap:
    def test_title_replaces_a_word(self):
        child = _on("swap", "title", Terms.written(title="production"))
        assert child.title == {"manager"}

    def test_another_state(self):
        pools = _pools(states=("Ohio", "Texas"))
        child = _on("swap", "place", Terms.written(title="f", place="Ohio, United States"), pools)
        assert child.place == "Texas, United States"

    def test_another_seniority(self):
        child = _on("swap", "seniority", Terms.written(title="f", seniority="vp"))
        assert child.seniority not in ("", "vp")


class TestCrossover:
    def test_two_searches_exchange_a_fields_terms(self):
        """`production × vapes` and `purchasing × cannabis` give the two other pairs."""
        a = Terms.written(title="production", tag="vapes")
        b = Terms.written(title="purchasing", tag="cannabis")

        children = set(moves.crossover(a, b, random.Random(0)))

        assert children in (
            {Terms.written(title="production", tag="cannabis"),
             Terms.written(title="purchasing", tag="vapes")},
        )

    def test_identical_searches_have_no_children(self):
        a = Terms.written(title="production")
        assert moves.crossover(a, a, random.Random(0)) == []


class TestMutate:
    def test_a_child_is_legal_and_differs_from_its_parent(self):
        terms = Terms.written(title="purchasing", tag="cannabis", place=US)
        for move in ("stricter", "wider", "swap"):
            for seed in range(10):
                child = moves.mutate(move, terms, _pools(), random.Random(seed))
                assert child is None or (child.is_legal and child != terms)


@pytest.mark.django_db
class TestPools:
    def test_terms_are_drawn_toward_the_people_who_fit(self):
        from openoutfind.crm.models import Lead

        fit = Lead.objects.create(profile_url="https://x/1", source_fields={
            "contact_job_title": "Purchasing Manager", "company_keywords": "thc vape",
            "contact_location_state": "Ohio", "contact_location_country": "United States"})
        miss = Lead.objects.create(profile_url="https://x/2", source_fields={
            "contact_job_title": "Chocolatier", "company_keywords": "chocolate",
            "contact_location_state": "Ontario", "contact_location_country": "Canada"})

        pools = TermPools.load({fit.pk: 1.0, miss.pk: 0.0},
                               [Terms.written(title="buyer", tag="cannabis")], "united states")

        assert pools.titles["purchasing"] > pools.titles["chocolatier"] > 0
        assert pools.tags[("thc", "vape")] == 1.0
        assert pools.titles["buyer"] == 1.0 and pools.tags[("cannabis",)] == 1.0
        assert "Ontario" not in pools.states  # outside the market
        assert pools.states["Ohio"] == 1.0
