# tests/test_compose.py
"""The composition — the searches worth the most on one draw, from every keyword."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from openoutfind.core.pipeline import compose
from openoutfind.core.pipeline.belief import query_keywords
from openoutfind.core.pipeline.compose import Vocabulary
from openoutfind.core.pipeline.terms import Terms

US = "united states"
LOG_MARKET = math.log(1_000_000)


def _worth(lift, share):
    """A page's fits per dollar: a share of the market at most a page, a fixed cost."""
    theta = 1.0 / (1.0 + np.exp(-(-3.0 + lift)))
    new = np.minimum(200.0, np.exp(LOG_MARKET + share))
    return theta * new / (5.0 + 0.05 * new)


def _vocabulary(titles=("founder",), tags=(("cannabis",),), seniorities=(), places=()):
    return Vocabulary(tuple(("title", w) for w in titles), tuple(("tag", t) for t in tags),
                      tuple(("seniority", s) for s in seniorities), tuple(places))


def _all(vocabulary: Vocabulary):
    """Every search the vocabulary holds: a title set, a tag, a seniority or none, a place."""
    titles = [kw for kw in vocabulary.titles]
    seniorities = [None, *vocabulary.seniorities]
    places = list(vocabulary.places) or [None]
    for size in range(1, len(titles) + 1):
        for words in itertools.combinations(titles, size):
            for tag in vocabulary.tags:
                for seniority in seniorities:
                    for place in places:
                        yield tuple(k for k in (*words, tag, seniority, place) if k)


def _score(keywords, lift, share):
    return float(_worth(np.array(sum(lift[k] for k in keywords)),
                        np.array(sum(share[k] for k in keywords))))


class TestCompose:
    def test_the_best_search_is_the_best_of_every_combination(self):
        """Six title words, three tags, two seniorities, three places: 3,276 searches."""
        rng = np.random.default_rng(7)
        vocabulary = _vocabulary(
            titles=("founder", "owner", "extraction", "production", "manager", "lab"),
            tags=(("cannabis",), ("thc", "vape"), ("food",)),
            seniorities=("founder", "director"),
            places=(("country", US), ("state", "ohio", US), ("state", "texas", US)))
        for draw in range(20):
            lift = {k: float(rng.normal(0, 1.5)) for k in vocabulary.keywords()}
            share = {k: float(-rng.exponential(2.5)) for k in vocabulary.keywords()}

            best = compose.compose(vocabulary, lift, share, LOG_MARKET, _worth, limit=5)[0]

            truth = max(_score(k, lift, share) for k in _all(vocabulary))
            got = _score(query_keywords(best), lift, share)
            assert got == pytest.approx(truth, rel=0.05), draw

    def test_every_search_holds_a_title_and_one_tag_and_the_place(self):
        vocabulary = _vocabulary(titles=("founder", "owner"), tags=(("cannabis",), ("vapes",)),
                                 places=(("state", "ohio", US),))
        lift = dict.fromkeys(vocabulary.keywords(), 0.5)
        share = dict.fromkeys(vocabulary.keywords(), -1.0)

        searches = compose.compose(vocabulary, lift, share, LOG_MARKET, _worth, limit=50)

        assert searches
        assert all(s.title and len(s.tag) == 1 and s.place == "Ohio, United States"
                   for s in searches)

    def test_a_word_that_lowers_the_rate_ranks_below_the_search_without_it(self):
        vocabulary = _vocabulary(titles=("founder", "clerk"))
        lift = {("title", "founder"): 2.0, ("title", "clerk"): -2.0, ("tag", ("cannabis",)): 0.0}
        share = dict.fromkeys(vocabulary.keywords(), -1.0)

        titles = [s.title for s in compose.compose(vocabulary, lift, share, LOG_MARKET,
                                                   _worth, limit=10)]

        assert titles[0] == {"founder"}
        assert titles.index({"founder"}) < titles.index({"founder", "clerk"})

    def test_a_search_matching_under_one_person_is_never_composed(self):
        vocabulary = _vocabulary(titles=("founder", "owner"))
        lift = dict.fromkeys(vocabulary.keywords(), 3.0)
        share = {("title", "founder"): -8.0, ("title", "owner"): -8.0,
                 ("tag", ("cannabis",)): -1.0}

        searches = compose.compose(vocabulary, lift, share, LOG_MARKET, _worth, limit=10)

        assert {frozenset(s.title) for s in searches} == {frozenset({"founder"}),
                                                          frozenset({"owner"})}

    def test_with_no_title_or_no_tag_nothing_is_composed(self):
        assert compose.compose(_vocabulary(titles=()), {}, {}, LOG_MARKET, _worth, 10) == []
        assert compose.compose(_vocabulary(tags=()), {}, {}, LOG_MARKET, _worth, 10) == []

    def test_a_composed_search_reads_back_as_its_keywords(self):
        keywords = (("title", "head"), ("title", "procurement"), ("tag", ("cannabis", "extraction")),
                    ("seniority", "director"), ("state", "new york", US))

        terms = compose.as_terms(keywords)

        assert terms == Terms(title=frozenset({"head", "procurement"}),
                              tag=("cannabis", "extraction"), place="New York, United States",
                              seniority="director")
        assert set(query_keywords(terms)) == set(keywords)
