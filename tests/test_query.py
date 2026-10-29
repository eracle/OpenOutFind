# tests/test_query.py
"""A search's terms — the provider's analyzers, copied, and who a search matches.

Every expectation here is a measured provider answer (the probe table above
``discovery.SEARCH_FIELDS``): a local matcher that disagrees with the provider scores a
page on people it did not ask for.
"""
import pytest

from openoutfind.core.pipeline.query import (
    Terms, company_tags, holds_phrase, phrase_words, query_title_words, split_place,
    title_words,
)


class TestTitleAnalyzer:
    def test_stopwords_are_dropped_and_order_ignored(self):
        """`head of procurement` = `head the procurement` = `procurement head` (89)."""
        assert (query_title_words("head of procurement")
                == query_title_words("procurement head")
                == {"head", "procurement"})

    def test_a_stopword_only_title_has_no_words(self):
        assert query_title_words("of the") == frozenset()

    def test_a_hyphen_splits_and_also_matches_the_joined_form(self):
        """`co-founder` = `co founder` = `cofounder` (161,369)."""
        held = title_words("Co-Founder & CEO")
        for spelling in ("co-founder", "co founder", "cofounder"):
            assert Terms.written(title=spelling).title <= held

    def test_nothing_is_stemmed(self):
        """`manager` 1,140,870, `managers` 231."""
        assert not Terms.written(title="managers").title <= title_words("Production Manager")


class TestTagAnalyzer:
    def test_stopwords_are_kept_as_words_of_the_phrase(self):
        """`food and beverage` 187,794 vs `food beverage` 162,754."""
        assert phrase_words("Food and Beverage") == ("food", "and", "beverage")

    def test_a_hyphen_only_splits(self):
        """`e-commerce` = `e commerce`; `ecommerce` is another term."""
        assert phrase_words("e-commerce") == ("e", "commerce")
        assert phrase_words("ecommerce") == ("ecommerce",)

    def test_ampersand_is_stripped(self):
        assert phrase_words("food & beverage") == ("food", "beverage")

    def test_a_phrase_is_adjacent_and_in_order(self):
        """`cannabis vapes` 76, `vapes cannabis` 0."""
        tag = ("cannabis", "vapes", "and", "edibles")
        assert holds_phrase(tag, ("cannabis", "vapes"))
        assert not holds_phrase(tag, ("vapes", "cannabis"))
        assert not holds_phrase(tag, ("cannabis", "edibles"))

    def test_a_tag_list_is_split_on_commas(self):
        assert company_tags("cannabis, thc vape, ") == [("cannabis",), ("thc", "vape")]

    def test_a_stopword_only_tag_is_never_a_term(self):
        assert Terms.written(title="founder", tag="and").tag == ()


class TestPlace:
    def test_a_state_carries_its_country(self):
        assert split_place("California, United States") == ("california", "united states")

    def test_a_country_alone(self):
        assert split_place("United States") == ("", "united states")


class TestLegal:
    def test_a_tag_never_fires_without_a_title(self):
        assert not Terms.written(tag="cannabis", seniority="director").is_legal
        assert Terms.written(title="purchasing", tag="cannabis").is_legal

    def test_a_place_alone_is_the_whole_market(self):
        assert not Terms.written(place="United States").is_legal

    def test_a_seniority_with_a_place_is_a_search(self):
        assert Terms.written(seniority="director", place="United States").is_legal


class TestMatches:
    PERSON = {
        "contact_job_title": "Head of Procurement",
        "company_keywords": "cannabis extraction, thc vape, food & beverage",
        "contact_location_state": "California",
        "contact_location_country": "United States",
        "contact_seniority": "Head",
    }

    def test_every_term_in_its_own_field(self):
        terms = Terms.written(title="procurement head", tag="thc vape",
                              place="California, United States", seniority="head")
        assert terms.matches(self.PERSON)

    def test_a_title_word_elsewhere_in_the_profile_does_not_count(self):
        """The regulator with "cannabis" in a tag, not a title, is not a title match."""
        assert not Terms.written(title="cannabis").matches(self.PERSON)

    def test_a_tag_word_in_the_title_does_not_count(self):
        assert not Terms.written(title="head", tag="procurement").matches(self.PERSON)

    def test_a_tag_phrase_must_sit_inside_one_tag(self):
        """Two tags cannot be ANDed: `extraction thc` spans two of them."""
        assert not Terms.written(title="head", tag="extraction thc").matches(self.PERSON)

    def test_another_state_of_the_country_does_not_match(self):
        terms = Terms.written(title="head", place="Texas, United States")
        assert not terms.matches(self.PERSON)
        assert Terms.written(title="head", place="United States").matches(self.PERSON)

    def test_a_person_without_the_field_does_not_match(self):
        assert not Terms.written(seniority="head").matches({"contact_job_title": "Head"})


class TestHoldsAllOf:
    """A stricter search matches a subset of people, so a dead one convicts it."""

    DEAD = Terms.written(title="purchasing", tag="cannabis", place="United States")

    @pytest.mark.parametrize("child", [
        Terms.written(title="purchasing manager", tag="cannabis", place="United States"),
        Terms.written(title="purchasing", tag="cannabis edibles", place="United States"),
        Terms.written(title="purchasing", tag="cannabis", place="Ohio, United States"),
        Terms.written(title="purchasing", tag="cannabis", place="United States",
                      seniority="director"),
    ])
    def test_a_stricter_search_holds_the_dead_ones_terms(self, child):
        assert child.holds_all_of(self.DEAD)

    @pytest.mark.parametrize("child", [
        Terms.written(title="purchasing", place="United States"),
        Terms.written(title="purchasing", tag="vapes", place="United States"),
        Terms.written(title="purchasing", tag="cannabis", place="Canada"),
    ])
    def test_a_wider_or_sibling_search_does_not(self, child):
        assert not child.holds_all_of(self.DEAD)


class TestProviderJson:
    def test_each_field_as_its_match_type_wants(self):
        terms = Terms.written(title="procurement head", tag="Cannabis Extraction",
                              place="California, United States", seniority="head")

        assert terms.to_filters((1, 50)) == {
            "company_headcount_min": 1, "company_headcount_max": 50,
            "lead_job_title": {"include": ["head procurement"], "exact_match": False},
            "company_keywords": {"include": ["cannabis extraction"], "exact_match": False},
            "lead_location": {"include": ["California, United States"]},
            "lead_seniority": {"include": ["head"]},
        }

    def test_an_empty_field_is_absent(self):
        assert set(Terms.written(title="founder").to_filters()) == {"lead_job_title"}


class TestIdentity:
    def test_one_search_however_it_was_written(self):
        a = Terms.written(title="head of procurement", tag="Cannabis")
        b = Terms.written(title="procurement head", tag="cannabis")
        assert a.key == b.key

    def test_the_json_round_trips(self):
        terms = Terms.written(title="head procurement", tag="thc vape",
                              place="Ohio, United States", seniority="head")
        assert Terms.from_json(terms.as_json()) == terms
