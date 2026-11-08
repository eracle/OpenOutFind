# tests/test_terms.py
"""A search's terms and who they match.

Every expectation is a measured provider answer (the probe table above
``discovery.SEARCH_FIELDS``): a matcher that disagrees with the provider scores a page on
people the search did not ask for.
"""
import pytest

from openoutfind.core.pipeline.analyzers import title_words
from openoutfind.core.pipeline.terms import Terms, split_place


class TestWritten:
    def test_a_hyphenated_title_matches_every_spelling(self):
        """`co-founder` = `co founder` = `cofounder` (179,644)."""
        held = title_words("Co-Founder & CEO")
        for spelling in ("co-founder", "co founder", "cofounder"):
            assert Terms.written(title=spelling).title <= held

    def test_nothing_is_stemmed(self):
        """`manager` 1,140,870, `managers` 231."""
        assert not Terms.written(title="managers").title <= title_words("Production Manager")

    def test_a_stopword_only_tag_is_never_a_term(self):
        assert Terms.written(title="founder", tag="and").tag == ()

    def test_accented_terms_reach_the_provider_whole(self):
        terms = Terms.written(title="directeur général", tag="agence immobilière",
                              place="France")
        assert terms.to_filters()["lead_job_title"]["include"] == ["directeur général"]
        assert terms.to_filters()["company_keywords"]["include"] == ["agence immobilière"]

    def test_an_accented_search_matches_its_own_spelling_only(self):
        """`gérant` 35,377, `gerant` 3,972: two terms."""
        row = {"contact_job_title": "Gérant"}
        assert Terms.written(title="gérant").matches(row)
        assert not Terms.written(title="gerant").matches(row)


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

    def test_a_row_with_no_state_matches_a_state_of_its_own_country_only(self):
        """The provider returns a state on US rows alone."""
        row = {"contact_location_country": "Indonesia"}
        assert Terms.written(place="Bali, Indonesia").matches(row)
        assert not Terms.written(place="Bali, Malaysia").matches(row)

    def test_a_row_with_another_state_does_not_match(self):
        assert not Terms.written(place="Ohio, United States").matches(self.PERSON)

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
