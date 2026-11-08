# tests/test_analyzers.py
"""The provider's analyzers, copied. Every expectation is a measured Lead Finder count."""
import pytest

from openoutfind.core.pipeline.analyzers import (
    company_tags, holds_phrase, phrase_words, query_title_words, standalone_title_words,
    title_words, words,
)


class TestWords:
    @pytest.mark.parametrize("text, expected", [
        ("Gérant", ["gérant"]),
        ("Directeur Général", ["directeur", "général"]),
        ("Geschäftsführer, Straße", ["geschäftsführer", "straße"]),
        ("Właściciel", ["właściciel"]),
        ("Генеральный директор", ["генеральный", "директор"]),
        ("Γενικός Διευθυντής", ["γενικός", "διευθυντής"]),
    ])
    def test_accented_letters_stay_inside_their_word(self, text, expected):
        """`gérant` 35,377, `gerant` 3,972: the provider keeps the accent."""
        assert words(text) == expected

    def test_a_straight_apostrophe_joins_and_a_curly_one_splits(self):
        """`d'affaires` = `daffaires` (44,226); `d’affaires` = `d affaires` (24,122)."""
        assert words("Chargé d'affaires") == ["chargé", "daffaires"]
        assert words("Chargé d’affaires") == ["chargé", "d", "affaires"]

    def test_dots_inside_a_word_join_it(self):
        """`c.e.o` = `ceo` (395,720)."""
        assert words("C.E.O.") == ["ceo"]

    def test_the_turkish_dotted_capital_keeps_its_dot(self):
        """`İşletme` 3,984, `işletme` 486: lowercased as `str.lower` does, one word."""
        assert words("İşletme Sahibi") == ["i̇şletme", "sahibi"]
        assert words("İşletme") != words("işletme")

    def test_a_thai_run_is_one_word(self):
        """`ผู้จัดการ` 71, `จัดการ` 2: no dictionary split."""
        assert words("ผู้จัดการทั่วไป") == ["ผู้จัดการทั่วไป"]

    def test_a_chinese_character_is_a_word(self):
        """`经理` = `理经` (242,808)."""
        assert words("总经理") == ["总", "经", "理"]
        assert set(words("经理")) == set(words("理经"))

    def test_no_text_is_no_words(self):
        assert words("") == []
        assert words(None) == []
        assert words(" & / - ") == []


class TestTitleAnalyzer:
    def test_stopwords_are_dropped_and_order_ignored(self):
        """`head of procurement` = `head the procurement` = `procurement head` (89)."""
        assert (query_title_words("head of procurement")
                == query_title_words("procurement head")
                == {"head", "procurement"})

    def test_a_stopword_only_title_has_no_words(self):
        assert query_title_words("of the") == frozenset()

    def test_a_hyphen_splits_and_also_joins(self):
        assert title_words("CEO/Co-Founder") == {"ceo", "co", "founder", "cofounder"}

    def test_an_accented_hyphenated_title_joins_whole(self):
        assert "directricegénérale" in title_words("Directrice-Générale")

    def test_a_piece_of_a_hyphenated_word_is_no_keyword(self):
        """`co` alone 283,068: co-owners and co-CEOs beyond the 179,644 co-founders."""
        assert standalone_title_words("Co-Founder & CEO") == {"ceo"}
        assert standalone_title_words("Founder") == {"founder"}


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

    def test_an_underscore_joins(self):
        """`home_and_garden` 1,625 vs `home and garden` 431."""
        assert phrase_words("home_and_garden/home_and_garden") == (
            "home_and_garden", "home_and_garden")

    def test_an_accented_tag_is_whole(self):
        assert phrase_words("Agence Immobilière") == ("agence", "immobilière")

    def test_a_phrase_is_adjacent_and_in_order(self):
        """`cannabis vapes` 76, `vapes cannabis` 0."""
        tag = ("cannabis", "vapes", "and", "edibles")
        assert holds_phrase(tag, ("cannabis", "vapes"))
        assert not holds_phrase(tag, ("vapes", "cannabis"))
        assert not holds_phrase(tag, ("cannabis", "edibles"))

    def test_a_tag_list_is_split_on_commas(self):
        assert company_tags("cannabis, thc vape, ") == [("cannabis",), ("thc", "vape")]
