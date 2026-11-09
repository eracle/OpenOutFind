# tests/test_keyword_guess.py
"""The LLM's guess about the market and every keyword of the walk: the market once, each
field's keywords in one answer against it, refused where a count breaks it, and read by the
belief as its prior."""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from openoutfind.core import agent_qualify
from openoutfind.core.models import KeywordGuess, MarketGuess
from openoutfind.core.pipeline import belief, keyword_guess
from openoutfind.core.pipeline.keyword_guess import (
    FieldAnswer, KeywordAnswer, KeywordGuesses, MarketAnswer,
)
from openoutfind.core.pipeline.qualify import KeywordGuessPending

MANAGER, ROSIN, US = ("title", "manager"), ("tag", "rosin"), ("country", "united states")
MARKET = MarketAnswer(people=10_000_000, fits=5_000)


@pytest.fixture(autouse=True)
def _reset_agent_qualify():
    """A contextvar carries state past a test's own scope otherwise."""
    yield
    agent_qualify._active.set(False)
    agent_qualify._keyword_guesses.set(None)


@pytest.fixture
def described(configure, operator):
    return configure(product_docs="We sell cart hardware.", campaign_target="Cannabis brands.",
                     ai_model="anthropic:claude-haiku-5-5")


def keywords(*rows):
    return [KeywordAnswer(keyword=k, people=p, fits=f) for k, p, f in rows]


def store(*rows, written_by="earlier"):
    if keyword_guess.stored_market() is None:
        keyword_guess.store_market(MARKET, written_by)
    asked = {row[0] for row in rows}
    return keyword_guess.store_keywords(keywords(*rows), asked, written_by)


def written_by(keyword):
    return KeywordGuess.objects.get(keyword=keyword).written_by


@pytest.mark.django_db
class TestGuessing:
    def test_the_market_is_guessed_once_before_any_keyword(self, described):
        def field(site_config, field, texts):
            assert MarketGuess.objects.exists()
            return FieldAnswer(keywords=keywords(*[(text, 400.0, 40.0) for text in texts]))

        with patch.object(keyword_guess, "guess_market_with_llm", return_value=MARKET) as market, \
                patch.object(keyword_guess, "guess_field_with_llm", side_effect=field):
            keyword_guess.ensure(described, [ROSIN])
            keyword_guess.ensure(described, [ROSIN, MANAGER])

        market.assert_called_once()
        assert MarketGuess.objects.count() == 1

    def test_each_field_with_unguessed_keywords_is_asked_once_for_them_alone(self, described):
        store(("title|manager", 9e6, 500.0))

        def field(site_config, field, texts):
            return FieldAnswer(keywords=keywords(*[(text, 400.0, 40.0) for text in texts]))

        with patch.object(keyword_guess, "guess_field_with_llm", side_effect=field) as asked:
            keyword_guess.ensure(described, [MANAGER, ROSIN, US])

        assert sorted(call.args[1:] for call in asked.call_args_list) == [
            ("place", ["country|united states"]), ("tag", ["tag|rosin"])]
        assert written_by("tag|rosin") == described.ai_model
        assert written_by("title|manager") == "earlier"

    def test_a_keyword_is_guessed_once(self):
        store(("tag|rosin", 400.0, 40.0))

        store(("tag|rosin", 9e6, 1.0), ("tag|live resin", 300.0, 90.0), written_by="later")

        assert KeywordGuess.objects.get(keyword="tag|rosin").people == 400.0
        assert written_by("tag|live resin") == "later"

    def test_a_count_beyond_the_market_is_refused(self):
        refused = store(("title|manager", 2e7, 10.0), ("tag|rosin", 400.0, 400.0),
                        ("tag|live resin", 9_000.0, 6_000.0))

        assert set(refused) == {"title|manager", "tag|live resin"}
        assert list(KeywordGuess.objects.values_list("keyword", flat=True)) == ["tag|rosin"]

    def test_a_keyword_not_asked_for_is_refused(self):
        keyword_guess.store_market(MARKET, "earlier")

        refused = keyword_guess.store_keywords(keywords(("tag|resin", 400.0, 40.0)),
                                               {"tag|rosin"}, "earlier")

        assert "tag|resin" in refused
        assert not KeywordGuess.objects.exists()

    def test_more_fits_than_people_does_not_parse(self):
        with pytest.raises(ValidationError):
            KeywordAnswer(keyword="tag|rosin", people=10.0, fits=11.0)

    def test_an_unknown_key_does_not_parse(self):
        with pytest.raises(ValidationError):
            KeywordGuesses.model_validate_json('{"fields": []}')

    def test_a_keyword_the_model_left_out_is_warned_about(self, described, caplog):
        with patch.object(keyword_guess, "guess_field_with_llm",
                          return_value=FieldAnswer(keywords=[])):
            keyword_guess.ensure(described, [ROSIN])

        assert "tag|rosin" in caplog.text

    def test_nothing_is_asked_when_every_keyword_has_a_guess(self, described):
        store(("title|manager", 9e6, 500.0))

        with patch.object(keyword_guess, "guess_field_with_llm") as asked:
            keyword_guess.ensure(described, [MANAGER])

        asked.assert_not_called()

    def test_the_calling_agent_is_asked_for_the_market_and_every_field(self, described):
        agent_qualify.enable(None)

        with pytest.raises(KeywordGuessPending) as stop:
            keyword_guess.ensure(described, [MANAGER, ROSIN])

        payload = stop.value.payload
        assert payload["market"] is None
        assert payload["fields"]["tag"]["keywords"] == ["tag|rosin"]
        assert payload["schema"]["title"] == "KeywordGuesses"

    def test_the_agents_answer_is_kept_and_the_run_goes_on(self, described):
        agent_qualify.enable(None, keyword_guesses=KeywordGuesses(
            market=MARKET, keywords=keywords(("tag|rosin", 400.0, 40.0))))

        keyword_guess.ensure(described, [ROSIN])

        assert written_by("tag|rosin") == keyword_guess.GIVEN
        assert MarketGuess.objects.get().written_by == keyword_guess.GIVEN

    def test_the_agent_is_told_what_was_refused(self, described):
        agent_qualify.enable(None, keyword_guesses=KeywordGuesses(
            market=MARKET, keywords=keywords(("tag|rosin", 2e7, 40.0))))

        with pytest.raises(KeywordGuessPending) as stop:
            keyword_guess.ensure(described, [ROSIN])

        assert "tag|rosin" in stop.value.payload["refused"]
        assert stop.value.payload["market"] == {"people": MARKET.people, "fits": MARKET.fits}

    def test_a_replaced_brief_clears_the_guesses(self):
        from openoutfind.core.pipeline.brief import adopt

        store(("tag|rosin", 400.0, 40.0))
        adopt("A fit is: extractors.")

        assert not KeywordGuess.objects.exists()
        assert not MarketGuess.objects.exists()

    def test_the_prompt_shows_the_market_and_the_guessed_beside_the_new(self, described):
        store(("tag|rosin", 400.0, 40.0))

        prompt = keyword_guess.render("guess_keywords.j2", keyword_guess.prompt_context(
            described, "tag", ["tag|live resin"]))

        assert "10,000,000 people, of whom about 5,000 fit" in prompt
        assert "tag|rosin: 400 people, 40 fit" in prompt
        assert "- tag|live resin" in prompt

    def test_the_market_prompt_renders(self, described):
        prompt = keyword_guess.render("guess_market.j2", keyword_guess.prompt_context(described))

        assert "We sell cart hardware." in prompt


class TestInTheModelsTerms:
    def test_a_guess_is_a_log_share_and_a_lift_against_the_market(self):
        guesses = belief.Guesses.from_counts((10_000_000, 5_000), [("tag|rosin", 400.0, 40.0)])

        assert guesses.log_share(ROSIN, None) == pytest.approx(math.log(400 / 10_000_000))
        assert guesses.lift(ROSIN) == pytest.approx(
            math.log(40.5 / 360.5) - math.log(5_000.5 / 9_995_000.5))

    def test_a_keyword_not_guessed_reads_the_markets_rate_and_the_stores_share(self):
        class Store:
            everyone = range(98)

            def holding(self, keyword):
                return range(9)

        assert belief.Guesses().lift(ROSIN) == 0.0
        assert belief.Guesses().log_share(ROSIN, Store()) == pytest.approx(math.log(0.1))

    @pytest.mark.django_db
    def test_no_market_guess_means_no_guess_is_read(self):
        KeywordGuess.objects.create(keyword="tag|rosin", people=400.0, fits=40.0,
                                    written_by="earlier")

        assert keyword_guess.load() == belief.Guesses()
