# openoutfind/core/pipeline/keyword_guess.py
"""The LLM's guess about the market and every keyword of the walk — the prior the belief
starts from.

Before a search holds a keyword the belief knows nothing about it: every unknown lift read
the market's rate, and every unknown size the store's share, which over-counts the words
of the market the store was searched from (``retail`` was put at 338 people and the
provider counted 103,785). A model reading the brief knows ``live resin`` is a cart buyer's
tag and ``retail`` is not, and that ``manager`` is held by millions.

**The guess is plain counts, over what a search runs on** (``COUNTS_MEAN``): the market is
everyone on LinkedIn in the customer's country, not the customer's buyers. First the
market, once: how many people it holds and how many of them fit. Then per keyword: how many
people in the market hold it, inside the customer's trade or outside it, and how many of
those fit — so a word shared with other trades (``processing``) is guessed large and thin,
as its searches come back. Every keyword is guessed against the stored market, so all of them share its
scale, and each field's keywords (titles, tags, seniorities, places) are guessed in one
answer beside the ones of that field already guessed. The belief turns the counts into its
own terms (``belief.Guesses``) and learns how far keywords stray from them. A keyword is
guessed once: its own searches correct it from there. A count that breaks the market
(more people than it holds, more fits than it has) or a keyword nobody asked for is
refused, and the keyword is asked again.

**When:** at the start of a composed round, for every keyword the frontier could compose
or has fired that has no guess yet — the whole vocabulary the first time, then the titles
and tags new fits bring. ``AI_MODEL`` writes it, or under ``--agent-qualify`` the run stops
with ``keyword_guess_pending`` and the calling agent answers with ``--keyword-guesses``.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import jinja2
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openoutfind.core.conf import PROMPTS_DIR
from openoutfind.core.pipeline import belief

logger = logging.getLogger(__name__)

# What a guess handed in with ``--keyword-guesses`` is recorded as having been written by.
GIVEN = "--keyword-guesses"

# What the counts are counted over, for whoever guesses them: the population a search runs
# on, not the customer's market. Batch 6's session, handed the payload without it, read the
# market as the customer's (60,000 people, `country|united states` at 60,000) and put
# `processing` at 2,000 people and 110 fits: cannabis processors only. Its search counted
# 61× what that implied, food and payments companies.
COUNTS_MEAN = (
    "The market is everyone on LinkedIn in the customer's country (or the states the rules "
    "name), at companies of any size and in any job, not only the customer's buyers: a "
    "search runs over all of them, and its count counts all of them. Its fits are the few "
    "of them the rules accept. A keyword's people are everyone in that market holding it, "
    "wherever they work: a word the customer's trade shares with others (`processing`, "
    "`manufacturing`, `distribution`, `ceo`) is held mostly outside that trade, and every "
    "one of those holders counts. Its fits are the ones among them who fit, so such a word "
    "holds few fits for its people."
)

# What each field's keywords are, for whoever guesses them.
FIELD_MEANINGS = {
    "title": "a word in the person's job title (every word of a search's title must be in it)",
    "tag": "a whole tag on the person's company's LinkedIn page (what it does, sells, makes)",
    "seniority": "the person's seniority level, as LinkedIn classifies it",
    "place": "where the person is: a country, or a state inside it",
}


class Counts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    people: float = Field(gt=0, description="How many people.")
    fits: float = Field(ge=0, description="How many of those people fit.")

    @model_validator(mode="after")
    def _fits_among_the_people(self):
        if self.fits > self.people:
            raise ValueError(f"{self.fits:g} fits among {self.people:g} people")
        return self


class MarketAnswer(Counts):
    """The whole market: everyone a search could reach, and how many of them fit."""


class KeywordAnswer(Counts):
    keyword: str = Field(description="The keyword exactly as it was given.")


class FieldAnswer(BaseModel):
    """One field's keywords, guessed in one answer against the stored market."""

    keywords: list[KeywordAnswer]


class KeywordGuesses(BaseModel):
    """What ``--keyword-guesses`` carries: the market when the store has none yet, and the
    keywords asked for. An unknown key is refused, so a misspelled answer is not read as an
    empty one."""

    model_config = ConfigDict(extra="forbid")

    market: MarketAnswer | None = None
    keywords: list[KeywordAnswer] = []


# ── reading them ─────────────────────────────────────────────────────

def stored_market():
    """The market's guess, or ``None`` before it is guessed."""
    from openoutfind.core.models import MarketGuess

    return MarketGuess.objects.first()


def load() -> belief.Guesses:
    """Every stored guess, in the belief's terms."""
    from openoutfind.core.models import KeywordGuess

    market = stored_market()
    if market is None:
        return belief.Guesses()
    return belief.Guesses.from_counts(
        (market.people, market.fits),
        KeywordGuess.objects.values_list("keyword", "people", "fits"))


def unguessed(keywords) -> dict[str, list[str]]:
    """``keywords`` with no stored guess, as text, per field."""
    from openoutfind.core.models import KeywordGuess

    known = set(KeywordGuess.objects.values_list("keyword", flat=True))
    missing: dict[str, list[str]] = defaultdict(list)
    for keyword in sorted(set(keywords)):
        text = belief.keyword_text(keyword)
        if text not in known:
            missing[belief.keyword_field(keyword)].append(text)
    return dict(missing)


def guessed(field: str) -> list[dict]:
    """The keywords of ``field`` already guessed, to show beside new ones."""
    from openoutfind.core.models import KeywordGuess

    return [{"keyword": row.keyword, "people": row.people, "fits": row.fits}
            for row in KeywordGuess.objects.order_by("keyword")
            if belief.keyword_field(tuple(row.keyword.split("|"))) == field]


# ── writing them ─────────────────────────────────────────────────────

def ensure(site_config, keywords) -> None:
    """Guess the market, if it has no guess, and every one of ``keywords`` that has none."""
    from openoutfind.core import agent_qualify

    keywords = set(keywords)
    if stored_market() is not None and not unguessed(keywords):
        return
    if agent_qualify.active():
        refused = take_answer(agent_qualify.keyword_guesses_answer(), keywords)
        if stored_market() is None or unguessed(keywords):
            raise_pending(site_config, unguessed(keywords), refused)
        return
    guess_with_llm(site_config, keywords)


def take_answer(answer: KeywordGuesses | None, keywords) -> dict[str, str]:
    """Store what ``answer`` holds of the market and of ``keywords``; what was refused,
    with why."""
    if answer is None:
        return {}
    if answer.market is not None and stored_market() is None:
        store_market(answer.market, GIVEN)
    asked = {text for texts in unguessed(keywords).values() for text in texts}
    return store_keywords(answer.keywords, asked, GIVEN)


def guess_with_llm(site_config, keywords) -> None:
    """Have ``AI_MODEL`` guess the market, if it has no guess, then each field holding
    keywords with none."""
    from openoutfind.core.pipeline.brief import ensure_rules

    ensure_rules(site_config)
    if stored_market() is None:
        store_market(guess_market_with_llm(site_config), site_config.ai_model)
    for field, texts in unguessed(keywords).items():
        answer = guess_field_with_llm(site_config, field, texts)
        store_keywords(answer.keywords, set(texts), site_config.ai_model)
    left = unguessed(keywords)
    if left:
        logger.warning("keyword guesses: %d keyword(s) still have none, asked again next "
                       "round: %s", sum(map(len, left.values())),
                       ", ".join(text for texts in left.values() for text in texts))


def store_market(answer: MarketAnswer, written_by: str):
    from openoutfind.core.models import MarketGuess

    logger.debug("keyword guesses: market ~%s people, ~%s fit · by %s",
                 f"{answer.people:,.0f}", f"{answer.fits:,.0f}", written_by)
    return MarketGuess.objects.create(people=answer.people, fits=answer.fits,
                                      written_by=written_by)


def store_keywords(answers: list[KeywordAnswer], asked: set[str],
                   written_by: str) -> dict[str, str]:
    """Keep the answers for ``asked`` keywords that fit inside the stored market; a keyword
    already guessed is kept as it was. Returns the refused ones, with why."""
    from openoutfind.core.models import KeywordGuess

    market = stored_market()
    known = set(KeywordGuess.objects.values_list("keyword", flat=True))
    refused, new = {}, {}
    for answer in answers:
        if answer.keyword in known:
            continue
        reason = refusal(answer, asked, market)
        if reason:
            refused[answer.keyword] = reason
        else:
            new[answer.keyword] = answer
    KeywordGuess.objects.bulk_create(
        KeywordGuess(keyword=text, people=answer.people, fits=answer.fits,
                     written_by=written_by)
        for text, answer in new.items())
    logger.debug("keyword guesses: %d kept, %d refused · by %s", len(new), len(refused),
                 written_by)
    for text, reason in refused.items():
        logger.warning("keyword guess refused: %s — %s", text, reason)
    return refused


def refusal(answer: KeywordAnswer, asked: set[str], market) -> str:
    """Why ``answer`` cannot be kept, or ``""``."""
    if answer.keyword not in asked:
        return "not one of the keywords asked for"
    if market is None:
        return "the market has no guess yet"
    if answer.people > market.people:
        return f"{answer.people:,.0f} people, more than the market's {market.people:,.0f}"
    if answer.fits > market.fits:
        return f"{answer.fits:,.0f} fits, more than the market's {market.fits:,.0f}"
    return ""


# ── asking ───────────────────────────────────────────────────────────

def brief_rules() -> str:
    from openoutfind.core.pipeline.brief import stored

    brief = stored()
    return brief.rules if brief is not None else ""


def market_counts() -> dict | None:
    market = stored_market()
    return None if market is None else {"people": market.people, "fits": market.fits}


def prompt_context(site_config, field: str | None = None, texts=()) -> dict:
    """Everything a guess is written from: the market's prompt without ``field``."""
    context = {"product_docs": site_config.product_docs,
               "campaign_target": site_config.campaign_target, "rules": brief_rules(),
               "market": market_counts(), "counts_mean": COUNTS_MEAN}
    if field is not None:
        context |= {"field": field, "meaning": FIELD_MEANINGS[field],
                    "guessed": guessed(field), "keywords": list(texts)}
    return context


def render(template: str, context: dict) -> str:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    return env.get_template(template).render(**context)


def ask(prompt: str, output_type):
    from pydantic_ai import Agent

    from openoutfind.core.llm import get_llm_model, run_agent_sync

    agent = Agent(get_llm_model(), output_type=output_type, model_settings={"timeout": 300})
    return run_agent_sync(agent.run(prompt)).output


def guess_market_with_llm(site_config) -> MarketAnswer:
    """Ask ``AI_MODEL`` for the market's guess."""
    return ask(render("guess_market.j2", prompt_context(site_config)), MarketAnswer)


def guess_field_with_llm(site_config, field: str, texts: list[str]) -> FieldAnswer:
    """Ask ``AI_MODEL`` for ``field``'s guesses, against the stored market."""
    return ask(render("guess_keywords.j2", prompt_context(site_config, field, texts)),
               FieldAnswer)


def raise_pending(site_config, missing: dict[str, list[str]], refused: dict[str, str]) -> None:
    """Stop the run for the calling agent to guess the market, if it has none, and
    ``missing``."""
    from openoutfind.core.pipeline.qualify import KeywordGuessPending

    count = sum(len(texts) for texts in missing.values())
    market = market_counts()
    ask_for = (f"{count} keyword(s)" if market is not None
               else f"the market and {count} keyword(s)")
    raise KeywordGuessPending(
        f"{ask_for} have no guess yet — guess how many people on LinkedIn hold each, "
        f"inside the customer's trade or not, and how many of them fit (see counts_mean)",
        payload={
            "counts_mean": COUNTS_MEAN,
            "product_docs": site_config.product_docs,
            "campaign_target": site_config.campaign_target,
            "rules": brief_rules(),
            "market": market,
            "fields": {field: {"meaning": FIELD_MEANINGS[field],
                               "guessed": guessed(field), "keywords": texts}
                       for field, texts in missing.items()},
            "refused": refused,
            "schema": KeywordGuesses.model_json_schema(),
        })
