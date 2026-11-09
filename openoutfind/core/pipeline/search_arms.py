# openoutfind/core/pipeline/search_arms.py
"""The search arms — every search the walk could fire next, each valued, and the one it
fires.

**An arm is valued on the fits the one page it would bring holds, from what is still out
there:**

    fits(arm) = new · (F − held) / (N − seen)

- **An arm** is a search at a page: the next page of a fired search (``deepen``), a written
  search not yet fired, or one of the ``COMPOSED`` searches worth the most at the mean
  belief, composed from the qualified profiles' keywords (``compose``) — never a search
  already fired, dead, or holding all the terms of one that is, and never a search of the
  whole country where every written search names a state.
- **N** is how many people the market holds for the search: the provider's count where it
  fired, otherwise the belief's estimate, capped by the count of any fired search it
  narrows and kept above the people the store holds for it. **F** is ``N · θ(q)``.
- **seen** and **held** are the store's people the search matches, and the fits among them.
- **new** is how many people the page would bring that the store does not hold. A search
  holding all the terms of a drained one can only re-find people, so it is never an arm.
  A next page brings as many as its latest page was first to bring in.
- **the cost** is dollars (``calibration.dollars``): a search's measured ready time, and
  for each new person the measured ingest time and a verdict.

**The search fired is the best at its mean plus one σ** (``Frontier.pick``), read on
``DRAWS`` draws of the belief seeded by the round's number so a rerun repeats. A search
little is known about is worth trying once; the five random verdicts ``probe`` takes from
its page then settle it. Searches are composed at the mean only: composed on each draw,
across 10,000 keywords near their prior one always drew high, and only 9 of 50 draws
stayed in the customer's market.

``judge_or_search`` weighs the arm worth the most at the mean against judging the pool,
and the walk stops when no arm is expected to bring a new fit. In the cold phase nothing
is yet known to stop on, so it fires while an arm brings somebody new.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, replace

import numpy as np

from openoutfind.core.conf import CAMPAIGN_CONFIG
from openoutfind.core.pipeline import belief, compose
from openoutfind.core.pipeline.store_snapshot import (
    FiredSearch, People, fired_searches, fit_rate_map, judged_fits, judged_leads,
    lead_fit_rates,
)
from openoutfind.core.pipeline.terms import Terms, split_place

logger = logging.getLogger(__name__)

# An estimate below the people the store holds for a search is disproved, and how far is
# unknown: the search is taken to hold a tenth more, so it stays worth a look.
SEEN_MARGIN = 1.1
# The fits a page must be expected to bring for the walk to go on.
FITS_WORTH_A_SEARCH = 1.0
# The composed searches valued exactly: room for the ones the frontier refuses (fired,
# dead, drained) below the best.
COMPOSED = 200
# The draws of the belief an arm's mean and σ are read off.
DRAWS = 200
# How many σ above its mean an arm's fits are taken at: at two σ an untried search read
# three times the market's rate.
OPTIMISM = 1.0


@dataclass
class Arm:
    """A search at a page the walk could fire, and what it is expected to bring."""

    terms: Terms
    number: int
    move: str
    search: object | None = None
    keywords: tuple = ()
    count: float = 0.0       # N: the people the market holds for it
    seen: int = 0            # the store's people it matches
    held_fits: float = 0.0   # the fits expected among them, each at their rate
    new: float = 0.0         # people the page would bring that the store does not hold
    theta: float | None = None  # θ(q) at the mean belief, None on an arm chosen unvalued
    rate: float = 0.0        # the fit rate expected of the new people, at the mean belief
    cost: float = 0.0        # dollars to fire it, take its new people in and judge them
    score: float | None = None  # fits on its page at mean + OPTIMISM σ, set when picked

    @property
    def is_next_page(self) -> bool:
        return self.move == "deepen"

    @property
    def fired_key(self) -> str | None:
        """The fired search whose own ``θ`` a next page is valued on."""
        return self.search.key if self.is_next_page else None

    @property
    def expected_fits(self) -> float:
        return self.new * self.rate

    def density(self, theta: float) -> float:
        """The fit rate of the people still out there, were ``theta`` of the market's to fit."""
        out_there = self.count - self.seen
        if out_there <= 0:
            return 0.0
        return min(max((self.count * theta - self.held_fits) / out_there, 0.0), 1.0)

    def __repr__(self):
        return f"{self.terms.as_text()} @ page {self.number}"


class Frontier:
    """Everything the next search is chosen from, read from the store once."""

    def __init__(self, qualifier):
        from openoutfind.core import meter
        from openoutfind.core.ml import calibration
        from openoutfind.core.models import Page, PageLead, Search
        from openoutfind.core.pipeline.walk import FIRST_POLL_SECONDS

        self.people = People.load(*lead_fit_rates(qualifier, fit_rate_map()))
        self.judged = judged_leads()
        self.fits = judged_fits()
        self.fired = fired_searches()
        self.beliefs = belief.Beliefs(self.people, self.fired, self.judged,
                                      self.people.base_rate())
        self.written = list(Search.objects.filter(origin=Search.Origin.WRITTEN))
        self.market_is_states = bool(self.written) and all(
            split_place(search.query.place)[0] for search in self.written)
        self.taken = set(Search.objects.filter(pages__isnull=False).values_list("key", flat=True))
        self.taken |= {search.key for search in self.written}
        self.dead = [search.query for search in Search.objects.filter(dead=True)]
        self.taken |= set(Search.objects.filter(dead=True).values_list("key", flat=True))
        self.closed = self.dead + [f.terms for f in self.fired if f.drained]
        self.loaded_vocabulary: compose.Vocabulary | None = None

        returned = sum(Page.objects.values_list("rows", flat=True))
        self.kept = PageLead.objects.count() / returned if returned else 1.0

        self.fire_cost = calibration.dollars(meter.mean(meter.SEARCH_READY_SECONDS,
                                                        FIRST_POLL_SECONDS))
        self.person_cost = calibration.verdict_cost() + calibration.dollars(
            meter.mean(meter.INGEST_SECONDS_PER_LEAD, CAMPAIGN_CONFIG["ingest_seconds_per_lead"]))

    # ── the arms ──

    def arms(self) -> list[Arm]:
        """Every arm, valued at the mean belief: the next pages, the unfired written
        searches, and the composed searches."""
        arms = [self.next_page_arm(f) for f in self.fired if not f.latest.at_bottom]
        fired_keys = {f.search.key for f in self.fired}
        arms += [self.first_page_arm(search.query, "written", search) for search in self.written
                 if not search.dead and search.key not in fired_keys]
        arms = [arm for arm in arms if self.in_market(arm.terms)]
        composed = self.composed_arms(self.beliefs.lifts(), self.beliefs.mean_sizes())
        arms = distinct_arms(arms + composed)
        self.rate_at_mean(arms)
        return arms

    def composed_arms(self, lift: dict, sizes: belief.Sizes) -> list[Arm]:
        """Page 1 of the ``COMPOSED`` searches worth the most on ``lift`` and ``sizes``
        that the frontier may fire."""
        vocabulary = self.vocabulary
        share = self.beliefs.shares(vocabulary.keywords(), sizes)
        searches = compose.compose(vocabulary, lift, share, sizes.log_market,
                                   self.page_worth(sizes.log_market), COMPOSED)
        return [self.first_page_arm(terms, "drawn") for terms in searches
                if terms.key not in self.taken and terms.is_legal
                and self.in_market(terms) and not self.is_closed(terms)]

    def page_worth(self, log_market: float) -> compose.Worth:
        """The fits a page of a search brings from its summed lift and share, with nobody
        yet seen — what ``pick`` ranks on, for the composition to sort by. Per dollar it
        would rank on the rate alone, since a verdict a person is most of a page's cost,
        and pick searches of 1 to 6 people."""
        from openoutfind.core.models import PAGE_SIZE

        market = belief.log_odds(self.beliefs.market_rate)

        def worth(lift: np.ndarray, share: np.ndarray) -> np.ndarray:
            theta = 1.0 / (1.0 + np.exp(-(market + lift)))
            return theta * self.kept * np.minimum(PAGE_SIZE, np.exp(log_market + share))

        return worth

    def next_page_arm(self, fired: FiredSearch) -> Arm:
        """The next page of a fired search: as fresh as its latest one."""
        from openoutfind.core.models import PAGE_SIZE

        keywords = fired.keywords
        count = fired.found if fired.found is not None else self.beliefs.count(keywords)
        remaining = count - fired.latest.number * PAGE_SIZE
        novelty = len(fired.fresh) / fired.latest.rows if fired.latest.rows else 0.0
        page = max(0.0, min(PAGE_SIZE, remaining)) * novelty
        return self.valued_arm(fired.terms, fired.latest.number + 1, "deepen", fired.search,
                               count, self.people.match(fired.terms) | fired.people, page)

    def first_page_arm(self, terms: Terms, move: str, search=None) -> Arm:
        """Page 1 of an unfired search, its count estimated."""
        matched = self.people.match(terms)
        count = self.estimated_count(terms, len(matched))
        return self.valued_arm(terms, 1, move, search, count, matched,
                               self.first_page_people(count, len(matched)))

    def valued_arm(self, terms: Terms, number: int, move: str, search, count: float,
                   seen: set[int], page: float) -> Arm:
        new = min(page, max(0.0, count - len(seen)))
        return Arm(terms, number, move, search, keywords=belief.query_keywords(terms),
                   count=count, seen=len(seen), held_fits=self.expected_fits_among(seen),
                   new=new, cost=self.fire_cost + self.person_cost * new)

    # ── the vocabulary and the market ──

    @property
    def vocabulary(self) -> compose.Vocabulary:
        """Every keyword a composed search may hold: the qualified profiles' title words and
        whole tags (the customer's good marks and the run's fits), the written searches'
        terms, every seniority, and the market's places — the country and its states, or
        only the written states where every written search names one. No written search, no
        market, nothing composed.

        **The titles and tags come from the fits alone**: with every lift centred on the
        market's rate, a keyword no search has held reads that rate, and the store's 10,000
        would be ranked on size alone. The fits are how the marks reach the walk."""
        if self.loaded_vocabulary is None:
            self.loaded_vocabulary = self.load_vocabulary()
        return self.loaded_vocabulary

    def load_vocabulary(self) -> compose.Vocabulary:
        from openoutfind.discovery import LEAD_SENIORITIES

        written = [search.query for search in self.written]
        if not written:
            return compose.Vocabulary((), (), (), ())
        country = next((split_place(t.place)[1] for t in written if t.place), "")
        titles = self.people.title_vocabulary(self.fits) | {w for t in written for w in t.title}
        tags = self.people.tag_vocabulary(self.fits) | {t.tag for t in written if t.tag}
        written_states = {split_place(t.place)[0] for t in written} - {""}
        states = written_states if self.market_is_states else (
            self.people.states_in(country) | written_states)
        places = [("state", state, country) for state in sorted(states)] if country else []
        if country and not self.market_is_states:
            places.insert(0, ("country", country))
        return compose.Vocabulary(
            titles=tuple(("title", word) for word in sorted(titles)),
            tags=tuple(("tag", tag) for tag in sorted(tags)),
            seniorities=tuple(("seniority", value) for value in LEAD_SENIORITIES),
            places=tuple(places))

    def in_market(self, terms: Terms) -> bool:
        """Whether ``terms`` stay in the market the written searches drew: where every one
        of them names a state, a search of the whole country is outside it. A store written
        partly on the country (batch 5: 8 of 10) has no such limit."""
        return bool(split_place(terms.place)[0]) or not self.market_is_states

    def is_closed(self, terms: Terms) -> bool:
        """Whether a dead or drained search holds all of ``terms``: then ``terms`` match
        nobody, or only people the store already holds."""
        return any(terms.holds_all_of(closed) for closed in self.closed)

    # ── sizes and fits ──

    def first_page_people(self, count: float, seen: int) -> float:
        """The people page 1 of an unfired search would bring, were ``count`` its size."""
        from openoutfind.core.models import PAGE_SIZE

        return min(PAGE_SIZE, max(0.0, count - seen)) * self.kept

    def estimated_count(self, terms: Terms, seen: int, sizes: belief.Sizes | None = None) -> float:
        """``N(q)`` for an unfired search on the mean or on ``sizes``, kept above the people
        the store holds for it and within the count of every fired search it narrows."""
        estimate = max(self.beliefs.count(belief.query_keywords(terms), sizes),
                       seen * SEEN_MARGIN)
        return min([estimate] + [f.found for f in self.fired
                                 if f.found is not None and terms.holds_all_of(f.terms)])

    def resized(self, arm: Arm, sizes: belief.Sizes) -> Arm:
        """``arm`` on ``sizes``. A next page's size is the provider's count, so it stays."""
        if arm.is_next_page:
            return arm
        count = self.estimated_count(arm.terms, arm.seen, sizes)
        new = self.first_page_people(count, arm.seen)
        return replace(arm, count=count, new=new, cost=self.fire_cost + self.person_cost * new)

    def expected_fits_among(self, people: set[int]) -> float:
        """The fits among ``people``: a verdict where judged, else a fit rate. Counting only
        the judged left an unjudged person's share of ``F`` on those still out there, and
        36 of 515's 202 arms read a density of 1 and tied."""
        rates = self.people.rates
        return sum(rates.get(pk, self.beliefs.store_rate) for pk in people)

    # ── the belief ──

    def rate_at_mean(self, arms: list[Arm]) -> None:
        """Each arm's ``θ(q)`` and fit rate at the mean belief."""
        lifts = self.beliefs.lifts()
        for arm in arms:
            arm.theta = self.beliefs.theta_at_mean(arm.keywords, lifts, arm.fired_key)
            arm.rate = arm.density(arm.theta)

    # ── the pick ──

    def pick(self, arms: list[Arm], round_number: int) -> Arm:
        """The arm whose page brings the most fits at its mean plus ``OPTIMISM`` σ over
        the draws of round ``round_number``. Sets the arm's ``score``."""
        arms = in_one_order(distinct_arms(arms))
        fits = self.fits_on_draws(arms, round_number)
        scores = fits.mean(axis=0) + OPTIMISM * fits.std(axis=0)
        chosen = arms[int(np.argmax(scores))]
        chosen.score = float(scores.max())
        return chosen

    def fits_on_draws(self, arms: list[Arm], round_number: int) -> np.ndarray:
        """``(DRAWS, arms)``: the fits each arm's page brings on each draw."""
        keywords = {keyword for arm in arms for keyword in arm.keywords}
        fits = np.zeros((DRAWS, len(arms)))
        for n in range(DRAWS):
            drawn = self.beliefs.draw((round_number, n), keywords)
            for i, arm in enumerate(arms):
                theta = self.beliefs.theta_on(drawn, arm.keywords, arm.fired_key)
                sized = self.resized(arm, drawn.sizes)
                fits[n, i] = sized.new * sized.density(theta)
        return fits


def distinct_arms(arms: list[Arm]) -> list[Arm]:
    """``arms`` with one arm per search and page, the first kept."""
    seen: dict[tuple[str, int], Arm] = {}
    for arm in arms:
        seen.setdefault((arm.terms.key, arm.number), arm)
    return list(seen.values())


def in_one_order(arms: list[Arm]) -> list[Arm]:
    """The arms in one order whatever made them, so a rerun breaks a tie the same way."""
    return sorted(arms, key=lambda arm: (arm.terms.key, arm.number))


def worth_firing(arms: list[Arm], qualifier) -> list[Arm]:
    """The arms bringing somebody new, or none once no arm is expected to bring a fit. The
    cold phase has nothing yet to stop on: only a frontier with nobody new ends it."""
    bringing = [arm for arm in arms if arm.new > 0]
    if qualifier is not None and qualifier.is_cold:
        return bringing
    if any(arm.expected_fits >= FITS_WORTH_A_SEARCH for arm in bringing):
        return bringing
    return []


def next_round(qualifier, round_number: int) -> list[Arm]:
    """The one search to fire next, or ``[]`` when none is worth firing."""
    frontier = Frontier(qualifier)
    arms = worth_firing(frontier.arms(), qualifier)
    log_last_page(frontier)
    if not arms:
        return []
    chosen = frontier.pick(arms, round_number)
    log_chosen(chosen, len(arms))
    return [chosen]


def best_expected(qualifier) -> Arm | None:
    """The arm worth the most at the mean belief, what ``judge_or_search`` weighs against
    the pool — ``None`` when none is worth firing."""
    arms = worth_firing(Frontier(qualifier).arms(), qualifier)
    if not arms:
        return None
    return max(in_one_order(arms), key=lambda arm: arm.expected_fits / arm.cost)


def written_arms() -> list[Arm]:
    """The written searches not yet fired — the whole first round, fired unvalued: there
    is nothing yet to value them on."""
    from openoutfind.core.models import Search

    return [Arm(search.query, 1, "written", search)
            for search in Search.objects.filter(origin=Search.Origin.WRITTEN, dead=False,
                                                pages__isnull=True)]


# ── the log ──────────────────────────────────────────────────────────

def log_chosen(chosen: Arm, candidates: int) -> None:
    logger.debug("  frontier: %d arm(s) · %-9s θ %.2f · density %.2f · N ~%.0f · seen %d · "
                 "~%.0f new · %.1f fits at +%gσ  %s", candidates, chosen.move, chosen.theta,
                 chosen.rate, chosen.count, chosen.seen, chosen.new, chosen.score, OPTIMISM,
                 chosen)


def log_last_page(frontier: Frontier) -> None:
    """What became of the last fired page, to read θ against and to catch pages nobody
    judges."""
    if not frontier.fired:
        return
    last = max(frontier.fired, key=lambda fired: fired.latest.round)
    people = last.latest_people
    rates = [frontier.people.rates[pk] for pk in people if pk in frontier.people.rates]
    logger.debug("  last page: %s · %d verdict(s), %d fit(s) of %d · observed rate %.2f",
                 last.terms.as_text(), len(people & frontier.judged),
                 len(people & frontier.fits), len(people),
                 statistics.fmean(rates) if rates else 0.0)
