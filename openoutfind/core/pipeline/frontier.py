# openoutfind/core/pipeline/frontier.py
"""The frontier — every search the walk could fire next, and the one it fires.

**An arm is valued on the fits the one page it would bring holds, from what is still out
there:**

    fits(arm) = new · (F − held) / (N − seen)

- **An arm** is a search at a page: the next page of a fired search (``deepen``), a written
  search not yet fired, or one of the ``COMPOSED`` searches worth the most at the mean
  belief, composed from every keyword the store holds (``compose``) — never a search
  already fired, dead, or holding all the terms of one that is, and never a search of the
  whole country where every written search names a state.
- **N** is how many people the market holds for the search: the provider's count where it
  fired, otherwise the belief's estimate (``belief``), capped by the count of any fired
  search it narrows and kept above the people the store holds for it. **F** is ``N · θ``,
  with ``θ`` the share of them expected to fit: a fired search's own (its pages' people),
  otherwise the product of its keywords'.
- **seen** and **held** are the store's people the search matches, and the fits among them.
  An arm with nobody new values at 0, and so does an area whose fits the store holds.
- **new** is how many people the page would bring that the store does not hold. A person
  already seen adds nothing, so **a search holding all the terms of a drained one** (its
  pages returned everything it counts) is never an arm: it matches a subset of those
  people, so it can only re-find them. A next page brings as many as its latest page was first to bring in.
- **the costs** are dollars (``calibration.dollars``): a search's measured ready time, each
  new person's measured ingest time — embedding is most of a round's — and a verdict on
  each of them, so a page of 200 strangers pays for every one.

**The search fired is the best at its mean plus one σ** (``Frontier.pick``): each arm's
fits are read on ``DRAWS`` draws of the belief — every ``θ_t``, every fired search's own
``θ``, the sizes — seeded by the round's number so a rerun repeats, and the arm with the
highest mean plus ``OPTIMISM`` σ fires. A search little is known about is worth trying
once; the five random verdicts ``probe`` takes from its page then settle it. It replaced
Thompson sampling over searches composed on each draw: across 10,000 keywords near their
prior one always drew high, and only 9 of 50 draws stayed in the customer's market. A
search the provider has counted keeps its count on every draw.

``judge_or_search`` weighs the arm with the highest value at the mean ``θ_t`` against judging
the pool, and the walk stops when no arm is expected to bring a new fit. In the cold phase
nothing is yet known to stop on, so it fires while an arm brings somebody new.

**The fit rate** is the GP's score through an isotonic map fitted on the store's verdicts
at their out-of-fold scores (``fit_rate_map``): the GP ranks well but is not calibrated
(0.15 where the truth was 0.28), and a fit rate is what can be weighed against a cost. With
one class of verdicts the score stands in.
"""
from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

from openoutfind.core.conf import CAMPAIGN_CONFIG
from openoutfind.core.pipeline import belief, compose
from openoutfind.core.pipeline.query import (
    Terms, company_tags, is_stopword_phrase, query_title_words, split_place, title_words,
)

logger = logging.getLogger(__name__)

_EMPTY: frozenset = frozenset()

# What a market reads as with nobody scored: nothing is known either way.
NEUTRAL_RATE = 0.5
# An estimate below the people the store holds for a search is disproved, and how far is
# unknown: the search is taken to hold a tenth more, so it stays worth a look.
SEEN_MARGIN = 1.1
# The fits a page must be expected to bring for the walk to go on.
FITS_WORTH_A_SEARCH = 1.0
# The composed searches valued exactly, at the mean: room for the ones the frontier
# refuses (fired, dead, drained) below the best.
COMPOSED = 200
# The draws of the belief an arm's mean and σ are read off.
DRAWS = 200
# How many σ above its mean an arm's fits are taken at: a search little is known about is
# worth trying once, and a page that came back bad stops being worth it.
OPTIMISM = 1.0


# ── the fit rate ─────────────────────────────────────────────────────

# The fitted map, under the fingerprint of the verdicts it was fitted on.
_FIT_RATE: tuple[str, object] | None = None


def fit_rate_map():
    """``scores → fit rates``: an isotonic regression of the store's verdicts on their
    out-of-fold scores, refitted when a verdict lands.

    The verdicts are the marks and the run's own, which the run chose by score — and
    choosing by score does not bias the rate *at* a score. Scores outside the verdicts'
    range clip to its ends. With one class of verdicts there is nothing to fit, and the
    score stands in.
    """
    global _FIT_RATE
    from openoutlearn.calibration import fit_rate_curve

    from openoutfind.core.ml.calibration import FOLDS
    from openoutfind.core.ml.qualifier import build_qualifier, label_fingerprint
    from openoutfind.crm.models import Lead

    key = label_fingerprint()
    if _FIT_RATE is None or _FIT_RATE[0] != key:
        ids, X, y = Lead.get_labeled_rows()
        curve = None
        if len(np.unique(y)) == 2:
            oof = build_qualifier(X, y).out_of_fold_probs(np.arange(len(ids)), k=FOLDS)
            if oof is not None:
                curve = fit_rate_curve(oof, y)
        _FIT_RATE = (key, curve)
    curve = _FIT_RATE[1]
    if curve is None:
        return lambda scores: np.asarray(scores, dtype=np.float64)
    return curve.predict


def lead_rates(qualifier, fit_rate) -> tuple[dict[int, float], dict[int, float]]:
    """``(lead_id → rate, lead_id → confidence)`` for every stored person: their verdict
    if judged, else the GP's score through ``fit_rate`` and the GP's confidence there
    (``BayesianQualifier.confidence``). A person the GP cannot score is left out."""
    from openoutfind.crm.models import Deal, DealState, Lead, Outcome

    rates: dict[int, float] = {
        lead_id: 0.0 if (state == DealState.FAILED and outcome == Outcome.WRONG_FIT) else 1.0
        for lead_id, state, outcome in Deal.objects.filter(lead_id__isnull=False)
        .values_list("lead_id", "state", "outcome")
    }
    unjudged = list(Lead.objects.filter(synthetic=False, embedding__isnull=False)
                    .exclude(pk__in=list(rates)).values_list("pk", "embedding"))
    if qualifier is None or not unjudged:
        return rates, {}
    X = np.array([np.frombuffer(bytes(emb), dtype=np.float32) for _, emb in unjudged],
                 dtype=np.float64)
    probs = qualifier.predict_probs(X)
    if probs is None:
        return rates, {}
    pks = [pk for pk, _ in unjudged]
    rates.update(zip(pks, map(float, fit_rate(probs))))
    return rates, dict(zip(pks, map(float, qualifier.confidence(X))))


def judged_leads() -> set[int]:
    """Every lead holding a verdict, marks included."""
    from openoutfind.crm.models import Deal

    return set(Deal.objects.filter(lead_id__isnull=False).values_list("lead_id", flat=True))


def judged_fits() -> set[int]:
    """Every lead judged a fit: any deal but the ``wrong_fit`` rejection, marks included."""
    from openoutfind.crm.models import Deal, DealState, Outcome

    return set(Deal.objects.filter(lead_id__isnull=False)
               .exclude(state=DealState.FAILED, outcome=Outcome.WRONG_FIT)
               .values_list("lead_id", flat=True))


# ── the store's people ───────────────────────────────────────────────

class People:
    """The store's people, indexed per field the way the provider matches them.

    ``match`` gives what ``Terms.matches`` would over every person, by intersecting
    per-term sets: a title word, every contiguous phrase inside a tag, a state, a country,
    a seniority.
    """

    def __init__(self, rows, rates: dict[int, float], confidence: dict[int, float] | None = None):
        self.rates = rates
        self.confidence = confidence or {}
        self.company: dict[int, object] = {}
        self.everyone: set[int] = set()
        self._index: dict[tuple, set[int]] = defaultdict(set)
        self._title_words: set[str] = set()
        self._joined: set[str] = set()
        self._states: dict[str, set[str]] = defaultdict(set)
        for pk, company_id, fields in rows:
            self.everyone.add(pk)
            self.company[pk] = _company_key(pk, company_id, fields)
            for key in _index_keys(fields):
                self._index[key].add(pk)
            self._take_vocabulary(fields)

    def _take_vocabulary(self, fields: dict) -> None:
        title = fields.get("contact_job_title", "")
        searched = query_title_words(title)
        self._title_words |= searched
        self._joined |= title_words(title) - searched
        country = str(fields.get("contact_location_country") or "").strip().lower()
        state = str(fields.get("contact_location_state") or "").strip().lower()
        if country and state:
            self._states[country].add(state)

    def title_vocabulary(self) -> set[str]:
        """Every title word a search may hold. A hyphenated title's joined form is left
        out: the provider counts ``cofounder`` and ``co founder`` as one (161,369), so the
        split words already reach it."""
        return self._title_words - self._joined

    def tag_vocabulary(self) -> set[tuple[str, ...]]:
        """Every phrase inside a tag the store's people hold, never stopwords alone."""
        return {key[1] for key in self._index if key[0] == "tag"
                and not is_stopword_phrase(key[1])}

    def states_in(self, country: str) -> set[str]:
        """The states the store's people live in inside ``country``."""
        return set(self._states.get(country.lower(), ()))

    @classmethod
    def load(cls, rates: dict[int, float],
             confidence: dict[int, float] | None = None) -> "People":
        from openoutfind.crm.models import Lead

        return cls(Lead.objects.filter(synthetic=False).exclude(source_fields={})
                   .values_list("pk", "company_id", "source_fields"), rates, confidence)

    def sureness(self, pk: int) -> float:
        """How far a person's rate is to be believed: the GP's confidence at them, or 1
        for a rate given with none (a verdict, or a rate set by hand)."""
        return self.confidence.get(pk, 1.0)

    def match(self, terms: Terms) -> set[int]:
        """Every stored person matching every term of ``terms``, each in its own field."""
        keys = [("title", word) for word in terms.title]
        if terms.tag:
            keys.append(("tag", terms.tag))
        if terms.seniority:
            keys.append(("seniority", terms.seniority))
        if terms.place:
            state, country = split_place(terms.place)
            keys.append(("country", country))
        if not keys:
            return set(self.everyone)
        sets = [self._index.get(key, _EMPTY) for key in keys]
        if terms.place and state:
            sets.append(self._index.get(("state", state), _EMPTY)
                        | self._index.get(("no state", country), _EMPTY))
        sets.sort(key=len)
        return set(sets[0]).intersection(*sets[1:])

    def holding(self, keyword: tuple) -> set[int]:
        """Everyone holding ``keyword`` (``belief.query_keywords``); a state is held by its
        own rows and by the rows of its country that carry no state."""
        if keyword[0] == "state":
            _, state, country = keyword
            return set(self._index.get(("state", state), _EMPTY)
                       | self._index.get(("no state", country), _EMPTY))
        return set(self._index.get(keyword, _EMPTY))

    def base_rate(self) -> float:
        """The store's own rate: every scored person, per company, then over companies."""
        by_company: dict = defaultdict(list)
        for pk, rate in self.rates.items():
            if pk in self.company:
                by_company[self.company[pk]].append(rate)
        if not by_company:
            return NEUTRAL_RATE
        return statistics.fmean(statistics.fmean(rates) for rates in by_company.values())


def _company_key(pk: int, company_id, fields: dict):
    """Who a person counts with: their company, or with no company the people sharing
    their tag list — the provider's row for "Self employed" staples one list onto
    hundreds of strangers — or alone."""
    if company_id is not None:
        return company_id
    keywords = str(fields.get("company_keywords") or "").strip().lower()
    return ("tags", keywords) if keywords else ("alone", pk)


def _index_keys(fields: dict):
    for word in title_words(fields.get("contact_job_title", "")):
        yield ("title", word)
    for tag in company_tags(fields.get("company_keywords", "")):
        for start in range(len(tag)):
            for end in range(start + 1, len(tag) + 1):
                yield ("tag", tag[start:end])
    if seniority := str(fields.get("contact_seniority") or "").strip().lower():
        yield ("seniority", seniority)
    if country := str(fields.get("contact_location_country") or "").strip().lower():
        yield ("country", country)
    if state := str(fields.get("contact_location_state") or "").strip().lower():
        yield ("state", state)
    elif country:
        yield ("no state", country)


# ── the fired searches ───────────────────────────────────────────────

@dataclass
class Fired:
    """A search with pages: its latest, everyone its pages returned, and who its latest
    page returned and was first to bring in."""

    search: object
    latest: object
    people: set[int]
    latest_people: set[int]
    fresh: set[int]

    @property
    def terms(self) -> Terms:
        return self.search.query

    @property
    def keywords(self) -> tuple:
        return belief.query_keywords(self.terms)

    @property
    def found(self) -> int | None:
        return self.search.leads_found

    @property
    def drained(self) -> bool:
        """Whether its pages returned everything it counts."""
        from openoutfind.core.models import PAGE_SIZE

        return self.found is not None and self.latest.number * PAGE_SIZE >= self.found


def fired_searches() -> list[Fired]:
    """Every search with pages, as of its latest."""
    from openoutfind.core.models import Page, PageLead

    latest: dict[int, object] = {}
    for page in Page.objects.select_related("search").order_by("number"):
        latest[page.search_id] = page
    people: dict[int, set] = defaultdict(set)
    on_page: dict[int, set] = defaultdict(set)
    first_page: dict[int, int] = {}
    for page_id, search_id, lead_id in (PageLead.objects.order_by("page_id")
                                        .values_list("page_id", "page__search_id", "lead_id")):
        people[search_id].add(lead_id)
        on_page[page_id].add(lead_id)
        first_page.setdefault(lead_id, page_id)
    return [Fired(page.search, page, people[search_id], on_page[page.pk],
                  {pk for pk in on_page[page.pk] if first_page[pk] == page.pk})
            for search_id, page in latest.items()]


# ── the arms ─────────────────────────────────────────────────────────

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
    rate: float = 0.0        # the fit rate expected of them, at the mean belief
    cost: float = 0.0        # dollars to fire it, take its new people in and judge them
    score: float | None = None  # fits on its page at mean + OPTIMISM σ, set when picked

    def density(self, theta: float) -> float:
        """The fit rate of the people still out there, were ``theta`` of the market's to fit."""
        out_there = self.count - self.seen
        if out_there <= 0:
            return 0.0
        return min(max((self.count * theta - self.held_fits) / out_there, 0.0), 1.0)

    @property
    def expected_fits(self) -> float:
        return self.new * self.rate

    def __repr__(self):
        return f"{self.terms.as_text()} @ page {self.number}"


class Frontier:
    """Everything the next search is chosen from, read from the store once."""

    def __init__(self, qualifier):
        from openoutfind.core import meter
        from openoutfind.core.ml import calibration
        from openoutfind.core.models import Page, PageLead, Search
        from openoutfind.core.pipeline.walk import FIRST_POLL_SECONDS

        self.qualifier = qualifier
        self.fit_rate = fit_rate_map()
        self.people = People.load(*lead_rates(qualifier, self.fit_rate))
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
        self._vocabulary: compose.Vocabulary | None = None

        returned = sum(Page.objects.values_list("rows", flat=True))
        self.kept = PageLead.objects.count() / returned if returned else 1.0

        self.fire_cost = calibration.dollars(meter.mean(meter.SEARCH_READY_SECONDS,
                                                        FIRST_POLL_SECONDS))
        self.person_cost = calibration.verdict_cost() + calibration.dollars(
            meter.mean(meter.INGEST_SECONDS_PER_LEAD, CAMPAIGN_CONFIG["ingest_seconds_per_lead"]))

    # ── candidates ──

    def arms(self) -> list[Arm]:
        """Every arm, valued at the mean belief: the next pages, the unfired written
        searches, and the searches composed at the mean."""
        arms = [self._deepen(f) for f in self.fired if not f.latest.at_bottom]
        fired_keys = {f.search.key for f in self.fired}
        arms += [self._first_page(search.query, "written", search) for search in self.written
                 if not search.dead and search.key not in fired_keys]
        arms = [arm for arm in arms if self._in_market(arm.terms)]
        self._settle(arms)
        composed = self._composed(self.beliefs.means(), self.beliefs.sizes.mean)
        self._rate(composed)
        return _distinct(arms + composed)

    def _composed(self, thetas: dict, sizes: belief.Sizes) -> list[Arm]:
        """Page 1 of the ``COMPOSED`` searches worth the most on ``thetas`` and ``sizes``
        that the frontier may fire, valued at the mean belief."""
        vocabulary = self.vocabulary
        lift = self.beliefs.lifts(thetas)
        share = self.beliefs.shares(vocabulary.keywords(), sizes)
        searches = compose.compose(vocabulary, lift, share, sizes.log_market,
                                   self._worth(sizes.log_market), COMPOSED)
        return [self._first_page(terms, "drawn") for terms in searches
                if terms.key not in self.taken and terms.is_legal
                and self._in_market(terms) and not self._closed(terms)]

    def _worth(self, log_market: float) -> compose.Worth:
        """The fits a page of a search brings, from its summed lift and share, with nobody
        yet seen — what ``pick`` ranks on, for the composition to sort by. Per dollar it
        ranked on the rate alone, since a verdict a person is most of a page's cost, and
        its best were searches of 1 to 6 people."""
        from openoutfind.core.models import PAGE_SIZE

        market = belief._log_odds(self.beliefs.market_rate)

        def worth(lift: np.ndarray, share: np.ndarray) -> np.ndarray:
            theta = 1.0 / (1.0 + np.exp(-(market + lift)))
            return theta * self.kept * np.minimum(PAGE_SIZE, np.exp(log_market + share))

        return worth

    @property
    def vocabulary(self) -> compose.Vocabulary:
        """Every keyword a composed search may hold: the store's people's title words and
        tag phrases, the written searches' terms, every seniority, and the market's places
        — the country and its states, or only the written states where every written
        search names one. No written search, no market, nothing composed."""
        if self._vocabulary is None:
            self._vocabulary = self._load_vocabulary()
        return self._vocabulary

    def _load_vocabulary(self) -> compose.Vocabulary:
        from openoutfind.discovery import LEAD_SENIORITIES

        written = [search.query for search in self.written]
        country = next((split_place(t.place)[1] for t in written if t.place), "")
        if not written:
            return compose.Vocabulary((), (), (), ())
        titles = self.people.title_vocabulary() | {w for t in written for w in t.title}
        tags = self.people.tag_vocabulary() | {t.tag for t in written if t.tag}
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

    def _in_market(self, terms: Terms) -> bool:
        """Whether ``terms`` stay in the market the written searches drew: where every one
        of them names a state, a search of the whole country is outside it. A store written
        partly on the country (batch 5: 8 of 10) has no such limit."""
        return bool(split_place(terms.place)[0]) or not self.market_is_states

    def _closed(self, terms: Terms) -> bool:
        """Whether a dead search or a drained one convicts ``terms``: it holds all their
        terms, so it matches nobody, or only people the store already holds."""
        return any(terms.holds_all_of(closed) for closed in self.closed)

    # ── valuing ──

    def _deepen(self, fired: Fired) -> Arm:
        """The next page of a fired search: as fresh as its latest one."""
        from openoutfind.core.models import PAGE_SIZE

        keywords = fired.keywords
        count = fired.found if fired.found is not None else self.beliefs.count(keywords)
        remaining = count - fired.latest.number * PAGE_SIZE
        novelty = len(fired.fresh) / fired.latest.rows if fired.latest.rows else 0.0
        page = max(0.0, min(PAGE_SIZE, remaining)) * novelty
        return self._valued(fired.terms, fired.latest.number + 1, "deepen", fired.search,
                            count, self.people.match(fired.terms) | fired.people, page)

    def _first_page(self, terms: Terms, move: str, search=None) -> Arm:
        """Page 1 of an unfired search, its count estimated."""
        matched = self.people.match(terms)
        count = self._count(terms, len(matched))
        return self._valued(terms, 1, move, search, count, matched,
                            self._first_page_people(count, len(matched)))

    def _first_page_people(self, count: float, seen: int) -> float:
        """The people page 1 of an unfired search would bring, were ``count`` its size."""
        from openoutfind.core.models import PAGE_SIZE

        return min(PAGE_SIZE, max(0.0, count - seen)) * self.kept

    def _count(self, terms: Terms, seen: int, sizes: belief.Sizes | None = None) -> float:
        """``N(q)`` for an unfired search: the belief's estimate, on the mean or on
        ``sizes``, kept above the people the store holds for it and within the count of
        every fired search it narrows."""
        estimate = max(self.beliefs.count(belief.query_keywords(terms), sizes),
                       seen * SEEN_MARGIN)
        return min([estimate] + [f.found for f in self.fired
                                 if f.found is not None and terms.holds_all_of(f.terms)])

    def _resized(self, arm: Arm, sizes: belief.Sizes) -> Arm:
        """``arm`` on ``sizes``. A next page's size is the provider's count, so it stays."""
        if arm.move == "deepen":
            return arm
        count = self._count(arm.terms, arm.seen, sizes)
        new = self._first_page_people(count, arm.seen)
        return replace(arm, count=count, new=new, cost=self.fire_cost + self.person_cost * new)

    def _valued(self, terms: Terms, number: int, move: str, search, count: float,
                seen: set[int], page: float) -> Arm:
        new = min(page, max(0.0, count - len(seen)))
        return Arm(terms, number, move, search, keywords=belief.query_keywords(terms),
                   count=count, seen=len(seen), held_fits=self._expected_fits(seen), new=new,
                   cost=self.fire_cost + self.person_cost * new)

    def _expected_fits(self, people: set[int]) -> float:
        """The fits among ``people``: a verdict where judged, else a fit rate. Counting only
        the judged left an unjudged person's share of ``F`` on those still out there, and
        36 of 515's 202 arms read a density of 1 and tied."""
        rates = self.people.rates
        return sum(rates.get(pk, self.beliefs.market_rate) for pk in people)

    def _settle(self, arms: list[Arm]) -> None:
        """Read the belief of every keyword — the whole vocabulary's and ``arms``' — in one
        pass, and each arm's rate at its mean."""
        keywords = sorted({*self.vocabulary.keywords(),
                           *(keyword for arm in arms for keyword in arm.keywords)})
        self.beliefs.settle(belief.keyword_priors(keywords, self.qualifier, self.fit_rate,
                                                  self.beliefs.market_rate))
        self._rate(arms)

    def _rate(self, arms: list[Arm]) -> None:
        """Each arm's ``θ(q)`` and fit rate at the mean belief."""
        means = self.beliefs.means()
        for arm in arms:
            arm.theta = (self.beliefs.search_mean(arm.search.key) if _is_next_page(arm)
                         else self.beliefs.rate(arm.keywords, means))
            arm.rate = arm.density(arm.theta)

    def _theta_on(self, arm: Arm, thetas: dict, searches: dict) -> float:
        """``θ(q)`` on one draw: a fired search's own, else its keywords'."""
        if _is_next_page(arm):
            return searches[arm.search.key]
        return self.beliefs.rate(arm.keywords, thetas)

    # ── the next search ──

    def pick(self, arms: list[Arm], round_number: int) -> Arm:
        """The arm whose page brings the most fits at its mean plus ``OPTIMISM`` σ, both
        read off ``DRAWS`` draws of the belief seeded by ``round_number``. Sets the arm's
        ``score``."""
        arms = _in_one_order(_distinct(arms))
        fits = self._fits_on_draws(arms, round_number)
        scores = fits.mean(axis=0) + OPTIMISM * fits.std(axis=0)
        chosen = arms[int(np.argmax(scores))]
        chosen.score = float(scores.max())
        return chosen

    def _fits_on_draws(self, arms: list[Arm], round_number: int) -> np.ndarray:
        """``(DRAWS, arms)``: the fits each arm's page brings on each draw."""
        keywords = {keyword for arm in arms for keyword in arm.keywords}
        fits = np.zeros((DRAWS, len(arms)))
        for n in range(DRAWS):
            seed = (round_number, n)
            thetas = self.beliefs.draw(seed, keywords)
            searches = self.beliefs.draw_searches(seed)
            sizes = self.beliefs.draw_sizes(seed, keywords)
            for i, arm in enumerate(arms):
                drawn = self._resized(arm, sizes)
                fits[n, i] = drawn.new * drawn.density(self._theta_on(arm, thetas, searches))
        return fits


def _is_next_page(arm: Arm) -> bool:
    return arm.move == "deepen"


def _distinct(arms: list[Arm]) -> list[Arm]:
    """``arms`` with one arm per search and page, the first kept."""
    seen: dict[tuple[str, int], Arm] = {}
    for arm in arms:
        seen.setdefault((arm.terms.key, arm.number), arm)
    return list(seen.values())


def _in_one_order(arms: list[Arm]) -> list[Arm]:
    """The arms in one order whatever made them, so a rerun breaks a tie the same way."""
    return sorted(arms, key=lambda arm: (arm.terms.key, arm.number))


def _worth_firing(arms: list[Arm], qualifier) -> list[Arm]:
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
    f = Frontier(qualifier)
    arms = _worth_firing(f.arms(), qualifier)
    _log_last_page(f)
    if not arms:
        return []
    chosen = f.pick(arms, round_number)
    _log(chosen, len(arms))
    return [chosen]


def best_expected(qualifier) -> Arm | None:
    """The arm worth the most at the mean belief, what ``judge_or_search`` weighs against
    the pool — ``None`` when none is worth firing."""
    f = Frontier(qualifier)
    arms = _worth_firing(f.arms(), qualifier)
    if not arms:
        return None
    return max(_in_one_order(arms), key=lambda arm: arm.expected_fits / arm.cost)


def written_arms() -> list[Arm]:
    """The initial population's searches not yet fired — the whole first round, fired
    unvalued: there is nothing yet to value them on."""
    from openoutfind.core.models import Search

    return [Arm(search.query, 1, "written", search)
            for search in Search.objects.filter(origin=Search.Origin.WRITTEN, dead=False,
                                                pages__isnull=True)]


def _log(chosen: Arm, candidates: int) -> None:
    logger.debug("  frontier: %d arm(s) · %-9s θ %.2f · density %.2f · N ~%.0f · seen %d · "
                 "~%.0f new · %.1f fits at +%gσ  %s", candidates, chosen.move, chosen.theta,
                 chosen.rate, chosen.count, chosen.seen, chosen.new, chosen.score, OPTIMISM,
                 chosen)


def _log_last_page(f: Frontier) -> None:
    """What became of the last fired page: the first runs read θ against this, and watch
    for pages nobody judges."""
    if not f.fired:
        return
    last = max(f.fired, key=lambda fired: fired.latest.round)
    people = last.latest_people
    rates = [f.people.rates[pk] for pk in people if pk in f.people.rates]
    logger.debug("  last page: %s · %d verdict(s), %d fit(s) of %d · observed rate %.2f",
                 last.terms.as_text(), len(people & f.judged), len(people & f.fits), len(people),
                 statistics.fmean(rates) if rates else 0.0)
