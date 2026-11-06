# openoutfind/core/pipeline/frontier.py
"""The frontier — every search the walk could fire next, and the round chosen from them.

**The round covers the fits** (``Frontier.batch``). The promising people are every lead
judged a fit — the customer's marked fits among them — and the anchors, the ICP written
as profiles. An unfired search holds the promising people whose own fields match all its
terms; a next page holds the fits on its search's latest page. The batch is chosen
greedily: take the arm holding the most promising people, mark them covered, recount the
rest on the people not yet covered, and take the best again, ties to the arm matching
fewer of the store's people (the narrower search), until no arm holds anyone new. The
slots left are drawn at random from the other arms, seeded by the round's number so a
replay repeats. No rate, floor or cost orders the batch, and the GP only orders the
judging: an offline replay of seven ways to pick a round on four stores found this one
reached the most later fits by far — on batch 5, 55 over 15 rounds against the walk's 0
— where the rate below picked small areas and found none (openoutreach-docs card
``p2-e3-the-round-fires-searches-that-pay-or-teach``, finding 8).

**What each arm is expected to bring** is still valued, for ``judge_or_search``'s
decision of *when* to fire a round, and for the debug log:

    value(arm) = new(arm) · max(0, rate(arm) − floor)  ÷  (c_fire + c_ingest · new(arm))

- **An arm** is a search at a page: the next page of a fired search (``deepen``), a written
  search not yet fired, or a neighbour of a fired one on the lattice (``lattice``) — never
  a search already fired, dead, or holding all the terms of one that is, and never a
  search of the whole country where every written search names a state.
- **new** is how many people the page would bring that the store does not hold. A person
  already seen adds nothing, so **a search holding all the terms of a drained one** (its
  pages returned everything it counts) is never an arm: the lattice proves it can only
  re-find people. For an unfired search the count is estimated from the fired searches it
  is stricter than — their count times the share of their people it matches — or, with
  none, from the fired searches stricter than it, or the store's typical search; the
  people of the store it already matches are taken off as seen.
- **rate** is the fit rate expected of those people: the store's people the search matches
  locally (``query.Terms.matches``, through ``People``'s per-field index), each their
  verdict if judged, else their fit rate, averaged per company, then shrunk toward the
  store's base rate by ``prior_companies`` pseudo-companies, so a search matching two
  companies does not rank on their luck. A next page's rate is its latest page's people.
- **the fit rate** is the GP's score through an isotonic map fitted on the store's
  verdicts at their out-of-fold scores (``fit_rate_map``): the GP ranks well but is not
  calibrated (0.15 where the truth was 0.28), and a fit rate is what can be weighed
  against a cost. With one class of verdicts the score stands in.
- **floor** is the fit rate of the best lead waiting: a new person adds to the top of the
  pool only by beating it. With nothing waiting it is 0.
- **the costs** are run time in dollars (``calibration.dollars``): a search's measured
  ready time, and each new person's measured ingest time — embedding is most of a
  round's — so a page of 200 strangers pays for every one.

``judge_or_search`` weighs the best of the arms beating the floor against judging the
pool, and a round fires only while one does.
"""
from __future__ import annotations

import logging
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from openoutfind.core.conf import CAMPAIGN_CONFIG, WALK_CONFIG
from openoutfind.core.pipeline import lattice
from openoutfind.core.pipeline.query import Terms, company_tags, split_place, title_words

logger = logging.getLogger(__name__)

_EMPTY: frozenset = frozenset()


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


def lead_rates(qualifier, fit_rate) -> dict[int, float]:
    """``lead_id → rate`` for every stored person: their verdict if judged, else the GP's
    score through ``fit_rate``. A person the GP cannot score is left out."""
    from openoutfind.crm.models import Deal, DealState, Lead, Outcome

    rates: dict[int, float] = {
        lead_id: 0.0 if (state == DealState.FAILED and outcome == Outcome.WRONG_FIT) else 1.0
        for lead_id, state, outcome in Deal.objects.filter(lead_id__isnull=False)
        .values_list("lead_id", "state", "outcome")
    }
    unjudged = list(Lead.objects.filter(synthetic=False, embedding__isnull=False)
                    .exclude(pk__in=list(rates)).values_list("pk", "embedding"))
    if qualifier is None or not unjudged:
        return rates
    X = np.array([np.frombuffer(bytes(emb), dtype=np.float32) for _, emb in unjudged],
                 dtype=np.float64)
    probs = qualifier.predict_probs(X)
    if probs is not None:
        rates.update((pk, float(r)) for (pk, _), r in zip(unjudged, fit_rate(probs)))
    return rates


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

    def __init__(self, rows, rates: dict[int, float]):
        self.rates = rates
        self.company: dict[int, object] = {}
        self.everyone: set[int] = set()
        self._index: dict[tuple, set[int]] = defaultdict(set)
        for pk, company_id, fields in rows:
            self.everyone.add(pk)
            self.company[pk] = _company_key(pk, company_id, fields)
            for key in _index_keys(fields):
                self._index[key].add(pk)

    @classmethod
    def load(cls, rates: dict[int, float]) -> "People":
        from openoutfind.crm.models import Lead

        return cls(Lead.objects.filter(synthetic=False).exclude(source_fields={})
                   .values_list("pk", "company_id", "source_fields"), rates)

    @classmethod
    def promising(cls, fits: set[int]) -> "People":
        """The people a round covers: the ``fits`` and the anchors, each matched on its own
        fields — an anchor's are the model's own assignment of value to field."""
        from django.db.models import Q

        from openoutfind.crm.models import Lead

        return cls(Lead.objects.filter(Q(pk__in=fits) | Q(synthetic=True))
                   .exclude(source_fields={})
                   .values_list("pk", "company_id", "source_fields"), {})

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

    def rate_of(self, pks, prior: float) -> float:
        """The fit rate of ``pks``' region: per company, then over companies, shrunk
        toward ``prior`` by ``prior_companies`` pseudo-companies."""
        by_company: dict = defaultdict(list)
        for pk in pks:
            if pk in self.rates:
                by_company[self.company[pk]].append(self.rates[pk])
        strength = WALK_CONFIG["prior_companies"]
        total = sum(statistics.fmean(rates) for rates in by_company.values())
        return (total + strength * prior) / (len(by_company) + strength)

    def base_rate(self) -> float:
        """The store's own rate: every scored person, per company, then over companies."""
        by_company: dict = defaultdict(list)
        for pk, rate in self.rates.items():
            if pk in self.company:
                by_company[self.company[pk]].append(rate)
        if not by_company:
            return 0.0
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
    def found(self) -> int | None:
        return self.search.leads_found

    @property
    def drained(self) -> bool:
        """Whether its pages returned everything it counts."""
        from openoutfind.core.models import PAGE_SIZE

        return self.found is not None and self.latest.number * PAGE_SIZE >= self.found


def _fired() -> list[Fired]:
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
    size: float = 0.0        # people the provider would return on this page
    new: float = 0.0         # of them, people the store does not hold, after drops
    rate: float = 0.0        # the fit rate expected of them
    cost: float = 0.0        # dollars to fire it and take its new people in
    covers: int = 0          # promising people it was the first in its batch to hold
    # The store's people that stand for its region: a next page's latest page, an
    # unfired search's matched people.
    sample: set[int] = field(default_factory=set, repr=False)

    def beats(self, floor: float) -> bool:
        """Whether its new people are expected to fit more often than ``floor``."""
        return self.new * (self.rate - floor) > 0

    def cost_per_fit(self, verdict_cost: float) -> float:
        """Dollars per fit to fire it and judge every new person it brings."""
        fits = self.new * self.rate
        return (self.cost + verdict_cost * self.new) / fits if fits > 0 else math.inf

    def __repr__(self):
        return f"{self.terms.as_text()} @ page {self.number}"


class Frontier:
    """Everything the next batch is chosen from, read from the store once."""

    def __init__(self, qualifier):
        from openoutfind.core import meter
        from openoutfind.core.ml.calibration import dollars
        from openoutfind.core.models import PAGE_SIZE, Page, PageLead, Search
        from openoutfind.core.pipeline.walk import FIRST_POLL_SECONDS

        self.fit_rate = fit_rate_map()
        self.people = People.load(lead_rates(qualifier, self.fit_rate))
        self.prior = self.people.base_rate()
        self.fits = judged_fits()
        self.promising = People.promising(self.fits)
        self.fired = _fired()
        self.written = list(Search.objects.filter(origin=Search.Origin.WRITTEN))
        self.market_is_states = bool(self.written) and all(
            split_place(search.query.place)[0] for search in self.written)
        self.taken = set(Search.objects.filter(pages__isnull=False).values_list("key", flat=True))
        self.taken |= {search.key for search in self.written}
        self.dead = [search.query for search in Search.objects.filter(dead=True)]
        self.taken |= set(Search.objects.filter(dead=True).values_list("key", flat=True))
        self.closed = self.dead + [f.terms for f in self.fired if f.drained]

        returned = sum(Page.objects.values_list("rows", flat=True))
        self.kept = PageLead.objects.count() / returned if returned else 1.0
        counts = [f.found for f in self.fired if f.found]
        self.typical = statistics.median(counts) if counts else float(PAGE_SIZE)

        self.fire_cost = dollars(meter.mean(meter.SEARCH_READY_SECONDS, FIRST_POLL_SECONDS))
        self.ingest_cost = dollars(meter.mean(meter.INGEST_SECONDS_PER_LEAD,
                                              CAMPAIGN_CONFIG["ingest_seconds_per_lead"]))

    # ── candidates ──

    def arms(self) -> list[Arm]:
        """Every arm, each valued alone."""
        arms = [self._deepen(f) for f in self.fired if not f.latest.at_bottom]
        fired_keys = {f.search.key for f in self.fired}
        arms += [self._first_page(search.query, "written", search) for search in self.written
                 if not search.dead and search.key not in fired_keys]

        near: dict[str, tuple[Terms, str]] = {}
        pools = self._pools()
        live = [f.terms for f in self.fired if not f.search.dead]
        for terms in live:
            for neighbour, move in lattice.neighbours(terms, pools):
                near.setdefault(neighbour.key, (neighbour, move))
        for neighbour, move in lattice.exchanges(live):
            near.setdefault(neighbour.key, (neighbour, move))
        arms += [self._first_page(terms, move) for key, (terms, move) in near.items()
                 if key not in self.taken and not self._closed(terms)]
        return [arm for arm in arms if self._in_market(arm.terms)]

    def _in_market(self, terms: Terms) -> bool:
        """Whether ``terms`` stay in the market the written searches drew: where every one
        of them names a state, a search of the whole country is outside it. A store written
        partly on the country (batch 5: 8 of 10) has no such limit."""
        return bool(split_place(terms.place)[0]) or not self.market_is_states

    def _closed(self, terms: Terms) -> bool:
        """Whether a dead search or a drained one convicts ``terms``: it holds all their
        terms, so it matches nobody, or only people the store already holds."""
        return any(terms.holds_all_of(closed) for closed in self.closed)

    def _pools(self) -> lattice.TermPools:
        written = [search.query for search in self.written]
        country = next((split_place(t.place)[1] for t in written if t.place), "")
        return lattice.TermPools.load(self.people.rates, self.people.company, written, country)

    # ── valuing ──

    def _deepen(self, fired: Fired) -> Arm:
        """The next page of a fired search, like its latest one: as fresh, as good."""
        from openoutfind.core.models import PAGE_SIZE

        remaining = (fired.found - fired.latest.number * PAGE_SIZE
                     if fired.found is not None else PAGE_SIZE)
        size = max(0, min(PAGE_SIZE, remaining))
        novelty = len(fired.fresh) / fired.latest.rows if fired.latest.rows else 0.0
        return self._priced(Arm(fired.terms, fired.latest.number + 1, "deepen", fired.search,
                                size=size, new=size * novelty,
                                rate=self.people.rate_of(fired.latest_people, self.prior),
                                sample=fired.latest_people))

    def _first_page(self, terms: Terms, move: str, search=None) -> Arm:
        """Page 1 of an unfired search: its count estimated from the fired searches next
        to it, less the people of the store it already matches."""
        from openoutfind.core.models import PAGE_SIZE

        matched = self.people.match(terms)
        wider = [f for f in self.fired if f.found is not None and terms.holds_all_of(f.terms)]
        count = self._count(terms, matched, wider)
        size = min(PAGE_SIZE, count)
        novelty = max(0.0, 1.0 - len(matched) / count) if count > 0 else 0.0
        return self._priced(Arm(terms, 1, move, search, size=size,
                                new=size * self.kept * novelty,
                                rate=self.people.rate_of(self._evidence(matched, wider),
                                                         self.prior),
                                sample=matched))

    @staticmethod
    def _evidence(matched: set[int], wider: list[Fired]) -> set[int]:
        """Whose rates say what a search's people are like.

        The pages of the fired searches it narrows, where they hold any of its people:
        those are the provider's own sample of it. Otherwise everyone the store holds that
        it matches — a sample leaning toward the searches the store came from, which is
        why a tag-less ``founder officer`` read 0.26 on batch 5's cannabis founders while
        the tag-less ``founder`` page it narrows held none in the industry.
        """
        sampled = set().union(*(f.people & matched for f in wider))
        return sampled or matched

    def _count(self, terms: Terms, matched: set[int], wider: list[Fired]) -> float:
        """How many people the provider would count for ``terms``, given the fired
        searches it narrows (``wider``).

        Where their pages hold some of its people: a search's count times their share,
        the tightest such. Otherwise the store's typical count, raised to the largest
        fired search stricter than it and capped at the smallest it narrows. A share of
        none is no estimate: a page of a 49,004-person search is that search's
        famous-company head, and holding none of a narrow search's people says nothing
        about how many it has.
        """
        shares = [f.found * len(f.people & matched) / len(f.people)
                  for f in wider if f.people & matched]
        if shares:
            return min(shares)
        narrower = [f.found for f in self.fired
                    if f.found is not None and f.terms.holds_all_of(terms)]
        return min([max([self.typical] + narrower)] + [f.found for f in wider])

    def _priced(self, arm: Arm) -> Arm:
        arm.cost = self.fire_cost + self.ingest_cost * arm.new
        return arm

    # ── the batch ──

    def holds(self, arm: Arm) -> set[int]:
        """The promising people ``arm`` holds: for a next page, the fits on its search's
        latest page; for an unfired search, the fits and anchors matching all its terms."""
        if arm.move == "deepen":
            return arm.sample & self.fits
        return self.promising.match(arm.terms)

    def batch(self, arms: list[Arm], round_number: int, slots: int | None = None) -> list[Arm]:
        """The ``slots`` arms of a round: those that between them hold the most promising
        people, chosen greedily, then a random fill seeded by ``round_number``."""
        slots = slots or WALK_CONFIG["batch"]
        # One order however the lattice enumerated them, so a replay draws the same fill.
        rest = sorted(arms, key=lambda arm: (arm.terms.key, arm.number))
        holding = {id(arm): self.holds(arm) for arm in rest}

        chosen: list[Arm] = []
        covered: set[int] = set()
        while rest and len(chosen) < slots:
            best = max(rest, key=lambda arm: (len(holding[id(arm)] - covered),
                                              -len(arm.sample)))
            gained = holding[id(best)] - covered
            if not gained:
                break
            best.covers = len(gained)
            chosen.append(best)
            rest.remove(best)
            covered |= gained

        drawn = random.Random(round_number).sample(rest, min(len(rest), slots - len(chosen)))
        _log(chosen, drawn, len(arms), len(self.promising.everyone))
        return chosen + drawn


def arms_beating(qualifier, floor: float) -> list[Arm]:
    """Every arm whose new people are expected to fit more often than ``floor``, each
    valued alone — what ``judge_or_search`` weighs a search on."""
    return [arm for arm in Frontier(qualifier).arms() if arm.beats(floor)]


def next_round(qualifier, round_number: int, floor: float = 0.0) -> list[Arm]:
    """The round's batch, or ``[]`` when no arm beats ``floor``, the fit rate of the best
    lead waiting: ``judge_or_search`` decides *when* a round fires, the batch *what*."""
    f = Frontier(qualifier)
    arms = f.arms()
    if not any(arm.beats(floor) for arm in arms):
        return []
    return f.batch(arms, round_number)


def written_arms() -> list[Arm]:
    """The initial population's searches not yet fired — the whole first round, fired
    unvalued: there is nothing yet to value them on."""
    from openoutfind.core.models import Search

    return [Arm(search.query, 1, "written", search)
            for search in Search.objects.filter(origin=Search.Origin.WRITTEN, dead=False,
                                                pages__isnull=True)]


def _log(chosen: list[Arm], drawn: list[Arm], candidates: int, promising: int) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("  frontier: %d arm(s), %d cover %d of %d promising, %d drawn",
                 candidates, len(chosen), sum(arm.covers for arm in chosen), promising,
                 len(drawn))
    for arm in chosen:
        logger.debug("  %-9s covers %d · rate %.2f · ~%.0f new  %s", arm.move, arm.covers,
                     arm.rate, arm.new, arm)
    for arm in drawn:
        logger.debug("  %-9s drawn · rate %.2f · ~%.0f new  %s", arm.move, arm.rate,
                     arm.new, arm)
