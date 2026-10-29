# openoutfind/core/pipeline/walk.py
"""The walk — a genetic algorithm over searches, on DEAP.

An **individual** is a search and the page it is on; its **fitness** is how good the
people on that page are (``fitness``). The initial population is written whole by the LLM
(``icp.write_initial_population``) instead of drawn at random. Each generation is one
blocking step of the run:

1. **Rank** the last survivors and the last generation's children together by fitness,
   on the GP as it stands now, and **keep the top 10** (μ + λ, ``tools.selBest``). The
   first generation has nothing to rank.
2. **Breed 10 children** from them with the moves (``moves``): identity pages the
   fittest survivors one page deeper, the other moves write new searches.
3. **Fire all 10 at once** — page 1 for a new search, the next page for an identity
   child. The worker threads only make the API calls.
4. **Wait for all of them**, then, in the main thread, record each page and take every
   page's people into the store together (``discover.harvest``): one embedding pass over
   the whole generation, shortest profile first, since embedding is most of its time.
   The children join the population unranked.
5. Hand back to the run, which goes on judging until it asks for the next generation
   (``judge_or_search``).

**A generation's children are ranked at the start of the next one, not when they land**:
by then the run has judged the people they brought in, and the GP reads every page with
those verdicts. Ranked on landing, a child would be cut on a GP that had not yet seen
any of its people judged, and a cut page never comes back.

DEAP supplies the loop's parts — the fitness type, the toolbox the operators are
registered on, ``toolbox.map`` as a thread pool's ``map`` so a generation fires
concurrently, tournament selection of parents and ``selBest`` over parents and children.
The walk supplies what is particular to it: the individual, the operators, the
evaluation (fire the page, score its people) and the initial population.

**Every child is fired before it is ranked**, so nothing is guessed for a search that has
not been fired. A search whose page 1 comes back empty is **dead**, and so is every search
holding all its terms. A search at the **bottom** — the page past its count, or page 50,
where the provider's 10,000-row reach ends — no longer pages, but still breeds through the
other moves, since a narrower or sibling search opens a fresh window. Nothing else is
pruned: a stricter child of a drained search fires like any other and is ranked on its
page.
"""
from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from deap import base, creator, tools
from termcolor import colored

from openoutfind.core.conf import WALK_CONFIG
from openoutfind.core.pipeline import fitness, moves

logger = logging.getLogger(__name__)

# Where a search's first poll goes until the store has measured its own ready time: the
# probe saw searches ready 1.1–3.6s after the submit.
FIRST_POLL_SECONDS = 1.5

if not hasattr(creator, "PageFitness"):
    creator.create("PageFitness", base.Fitness, weights=(1.0,))


class Individual:
    """A search at a page — about to be fired, or fired and in the population.

    ``search`` is the ``Search`` row once there is one (a bred child gets its row when it
    is fired), ``page`` the fired ``Page``.
    """

    def __init__(self, terms, number: int, search=None, move: str = "", page=None):
        self.terms = terms
        self.number = number
        self.search = search
        self.move = move
        self.page = page
        self.fitness = creator.PageFitness()

    @classmethod
    def of(cls, page) -> "Individual":
        return cls(page.search.query, page.number, search=page.search, page=page)

    def __repr__(self):
        return f"{self.terms.as_text()} @ page {self.number}"


@dataclass(frozen=True)
class Generation:
    """What one generation did."""

    number: int
    fired: int
    new_leads: int


# ── the population ───────────────────────────────────────────────────

def population():
    """The pages the last generation kept, as individuals."""
    from openoutfind.core.models import Page

    return [Individual.of(page) for page in
            Page.objects.filter(in_population=True).select_related("search")]


def evaluate(individuals, qualifier) -> None:
    """Give each fired individual its page's fitness, on the GP as it stands now."""
    scores = fitness.page_fitness([ind.page for ind in individuals], qualifier)
    for ind in individuals:
        ind.fitness.values = (scores[ind.page.pk],)


def select(candidates, qualifier) -> list[Individual]:
    """The fittest ``WALK_CONFIG["population"]`` of the last survivors and the last
    children, fittest first — the parents of the next generation.

    Ranked here, at the start of a generation, and nowhere else: the run has judged since
    the last one fired, so the GP reads its children's people with those verdicts in.
    """
    from openoutfind.core.models import Page

    if not candidates:
        return []
    evaluate(candidates, qualifier)
    survivors = tools.selBest(candidates, WALK_CONFIG["population"])
    Page.objects.filter(in_population=True).exclude(
        pk__in=[ind.page.pk for ind in survivors]).update(in_population=False)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("  survivors of %d:", len(candidates))
        for ind in survivors:
            logger.debug("  %.3f  %s", ind.fitness.values[0], ind)
    return survivors


def last_children_fitness(qualifier) -> float | None:
    """The mean fitness of the last generation's children — what a generation brings.

    Not its best survivor, whose people are already in the pool. ``None`` before the
    first generation, or when it returned no people.
    """
    from django.db.models import Max

    from openoutfind.core.models import Page

    last = Page.objects.aggregate(Max("generation"))["generation__max"]
    if last is None:
        return None
    pages = list(Page.objects.filter(generation=last, rows__gt=0).select_related("search"))
    if not pages:
        return None
    scores = fitness.page_fitness(pages, qualifier)
    return sum(scores.values()) / len(scores)


# ── breeding ─────────────────────────────────────────────────────────

def _toolbox(pools, rng, executor) -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register("map", executor.map)
    for move in ("stricter", "wider", "swap"):
        toolbox.register(move, moves.mutate, move, pools=pools, rng=rng)
    toolbox.register("mate", moves.crossover, rng=rng)
    toolbox.register("pick", tools.selTournament, tournsize=3)
    return toolbox


class _Nursery:
    """The children of one generation, and what a new one must not repeat.

    A new search is one never fired and not dead, not convicted by a dead search it holds
    all the terms of, and not already bred this generation.
    """

    def __init__(self):
        from django.db.models import Q

        from openoutfind.core.models import Search

        self.children: list[Individual] = []
        self._taken = set(Search.objects.filter(Q(pages__isnull=False) | Q(dead=True))
                          .values_list("key", flat=True))
        self._dead = [search.query for search in Search.objects.filter(dead=True)]

    def admit(self, terms, move: str) -> bool:
        if terms is None or not terms.is_legal or terms.key in self._taken:
            return False
        if any(terms.holds_all_of(dead) for dead in self._dead):
            return False
        self._taken.add(terms.key)
        self.children.append(Individual(terms, 1, move=move))
        return True


def _identity(ranked, slots: int) -> list[Individual]:
    """The next page of the fittest survivors not at the bottom, one per search."""
    from openoutfind.core.models import Page

    children = []
    for ind in ranked:
        if len(children) >= slots:
            break
        if ind.page.at_bottom or Page.objects.filter(
                search=ind.search, number=ind.number + 1).exists():
            continue
        children.append(Individual(ind.terms, ind.number + 1, search=ind.search,
                                   move="identity"))
    return children


def _bred(move: str, ranked, slots: int, toolbox, nursery: _Nursery) -> int:
    """Up to ``slots`` new searches by ``move`` from tournament-picked parents."""
    made = 0
    for _ in range(slots * WALK_CONFIG["tries_per_slot"]):
        if made >= slots:
            break
        if move == "crossover":
            if len(ranked) < 2:
                break
            a, b = toolbox.pick(ranked, k=2)
            for child in toolbox.mate(a.terms, b.terms):
                if made < slots and nursery.admit(child, move):
                    made += 1
        else:
            (parent,) = toolbox.pick(ranked, k=1)
            made += nursery.admit(getattr(toolbox, move)(parent.terms), move)
    return made


def breed(ranked, toolbox) -> list[Individual]:
    """The children of a generation, split across the moves (``WALK_CONFIG["moves"]``).

    A move with no legal child gives its slot to the next one.
    """
    nursery = _Nursery()
    children: list[Individual] = []
    carry = 0
    for move in moves.MOVES:
        slots = WALK_CONFIG["moves"].get(move, 0) + carry
        if move == "identity":
            made = _identity(ranked, slots)
            children += made
            carry = slots - len(made)
        else:
            carry = slots - _bred(move, ranked, slots, toolbox, nursery)
    return children + nursery.children


def _unfired_written() -> list[Individual]:
    """The initial population's searches not yet fired — the whole first generation."""
    from openoutfind.core.models import Search

    return [Individual(search.query, 1, search=search, move="written")
            for search in Search.objects.filter(origin=Search.Origin.WRITTEN, dead=False,
                                                pages__isnull=True)]


# ── firing ───────────────────────────────────────────────────────────

def _shape() -> dict:
    """The size band and market code every search rides — the initial population's."""
    from openoutfind.core.models import Search

    first = Search.objects.filter(origin=Search.Origin.WRITTEN).order_by("pk").first()
    if first is None:
        return {}
    return {"headcount_min": first.headcount_min, "headcount_max": first.headcount_max,
            "country_code": first.country_code}


def _fire_all(children, toolbox, api_key: str) -> list:
    """Fire every child at once; return each one's ``Fetched`` or the exception it raised.

    The threads make the API calls and nothing else, so an outage on one search is
    handed back rather than raised in a worker.
    """
    from openoutfind.core import meter
    from openoutfind.core.models import PAGE_SIZE
    from openoutfind.discovery import search
    from openoutfind.enrichment.bettercontact import BetterContactUnavailable

    first_poll = meter.mean(meter.SEARCH_READY_SECONDS, FIRST_POLL_SECONDS)
    shape = _shape()
    headcount = (shape.get("headcount_min", 1), shape.get("headcount_max", 10000))

    def fire(child):
        try:
            return search(api_key, child.terms.to_filters(headcount), limit=PAGE_SIZE,
                          offset=(child.number - 1) * PAGE_SIZE, first_poll_s=first_poll)
        except BetterContactUnavailable as exc:
            return exc

    return list(toolbox.map(fire, children))


def _take_in(child: Individual, result, generation: int) -> list[dict]:
    """Record one fired child and its page. Returns the page's rows, empty when it has no
    page to rank; ``harvest`` takes the whole generation's rows in at once.

    A provider refusal (bad key, no credits, rate-limited through the backoff) is
    re-raised: every later search would repeat it. An outage is not evidence about the
    search and leaves no trace. An empty page 1 with a positive count is a transport
    artifact — the burst that once answered a 71-million-lead query with nothing in 0.0s
    — and leaves no trace either; with a count of 0 the search is dead.
    """
    from openoutfind.core import meter
    from openoutfind.core.errors import ErrorType
    from openoutfind.core.models import PAGE_SIZE, Page, Search
    from openoutfind.discovery import query_header, step_line
    from openoutfind.enrichment.bettercontact import BetterContactUnavailable

    if isinstance(result, BetterContactUnavailable):
        if result.error_type != ErrorType.PROVIDER_UNAVAILABLE:
            raise result
        logger.warning("%s", step_line("fetch", f"provider unavailable ({result}) — "
                                                f"{child} not fired", glyph="⚠", color="red"))
        return []

    page, ready = result
    meter.record(meter.SEARCH_READY_SECONDS, ready)
    logger.info("%s", query_header(child.terms.to_filters(), (child.number - 1) * PAGE_SIZE))
    found = f" · {page.leads_found:,} in the index" if page.leads_found else ""
    logger.info("%s", step_line("leadfinder", f"{len(page.leads)} leads{found} "
                                              f"· {child.move}"))

    if child.search is None:
        child.search, _ = Search.objects.get_or_create(key=child.terms.key, defaults={
            "terms": child.terms.as_json(), "move": child.move, **_shape()})

    if child.number == 1 and not page.leads:
        if page.leads_found:
            logger.warning("%s", step_line("fetch", f"empty page but {page.leads_found:,} "
                                                    f"in the index — transport artifact",
                                           glyph="⚠", color="yellow"))
            return []
        child.search.dead = True
        child.search.save(update_fields=["dead"])
        logger.debug("%s", step_line("fetch", "nobody matches — the search is dead, and so "
                                              "is any holding all its terms",
                                     glyph="✗", color="yellow"))
        return []

    if child.number == 1:
        child.search.leads_found = page.leads_found
        child.search.save(update_fields=["leads_found"])

    child.page = Page.objects.create(search=child.search, number=child.number,
                                     generation=generation, rows=len(page.leads))
    return page.leads  # empty past the end: the search is at the bottom


# ── one generation ───────────────────────────────────────────────────

def run_generation(site_config, qualifier, rng: random.Random | None = None) -> Generation | None:
    """Breed, fire, take in, rank and keep one generation. ``None`` when nothing is left
    to fire — no unfired written search, and no population to breed from.

    The first call writes the initial population (one LLM call, or ``population_pending``
    under ``--agent-qualify``) and fires it whole.
    """
    from django.db.models import Max

    from openoutfind.core import meter
    from openoutfind.core.models import Page, Search
    from openoutfind.core.pipeline.discover import harvest
    from openoutfind.core.pipeline.icp import write_initial_population
    from openoutfind.crm.models import Lead

    rng = rng or random.Random()
    if not Search.objects.exists():
        write_initial_population(site_config)

    started = time.monotonic()
    before = Lead.objects.filter(synthetic=False, marked=False).count()
    last = Page.objects.aggregate(Max("generation"))["generation__max"]
    number = 0 if last is None else last + 1

    parents = select(population(), qualifier)

    with ThreadPoolExecutor(max_workers=WALK_CONFIG["population"]) as executor:
        toolbox = _toolbox(_pools(qualifier), rng, executor)
        children = _unfired_written() + (breed(parents, toolbox) if parents else [])
        if not children:
            return None
        results = _fire_all(children, toolbox, site_config.bettercontact_api_key)

    pages = [(child, rows) for child, result in zip(children, results)
             if (rows := _take_in(child, result, number))]
    fired = [child for child, _ in pages]
    if fired:
        harvest([(child.page, rows) for child, rows in pages], fired[0].search.country_code)
        # Unranked until the next generation: its people have not been judged yet.
        Page.objects.filter(pk__in=[child.page.pk for child in fired]).update(
            in_population=True)

    new_leads = Lead.objects.filter(synthetic=False, marked=False).count() - before
    meter.record(meter.GENERATION_SECONDS, time.monotonic() - started)
    meter.record(meter.GENERATION_LEADS, new_leads)
    if new_leads:
        # New leads make a new pool, so how often its verdicts fit starts over.
        meter.reset(meter.POOL_FIT)

    return _log(number, children, fired, new_leads)


def _pools(qualifier) -> moves.TermPools:
    """The terms the moves draw on: every stored person's, weighted by their score, and
    the written searches'."""
    from openoutfind.core.models import Search
    from openoutfind.core.pipeline.query import split_place
    from openoutfind.crm.models import Lead

    written = [search.query for search in Search.objects.filter(origin=Search.Origin.WRITTEN)]
    country = next((split_place(t.place)[1] for t in written if t.place), "")
    scores = fitness.lead_scores(qualifier, Lead.objects.filter(synthetic=False)
                                 .exclude(source_fields={}).values_list("pk", flat=True))
    return moves.TermPools.load(scores, written, country)


def _log(number: int, children, fired, new_leads: int) -> Generation:
    """Say what the generation did. Its children are not scored until the next one."""
    logger.info("%s", colored(
        f"■ generation {number}: {len(fired)} of {len(children)} search(es) fired · "
        f"+{new_leads} new lead(s)", "blue"))
    return Generation(number, len(fired), new_leads)
