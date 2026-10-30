# openoutfind/core/pipeline/lattice.py
"""The lattice of searches — the neighbours of a fired search, the frontier's candidates.

Searches ordered by their terms form a lattice: adding a term never adds people
(``S(q ∧ t) ⊆ S(q)``), and neither does extending a tag phrase or narrowing the country to
a state. **Stricter** and **wider** move along it, **swap** and **exchange** across it.
Each move works within one field's type (``query``), and draws its terms from what the
store holds: the title words and company tags of the store's people (the customer's marks
among them), the initial population's terms, and the states the store's people live in inside
the market. No LLM is called.

    move        title (words AND)   company tag (phrase)            place            seniority
    stricter    add a word          extend the phrase to a real     country → a      add one
                                    tag holding it, or add a tag    state in it
    wider       drop a word         shorten the phrase              state → country  drop it
    swap        replace a word      replace the phrase              another state    another
    exchange    two fired searches exchange one field's terms

**A neighbour always holds a tag**, so wider never drops one, and a tag-less search
(only ever written) has as neighbours only the searches that add one. A search with no
tag is the whole market's head for its title: on CaleafTech's batch 5 the two tag-less
searches brought 334 of a round's 426 new people, none of them in cannabis.

The neighbours are enumerated, not drawn: each field offers its ``terms_per_field``
heaviest terms (``conf.WALK_CONFIG``), a term weighing what the companies holding it look
like — each company once, at its people's mean verdict or fit rate — so the candidates
lean toward the words of the store's fits, and the provider's "Self employed" row, one
nonsense tag list on 289 people, weighs as one company. The frontier (``frontier``) values every one and fires the best;
nothing beyond the fired searches' neighbours is ever made.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from openoutfind.core.conf import WALK_CONFIG
from openoutfind.core.pipeline.query import (
    Terms, company_tags, holds_phrase, is_stopword_phrase, place_of, split_place,
    title_words,
)

# A term's weight never drops to nothing: a held term stays offered even if every person
# holding it was judged a miss, at a twentieth of a fit's pull.
_FLOOR_WEIGHT = 0.05

FIELDS = ("title", "tag", "place", "seniority")


@dataclass
class TermPools:
    """The terms the neighbours are made from, per field, each with a weight."""

    titles: Counter
    tags: Counter
    states: Counter
    seniorities: tuple[str, ...]
    country: str

    @classmethod
    def load(cls, scores: dict[int, float], company: dict[int, object], written: list[Terms],
             country: str) -> "TermPools":
        """The store's people's terms, weighted by ``scores`` with each company counted
        once (``company``: lead → who they count with), plus the written searches'.

        ``country`` is the market's, so only states inside it are offered.
        """
        from openoutfind.crm.models import Lead
        from openoutfind.discovery import LEAD_SENIORITIES, as_place

        rows = list(Lead.objects.filter(synthetic=False).exclude(source_fields={})
                    .values_list("pk", "source_fields"))
        headcount = Counter(company.get(pk, pk) for pk, _ in rows)
        titles, tags, states = Counter(), Counter(), Counter()
        for pk, fields in rows:
            weight = max(scores.get(pk, 0.0), _FLOOR_WEIGHT) / headcount[company.get(pk, pk)]
            for word in title_words(fields.get("contact_job_title", "")):
                titles[word] += weight
            for tag in company_tags(fields.get("company_keywords", "")):
                if not is_stopword_phrase(tag):
                    tags[tag] += weight
            state = str(fields.get("contact_location_state") or "").strip()
            if state and str(fields.get("contact_location_country") or "").strip().lower() == country.lower():
                states[as_place(state)] += weight

        for terms in written:
            for word in terms.title:
                titles[word] += 1.0
            if terms.tag:
                tags[terms.tag] += 1.0
            state, _ = split_place(terms.place)
            if state:
                states[as_place(state)] += 1.0
        return cls(titles, tags, states, tuple(LEAD_SENIORITIES), country)


def _top(weighted: Counter, exclude=()) -> list:
    """The ``terms_per_field`` heaviest keys of ``weighted``, skipping ``exclude``."""
    keys = [key for key, w in weighted.most_common() if key not in exclude and w > 0]
    return keys[:WALK_CONFIG["terms_per_field"]]


def _extensions(tag: tuple[str, ...], pools: TermPools) -> Counter:
    """Phrases one word longer than ``tag`` that a real company tag holds, and holds it in."""
    longer: Counter = Counter()
    span = len(tag) + 1
    for real, weight in pools.tags.items():
        for start in range(len(real) - span + 1):
            window = real[start:start + span]
            if holds_phrase(window, tag):
                longer[window] += weight
    return longer


# ── one field at a time ──────────────────────────────────────────────

def _stricter(terms: Terms, field: str, pools: TermPools) -> list[Terms]:
    if field == "title":
        return [terms.with_(title=terms.title | {word})
                for word in _top(pools.titles, exclude=terms.title)]
    if field == "tag":
        phrases = _top(_extensions(terms.tag, pools)) if terms.tag else _top(pools.tags)
        return [terms.with_(tag=phrase) for phrase in phrases]
    if field == "place":
        state, country = split_place(terms.place)
        if state or not country:
            return []
        return [terms.with_(place=place_of(new_state, country))
                for new_state in _top(pools.states)]
    if field == "seniority":
        if terms.seniority:
            return []
        return [terms.with_(seniority=value) for value in pools.seniorities]
    return []


def _wider(terms: Terms, field: str, pools: TermPools) -> list[Terms]:
    if field == "title":
        return [terms.with_(title=terms.title - {word}) for word in sorted(terms.title)]
    if field == "tag":
        if len(terms.tag) < 2:
            return []
        return [terms.with_(tag=terms.tag[1:]), terms.with_(tag=terms.tag[:-1])]
    if field == "place":
        state, country = split_place(terms.place)
        return [terms.with_(place=place_of("", country))] if state else []
    if field == "seniority":
        return [terms.with_(seniority="")] if terms.seniority else []
    return []


def _swap(terms: Terms, field: str, pools: TermPools) -> list[Terms]:
    if field == "title":
        others = _top(pools.titles, exclude=terms.title)
        return [terms.with_(title=(terms.title - {old}) | {new})
                for old in sorted(terms.title) for new in others]
    if field == "tag":
        if not terms.tag:
            return []
        return [terms.with_(tag=phrase) for phrase in _top(pools.tags, exclude={terms.tag})]
    if field == "place":
        state, country = split_place(terms.place)
        if not state:
            return []
        same = {name for name in pools.states if name.lower() == state}
        return [terms.with_(place=place_of(other, country))
                for other in _top(pools.states, exclude=same)]
    if field == "seniority":
        if not terms.seniority:
            return []
        return [terms.with_(seniority=value) for value in pools.seniorities
                if value != terms.seniority]
    return []


_ONE_FIELD = {"stricter": _stricter, "wider": _wider, "swap": _swap}
MOVES = (*_ONE_FIELD, "exchange")


def neighbours(terms: Terms, pools: TermPools) -> list[tuple[Terms, str]]:
    """Every legal search holding a tag one move from ``terms`` on one field, with the
    move."""
    out = []
    for move, operator in _ONE_FIELD.items():
        for field in FIELDS:
            out += [(child, move) for child in operator(terms, field, pools)
                    if child != terms and child.tag and child.is_legal]
    return out


def exchange(a: Terms, b: Terms) -> list[Terms]:
    """Two searches exchange one field's terms — every legal result, for every field
    they differ in.

    ``production × vapes`` and ``purchasing × cannabis`` give ``production × cannabis``
    and ``purchasing × vapes``. A field is a ``Terms`` attribute of the same name.
    """
    out = []
    for field in FIELDS:
        if getattr(a, field) == getattr(b, field):
            continue
        out += [a.with_(**{field: getattr(b, field)}),
                b.with_(**{field: getattr(a, field)})]
    return [terms for terms in dict.fromkeys(out)
            if terms.tag and terms.is_legal and terms not in (a, b)]


def exchanges(searches: list[Terms]) -> list[tuple[Terms, str]]:
    """What every pair of ``searches`` gives by ``exchange``."""
    return [(terms, "exchange") for a, b in combinations(searches, 2)
            for terms in exchange(a, b)]
