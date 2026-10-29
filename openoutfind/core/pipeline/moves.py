# openoutfind/core/pipeline/moves.py
"""The walk's moves — how a child search is bred from the population.

Each move works within one field's type (``query``), and draws its terms from what the
store holds: the title words and company tags of the store's people (the customer's marks
among them), the initial population's terms, and the states the store's people live in
inside the market. No LLM is called.

    move        title (words AND)   company tag (phrase)            place            seniority
    identity    the same search, one page deeper
    stricter    add a word          extend the phrase to a real     country → a      add one
                                    tag holding it, or add a tag    state in it
    wider       drop a word         shorten the phrase, or drop it  state → country  drop it
    swap        replace a word      replace the phrase              another state    another
    crossover   two searches exchange one field's terms

A term is drawn in proportion to how good the people holding it look — their verdict if
judged, else the GP's score — so the moves lean toward the words of the store's fits
without being confined to them. The space is never enumerated: only neighbours of what was
fired are made.

Identity is the walk's (``walk.breed``), since it pages a search rather than writing one.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from openoutfind.core.pipeline.query import (
    Terms, company_tags, holds_phrase, is_stopword_phrase, place_of, split_place,
    title_words,
)

# A term's weight never drops to nothing: a held term stays drawable even if every
# person holding it was judged a miss, at a twentieth of a fit's pull.
_FLOOR_WEIGHT = 0.05


@dataclass
class TermPools:
    """The terms the moves draw on, per field, each with a weight."""

    titles: Counter
    tags: Counter
    states: Counter
    seniorities: tuple[str, ...]
    country: str

    @classmethod
    def load(cls, scores: dict[int, float], written: list[Terms], country: str) -> "TermPools":
        """The store's people's terms, weighted by ``scores``, plus the written searches'.

        ``country`` is the market's, so only states inside it are offered.
        """
        from openoutfind.crm.models import Lead
        from openoutfind.discovery import LEAD_SENIORITIES, as_place

        titles, tags, states = Counter(), Counter(), Counter()
        for pk, fields in (Lead.objects.filter(synthetic=False)
                           .exclude(source_fields={}).values_list("pk", "source_fields")):
            weight = max(scores.get(pk, 0.0), _FLOOR_WEIGHT)
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


def _draw(rng: random.Random, weighted: Counter, exclude=()):
    """One key of ``weighted`` in proportion to its weight, skipping ``exclude``."""
    options = [(key, w) for key, w in weighted.items() if key not in exclude and w > 0]
    if not options:
        return None
    keys, weights = zip(*options)
    return rng.choices(keys, weights=weights, k=1)[0]


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

def _stricter(terms: Terms, field: str, pools: TermPools, rng) -> Terms | None:
    if field == "title":
        word = _draw(rng, pools.titles, exclude=terms.title)
        return terms.with_(title=terms.title | {word}) if word else None
    if field == "tag":
        if terms.tag:
            phrase = _draw(rng, _extensions(terms.tag, pools))
        else:
            phrase = _draw(rng, pools.tags)
        return terms.with_(tag=phrase) if phrase else None
    if field == "place":
        state, country = split_place(terms.place)
        if state or not country:
            return None
        new_state = _draw(rng, pools.states)
        return terms.with_(place=place_of(new_state, country)) if new_state else None
    if field == "seniority":
        if terms.seniority:
            return None
        return terms.with_(seniority=rng.choice(pools.seniorities))
    return None


def _wider(terms: Terms, field: str, pools: TermPools, rng) -> Terms | None:
    if field == "title":
        if not terms.title:
            return None
        return terms.with_(title=terms.title - {rng.choice(sorted(terms.title))})
    if field == "tag":
        if not terms.tag:
            return None
        if len(terms.tag) == 1 or rng.random() < 0.5:
            return terms.with_(tag=())
        return terms.with_(tag=terms.tag[1:] if rng.random() < 0.5 else terms.tag[:-1])
    if field == "place":
        state, country = split_place(terms.place)
        return terms.with_(place=place_of("", country)) if state else None
    if field == "seniority":
        return terms.with_(seniority="") if terms.seniority else None
    return None


def _swap(terms: Terms, field: str, pools: TermPools, rng) -> Terms | None:
    if field == "title":
        if not terms.title:
            return None
        word = _draw(rng, pools.titles, exclude=terms.title)
        if word is None:
            return None
        return terms.with_(title=(terms.title - {rng.choice(sorted(terms.title))}) | {word})
    if field == "tag":
        if not terms.tag:
            return None
        phrase = _draw(rng, pools.tags, exclude={terms.tag})
        return terms.with_(tag=phrase) if phrase else None
    if field == "place":
        state, country = split_place(terms.place)
        if not state:
            return None
        other = _draw(rng, pools.states,
                      exclude={name for name in pools.states if name.lower() == state})
        return terms.with_(place=place_of(other, country)) if other else None
    if field == "seniority":
        if not terms.seniority:
            return None
        return terms.with_(seniority=rng.choice(
            [s for s in pools.seniorities if s != terms.seniority]))
    return None


FIELDS = ("title", "tag", "place", "seniority")
_ONE_FIELD = {"stricter": _stricter, "wider": _wider, "swap": _swap}
MOVES = ("identity", "stricter", "wider", "swap", "crossover")


def mutate(move: str, terms: Terms, pools: TermPools, rng: random.Random) -> Terms | None:
    """One child of ``terms`` by ``move`` on one field drawn at random, or ``None``.

    Each field is tried in a random order until one gives a legal child.
    """
    operator = _ONE_FIELD[move]
    for field in rng.sample(FIELDS, len(FIELDS)):
        child = operator(terms, field, pools, rng)
        if child is not None and child != terms and child.is_legal:
            return child
    return None


def crossover(a: Terms, b: Terms, rng: random.Random) -> list[Terms]:
    """Two searches exchange one field's terms — the legal children among the two.

    ``production × vapes`` and ``purchasing × cannabis`` give ``production × cannabis``
    and ``purchasing × vapes``. A field is a ``Terms`` attribute of the same name.
    """
    differing = [field for field in FIELDS if getattr(a, field) != getattr(b, field)]
    if not differing:
        return []
    field = rng.choice(differing)
    children = [a.with_(**{field: getattr(b, field)}), b.with_(**{field: getattr(a, field)})]
    return [child for child in children if child.is_legal and child not in (a, b)]
