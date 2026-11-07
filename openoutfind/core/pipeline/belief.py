# openoutfind/core/pipeline/belief.py
"""The walk's belief about the market — how many people hold each keyword, and how many
of them the customer would like.

Two numbers per keyword, a title word, a tag phrase, a seniority or a place:

- ``N_t``, the people in the market holding it;
- ``θ_t``, the share of them who fit.

**``θ_t`` is a Beta, so an update is a count.** A person is a sample of every keyword they
hold, and of the keywords of every search that brought them. A judged person adds their
verdict; an unjudged one adds their fit rate at ``SCORE_WEIGHT`` of a verdict, since every
score comes from the same few verdicts: a page of bad people lowers its keywords with
nobody judged, and 200 scores do not pass for 200 verdicts. A company adds the mean of its
people once, because colleagues share tags.

**The prior** is the GP's score of the keyword's own embedding, through the fit-rate map,
at a fixed ``prior_companies`` pseudo-companies: the GP was trained on the verdicts the
counts hold, so it is kept weak enough for them to overtake it. The anchors reach a keyword
through the GP alone.

**A query is an AND of keywords**, taken as independent, with ``π`` the market's fit rate:

    N(q)      = N₀ · Π (N_t / N₀)
    odds θ(q) = odds π · Π (odds θ_t / odds π)

**The provider counts queries, never a keyword**, and a count costs what a page costs, so
``log N₀`` and the ``log N_t`` are fitted to the fired searches' counts by ridge regression:
``log N₀`` is the intercept, and each keyword is pulled toward its share of the store's
people, where one no fired search held stays.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

import numpy as np

from openoutfind.core.conf import WALK_CONFIG
from openoutfind.core.pipeline.query import Terms, split_place

# What an unjudged person's fit rate weighs against a verdict.
SCORE_WEIGHT = 0.25
# How hard a keyword's count is held to its share of the store, on a log scale.
RIDGE = 1.0
# A rate is kept off 0 and 1, where its odds would end the product.
_MIN_RATE = 0.001

Keyword = tuple


def query_keywords(terms: Terms) -> tuple[Keyword, ...]:
    """A query's keywords, the keys ``People.holding`` reads. A place is one keyword, the
    state where the query names one and the country otherwise: a state implies its country."""
    keywords: list[Keyword] = [("title", word) for word in sorted(terms.title)]
    if terms.tag:
        keywords.append(("tag", terms.tag))
    if terms.seniority:
        keywords.append(("seniority", terms.seniority))
    if terms.place:
        state, country = split_place(terms.place)
        keywords.append(("state", state, country) if state else ("country", country))
    return tuple(keywords)


def keyword_text(keyword: Keyword) -> str:
    """What a keyword reads as, for its embedding."""
    value = keyword[1]
    return " ".join(value) if isinstance(value, tuple) else value


def evidence(pks, people, judged: set[int]) -> tuple[float, float]:
    """``(fits, no-fits)`` that ``pks`` add to a keyword, each company once: the mean of its
    verdicts at full weight, or with none judged the mean of its fit rates at ``SCORE_WEIGHT``."""
    by_company: dict = defaultdict(list)
    for pk in pks:
        if pk in people.rates and pk in people.company:
            by_company[people.company[pk]].append(pk)
    fits = misses = 0.0
    for members in by_company.values():
        verdicts = [people.rates[pk] for pk in members if pk in judged]
        rate = statistics.fmean(verdicts or [people.rates[pk] for pk in members])
        weight = 1.0 if verdicts else SCORE_WEIGHT
        fits += weight * rate
        misses += weight * (1.0 - rate)
    return fits, misses


# ── the prior ────────────────────────────────────────────────────────

# A keyword's vector under its text: the frontier is read before every search, and the
# same few hundred keywords come back each time.
_VECTORS: dict[str, np.ndarray] = {}


def _vectors(texts: list[str]) -> np.ndarray:
    from openoutfind.core.ml.embeddings import embed_texts

    missing = [text for text in dict.fromkeys(texts) if text not in _VECTORS]
    if missing:
        _VECTORS.update(zip(missing, embed_texts(missing)))
    return np.array([_VECTORS[text] for text in texts], dtype=np.float64)


def keyword_priors(keywords: list[Keyword], qualifier, fit_rate,
                   market_rate: float) -> dict[Keyword, float]:
    """``g_t`` for each keyword: the GP's score of its embedding through ``fit_rate``, or
    ``market_rate`` where the GP cannot score."""
    if qualifier is None or not keywords:
        return dict.fromkeys(keywords, market_rate)
    probs = qualifier.predict_probs(_vectors([keyword_text(k) for k in keywords]))
    if probs is None:
        return dict.fromkeys(keywords, market_rate)
    return {keyword: float(rate) for keyword, rate in zip(keywords, fit_rate(probs))}


# ── the market's size ────────────────────────────────────────────────

def _log_share(keyword: Keyword, people) -> float:
    return math.log((len(people.holding(keyword)) + 1) / (len(people.everyone) + 2))


def fit_sizes(fired, people) -> tuple[float, dict[Keyword, float]]:
    """``(log N₀, {keyword: log(N_t / N₀)})``, fitted to the fired searches' counts. Only
    keywords a counted search held appear; any other stays at its share of the store."""
    counted = [f for f in fired if f.found]
    if not counted:
        return math.log(max(len(people.everyone), 1)), {}

    keywords = sorted({keyword for f in counted for keyword in f.keywords})
    column = {keyword: i + 1 for i, keyword in enumerate(keywords)}
    shares = np.array([_log_share(keyword, people) for keyword in keywords])

    rows = np.zeros((len(counted) + len(keywords), len(keywords) + 1))
    target = np.zeros(len(rows))
    for i, f in enumerate(counted):
        rows[i, 0] = 1.0
        held = [column[keyword] for keyword in f.keywords]
        rows[i, held] = 1.0
        target[i] = math.log(f.found) - shares[[c - 1 for c in held]].sum()
    for i in range(len(keywords)):
        rows[len(counted) + i, i + 1] = math.sqrt(RIDGE)

    solution = np.linalg.lstsq(rows, target, rcond=None)[0]
    # A keyword cannot hold more of the market than the market.
    fitted = np.minimum(shares + solution[1:], 0.0)
    return float(solution[0]), dict(zip(keywords, fitted.tolist()))


# ── the belief ───────────────────────────────────────────────────────

class Beliefs:
    """What the store says about every keyword, read once per frontier."""

    def __init__(self, people, fired, judged: set[int], market_rate: float):
        self.people = people
        self.judged = judged
        self.market_rate = _clip(market_rate)
        self.log_market, self.log_sizes = fit_sizes(fired, people)
        self.brought: dict[Keyword, set[int]] = defaultdict(set)
        for f in fired:
            for keyword in f.keywords:
                self.brought[keyword] |= f.people
        self.posterior: dict[Keyword, tuple[float, float]] = {}

    def count(self, keywords) -> float:
        """``N(q)``: the people the market holds with every one of ``keywords``."""
        log_size = self.log_market + sum(
            self.log_sizes.get(keyword, _log_share(keyword, self.people))
            for keyword in keywords)
        return math.exp(log_size)

    def settle(self, priors: dict[Keyword, float]) -> None:
        """Each keyword's Beta: ``priors`` at ``prior_companies``, plus the counts."""
        strength = WALK_CONFIG["prior_companies"]
        for keyword, prior in priors.items():
            held = self.people.holding(keyword) | self.brought.get(keyword, set())
            fits, misses = evidence(held, self.people, self.judged)
            prior = _clip(prior)
            self.posterior[keyword] = (strength * prior + fits,
                                       strength * (1.0 - prior) + misses)

    def means(self) -> dict[Keyword, float]:
        """Each keyword's expected ``θ_t``."""
        return {keyword: a / (a + b) for keyword, (a, b) in self.posterior.items()}

    def draw(self, round_number: int) -> dict[Keyword, float]:
        """One ``θ_t`` per keyword from its Beta, seeded by ``round_number`` so a rerun
        repeats. A keyword shared by two arms has one draw."""
        rng = np.random.default_rng(round_number)
        return {keyword: float(rng.beta(*self.posterior[keyword]))
                for keyword in sorted(self.posterior)}

    def rate(self, keywords, thetas: dict[Keyword, float]) -> float:
        """``θ(q)`` for a query of ``keywords``, given a ``θ_t`` for each."""
        log_odds = _log_odds(self.market_rate)
        for keyword in keywords:
            log_odds += _log_odds(_clip(thetas[keyword])) - _log_odds(self.market_rate)
        return 1.0 / (1.0 + math.exp(-log_odds))


def _clip(rate: float) -> float:
    return min(max(rate, _MIN_RATE), 1.0 - _MIN_RATE)


def _log_odds(rate: float) -> float:
    return math.log(rate / (1.0 - rate))
