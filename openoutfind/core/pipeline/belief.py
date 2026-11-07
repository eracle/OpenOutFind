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
people, where one no fired search held stays. That fit is a Normal posterior, so the sizes
are drawn with ``θ_t``: its mean is the ridge solution, its covariance ``σ² (RᵀR)⁻¹``, with
``σ²`` the fit's own residual variance. A count pins down a product, so the keywords of one
search are drawn together. With one count there are no residuals, so the sizes stay at the fit.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from openoutfind.core.conf import WALK_CONFIG
from openoutfind.core.pipeline.query import Terms, split_place

logger = logging.getLogger(__name__)
# What an unjudged person's fit rate weighs against a verdict.
SCORE_WEIGHT = 0.25
# How hard a keyword's count is held to its share of the store, on a log scale.
RIDGE = 1.0
# Keeps the sizes' draw apart from the θ draw of the same round.
_SIZE_STREAM = 1
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

# A keyword's vector under its text, for this process: the frontier is read twice a
# step, and the store's copy (``core.KeywordVector``) is read once.
_VECTORS: dict[str, np.ndarray] = {}


def _vectors(texts: list[str]) -> np.ndarray:
    """Each text's vector: from this process, else the store, else embedded and kept."""
    missing = [text for text in dict.fromkeys(texts) if text not in _VECTORS]
    if missing:
        _VECTORS.update(_stored_or_embedded(missing))
    return np.array([_VECTORS[text] for text in texts], dtype=np.float64)


def _stored_or_embedded(texts: list[str]) -> dict[str, np.ndarray]:
    """``texts``' vectors, embedding and storing only those the store does not hold."""
    from openoutfind.core.conf import CAMPAIGN_CONFIG
    from openoutfind.core.ml.embeddings import embed_texts
    from openoutfind.core.models import KeywordVector

    model = CAMPAIGN_CONFIG["embedding_model"]
    found: dict[str, np.ndarray] = {}
    wanted = set(texts)
    for text, vector in (KeywordVector.objects.filter(model=model)
                         .values_list("text", "vector").iterator(chunk_size=5000)):
        if text in wanted:
            found[text] = np.frombuffer(bytes(vector), dtype=np.float32)
    new = [text for text in texts if text not in found]
    if new:
        logger.info("  embedding %d new keyword(s), once for this store", len(new))
        vectors = np.asarray(embed_texts(new), dtype=np.float32)
        KeywordVector.objects.bulk_create(
            [KeywordVector(text=text, model=model, vector=vector.tobytes())
             for text, vector in zip(new, vectors)],
            batch_size=1000, ignore_conflicts=True)
        found.update(zip(new, vectors))
    return found


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


class Sizes(NamedTuple):
    """The market's size and each keyword's share of it, on a log scale: ``log N₀`` and
    ``{keyword: log(N_t / N₀)}``. Either the fit's mean or one draw of it."""

    log_market: float
    log_keywords: dict


@dataclass(frozen=True)
class SizeFit:
    """The Normal posterior over the sizes: its mean, and how to draw from it."""

    mean: Sizes
    keywords: tuple = ()
    shares: np.ndarray | None = None
    solution: np.ndarray | None = None
    root: np.ndarray | None = None  # R with R·Rᵀ the solution's covariance
    sigma: float = 0.0              # the fit's residual standard deviation

    def draw(self, rng) -> Sizes:
        """One joint draw of the fitted sizes, or the mean where no count was fitted."""
        if self.root is None:
            return Sizes(self.mean.log_market, dict(self.mean.log_keywords))
        sample = self.solution + self.root @ rng.standard_normal(len(self.solution))
        return Sizes(float(sample[0]), _within_the_market(self.keywords, self.shares, sample))


def _within_the_market(keywords, shares: np.ndarray, solution: np.ndarray) -> dict:
    """``{keyword: log(N_t / N₀)}``: a keyword cannot hold more of the market than the market."""
    return dict(zip(keywords, np.minimum(shares + solution[1:], 0.0).tolist()))


def fit_sizes(fired, people) -> SizeFit:
    """The sizes, fitted to the fired searches' counts. Only keywords a counted search held
    appear; any other stays at its share of the store."""
    counted = [f for f in fired if f.found]
    if not counted:
        return SizeFit(Sizes(math.log(max(len(people.everyone), 1)), {}))

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
    mean = Sizes(float(solution[0]), _within_the_market(keywords, shares, solution))
    sigma = _residual_sigma(rows, target, solution)
    root = sigma * np.linalg.cholesky(np.linalg.inv(rows.T @ rows))
    return SizeFit(mean, tuple(keywords), shares, solution, root, sigma)


def _residual_sigma(rows: np.ndarray, target: np.ndarray, solution: np.ndarray) -> float:
    """The fit's residual standard deviation, the ridge rows counted as observations. One
    count leaves no degree of freedom, and no spread."""
    freedom = rows.shape[0] - rows.shape[1]
    if freedom < 1:
        return 0.0
    return math.sqrt(float(np.sum((target - rows @ solution) ** 2)) / freedom)


# ── the belief ───────────────────────────────────────────────────────

class Beliefs:
    """What the store says about every keyword, read once per frontier."""

    def __init__(self, people, fired, judged: set[int], market_rate: float):
        self.people = people
        self.judged = judged
        self.market_rate = _clip(market_rate)
        self.sizes = fit_sizes(fired, people)
        self.brought: dict[Keyword, set[int]] = defaultdict(set)
        for f in fired:
            for keyword in f.keywords:
                self.brought[keyword] |= f.people
        self.posterior: dict[Keyword, tuple[float, float]] = {}

    def count(self, keywords, sizes: Sizes | None = None) -> float:
        """``N(q)``: the people the market holds with every one of ``keywords``, on the
        fit's mean or on ``sizes``."""
        sizes = sizes or self.sizes.mean
        log_size = sizes.log_market + sum(
            sizes.log_keywords.get(keyword, _log_share(keyword, self.people))
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

    def draw_sizes(self, round_number: int) -> Sizes:
        """One draw of the sizes, seeded by ``round_number``. A keyword no count held
        draws from its prior, around its share of the store."""
        rng = np.random.default_rng([round_number, _SIZE_STREAM])
        drawn = self.sizes.draw(rng)
        spread = self.sizes.sigma / math.sqrt(RIDGE)
        for keyword in sorted(self.posterior):
            if keyword not in drawn.log_keywords:
                drawn.log_keywords[keyword] = min(
                    _log_share(keyword, self.people) + spread * rng.standard_normal(), 0.0)
        return drawn

    def lifts(self, thetas: dict[Keyword, float]) -> dict[Keyword, float]:
        """Each keyword's lift on ``thetas``: ``logit θ_t − logit π``, what ``rate`` adds up."""
        market = _log_odds(self.market_rate)
        return {keyword: _log_odds(_clip(theta)) - market for keyword, theta in thetas.items()}

    def shares(self, keywords, sizes: Sizes | None = None) -> dict[Keyword, float]:
        """Each keyword's ``log(N_t / N₀)`` on the fit's mean or on ``sizes``, what
        ``count`` adds up."""
        sizes = sizes or self.sizes.mean
        return {keyword: sizes.log_keywords.get(keyword, _log_share(keyword, self.people))
                for keyword in keywords}

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
