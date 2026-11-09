# openoutfind/core/pipeline/belief.py
"""The walk's belief about the market — how many people hold each keyword, and how many
of them the customer would like — and every read of it the walk makes, at the mean or on
a draw.

Two numbers per keyword, a title word, a tag phrase, a seniority or a place: ``N_t``, the
people in the market holding it, and ``θ_t``, the share of them who fit. **A fired search
is one sample of the AND of its keywords**, taken as independent, and both halves are
hierarchical models fitted to those samples:

    log N(q)     = log N₀ + Σ log s_t           s_t = N_t / N₀
    logit θ(q)   = α + Σ ℓ_t + e_q              α = logit π, ℓ_t the keyword's lift

**The sizes are a log-linear count model.** The provider's count is a Poisson draw around
``N(q)`` with a log-normal error of its own (``COUNT_NOISE``, what independence gets
wrong), read on ``log(y + ½)``. A keyword's log share is

    log s_t = μ_field + γ · log(store share of t) + u_t      u_t ~ Normal(0, τ²)

The store is not the market, so its share is evidence whose weight ``γ`` the counts
learn, not the prior's centre. ``log N₀`` is anchored broadly at ``MARKET_PEOPLE``: no
search counts the market itself. Everything is Normal, so the posterior is exact.

**The fit rate is a logistic model of what searches brought.** Each fired search's people
add ``evidence``, a binomial sample of its ``θ(q)``. ``α`` is learned from the searched
people only: the store's rate, the customer's marks above all, is its prior and never its
evidence. A keyword's lift is centred on its prior, the GP's score of the keyword's own
embedding moved by the store's other people holding it, and ``e_q`` is a fired search's
own effect, so a page that came back bad says so about the search that brought it. The
posterior is the Laplace approximation at the mode, found by Newton's method.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from openoutfind.core.conf import WALK_CONFIG
from openoutfind.core.pipeline.terms import Terms, split_place

logger = logging.getLogger(__name__)
# What an unjudged person's fit rate weighs against a verdict.
SCORE_WEIGHT = 0.25
# The random streams of one seed, so its draws stay apart.
LIFT_STREAM, SIZE_STREAM, RATE_STREAM = 0, 1, 2
# A rate is kept off 0 and 1, where its odds would end the sum.
MIN_RATE = 0.001

# ── the sizes' priors ──
# Where log N₀ is anchored, and how loosely.
MARKET_PEOPLE = 10_000_000
MARKET_SPREAD = 3.0
# Each field's mean log share, before any count.
FIELD_SPREAD = 3.0
# γ, how far the store's share of a keyword says its market share: taken as said, at first.
STORE_SHARE_MEAN, STORE_SHARE_SPREAD = 1.0, 1.0
# τ, a keyword's log share off its field's line: ×4.5, the median miss of the product of
# true keyword counts measured on 37 queries.
KEYWORD_SPREAD = 1.5
# ε, the AND's own error on a log count.
COUNT_NOISE = 0.5

# ── the fit rate's priors ──
# α, the searched market's log odds, about the store's.
MARKET_RATE_SPREAD = 1.0
# ℓ_t, a keyword's lift about its prior.
LIFT_SPREAD = 1.0
# e_q, a fired search's own effect.
SEARCH_SPREAD = 0.5
NEWTON_STEPS, NEWTON_TOLERANCE = 50, 1e-8

FIELDS = ("title", "tag", "seniority", "place")

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


def keyword_field(keyword: Keyword) -> str:
    """The field a keyword's size is pooled in: a state and a country are both a place."""
    return "place" if keyword[0] in ("state", "country") else keyword[0]


def evidence(pks, people, judged: set[int]) -> tuple[float, float]:
    """``(fits, no-fits)`` that ``pks`` add to a keyword, each company once: the mean of its
    verdicts at full weight, or with none judged the mean of its fit rates at
    ``SCORE_WEIGHT`` times the GP's mean confidence at them."""
    by_company: dict = defaultdict(list)
    for pk in pks:
        if pk in people.rates and pk in people.company:
            by_company[people.company[pk]].append(pk)
    fits = misses = 0.0
    for members in by_company.values():
        verdicts = [people.rates[pk] for pk in members if pk in judged]
        if verdicts:
            rate, weight = statistics.fmean(verdicts), 1.0
        else:
            rate = statistics.fmean(people.rates[pk] for pk in members)
            weight = SCORE_WEIGHT * statistics.fmean(people.confidence_in(pk) for pk in members)
        fits += weight * rate
        misses += weight * (1.0 - rate)
    return fits, misses


# ── the prior ────────────────────────────────────────────────────────

# A keyword's vector under its text, for this process; the store keeps them across
# processes (``core.KeywordVector``).
KEYWORD_VECTORS: dict[str, np.ndarray] = {}


def keyword_vectors(texts: list[str]) -> np.ndarray:
    """Each text's vector: from this process, else the store, else embedded and kept."""
    missing = [text for text in dict.fromkeys(texts) if text not in KEYWORD_VECTORS]
    if missing:
        KEYWORD_VECTORS.update(load_or_embed_vectors(missing))
    return np.array([KEYWORD_VECTORS[text] for text in texts], dtype=np.float64)


def load_or_embed_vectors(texts: list[str]) -> dict[str, np.ndarray]:
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
                   store_rate: float) -> dict[Keyword, float]:
    """Each keyword's prior rate: the GP's score of its embedding through ``fit_rate``,
    or ``store_rate`` where the GP cannot score."""
    if qualifier is None or not keywords:
        return dict.fromkeys(keywords, store_rate)
    probs = qualifier.predict_probs(keyword_vectors([keyword_text(k) for k in keywords]))
    if probs is None:
        return dict.fromkeys(keywords, store_rate)
    return {keyword: float(rate) for keyword, rate in zip(keywords, fit_rate(probs))}


def prior_moved_by_store(score: float, fits: float, misses: float) -> float:
    """A keyword's prior rate: its GP ``score`` at ``prior_companies`` companies, moved by
    the ``fits`` and ``misses`` of the store's people holding it."""
    strength = WALK_CONFIG["prior_companies"]
    return (strength * clip_rate(score) + fits) / (strength + fits + misses)


# ── the market's size ────────────────────────────────────────────────

def store_log_share(keyword: Keyword, people) -> float:
    """``keyword``'s log share of the store's people."""
    return math.log((len(people.holding(keyword)) + 1) / (len(people.everyone) + 2))


# The fixed effects' columns: log N₀, each field's mean, γ. The keywords' own follow.
MARKET_COLUMN, GAMMA_COLUMN = 0, 1 + len(FIELDS)
FIXED_COLUMNS = GAMMA_COLUMN + 1


def size_prior() -> tuple[np.ndarray, np.ndarray]:
    """The fixed effects' prior means and precisions."""
    mean = np.zeros(FIXED_COLUMNS)
    precision = np.full(FIXED_COLUMNS, FIELD_SPREAD ** -2)
    mean[MARKET_COLUMN], precision[MARKET_COLUMN] = math.log(MARKET_PEOPLE), MARKET_SPREAD ** -2
    mean[GAMMA_COLUMN], precision[GAMMA_COLUMN] = STORE_SHARE_MEAN, STORE_SHARE_SPREAD ** -2
    return mean, precision


@dataclass(frozen=True)
class SizeFit:
    """The Normal posterior over the sizes: the fixed effects and the deviation of every
    keyword a count held."""

    keywords: tuple = ()
    solution: np.ndarray = field(default_factory=lambda: size_prior()[0])
    root: np.ndarray = field(default_factory=lambda: np.diag(size_prior()[1] ** -0.5))

    def __post_init__(self):
        object.__setattr__(self, "column",
                           {k: FIXED_COLUMNS + i for i, k in enumerate(self.keywords)})

    def mean(self) -> "Sizes":
        return Sizes(self, self.solution, {})

    def draw(self, rng, unfitted=()) -> "Sizes":
        """One joint draw of the sizes; each of ``unfitted`` (keywords no count held) draws
        its deviation at ``KEYWORD_SPREAD``."""
        params = self.solution + self.root @ rng.standard_normal(len(self.solution))
        unfitted = sorted(k for k in unfitted if k not in self.column)
        deviations = KEYWORD_SPREAD * rng.standard_normal(len(unfitted))
        return Sizes(self, params, dict(zip(unfitted, deviations.tolist())))


@dataclass(frozen=True)
class Sizes:
    """One reading of the sizes, the fit's mean or one draw of it."""

    fit: SizeFit
    params: np.ndarray
    deviations: dict

    @property
    def log_market(self) -> float:
        return float(self.params[MARKET_COLUMN])

    def share(self, keyword: Keyword, people) -> float:
        """``log(N_t / N₀)``, never above 0: the field's mean, γ times the store's share,
        the deviation."""
        store = store_log_share(keyword, people)
        column = self.fit.column.get(keyword)
        deviation = (self.params[column] if column is not None
                     else self.deviations.get(keyword, 0.0))
        share = (self.params[1 + FIELDS.index(keyword_field(keyword))]
                 + self.params[GAMMA_COLUMN] * store + deviation)
        return min(float(share), 0.0)


def fit_sizes(fired, people) -> SizeFit:
    """The sizes' posterior, fitted to the fired searches' counts. With none, the prior."""
    counted = [f for f in fired if f.found is not None]
    if not counted:
        return SizeFit()
    keywords = tuple(sorted({keyword for f in counted for keyword in f.keywords}))
    shares = {keyword: store_log_share(keyword, people) for keyword in keywords}

    column = {keyword: FIXED_COLUMNS + i for i, keyword in enumerate(keywords)}
    rows = np.zeros((len(counted), FIXED_COLUMNS + len(keywords)))
    target, weight = np.zeros(len(counted)), np.zeros(len(counted))
    for i, f in enumerate(counted):
        rows[i, MARKET_COLUMN] = 1.0
        for keyword in f.keywords:
            rows[i, 1 + FIELDS.index(keyword_field(keyword))] += 1.0
            rows[i, GAMMA_COLUMN] += shares[keyword]
            rows[i, column[keyword]] = 1.0
        target[i] = math.log(f.found + 0.5)
        weight[i] = 1.0 / (COUNT_NOISE ** 2 + 1.0 / (f.found + 0.5))

    fixed_mean, fixed_precision = size_prior()
    prior_mean = np.concatenate([fixed_mean, np.zeros(len(keywords))])
    precision = np.concatenate([fixed_precision, np.full(len(keywords), KEYWORD_SPREAD ** -2)])
    information = rows.T @ (weight[:, None] * rows) + np.diag(precision)
    solution = np.linalg.solve(information, rows.T @ (weight * target) + precision * prior_mean)
    root = np.linalg.cholesky(np.linalg.inv(information))
    return SizeFit(keywords, solution, root)


# ── the fit rate ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RateFit:
    """The Laplace posterior over the fit rate: ``α``, the lift of every keyword a fired
    search held, and every fired search's own effect."""

    keywords: tuple
    searches: tuple
    solution: np.ndarray
    root: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "column", {k: 1 + i for i, k in enumerate(self.keywords)})
        start = 1 + len(self.keywords)
        object.__setattr__(self, "own", {key: start + i for i, key in enumerate(self.searches)})

    def draw(self, rng) -> np.ndarray:
        return self.solution + self.root @ rng.standard_normal(len(self.solution))


def fit_rates(store_rate: float, lift_priors: dict[Keyword, float],
              samples: list[tuple[str, tuple, float, float]]) -> RateFit:
    """The fit rate's posterior mode and its Laplace covariance.

    ``samples`` are ``(search key, keywords, fits, no-fits)``, one per fired search."""
    keywords = tuple(sorted({keyword for _, held, _, _ in samples for keyword in held}))
    column = {keyword: 1 + i for i, keyword in enumerate(keywords)}
    size = 1 + len(keywords) + len(samples)
    rows = np.zeros((len(samples), size))
    fits, trials = np.zeros(len(samples)), np.zeros(len(samples))
    for i, (_, held, hits, misses) in enumerate(samples):
        rows[i, 0] = 1.0
        for keyword in held:
            rows[i, column[keyword]] = 1.0
        rows[i, 1 + len(keywords) + i] = 1.0
        fits[i], trials[i] = hits, hits + misses

    prior_mean = np.zeros(size)
    prior_mean[0] = log_odds(clip_rate(store_rate))
    for keyword, i in column.items():
        prior_mean[i] = lift_priors[keyword]
    precision = np.concatenate([[MARKET_RATE_SPREAD ** -2],
                                np.full(len(keywords), LIFT_SPREAD ** -2),
                                np.full(len(samples), SEARCH_SPREAD ** -2)])

    solution = prior_mean.copy()
    for _ in range(NEWTON_STEPS):
        rate = sigmoid(rows @ solution)
        gradient = rows.T @ (fits - trials * rate) - precision * (solution - prior_mean)
        information = rows.T @ ((trials * rate * (1.0 - rate))[:, None] * rows) + np.diag(precision)
        step = np.linalg.solve(information, gradient)
        solution += step
        if np.max(np.abs(step)) < NEWTON_TOLERANCE:
            break
    rate = sigmoid(rows @ solution)
    information = rows.T @ ((trials * rate * (1.0 - rate))[:, None] * rows) + np.diag(precision)
    root = np.linalg.cholesky(np.linalg.inv(information))
    return RateFit(keywords, tuple(key for key, *_ in samples), solution, root)


# ── the belief ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Draw:
    """One joint draw of the belief: ``α``, a lift per keyword asked for, every fired
    search's own ``θ``, and the sizes."""

    alpha: float
    lifts: dict
    searches: dict
    sizes: Sizes


class Beliefs:
    """What the store says about every keyword, fitted once per frontier."""

    def __init__(self, people, fired, judged: set[int], store_rate: float):
        self.people = people
        self.judged = judged
        self.fired = list(fired)
        self.store_rate = clip_rate(store_rate)
        self.sizes = fit_sizes(fired, people)
        self.brought: dict[Keyword, set[int]] = defaultdict(set)
        for f in fired:
            for keyword in f.keywords:
                self.brought[keyword] |= f.people
        self.search_keywords = {f.search.key: f.keywords for f in fired}
        self.prior: dict[Keyword, float] = {}
        self.rates: RateFit | None = None

    def settle(self, priors: dict[Keyword, float]) -> None:
        """Fit the rate: each keyword's prior, ``priors`` moved by the store's people
        holding it that no search brought, and every fired search's people as a sample of
        its keywords together."""
        for keyword, score in priors.items():
            brought = self.brought.get(keyword, set())
            self.prior[keyword] = prior_moved_by_store(score, *evidence(
                self.people.holding(keyword) - brought, self.people, self.judged))
        for f in self.fired:
            for keyword in f.keywords:
                self.prior.setdefault(keyword, self.store_rate)
        samples = [(f.search.key, f.keywords, *evidence(f.people, self.people, self.judged))
                   for f in self.fired]
        self.rates = fit_rates(self.store_rate, self.prior_lifts(), samples)

    # ── at the mean ──

    def count(self, keywords, sizes: Sizes | None = None) -> float:
        """``N(q)``: the people the market holds with every one of ``keywords``, on the
        fit's mean or on ``sizes``."""
        sizes = sizes or self.sizes.mean()
        return math.exp(sizes.log_market + sum(self.shares(keywords, sizes).values()))

    def shares(self, keywords, sizes: Sizes | None = None) -> dict[Keyword, float]:
        """Each keyword's ``log(N_t / N₀)`` on the fit's mean or on ``sizes``."""
        sizes = sizes or self.sizes.mean()
        return {keyword: sizes.share(keyword, self.people) for keyword in keywords}

    @property
    def market_rate(self) -> float:
        """``π``, the share of the searched market who fit; the store's before ``settle``."""
        if self.rates is None:
            return self.store_rate
        return float(sigmoid(self.rates.solution[0]))

    def prior_lifts(self) -> dict[Keyword, float]:
        """Each keyword's prior lift, against the store's rate its prior was read on."""
        store = log_odds(self.store_rate)
        return {keyword: log_odds(clip_rate(rate)) - store for keyword, rate in self.prior.items()}

    def lifts(self) -> dict[Keyword, float]:
        """Each keyword's lift at the posterior mean: ``logit θ_t − logit π``."""
        lifts = self.prior_lifts()
        for keyword, column in self.rates.column.items():
            lifts[keyword] = float(self.rates.solution[column])
        return lifts

    def means(self) -> dict[Keyword, float]:
        """Each keyword's expected ``θ_t``."""
        alpha = self.rates.solution[0]
        return {keyword: sigmoid(alpha + lift) for keyword, lift in self.lifts().items()}

    def search_mean(self, key: str) -> float:
        """A fired search's expected share of fits, its own effect included."""
        return sigmoid(self.search_log_odds(key, self.rates.solution))

    def rate(self, keywords, lifts: dict[Keyword, float], alpha: float | None = None) -> float:
        """``θ(q)`` for a query of ``keywords``, given a lift for each and ``α`` (the mean's
        where not given)."""
        if alpha is None:
            alpha = log_odds(self.market_rate)
        return sigmoid(alpha + sum(lifts[keyword] for keyword in keywords))

    def theta_at_mean(self, keywords, lifts: dict[Keyword, float],
                      search_key: str | None = None) -> float:
        """``θ(q)`` at the mean: a fired search's own where ``search_key`` names one, else
        its keywords' on ``lifts`` (``lifts()``, read once for many queries)."""
        if search_key is not None:
            return self.search_mean(search_key)
        return self.rate(keywords, lifts)

    # ── on a draw ──

    def draw(self, seed, keywords) -> Draw:
        """One joint draw, seeded by ``seed`` so a rerun repeats. The fitted lifts are
        drawn jointly; each of ``keywords`` no search held draws its lift about its prior
        and its size off its field's line."""
        params = self.rates.draw(seeded_rng(seed, RATE_STREAM))
        priors = self.prior_lifts()
        unfitted = sorted(k for k in keywords if k not in self.rates.column)
        noise = seeded_rng(seed, LIFT_STREAM).standard_normal(len(unfitted))
        lifts = {k: priors[k] + LIFT_SPREAD * z for k, z in zip(unfitted, noise)}
        for keyword in keywords:
            if keyword in self.rates.column:
                lifts[keyword] = float(params[self.rates.column[keyword]])
        searches = {key: sigmoid(self.search_log_odds(key, params)) for key in self.rates.searches}
        return Draw(float(params[0]), lifts, searches, self.draw_sizes(seed, keywords))

    def draw_sizes(self, seed, keywords=()) -> Sizes:
        """The sizes of the draw ``seed`` names."""
        return self.sizes.draw(seeded_rng(seed, SIZE_STREAM), keywords)

    def theta_on(self, drawn: Draw, keywords, search_key: str | None = None) -> float:
        """``θ(q)`` on ``drawn``: a fired search's own where ``search_key`` names one, else
        its keywords'."""
        if search_key is not None:
            return drawn.searches[search_key]
        return self.rate(keywords, drawn.lifts, drawn.alpha)

    def search_log_odds(self, key: str, params: np.ndarray) -> float:
        held = self.search_keywords[key]
        return float(params[0] + sum(params[self.rates.column[k]] for k in held)
                     + params[self.rates.own[key]])


def seeded_rng(seed, stream: int):
    """A generator for one stream of ``seed`` — a round's number, or a round and a draw."""
    return np.random.default_rng([*np.atleast_1d(seed).tolist(), stream])


def clip_rate(rate: float) -> float:
    return min(max(rate, MIN_RATE), 1.0 - MIN_RATE)


def log_odds(rate: float) -> float:
    return math.log(rate / (1.0 - rate))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
