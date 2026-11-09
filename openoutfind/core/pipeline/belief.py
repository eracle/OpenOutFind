# openoutfind/core/pipeline/belief.py
"""The walk's belief about the market — how many people hold each keyword, and how many
of them the customer would like — and every read of it the walk makes, at the mean or on
a draw.

Two numbers per keyword, a title word, a whole tag, a seniority or a place: ``N_t``, the
people in the market holding it, and ``θ_t``, the share of them who fit. **A fired search
is one sample of the AND of its keywords**, taken as independent, and both halves are one
hierarchical model, sampled whole by NUTS (PyMC, compiled by nutpie):

    log N(q)   = log N₀ + Σ_{t∈q} ( μ_field(t) + γ · x_t + u_t )     x_t = log store share of t
    logit θ(q) = α + Σ_{t∈q} v_t + e_q
    (u_t, v_t) ~ MVNormal(0, diag(τ, s_ℓ) · R(ρ) · diag(τ, s_ℓ))    one pair per keyword
    e_q        ~ Normal(0, s_e²)

- **The count** is the provider's, a negative binomial draw around ``N(q)`` whose
  dispersion ``φ`` is learned. The store is not the market, so its share of a keyword is
  evidence whose weight ``γ`` the counts learn. ``log N₀`` is anchored broadly at
  ``MARKET_PEOPLE``: no search counts the market itself.
- **The fit rate** is what searches brought: each fired search's people add
  ``evidence``, a binomial sample of its ``θ(q)``. ``α = logit π`` is learned from the
  searched people only, centred on the store's rate. ``e_q`` is a fired search's own
  effect, so a page that came back bad says so about the search that brought it.
- **A keyword's lift is centred on ``p_t``**, an LLM's prior lift for the keyword read
  against the brief (``OPENOUTFIND_KEYWORD_PRIOR``, every keyword in one output so they share
  a scale), in logits against a neutral keyword of its field. Unscored, it is 0.
- **The spreads are learned** — ``τ``, ``s_ℓ``, ``s_e``, ``φ`` and the correlation ``ρ``
  between a keyword's size and its lift — and a keyword no fired search holds draws its pair
  from the draw's own ``MVNormal(0, Σ)``.

Sampling has no mode to reach: the Newton fit this replaced fell into a 2-cycle on batch 6
and stopped at ``α`` = +17.86, every arm reading ``θ`` = 1. **The posterior is refitted only
when a page or a verdict lands** (``fitted``), about 3.5 s a refit.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from openoutfind.core.pipeline.terms import Terms, split_place

logger = logging.getLogger(__name__)
# What an unjudged person's fit rate weighs against a verdict.
SCORE_WEIGHT = 0.25
# The random streams of one seed, so its draws stay apart.
INDEX_STREAM, PAIR_STREAM = 0, 1
# A rate is kept off 0 and 1, where its odds would end the sum.
MIN_RATE = 0.001

# ── the priors ──
# Where log N₀ is anchored, and how loosely.
MARKET_PEOPLE = 10_000_000
MARKET_SPREAD = 3.0
# Each field's mean log share, before any count.
FIELD_SPREAD = 3.0
# γ, how far the store's share of a keyword says its market share: taken as said, at first.
STORE_SHARE_MEAN, STORE_SHARE_SPREAD = 1.0, 1.0
# α, the searched market's log odds, about the store's.
MARKET_RATE_SPREAD = 1.0
# The scales of the learned spreads (each a half-normal): τ, a keyword's log share off its
# field's line (×4.5 measured on 37 queries); s_ℓ, its lift; s_e, a search's own effect.
KEYWORD_SPREAD, LIFT_SPREAD, SEARCH_SPREAD = 1.5, 1.0, 0.5
# φ, the count's dispersion about N(q), what independence gets wrong. A negative binomial
# beat the log-normal on log(y + ½) inside this model on batch 6's counts: 16 of 20 inside
# the 80% interval against 14.
COUNT_DISPERSION = 5.0
# R's LKJ concentration: 2 leans away from a perfect correlation.
CORRELATION_CONCENTRATION = 2.0
# A JSON file of {"field|value": prior lift in logits}, every keyword of the vocabulary in one
# output against the brief.
KEYWORD_PRIOR_ENV = "OPENOUTFIND_KEYWORD_PRIOR"

# ── the sampler ──
CHAINS, TUNE, DRAWS = 4, 500, 500
# Smaller steps than the default 0.8, as a hierarchical model with few groups wants: its
# learned spreads make a funnel while few searches are fired.
TARGET_ACCEPT = 0.99
SAMPLER_SEED = 1

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


def store_log_share(keyword: Keyword, people) -> float:
    """``keyword``'s log share of the store's people."""
    return math.log((len(people.holding(keyword)) + 1) / (len(people.everyone) + 2))


# ── the LLM's prior on the lifts ──

def keyword_text(keyword: Keyword) -> str:
    """``keyword`` as the scores file spells it: ``tag|cannabis extraction``,
    ``state|new york|united states``."""
    return "|".join(" ".join(part) if isinstance(part, tuple) else str(part)
                    for part in keyword)


@functools.cache
def keyword_priors(path: str | None) -> dict[str, float]:
    """``p_t`` per keyword, as the file gives it."""
    if not path:
        return {}
    return {k: float(v) for k, v in json.loads(Path(path).read_text()).items()}


def centre(keyword: Keyword) -> float:
    """``p_t`` for ``keyword``, 0 where it was not scored."""
    return keyword_priors(os.environ.get(KEYWORD_PRIOR_ENV)).get(keyword_text(keyword), 0.0)


# ── the model's inputs ───────────────────────────────────────────────

@dataclass(frozen=True)
class Inputs:
    """The fired searches as the model reads them: which keywords each holds, its count and
    its evidence."""

    keywords: tuple
    searches: tuple
    member: np.ndarray        # (searches, keywords): 1 where a search holds a keyword
    fields: np.ndarray        # (searches, FIELDS): its keywords per field
    store_share: np.ndarray   # (keywords,): x_t
    centre: np.ndarray        # (keywords,): p_t, the LLM's prior lift
    counted: np.ndarray       # the searches with a provider count
    counts: np.ndarray        # their counts
    fits: np.ndarray          # (searches,), fractional
    misses: np.ndarray
    market_log_odds: float    # α's centre, the store's rate


def model_inputs(fired, people, judged: set[int], store_rate: float) -> Inputs:
    """The fired searches' keywords, counts and evidence."""
    keywords = tuple(sorted({keyword for f in fired for keyword in f.keywords}))
    column = {keyword: j for j, keyword in enumerate(keywords)}
    member = np.zeros((len(fired), len(keywords)))
    fields = np.zeros((len(fired), len(FIELDS)))
    fits, misses = np.zeros(len(fired)), np.zeros(len(fired))
    for i, f in enumerate(fired):
        for keyword in f.keywords:
            member[i, column[keyword]] = 1.0
            fields[i, FIELDS.index(keyword_field(keyword))] += 1.0
        fits[i], misses[i] = evidence(f.people, people, judged)
    counted = np.array([i for i, f in enumerate(fired) if f.found is not None], dtype=np.int64)
    return Inputs(
        keywords=keywords, searches=tuple(f.search.key for f in fired), member=member,
        fields=fields,
        store_share=np.array([store_log_share(keyword, people) for keyword in keywords]),
        centre=np.array([centre(keyword) for keyword in keywords]),
        counted=counted, counts=np.array([fired[i].found for i in counted], dtype=np.float64),
        fits=fits, misses=misses, market_log_odds=log_odds(clip_rate(store_rate)))


# ── the posterior ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Posterior:
    """The joint posterior as draws, one row each: the globals, every fitted keyword's pair
    and every fired search's own effect."""

    keywords: tuple
    searches: tuple
    log_market: np.ndarray    # (D,)
    field_mean: np.ndarray    # (D, FIELDS)
    gamma: np.ndarray         # (D,)
    alpha: np.ndarray         # (D,)
    chol: np.ndarray          # (D, 2, 2): Σ's Cholesky factor, size then lift
    size: np.ndarray          # (D, keywords): u_t
    lift: np.ndarray          # (D, keywords): v_t, p_t included
    own: np.ndarray           # (D, searches): e_q
    divergences: int = 0
    seconds: float = 0.0
    column: dict = field(init=False)
    search_column: dict = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "column", {k: j for j, k in enumerate(self.keywords)})
        object.__setattr__(self, "search_column", {k: i for i, k in enumerate(self.searches)})

    @property
    def draws(self) -> int:
        return len(self.alpha)

    def spreads(self) -> dict[str, float]:
        """The learned spreads and correlation at their posterior means."""
        tau = self.chol[:, 0, 0]
        s_lift = np.hypot(self.chol[:, 1, 0], self.chol[:, 1, 1])
        return {"tau": float(tau.mean()), "s_lift": float(s_lift.mean()),
                "rho": float((self.chol[:, 1, 0] / s_lift).mean())}


def sample_posterior(inputs: Inputs) -> Posterior:
    """The joint posterior of ``inputs``, sampled by nutpie; with no fired search, the prior."""
    started = time.perf_counter()
    if not inputs.searches:
        return prior_draws(inputs)
    import nutpie

    searches, keywords = len(inputs.searches), len(inputs.keywords)
    slots = capacity(searches, MIN_SEARCHES), capacity(keywords, MIN_KEYWORDS)
    compiled = compiled_model(*slots).with_data(**padded_data(inputs, *slots))
    trace = nutpie.sample(compiled, draws=DRAWS,
                          tune=TUNE, chains=CHAINS, seed=SAMPLER_SEED,
                          target_accept=TARGET_ACCEPT, progress_bar=False)
    posterior = trace.posterior

    def flat(name):
        values = posterior[name].values
        return values.reshape(-1, *values.shape[2:])

    pair = flat("pair")
    return Posterior(
        keywords=inputs.keywords, searches=inputs.searches, log_market=flat("log_market"),
        field_mean=flat("field_mean"), gamma=flat("gamma"), alpha=flat("alpha"),
        chol=flat("chol_matrix"), size=pair[:, 0, :keywords],
        lift=pair[:, 1, :keywords] + inputs.centre[None, :],
        own=flat("own")[:, :searches],
        divergences=int(trace.sample_stats["diverging"].values.sum()),
        seconds=time.perf_counter() - started)


# ── the compiled model ──
# Compiling takes 2.5-5 s and sampling 1.3, so the model is compiled once per size and the
# data swapped in: the searches and keywords are padded to a power of two, at least these.
# A padded keyword is in no search and a padded search holds no keyword, count or person,
# so neither moves the posterior.
MIN_SEARCHES, MIN_KEYWORDS = 8, 16
COMPILED: dict[tuple[int, int], object] = {}


def capacity(needed: int, least: int) -> int:
    """The power of two at least ``least`` that holds ``needed``."""
    return max(least, 1 << max(needed - 1, 0).bit_length())


def compiled_model(searches: int, keywords: int):
    """The model compiled for ``searches`` and ``keywords`` slots, once per process."""
    import nutpie

    if (searches, keywords) not in COMPILED:
        COMPILED[searches, keywords] = nutpie.compile_pymc_model(joint_model(searches, keywords))
    return COMPILED[searches, keywords]


def padded_data(inputs: Inputs, searches: int, keywords: int) -> dict[str, np.ndarray]:
    """``inputs`` padded to ``searches`` and ``keywords`` slots."""
    rows, columns = inputs.member.shape
    member = np.zeros((searches, keywords))
    member[:rows, :columns] = inputs.member
    fields = np.zeros((searches, len(FIELDS)))
    fields[:rows] = inputs.fields
    store_share, centres = np.zeros(keywords), np.zeros(keywords)
    store_share[:columns], centres[:columns] = inputs.store_share, inputs.centre
    counts, counted = np.zeros(searches), np.zeros(searches)
    counts[inputs.counted], counted[inputs.counted] = inputs.counts, 1.0
    fits, misses = np.zeros(searches), np.zeros(searches)
    fits[:rows], misses[:rows] = inputs.fits, inputs.misses
    return {"member": member, "fields": fields, "store_share": store_share,
            "centre": centres, "counts": counts,
            "counted": counted, "fits": fits, "misses": misses,
            "market_log_odds": np.array(inputs.market_log_odds)}


def joint_model(searches: int, keywords: int):
    """The sizes and the fit rate as one PyMC model, non-centred, over ``searches`` and
    ``keywords`` slots whose data ``padded_data`` gives."""
    import pymc as pm
    import pytensor.tensor as pt

    with pm.Model() as model:
        member = pm.Data("member", np.zeros((searches, keywords)))
        fields = pm.Data("fields", np.zeros((searches, len(FIELDS))))
        store_share = pm.Data("store_share", np.zeros(keywords))
        centres = pm.Data("centre", np.zeros(keywords))
        counts = pm.Data("counts", np.zeros(searches))
        counted = pm.Data("counted", np.zeros(searches))
        fits = pm.Data("fits", np.zeros(searches))
        misses = pm.Data("misses", np.zeros(searches))
        market_log_odds = pm.Data("market_log_odds", np.array(0.0))

        log_market = pm.Normal("log_market", math.log(MARKET_PEOPLE), MARKET_SPREAD)
        field_mean = pm.Normal("field_mean", 0.0, FIELD_SPREAD, shape=len(FIELDS))
        gamma = pm.Normal("gamma", STORE_SHARE_MEAN, STORE_SHARE_SPREAD)
        alpha = pm.Normal("alpha", market_log_odds, MARKET_RATE_SPREAD)
        chol, _, _ = pm.LKJCholeskyCov(
            "chol", n=2, eta=CORRELATION_CONCENTRATION, compute_corr=True,
            sd_dist=pm.HalfNormal.dist(sigma=np.array([KEYWORD_SPREAD, LIFT_SPREAD]), shape=2))
        pm.Deterministic("chol_matrix", chol)
        raw_pair = pm.Normal("raw_pair", 0.0, 1.0, shape=(2, keywords))
        pair = pm.Deterministic("pair", chol @ raw_pair)
        s_search = pm.HalfNormal("s_search", SEARCH_SPREAD)
        raw_own = pm.Normal("raw_own", 0.0, 1.0, shape=searches)
        own = pm.Deterministic("own", s_search * raw_own)
        dispersion = pm.HalfNormal("dispersion", COUNT_DISPERSION)

        log_count = (log_market + pt.dot(fields, field_mean)
                     + pt.dot(member, gamma * store_share + pair[0]))
        count_logp = pm.logp(pm.NegativeBinomial.dist(mu=pt.exp(log_count), alpha=dispersion),
                             counts)
        pm.Potential("count", pt.dot(counted, count_logp))
        log_odds_q = alpha + pt.dot(member, centres + pair[1]) + own
        # log σ(x) = −softplus(−x), stable at either end.
        pm.Potential("fit", -pt.dot(fits, pt.softplus(-log_odds_q))
                     - pt.dot(misses, pt.softplus(log_odds_q)))
    return model


def prior_draws(inputs: Inputs) -> Posterior:
    """The globals' prior as draws: what a store with no fired search believes."""
    rng = np.random.default_rng(SAMPLER_SEED)
    count = CHAINS * DRAWS
    sds = np.abs(rng.normal(0.0, [KEYWORD_SPREAD, LIFT_SPREAD], size=(count, 2)))
    # LKJ(η) on a 2×2 correlation: (ρ + 1) / 2 ~ Beta(η, η).
    rho = 2.0 * rng.beta(CORRELATION_CONCENTRATION, CORRELATION_CONCENTRATION, count) - 1.0
    chol = np.zeros((count, 2, 2))
    chol[:, 0, 0] = sds[:, 0]
    chol[:, 1, 0] = sds[:, 1] * rho
    chol[:, 1, 1] = sds[:, 1] * np.sqrt(1.0 - rho ** 2)
    return Posterior(
        keywords=(), searches=(),
        log_market=rng.normal(math.log(MARKET_PEOPLE), MARKET_SPREAD, count),
        field_mean=rng.normal(0.0, FIELD_SPREAD, (count, len(FIELDS))),
        gamma=rng.normal(STORE_SHARE_MEAN, STORE_SHARE_SPREAD, count),
        alpha=rng.normal(inputs.market_log_odds, MARKET_RATE_SPREAD, count),
        chol=chol, size=np.zeros((count, 0)), lift=np.zeros((count, 0)),
        own=np.zeros((count, 0)))


# The posterior, under the key of the pages and verdicts it was fitted on.
FITTED: tuple[str, Posterior] | None = None


def evidence_key(fired, judged: set[int]) -> str:
    """What a refit waits on: the fired pages and the judged people. Not the evidence
    itself — an unjudged person's share of it moves with every GP refit, and a verdict,
    which already changes this key, is what refits the GP."""
    pages = sorted((f.search.key, f.latest.pk, f.latest.number, f.found) for f in fired)
    return hashlib.sha256(repr((pages, sorted(judged))).encode()).hexdigest()


def fitted(fired, people, judged: set[int], store_rate: float) -> Posterior:
    """The posterior, refitted only when a page or a verdict has landed since the last fit."""
    global FITTED
    key = evidence_key(fired, judged)
    if FITTED is not None and FITTED[0] == key:
        return FITTED[1]
    posterior = sample_posterior(model_inputs(fired, people, judged, store_rate))
    FITTED = (key, posterior)
    log_refit(posterior)
    return posterior


def log_refit(posterior: Posterior) -> None:
    if not posterior.searches:
        return
    spreads = posterior.spreads()
    logger.debug("  belief: %d search(es), %d keyword(s) · π %.3f · γ %.2f · "
                 "τ %.2f · s_ℓ %.2f · ρ %+.2f · %d divergence(s) · %.1fs",
                 len(posterior.searches), len(posterior.keywords),
                 float(sigmoid(posterior.alpha).mean()), float(posterior.gamma.mean()),
                 spreads["tau"], spreads["s_lift"], spreads["rho"], posterior.divergences,
                 posterior.seconds)


# ── one reading: the mean, or a draw ─────────────────────────────────

class Pairs:
    """Each keyword's ``(u_t, v_t)`` on one reading of the posterior: a fitted keyword's own;
    for one no search held, 0 at the mean, or on a draw a pair from that draw's
    ``MVNormal(0, Σ)``, seeded by the draw's seed and the keyword so every read agrees."""

    def __init__(self, posterior: Posterior, index: int | None = None, seed=None):
        self.posterior, self.index, self.seed = posterior, index, seed
        self.known: dict[Keyword, tuple[float, float]] = {}

    def __call__(self, keyword: Keyword) -> tuple[float, float]:
        if keyword not in self.known:
            self.known[keyword] = self.read(keyword)
        return self.known[keyword]

    def read(self, keyword: Keyword) -> tuple[float, float]:
        posterior = self.posterior
        column = posterior.column.get(keyword)
        if self.index is None:
            if column is None:
                return 0.0, centre(keyword)
            return float(posterior.size[:, column].mean()), float(posterior.lift[:, column].mean())
        if column is not None:
            return (float(posterior.size[self.index, column]),
                    float(posterior.lift[self.index, column]))
        noise = np.random.default_rng(
            [*np.atleast_1d(self.seed).tolist(), PAIR_STREAM, keyword_number(keyword)]
        ).standard_normal(2)
        size, lift = posterior.chol[self.index] @ noise
        return float(size), float(lift + centre(keyword))


@dataclass(frozen=True)
class Sizes:
    """One reading of the sizes, the posterior's mean or one draw of it."""

    log_market: float
    field_mean: np.ndarray
    gamma: float
    pairs: Pairs

    def share(self, keyword: Keyword, people) -> float:
        """``log(N_t / N₀)``, never above 0: the field's mean, γ times the store's share,
        the keyword's own deviation."""
        share = (self.field_mean[FIELDS.index(keyword_field(keyword))]
                 + self.gamma * store_log_share(keyword, people) + self.pairs(keyword)[0])
        return min(float(share), 0.0)


@dataclass(frozen=True)
class Draw:
    """One draw of the belief: ``α``, a lift per keyword asked for, every fired search's own
    ``θ``, and the sizes — both halves from the same posterior draw, so ``F`` is joint."""

    alpha: float
    lifts: dict
    searches: dict
    sizes: Sizes


class Beliefs:
    """What the store says about every keyword: the posterior, and every read of it."""

    def __init__(self, people, fired, judged: set[int], store_rate: float):
        self.people = people
        self.store_rate = clip_rate(store_rate)
        self.search_keywords = {f.search.key: f.keywords for f in fired}
        self.posterior = fitted(list(fired), people, judged, self.store_rate)
        self.mean_pairs = Pairs(self.posterior)

    # ── at the mean ──

    def mean_sizes(self) -> Sizes:
        posterior = self.posterior
        return Sizes(float(posterior.log_market.mean()), posterior.field_mean.mean(axis=0),
                     float(posterior.gamma.mean()), self.mean_pairs)

    def count(self, keywords, sizes: Sizes | None = None) -> float:
        """``N(q)``: the people the market holds with every one of ``keywords``, on the
        posterior's mean or on ``sizes``."""
        sizes = sizes or self.mean_sizes()
        return math.exp(sizes.log_market + sum(self.shares(keywords, sizes).values()))

    def shares(self, keywords, sizes: Sizes | None = None) -> dict[Keyword, float]:
        """Each keyword's ``log(N_t / N₀)`` on the posterior's mean or on ``sizes``."""
        sizes = sizes or self.mean_sizes()
        return {keyword: sizes.share(keyword, self.people) for keyword in keywords}

    @property
    def market_rate(self) -> float:
        """``π``, the share of the searched market who fit."""
        return float(sigmoid(self.posterior.alpha.mean()))

    def lifts(self) -> dict[Keyword, float]:
        """Each keyword's lift at the posterior mean, ``logit θ_t − logit π``: ``p_t`` for a
        keyword no search held."""
        posterior = self.posterior
        return Lifts(zip(posterior.keywords, posterior.lift.mean(axis=0).tolist()))

    def means(self) -> dict[Keyword, float]:
        """Each fitted keyword's expected ``θ_t``."""
        alpha = self.posterior.alpha.mean()
        return {keyword: float(sigmoid(alpha + lift)) for keyword, lift in self.lifts().items()}

    def search_mean(self, key: str) -> float:
        """A fired search's expected share of fits, its own effect included."""
        return float(sigmoid(self.search_log_odds(key).mean()))

    def rate(self, keywords, lifts: dict[Keyword, float], alpha: float | None = None) -> float:
        """``θ(q)`` for a query of ``keywords``, given a lift for each and ``α`` (the mean's
        where not given)."""
        if alpha is None:
            alpha = float(self.posterior.alpha.mean())
        return float(sigmoid(alpha + sum(lifts[keyword] for keyword in keywords)))

    def theta_at_mean(self, keywords, lifts: dict[Keyword, float],
                      search_key: str | None = None) -> float:
        """``θ(q)`` at the mean: a fired search's own where ``search_key`` names one, else
        its keywords' on ``lifts`` (``lifts()``, read once for many queries)."""
        if search_key is not None:
            return self.search_mean(search_key)
        return self.rate(keywords, lifts)

    # ── on a draw ──

    def draw_index(self, seed) -> int:
        """The stored posterior draw ``seed`` names, so a rerun repeats."""
        return int(seeded_rng(seed, INDEX_STREAM).integers(self.posterior.draws))

    def draw(self, seed, keywords) -> Draw:
        """One stored posterior draw, chosen by ``seed``: ``α``, the lift of each of
        ``keywords``, every fired search's own ``θ`` and the sizes."""
        index = self.draw_index(seed)
        sizes = self.draw_sizes(seed)
        lifts = {keyword: sizes.pairs(keyword)[1] for keyword in keywords}
        posterior = self.posterior
        searches = {key: float(sigmoid(self.search_log_odds(key)[index]))
                    for key in posterior.searches}
        return Draw(float(posterior.alpha[index]), lifts, searches, sizes)

    def draw_sizes(self, seed) -> Sizes:
        """The sizes of the draw ``seed`` names."""
        index = self.draw_index(seed)
        posterior = self.posterior
        return Sizes(float(posterior.log_market[index]), posterior.field_mean[index],
                     float(posterior.gamma[index]), Pairs(posterior, index, seed))

    def theta_on(self, drawn: Draw, keywords, search_key: str | None = None) -> float:
        """``θ(q)`` on ``drawn``: a fired search's own where ``search_key`` names one, else
        its keywords'."""
        if search_key is not None:
            return drawn.searches[search_key]
        return self.rate(keywords, drawn.lifts, drawn.alpha)

    def search_log_odds(self, key: str) -> np.ndarray:
        """A fired search's ``logit θ(q)`` on every draw."""
        posterior = self.posterior
        held = [posterior.column[k] for k in self.search_keywords[key]]
        return (posterior.alpha + posterior.lift[:, held].sum(axis=1)
                + posterior.own[:, posterior.search_column[key]])


class Lifts(dict):
    """The fitted keywords' mean lifts; any other keyword reads its prior ``p_t``."""

    def __missing__(self, keyword: Keyword) -> float:
        return centre(keyword)


def keyword_number(keyword: Keyword) -> int:
    """A keyword as a stable number, to seed its draw."""
    return int.from_bytes(hashlib.sha256(repr(keyword).encode()).digest()[:8], "big")


def seeded_rng(seed, stream: int):
    """A generator for one stream of ``seed`` — a round's number, or a round and a draw."""
    return np.random.default_rng([*np.atleast_1d(seed).tolist(), stream])


def clip_rate(rate: float) -> float:
    return min(max(rate, MIN_RATE), 1.0 - MIN_RATE)


def log_odds(rate: float) -> float:
    return math.log(rate / (1.0 - rate))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
