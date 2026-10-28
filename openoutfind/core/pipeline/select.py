# openoutfind/core/pipeline/select.py
"""Query selection — the frontier is scored, drawn from, and fired.

The whole discovery walk, from first principles. A **node** is a set of ``(field, token)``
keywords; firing it pages Lead Finder for the conjunction. Its **children** are itself
plus one more token. There is no remove move: the frontier is global — every unfired child
of every fired node, all in one pool — so a shallow node's untried siblings are always
still reachable and removal would be a second way to say the same thing.

**A node's value is what its own people look like.**

    P̂(node) = (s + 10·P̂(parent)) / (n + 10)

where ``n`` is how many people the node's pages brought in and ``s`` what they add up to
as fits: a judged lead counts its verdict, any other the GP's ``P(f > 0.5)``
(``HarvestScores``). The prior is ten people at the **parent's level**, so a node that has
found few people stays near its parent and one that has found a hundred is mostly itself;
an unfired node holds none of its own and sits at its parent's level.

**Why not count words.** The walk used to count, for each node, the labelled profiles
holding all its words. That measures how well a query's words describe the people
already judged, not what firing it returns — and the judged people are a skewed sample:
on CaleafTech's batch 5, mostly the customer's marks. Rare tags on two marked fits
(``capsules``, ``cbga``) scored 0.8–0.9 and matched 1–15 people on LinkedIn, most of them
already in the store. Measured on the held 500-lead order (70 queries with 8 or more
verdicts, each scored by a GP that never saw that query's verdicts), the rank correlation
of a query's score with its real fit rate:

    store age (verdicts)      60     150    400    all
    word counts            −0.04      —      —    0.18
    GP over its people      0.32    0.14   0.26   0.52   (shrunk, weight 10)

Nothing ranks queries well on a young store, which is why a fresh walk opens on the ICP's
own words (``seed_frontier``) and the customer's words only narrow it. An earlier GP
attempt embedded a query's *keywords* and lost to the counts; this scores the people the
query returned. The label store still decides which children exist (``expand``).

**Sampling is Thompson over the same two numbers.** ``θ ~ Beta(s + 10·P̂(parent), (n − s)
+ 10·(1−P̂(parent)))`` — the Beta parameters *are* the shrunk estimate, so this is one line
rather than a mechanism. Width tracks evidence, so a node nobody has tried gets tried and a
node whose people keep looking poor stops appearing. Set ``THOMPSON = False`` to fall back
to greedy — defensible here, because depletion pushes the walk along on its own, and which
is better cannot be settled offline (simulating it needs corpus answers for queries never
fired).

**Retirement is a corpus fact, never a model fact.** Nothing is retired for scoring badly —
the qualifier refits constantly, and a barren yield is a verdict about a view, not about
whether anybody matches. Only emptiness retires a node, and *which* emptiness depends on
the offset, because the provider answers ``0`` both for a query matching nobody and for one
paged past its end (§7).
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter

import numpy as np

from openoutfind.discovery import (
    REQUIRES, SEARCH_FIELDS, describe_filters, filters_for, is_fireable,
)

logger = logging.getLogger(__name__)

DISCOVERY_PAGE_SIZE = 100

# How much of the frontier a debug run prints. The frontier is the whole decision, but
# printing thousands of rows per pass buries the answer it is meant to explain.
_DEBUG_FRONTIER_ROWS = 10

# Elasticsearch's ``index.max_result_window``, measured (§7): every query is worth at most
# 10,000 rows however many millions it counts. Reaching it means "capped", which is a very
# different fact from "drained" — a capped node's children still open fresh windows, a
# drained node's children are already in our DB.
REACH_CAP = 10_000

# Draw the frontier's order from each node's Beta, rather than taking its mean. One line,
# nothing to tune; see the module docstring.
THOMPSON = True

# How many values one field may hold in a single node lives with the provider contract it
# comes from (``discovery.SEARCH_FIELDS``): two for ``lead_job_title``, whose words AND
# inside the string, one for each closed axis, where a joined pair is a value nobody
# has, and one for ``company_keywords``, which reads two words as a phrase. Without a cap the walk descends: a live campaign reached 183 frontier nodes at
# depth 4 asking for titles like ``ai-powered cto founder user``, each costing a
# round-trip to learn it matches nobody. The cap is per field, so a node can still narrow
# on another axis.


# ── node identity ────────────────────────────────────────────────────

def token_key(keywords) -> str:
    """sha256 of the canonicalized keyword set — the node-identity key.

    Order-independent and field-aware, so ``{title:founder, title:cto}`` is one node
    whichever parent reached it first. Add-only expansion over three fields means most
    nodes *are* reachable several ways, so this is load-bearing rather than a dedup nicety.
    """
    canonical = json.dumps(sorted(tuple(k) for k in keywords), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ── the label store ──────────────────────────────────────────────────

class LabelStore:
    """Every labelled profile as a token set plus a verdict — the walk's whole evidence.

    Loaded once per discovery pass and held in memory: the store is hundreds of rows, so
    counting a node is a set-containment scan that costs microseconds, and re-deriving it
    each pass means there is no counter anywhere to drift out of step with the labels.
    """

    def __init__(self, tokens: list[frozenset[str]], labels: list[int]):
        self._tokens = tokens
        self._labels = labels

    @classmethod
    def load(cls) -> "LabelStore":
        """The install's labelled leads, **plus the anchors as positives**.

        Qualified = any deal that is not a rejection.

        **The anchors are what make the cold phase work at all**, here for the same
        reason they already anchor the GP. Expansion only offers a token that has shared
        a *qualified* profile with the node, and a campaign that has never accepted
        anybody has no qualified profile — so the frontier could not grow past its
        depth-1 seed nodes, which are one-token queries matching millions of the wrong
        people, so nothing qualified, so the frontier still could not grow. A closed
        loop, and the seed's own tokens could never be conjoined into the precise query
        (``"founder cto"``, not ``founder``) that the walk exists to find.

        The synthetic ideal profiles break it: they are written in ``profile_text``'s
        shape, so they tokenize like any lead and say which words describe the people
        this campaign wants — which is exactly the claim co-occurrence needs. It is the
        same bargain the GP already takes, on the same evidence: ``BayesianQualifier``
        keeps the anchor profiles permanently, so this store always counts them alongside
        whatever real qualified profiles have accumulated — a small, constant nudge that
        matters early and is swamped by real evidence as the campaign grows.

        They feed the **vocabulary** too (``vocabulary.refresh``), through a second field
        rather than through this one: an anchor is a ``Lead`` row, so its
        ``source_fields`` carry the same invented person with each value already under
        the field it is searchable in. The rule this replaces was that anchors must never reach the
        vocabulary, and the reason was sound — an anchor was one flat string, and
        splitting it by guess would file ``united states`` as a job title, the error
        ``discovery.KEYWORD_SOURCE_FIELDS`` exists to prevent. But that argues against
        *splitting*, not against asking: ``icp._AnchorProfile`` has the model write the
        fields out itself, so nothing is inferred from the line. What it buys is the case
        this store cannot reach — a campaign with no acceptances has no vocabulary beyond
        the seed, so its frontier cannot grow past depth-1 however well this ranks it.
        """
        from openoutfind.core.pipeline.vocabulary import label_tokens
        from openoutfind.crm.models import Deal, DealState, Lead, Outcome

        verdicts = {}
        for lead_id, state, outcome in (
            Deal.objects.filter(lead_id__isnull=False)
            .values_list("lead_id", "state", "outcome")
        ):
            verdicts[lead_id] = 0 if (state == DealState.FAILED
                                      and outcome == Outcome.WRONG_FIT) else 1

        tokens, labels = [], []
        for lead_id, text, fields in (
            Lead.objects.filter(pk__in=verdicts)
            .values_list("pk", "profile_text", "source_fields")
        ):
            if text:
                tokens.append(label_tokens(text, fields))
                labels.append(verdicts[lead_id])

        anchors = 0
        for text, fields in (
            Lead.objects.filter(synthetic=True).values_list("profile_text", "source_fields")
        ):
            if text:
                tokens.append(label_tokens(text, fields))
                labels.append(1)
                anchors += 1
        if anchors:
            logger.debug("label store: %d real verdict(s) + %d anchor(s) as positives",
                         len(labels) - anchors, anchors)
        return cls(tokens, labels)

    def __len__(self) -> int:
        return len(self._labels)

    @property
    def qualified_count(self) -> int:
        """How many labelled profiles were accepted — what expansion has to work from."""
        return sum(self._labels)

    @property
    def base_rate(self) -> float:
        """The level a depth-1 node inherits — what an unfiltered query would qualify at.

        The root's stand-in. The walk deliberately never *fires* the empty query (it
        matches everyone, and its 10k window is the provider's famous-company head), so
        the one thing a root would have supplied is taken from the labels instead.

        **Laplace-smoothed, and that is load-bearing rather than tidy.** The raw rate is 0
        for a campaign whose every verdict so far is a rejection — the exact state the
        anchors exist for, so a common one, not an edge case — and a level of 0 makes
        ``α = a + 2·0`` zero for every unlabelled node, which is not a distribution.
        Smoothing here fixes it for the whole walk by induction: with the level strictly
        inside (0, 1), every ``_beta_params`` is positive on both sides, so every
        ``estimate`` is again strictly inside (0, 1) for the children below it. An empty
        store still reads 0.5.
        """
        return (sum(self._labels) + 1) / (len(self._labels) + 2)

    def counts(self, pairs) -> tuple[int, int]:
        """``(a, b)`` — labelled profiles containing every one of a node's tokens.

        Title, place and seniority tokens are matched anywhere in ``profile_text`` (see
        ``vocabulary.profile_tokens``), which is how the estimator was measured and what
        lets rows predating per-field capture still count. A company token is matched
        among the lead's company tags only (``vocabulary.pair_words``). **Word-wise,
        too** — a keyword may be a phrase (``California, United States``), and a profile
        is a bag of words, so containment is over the phrase's words.
        """
        wanted = _words(pairs)
        a = b = 0
        for tokens, label in zip(self._tokens, self._labels):
            if wanted <= tokens:
                if label:
                    a += 1
                else:
                    b += 1
        return a, b

    def cooccurring(self, pairs, candidates) -> list[tuple[str, str]]:
        """Candidate keywords that appear alongside ``pairs`` in ≥1 *qualified* profile.

        The expansion rule, and the reason the frontier stays small without a top-K cap.
        A child whose tokens never co-occur in anything we have accepted is a guess the
        store cannot speak to at all: it would enter with ``a = b = 0``, inherit its
        parent's estimate exactly, and be indistinguishable from every one of its
        thousand siblings — so the draw among them would be noise, and most would be
        empty at the provider anyway. Requiring one real co-occurrence keeps every child
        a proposition the evidence has something to say about, and it self-limits with
        depth: a three-token node has few words that ever shared a profile with it.
        """
        from openoutfind.core.pipeline.vocabulary import pair_words

        wanted = _words(pairs)
        live = set()
        for tokens, label in zip(self._tokens, self._labels):
            if label and wanted <= tokens:
                live |= tokens
        held = set(pairs)
        return [(field, token) for field, token in candidates
                if (field, token) not in held
                and (words := pair_words(field, token)) and words <= live]


def _words(pairs) -> set[str]:
    """Every word a node's keywords are made of — what a profile is matched against.

    One keyword is one search value, and a search value is not always one word: a
    ``lead_location`` is a whole place. Counting is over profile *words*, so a phrase
    keyword contributes each of its words and all of them have to be present.
    """
    from openoutfind.core.pipeline.vocabulary import pair_words

    words: set[str] = set()
    for field, token in pairs:
        words |= pair_words(field, token)
    return words


# ── the harvest scores ───────────────────────────────────────────────

class HarvestScores:
    """What each fired node's own people look like: soft fit counts per node.

    A lead a node found counts as its verdict when it has one, and otherwise as the GP's
    ``P(f > 0.5)`` for it — a fraction of a fit. Summed per node that is ``(s, n)``: how
    many fits its people add up to, out of how many people. The root level is the same
    sum over every searched lead in the store, so the level and the counts are in one unit.

    Loaded once per pass and held in memory: one GP prediction over the searched leads.
    A lead the customer marked (``--labels``) was never found by a search, so it is in
    neither; an anchor neither.
    """

    def __init__(self, soft: dict[int, tuple[float, int]], root: float):
        self._soft = soft
        self.root = root

    @classmethod
    def load(cls, qualifier, fallback_level: float) -> "HarvestScores":
        """Score every searched lead, or count only the judged ones without a model.

        ``fallback_level`` is the root when nothing has been scored yet — the label
        store's base rate, so a fresh walk starts where it always has.
        """
        from openoutfind.crm.models import Deal, DealState, Lead, Outcome

        verdicts = {
            lead_id: 0 if (state == DealState.FAILED and outcome == Outcome.WRONG_FIT) else 1
            for lead_id, state, outcome in Deal.objects.filter(
                lead__discovered_by__isnull=False).values_list("lead_id", "state", "outcome")
        }
        rows = list(
            Lead.objects.filter(discovered_by__isnull=False, synthetic=False, marked=False)
            .values_list("pk", "discovered_by_id", "embedding")
        )
        scores = dict(verdicts)
        unjudged = [(pk, emb) for pk, _, emb in rows if pk not in verdicts and emb]
        if qualifier is not None and unjudged:
            X = np.array([np.frombuffer(emb, dtype=np.float32) for _, emb in unjudged],
                         dtype=np.float64)
            probs = qualifier.predict_probs(X)
            if probs is not None:
                scores.update(zip((pk for pk, _ in unjudged), (float(p) for p in probs)))

        soft: dict[int, tuple[float, int]] = {}
        for pk, node_id, _ in rows:
            if pk in scores:
                s, n = soft.get(node_id, (0.0, 0))
                soft[node_id] = (s + scores[pk], n + 1)
        total = sum(s for s, _ in soft.values())
        people = sum(n for _, n in soft.values())
        # Laplace-smoothed like ``LabelStore.base_rate``, and for the same reason: a store
        # whose searched leads are all fits (or all misses) would put the level at 1 (or
        # 0), and every unfired node's Beta would lose a side.
        return cls(soft, (total + 1) / (people + 2) if people else fallback_level)

    def counts(self, node) -> tuple[float, int]:
        """``(s, n)`` — the node's people as soft fits, and how many people it found."""
        return self._soft.get(node.pk, (0.0, 0))


# ── the estimator ────────────────────────────────────────────────────

# How many people of the parent's quality a node is assumed to hold before its own count:
# its score is (s + W·level) / (n + W). Measured on the held 500-lead order's store, as
# the rank correlation of each query's score with its real fit rate over 70 queries, GP
# never trained on that query's verdicts: 10 beat both the plain average and 2 at every
# store age — first 60 verdicts 0.32 (plain 0.27), 150 0.14 (0.09), 400 0.26 (0.18), all
# 0.52 (0.52), and 0.37 on the smaller half of queries (0.34).
PRIOR_WEIGHT = 10


def _beta_params(node, scores: HarvestScores, cache: dict) -> tuple[float, float]:
    """``(α, β)`` of a node's Beta — the shrunk estimate, in the form Thompson wants.

    ``PRIOR_WEIGHT`` people of prior mass at the parent's level, plus the node's own
    people as soft fits. A node nobody has fired holds none of its own, so it sits at its
    parent's level with the prior's width; a node that found 100 people is mostly itself.
    Both sides are strictly positive while the level is inside (0, 1).
    """
    s, n = scores.counts(node)
    level = estimate(node.parent, scores, cache) if node.parent_id else scores.root
    return s + PRIOR_WEIGHT * level, (n - s) + PRIOR_WEIGHT * (1 - level)


def estimate(node, scores: HarvestScores, cache: dict | None = None) -> float:
    """``P̂(node)`` — the node's shrunk fit rate, its people seen through the GP.

    Recurses to the parent for the level. Depth is the number of tokens in a node, so the
    recursion is a handful of frames; the cache keeps a frontier of thousands from
    re-walking shared ancestry.
    """
    cache = {} if cache is None else cache
    if node is None:
        return scores.root
    if node.pk in cache:
        return cache[node.pk]
    alpha, beta = _beta_params(node, scores, cache)
    cache[node.pk] = value = alpha / (alpha + beta)
    return value


# ── the frontier ─────────────────────────────────────────────────────

def frontier() -> list:
    """Every node still worth firing: unfired children, plus fired veins with pages left.

    One pool, deliberately. Deepening a proven vein and opening a fresh one are not two
    policies needing an alternation rule — they are two rows scored the same way, and the
    draw decides. A vein whose people stop looking like fits sinks on its own.
    """
    from openoutfind.core.models import QueryNode

    return list(
        QueryNode.objects
        .filter(state__in=(QueryNode.State.FRONTIER, QueryNode.State.FIRED))
        .prefetch_related("keywords").select_related("parent")
    )


def best_estimate(scores: HarvestScores) -> float:
    """The highest ``P̂`` on the frontier: the fit rate the walk expects of its best next
    page. ``0.0`` when there is nothing left to fire.

    The greedy value, not a Thompson draw — it answers *what would a search be worth*,
    which wants the estimate, not the exploration noise the draw adds to pick a node.
    """
    cache: dict = {}
    return max((estimate(node, scores, cache) for node in frontier()), default=0.0)


def next_node(scores: HarvestScores, rng=None):
    """The node to fire next, or ``None`` when the frontier is empty.

    ``None`` is the saturation signal: nothing left unfired and nothing left to deepen,
    which ``discover`` answers by refreshing the vocabulary and expanding again.
    """
    candidates = frontier()
    if not candidates:
        return None

    rng = np.random.default_rng() if rng is None else rng
    cache: dict = {}
    scored = []
    for node in candidates:
        alpha, beta = _beta_params(node, scores, cache)
        score = rng.beta(alpha, beta) if THOMPSON else alpha / (alpha + beta)
        scored.append((score, node))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    score, best = scored[0]

    if logger.isEnabledFor(logging.DEBUG):
        # Why *this* query: the whole decision is these numbers, so print them rather
        # than the conclusion. `s`/`n` are the node's own people as soft fits, `lvl` its
        # inherited level, `P̂` the shrunk estimate and `draw` what Thompson actually
        # sampled — a node winning on a low P̂ means its posterior was wide, which is the point.
        logger.debug("frontier: %d node(s), root level %.3f", len(candidates), scores.root)
        for rank, (drawn, node) in enumerate(scored[:_DEBUG_FRONTIER_ROWS], start=1):
            s, n = scores.counts(node)
            level = estimate(node.parent, scores, cache) if node.parent_id else scores.root
            marker = "→" if node.pk == best.pk else " "
            logger.debug("  %s #%d draw=%.3f P̂=%.3f  s=%.1f n=%d lvl=%.3f  %s@%d %s",
                         marker, rank, drawn, estimate(node, scores, cache), s, n, level,
                         describe_filters(filters_for(node.pairs)),
                         node.next_offset, node.state)
        if len(scored) > _DEBUG_FRONTIER_ROWS:
            logger.debug("  … %d more", len(scored) - _DEBUG_FRONTIER_ROWS)
    return best


# ── node lifecycle ───────────────────────────────────────────────────

def _dead_sets() -> list[frozenset]:
    """Keyword sets already proven to match nobody — the anti-monotone prune.

    A superset of an empty conjunction is empty, so a dead node convicts every node that
    contains it. Checking at *creation* rather than storing a separate blacklist is what
    lets the prune reach supersets the walk would otherwise reach through a different
    parent — dedup on ``token_key`` makes the lattice a DAG, not a tree, so following
    parent links alone would miss them.
    """
    from openoutfind.core.models import QueryNode

    return [
        frozenset(node.pairs) for node in
        QueryNode.objects.filter(state=QueryNode.State.DEAD)
        .prefetch_related("keywords")
    ]


def seed_frontier(keywords, seed) -> int:
    """Open the walk with one depth-1 node per keyword. Returns nodes created.

    ``discover`` passes the ICP's own words, not the whole vocabulary: on a store started
    from a customer's marks the vocabulary already holds every word two of their fits
    share, and opening on rare ones (``capsules``, ``cbga``) searched a few people the
    marks already named. Those words still reach the walk as children of the ICP's nodes.

    There is no root. The empty query is never fired — it matches everyone, so its one
    10k window is the provider's famous-company head and tells us nothing — and the level
    a root would have supplied comes from ``HarvestScores.root`` instead.

    ``seed`` carries what the ICP said about the *shape* of every query rather than its
    words — the size band and the country — and each opened node keeps its own copy, so a
    later re-seed cannot retroactively change what an already-fired node searched.

    A keyword that cannot fire alone (``discovery.REQUIRES`` — a company word) opens
    instead as a depth-2 node with each keyword of the field it needs: ``cannabis`` the
    company opens as {company cannabis, title purchasing}, {…, title founder}, and so on.

    **The ICP's place rides every opening node** when it names one (``seed.location``),
    and is not opened on its own. Location is one value per node and the walk only adds,
    so every node below inherits it: a US order whose company + title nodes carried no
    country put Canadian extraction managers up for a verdict. A depth-1 {United States}
    node matched everyone in the country and was never worth firing anyway.
    """
    place = seed.location if seed is not None else ""
    created = 0
    for node_pairs in _opening_nodes(keywords, place):
        if _upsert(node_pairs, parent=None, seed=seed) is not None:
            created += 1
    return created


def _opening_nodes(keywords, place: str = "") -> list[list[tuple[str, str]]]:
    """The keyword sets the walk opens with — one per keyword, paired where it must be.

    With a ``place``, other locations are not opened and every node carries that one.
    """
    if place:
        keywords = [pair for pair in keywords if pair[0] != "lead_location"]
    nodes = []
    for pair in keywords:
        if is_fireable([pair]):
            nodes.append([pair])
            continue
        partner_field = REQUIRES[pair[0]]
        nodes.extend(sorted([pair, partner]) for partner in keywords
                     if partner[0] == partner_field)
    if place:
        nodes = [sorted([*node, ("lead_location", place)]) for node in nodes]
    return nodes


def _upsert(pairs, parent, scores: HarvestScores | None = None, cache: dict | None = None,
            seed=None):
    """Create a node, or re-point an existing one at a better parent. ``None`` if pruned.

    A node reachable by several paths keeps the parent giving the **highest** estimate.
    Optimism, and it matches how the level is used: the parent is a claim about the region
    this node sits in, and the best-supported claim is the one worth carrying.

    A child searches its parent's band and country — it is that query plus one word, so
    changing either would make it a different query, not a narrower one. A depth-1 node
    has no parent and takes them from ``seed``.
    """
    from openoutfind.core.models import Keyword, QueryNode

    key = token_key(pairs)
    node = QueryNode.objects.filter(token_key=key).first()
    if node is not None:
        if (parent is not None and node.parent_id != parent.pk
                and node.state == QueryNode.State.FRONTIER and scores is not None):
            if estimate(parent, scores, cache) > estimate(node.parent, scores, cache):
                node.parent = parent
                node.save(update_fields=["parent"])
        return None

    node = QueryNode.objects.create(token_key=key, parent=parent,
                                    **_query_shape(parent, seed))
    node.keywords.set(Keyword.rows_for(pairs))
    return node


def _query_shape(parent, seed) -> dict:
    """The size band and country a new node inherits — from its parent, or from the seed.

    Neither is defaulted here: the field defaults are the answer when there is no parent
    and no seed, which is the case an install with a hand-built node is in.
    """
    if parent is not None:
        return {
            "headcount_min": parent.headcount_min,
            "headcount_max": parent.headcount_max,
            "country_code": parent.country_code,
        }
    if seed is None:
        return {}
    return {
        "headcount_min": seed.headcount[0],
        "headcount_max": seed.headcount[1],
        "country_code": seed.country_code,
    }


def expand(node, store: LabelStore, candidates, scores: HarvestScores | None = None) -> int:
    """Grow the frontier with this node's children. Returns how many were created.

    A child is this node plus one token that has shared a qualified profile with it
    (``LabelStore.cooccurring``), that no dead node is a subset of, that does not push
    any one field past what ``SEARCH_FIELDS`` allows it, and that can be fired
    (``discovery.is_fireable`` — a company word joins a node only if it holds a title).
    Which children exist is the label store's word; which parent a child reached two
    ways keeps is ``scores``'.
    """
    pairs = node.pairs
    dead = _dead_sets()
    cache: dict = {}

    per_field = Counter(field for field, _ in pairs)
    offered = store.cooccurring(pairs, candidates)
    created = pruned = capped = 0
    for pair in offered:
        child_pairs = sorted([*pairs, pair])
        if per_field[pair[0]] >= SEARCH_FIELDS[pair[0]] or not is_fireable(child_pairs):
            capped += 1
            continue
        child_set = frozenset(child_pairs)
        if any(empty <= child_set for empty in dead):
            pruned += 1
            continue
        if _upsert(child_pairs, parent=node, scores=scores, cache=cache) is not None:
            created += 1

    if logger.isEnabledFor(logging.DEBUG):
        # "+0 nodes" is the frontier's most confusing outcome, and it has three very
        # different causes: no vocabulary to offer, no *qualified* profile pairing any
        # of it with this node, or every child already known. Name which one.
        logger.debug("  expand [%s]: %d vocabulary → %d co-occurring → "
                     "%d new, %d dead-pruned, %d field-capped, %d already known",
                     describe_filters(filters_for(pairs)), len(candidates), len(offered),
                     created, pruned, capped,
                     len(offered) - created - pruned - capped)
        if candidates and not offered:
            logger.debug("    nothing co-occurs: the store holds %d qualified profile(s), "
                         "and expansion only offers tokens seen alongside this node in one",
                         store.qualified_count)
    return created


def advance(node, leads_found: int | None = None) -> None:
    """Record a productive page: move the offset on, keep the node fireable.

    ``leads_found`` is the provider's corpus count, stored when offset 0 reported one.
    Diagnostic only — the walk fires nodes rather than counting them first — but it is the
    number that tells an operator whether a vein is 400 rows or 40 million.
    """
    from openoutfind.core.models import QueryNode

    node.next_offset += DISCOVERY_PAGE_SIZE
    node.state = QueryNode.State.FIRED
    fields = ["next_offset", "state"]
    if leads_found is not None:
        node.leads_found = leads_found
        fields.append("leads_found")
    node.save(update_fields=fields)


def retire(node, *, at_offset: int) -> str:
    """Retire an empty node and prune what its emptiness convicts. Returns what happened.

    Three cases, and they are not interchangeable — the provider reports ``0`` for all
    three (§7):

    - **offset 0** — the index matches nobody. The node is ``dead``, and so is every node
      containing its tokens: a superset matches a subset of people.
    - **past the end, below the reach cap** — the node drained completely, so every one of
      its people is already a ``Lead`` here. Its supersets are drawn from that same
      exhausted population, so they are pruned too; there is nothing new down there.
    - **past the end, at the reach cap** — we hit Elasticsearch's 10,000-row window, not
      the end of the population. The node retires but its **subtree stays**: adding a
      token opens a fresh 10k window over the part we could not reach.
    """
    from openoutfind.core.models import QueryNode

    if at_offset == 0:
        node.state = QueryNode.State.DEAD
        node.save(update_fields=["state"])
        _prune_descendants(node)
        return "dead"

    node.state = QueryNode.State.DRAINED
    node.save(update_fields=["state"])
    if at_offset < REACH_CAP:
        _prune_descendants(node)
        return "drained"
    return "capped"


def _prune_descendants(node) -> int:
    """Mark every node below this one dead. Returns how many.

    Walks the ``parent`` links breadth-first. Supersets reached through a *different*
    parent are handled at creation time instead (``_dead_sets``), which is the half of the
    prune that survives the lattice being a DAG.
    """
    from openoutfind.core.models import QueryNode

    pruned, generation = 0, [node.pk]
    while generation:
        children = list(
            QueryNode.objects.filter(parent_id__in=generation)
            .exclude(state=QueryNode.State.DEAD).values_list("pk", flat=True)
        )
        if not children:
            break
        QueryNode.objects.filter(pk__in=children).update(state=QueryNode.State.DEAD)
        pruned += len(children)
        generation = children
    return pruned
