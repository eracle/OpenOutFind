# openoutfind/core/pipeline/compose.py
"""The searches worth the most on one draw of the belief, composed from every keyword the
store holds.

An LLM scores its whole vocabulary at every step and samples from all of it. The walk
does the same: no term is cut before it is valued. A search is title words (any number,
ANDed), one tag phrase, at most one seniority and one place (``query``), and on a draw of
the belief (``belief``) each keyword brings two numbers, both adding up over a search's
keywords:

    lift   logit θ_t − logit π     what it does to the share of people who fit
    share  log(N_t / N₀)           what it does to the people matched, never above 0

So a search's rate and size are sums, and the best searches on a draw are found without
listing every one. At one size — a bin of the summed share — only the highest summed lift
can be worth the most, so the fields are added one at a time, keeping the best per bin:
title words (any subset of those that lift, plus every word alone), then a seniority or
none, then the place, then every tag. Each finished search is valued at what a page of it
would bring per dollar (``worth``, from the frontier), and the best go back to the
frontier, which values them exactly — the store's people they match, the counts of fired
searches they narrow. A search the frontier refuses (fired, dead, drained) is what
``limit`` leaves room for.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from openoutfind.core.pipeline.query import Terms, place_of

# The width of a size bin, on the log scale: two searches a tenth apart in size are one size.
BIN = 0.1
# The fewest people a search can match and still be worth a page.
MIN_PEOPLE = 1.0

Keyword = tuple
Worth = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Vocabulary:
    """Every keyword a search may hold, per field.

    ``places`` is empty when the market has no country; a search then names no place.
    """

    titles: tuple[Keyword, ...]
    tags: tuple[Keyword, ...]
    seniorities: tuple[Keyword, ...]
    places: tuple[Keyword, ...]

    def keywords(self) -> list[Keyword]:
        return [*self.titles, *self.tags, *self.seniorities, *self.places]


def as_terms(keywords) -> Terms:
    """A composed search's keywords as ``Terms``."""
    title, tag, place, seniority = set(), (), "", ""
    for keyword in keywords:
        field = keyword[0]
        if field == "title":
            title.add(keyword[1])
        elif field == "tag":
            tag = keyword[1]
        elif field == "seniority":
            seniority = keyword[1]
        elif field == "state":
            place = place_of(keyword[1], keyword[2])
        elif field == "country":
            place = place_of("", keyword[1])
    return Terms(title=frozenset(title), tag=tag, place=place, seniority=seniority)


# ── the best per size ────────────────────────────────────────────────

@dataclass
class _Best:
    """The partial searches kept: at each size bin the ``PER_BIN`` highest summed lifts,
    each with its exact summed share and the keywords that reached it."""

    lift: np.ndarray
    share: np.ndarray
    parts: list

    @classmethod
    def start(cls) -> "_Best":
        return cls(np.zeros(1), np.zeros(1), [()])


# The partial searches kept per size bin: room below the best for the ones the frontier
# refuses, and for two words of one size and lift.
PER_BIN = 4


def _bin(share: np.ndarray) -> np.ndarray:
    return np.floor(-share / BIN + 1e-9).astype(np.int64)


def _add(best: _Best, options, keep: bool, floor: float, skip=None) -> _Best:
    """``best`` with one of ``options`` — ``(keyword, lift, share)`` — added to each
    entry, or left as it is too where ``keep``. ``skip(parts, keyword)`` refuses one."""
    entries = np.arange(len(best.lift))
    lifts, shares, origins, chosen = [], [], [], []
    if keep:
        lifts.append(best.lift)
        shares.append(best.share)
        origins.append(entries)
        chosen.append(np.full(len(entries), -1))
    for index, (keyword, lift, share) in enumerate(options):
        rows = entries if skip is None else np.array(
            [e for e in entries if not skip(best.parts[e], keyword)], dtype=np.int64)
        summed = best.share[rows] + share
        inside = summed >= floor
        lifts.append(best.lift[rows[inside]] + lift)
        shares.append(summed[inside])
        origins.append(rows[inside])
        chosen.append(np.full(int(inside.sum()), index))
    lift, share = np.concatenate(lifts), np.concatenate(shares)
    origin, option = np.concatenate(origins), np.concatenate(chosen)
    return _prune(lift, share, [
        best.parts[o] + (() if c < 0 else (options[c][0],)) for o, c in zip(origin, option)])


def _prune(lift: np.ndarray, share: np.ndarray, parts: list) -> _Best:
    """The ``PER_BIN`` highest lifts of each size bin, each search once."""
    target = _bin(share)
    kept, count, seen = [], {}, set()
    for i in np.lexsort((-lift, target)):
        b = target[i]
        key = frozenset(parts[i])
        if count.get(b, 0) >= PER_BIN or key in seen:
            continue
        seen.add(key)
        count[b] = count.get(b, 0) + 1
        kept.append(i)
    kept = np.array(kept, dtype=np.int64)
    return _Best(lift[kept], share[kept], [parts[i] for i in kept])


def _titles(options, floor: float) -> _Best:
    """Every word alone, then every word that lifts added where it pays, one at a time."""
    best = _add(_Best.start(), options, keep=False, floor=floor)
    for option in sorted((o for o in options if o[1] > 0), key=lambda o: -o[1]):
        best = _add(best, [option], keep=True, floor=floor,
                    skip=lambda parts, keyword: keyword in parts)
    return best


# ── the composition ──────────────────────────────────────────────────

def compose(vocabulary: Vocabulary, lift: dict, share: dict, log_market: float,
            worth: Worth, limit: int) -> list[Terms]:
    """The ``limit`` searches worth the most on one draw — ``lift`` and ``share`` per
    keyword — each holding title words and a tag; ``[]`` with no title or tag to hold."""
    if not vocabulary.titles or not vocabulary.tags:
        return []
    floor = math.log(MIN_PEOPLE) - log_market

    def options(keywords):
        return [(k, lift[k], share[k]) for k in keywords]

    best = _titles(options(vocabulary.titles), floor)
    best = _add(best, options(vocabulary.seniorities), keep=True, floor=floor)
    if vocabulary.places:
        best = _add(best, options(vocabulary.places), keep=False, floor=floor)
    return [as_terms(parts) for parts in _best_with_a_tag(best, options(vocabulary.tags),
                                                          floor, worth, limit)]


def _best_with_a_tag(best: _Best, tags, floor: float, worth: Worth, limit: int) -> list:
    """The ``limit`` entries of ``best`` with one tag added, worth the most."""
    rows = np.arange(len(best.lift))
    if not len(rows):
        return []
    tag_lift = np.array([t[1] for t in tags])
    tag_share = np.array([t[2] for t in tags])
    kept_value = np.empty(0)
    kept_at: list[tuple[int, int]] = []
    chunk = max(1, 2_000_000 // len(rows))
    for start in range(0, len(tags), chunk):
        stop = min(start + chunk, len(tags))
        summed_share = best.share[rows][:, None] + tag_share[None, start:stop]
        summed_lift = best.lift[rows][:, None] + tag_lift[None, start:stop]
        value = np.where(summed_share >= floor, worth(summed_lift, summed_share), -np.inf)
        flat = value.ravel()
        take = min(limit, flat.size)
        top = np.argpartition(-flat, take - 1)[:take]
        at = [(rows[i // value.shape[1]], start + i % value.shape[1]) for i in top]
        kept_value = np.concatenate([kept_value, flat[top]])
        kept_at += at
        if len(kept_at) > limit:
            keep = np.argsort(-kept_value, kind="stable")[:limit]
            kept_value, kept_at = kept_value[keep], [kept_at[i] for i in keep]
    order = np.argsort(-kept_value, kind="stable")
    return [best.parts[kept_at[i][0]] + (tags[kept_at[i][1]][0],)
            for i in order if np.isfinite(kept_value[i])]
