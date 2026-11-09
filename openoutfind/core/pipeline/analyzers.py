# openoutfind/core/pipeline/analyzers.py
"""The provider's analyzers, copied: how a job title, a tag and a search become words.

A search matches the people whose text the provider analyzes into the search's words, so
a matcher that splits otherwise scores a page on people the search never asked for. ICU
splits by Unicode's word rules (UAX #29); each rule on top is a measured Lead Finder count.
"""
from __future__ import annotations

import re
from functools import lru_cache

from icu4py.breakers import WordBreaker

# Elasticsearch's ``_english_`` list, which a title drops: ``head of procurement`` =
# ``procurement head`` (89), and a title of these alone matches nobody.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
    "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with",
})

# A straight apostrophe or a dot inside a word joins it: ``d'affaires`` = ``daffaires``
# (44,226), ``c.e.o`` = ``ceo``. A curly one splits it: ``d’affaires`` = ``d affaires``.
JOINERS = re.compile(r"['.]")
CURLY_APOSTROPHE = "’"

# Thai stays one word per run, where ICU splits it by dictionary: ``ผู้จัดการ`` 71,
# ``จัดการ`` 2. Lao, Burmese and Khmer are written the same way and taken alike.
UNSPACED = re.compile(r"[฀-໿က-႟ក-៿]+")

# A Chinese character is a word of its own: ``经理`` = ``理经`` (242,808).
HAN = r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]"
HAN_ALONE = re.compile(rf"{HAN}|(?:(?!{HAN}).)+")


# ── splitting ────────────────────────────────────────────────────────

def is_word(text: str) -> bool:
    return any(char.isalnum() for char in text)


def is_unspaced(text: str) -> bool:
    return bool(UNSPACED.fullmatch(text))


def spans(text: str) -> list[tuple[int, int]]:
    """Where ICU finds words in ``text``, punctuation and spaces dropped, a Thai run whole."""
    found: list[tuple[int, int]] = []
    for start, end in WordBreaker(text, "und").segments():
        if not is_word(text[start:end]):
            continue
        if found and found[-1][1] == start and is_unspaced(text[found[-1][0]:end]):
            found[-1] = (found[-1][0], end)
        else:
            found.append((start, end))
    return found


def pieces(segment: str) -> list[str]:
    """One ICU word as the provider's words."""
    return [piece for part in segment.split(CURLY_APOSTROPHE)
            for piece in HAN_ALONE.findall(JOINERS.sub("", part))]


@lru_cache(maxsize=65_536)
def hyphen_runs(text: str | None) -> tuple[tuple[str, ...], ...]:
    """``text`` lowercased into runs of words joined by one hyphen each.

    ``CEO, Co-Founder`` → ``(("ceo",), ("co", "founder"))``. ``str.lower`` is the
    provider's casing: ``İşletme`` (3,984) is not ``işletme`` (486). Cached: a company's
    tag list repeats on every one of its people.
    """
    text = (text or "").lower()
    runs: list[list[str]] = []
    previous_end = 0
    for start, end in spans(text):
        if runs and text[previous_end:start] == "-":
            runs[-1].append(text[start:end])
        else:
            runs.append([text[start:end]])
        previous_end = end
    return tuple(tuple(run) for run in runs)


def run_words(run: tuple[str, ...]) -> list[str]:
    return [piece for segment in run for piece in pieces(segment)]


def words(text: str) -> list[str]:
    """``text``'s words as the provider indexes them, lowercased, in order."""
    return [word for run in hyphen_runs(text) for word in run_words(run)]


# ── the title analyzers ──────────────────────────────────────────────

def title_words(text: str) -> frozenset[str]:
    """A profile's title as the words a search can match: stopwords dropped, and a
    hyphenated word also joined, since ``co-founder``, ``co founder`` and ``cofounder``
    each find the same 179,644."""
    runs = hyphen_runs(text)
    joined = {"".join(run_words(run)) for run in runs if len(run) > 1}
    return frozenset({word for run in runs for word in run_words(run)} - STOPWORDS | joined)


def query_title_words(text: str) -> frozenset[str]:
    """The words a title search requires: stopwords dropped, a hyphenated word split."""
    return frozenset(words(text)) - STOPWORDS


def standalone_title_words(text: str) -> frozenset[str]:
    """A title's words outside any hyphenated one: what a composed search may hold.

    ``co`` alone finds 283,068: the 179,644 co-founders, and co-owners, co-CEOs and
    ``& co`` besides.
    """
    return frozenset(word for run in hyphen_runs(text) if len(run) == 1
                     for word in run_words(run)) - STOPWORDS


# ── the tag analyzer ─────────────────────────────────────────────────

def phrase_words(text: str) -> tuple[str, ...]:
    """A tag, or a tag search, as its word sequence: stopwords kept, a hyphen only split.

    ``e-commerce`` is ``e commerce``, ``ecommerce`` another term, and ``home_and_garden``
    one word (1,625) apart from ``home and garden`` (431).
    """
    return tuple(words(text))


def company_tags(text: str) -> list[tuple[str, ...]]:
    """A company's stored tag list (``", "``-joined) as phrases."""
    return [phrase for tag in (text or "").split(",") if (phrase := phrase_words(tag))]


def sub_phrases(tag: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Every phrase a tag search matches inside ``tag``: its words adjacent and in order.

    For matching only, never a keyword: ``pre`` from ``pre rolls`` reaches
    ``pre-seed venture capital`` and ``pre-k to 5th grade``, 6,512 founders against the
    71 tagged ``pre rolls``. A search holds a whole tag (``company_tags``).
    """
    return [tag[start:end] for start in range(len(tag))
            for end in range(start + 1, len(tag) + 1)]


def holds_phrase(longer: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    """Whether ``phrase`` sits inside ``longer``, adjacent and in order."""
    span = len(phrase)
    return bool(span) and any(longer[i:i + span] == phrase
                              for i in range(len(longer) - span + 1))


def is_stopword_phrase(phrase: tuple[str, ...]) -> bool:
    """A phrase of stopwords alone, never a term a search may hold."""
    return all(word in STOPWORDS for word in phrase)
