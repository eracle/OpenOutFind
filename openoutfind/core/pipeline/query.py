# openoutfind/core/pipeline/query.py
"""A search — its terms per field, the provider's analyzers, and who it matches.

A search is a set of ``field:term`` pairs, each field holding what its match type allows
(the probe table above ``discovery.SEARCH_FIELDS``):

    title      words, ANDed (``match``)        any number, in the forms profiles carry
    tag        one phrase (``match_phrase``)   a word or a phrase a company's tags hold;
                                               never without a title
    place      ``state, country``              the market's country, or a state in it
    seniority  one closed-list value           at most one of the provider's 12

The size band and the market's country code are the **shape** the customer fixed: they
ride every search and never vary, so they live on the ``Search`` row beside the terms
rather than in them.

**A person matches a search when every term matches its own field**, read from the
lead's ``source_fields`` with that field's analyzer. The analyzers copy the provider's,
as measured: nothing is stemmed on either free-text field; a title drops stopwords and a
tag keeps them as words of its phrase; a hyphen splits a title word and also matches the
joined form, while in a tag it only splits; ``&`` and other punctuation are stripped; case
is ignored everywhere.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace

# Elasticsearch's ``_english_`` stopword list — the one a title's analyzer drops.
# ``head of procurement`` = ``head the procurement`` = ``procurement head`` (89), and a
# title made only of these counts 0.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
    "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with",
})

_WORD = re.compile(r"[a-z0-9]+")
_HYPHENATED = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")


# ── the analyzers ────────────────────────────────────────────────────

def title_words(text: str) -> frozenset[str]:
    """A job title's terms: words, stopwords dropped, a hyphenated word also joined.

    ``co-founder`` yields ``co``, ``founder`` and ``cofounder``, so a search for any of
    the three spellings finds it — the provider matched all three at 161,369.
    """
    text = (text or "").lower()
    words = {word for word in _WORD.findall(text) if word not in STOPWORDS}
    words |= {compound.replace("-", "") for compound in _HYPHENATED.findall(text)}
    return frozenset(words)


def query_title_words(text: str) -> frozenset[str]:
    """The words a title *search* requires: its words, stopwords dropped, never joined.

    The provider splits ``co-founder`` in a query into ``co`` and ``founder``; matching
    that against a profile's title words finds the hyphenated and spaced forms alike.
    """
    return frozenset(word for word in _WORD.findall((text or "").lower())
                     if word not in STOPWORDS)


def phrase_words(text: str) -> tuple[str, ...]:
    """A tag, or a tag search, as its word sequence: stopwords kept, punctuation gone.

    ``e-commerce`` is ``e commerce`` and ``food & beverage`` is ``food beverage``;
    ``ecommerce`` stays one word, a different term.
    """
    return tuple(_WORD.findall((text or "").lower()))


def company_tags(text: str) -> list[tuple[str, ...]]:
    """A company's stored tag list (``source_fields``, ``", "``-joined) as phrases."""
    return [words for tag in (text or "").split(",") if (words := phrase_words(tag))]


def holds_phrase(longer: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    """Whether ``phrase`` sits inside ``longer`` — adjacent and in order."""
    span = len(phrase)
    return bool(span) and any(longer[i:i + span] == phrase
                              for i in range(len(longer) - span + 1))


def is_stopword_phrase(phrase: tuple[str, ...]) -> bool:
    """A tag phrase made only of stopwords — never a term a search may hold."""
    return all(word in STOPWORDS for word in phrase)


# ── the place ────────────────────────────────────────────────────────

def split_place(place: str) -> tuple[str, str]:
    """``"California, United States"`` → ``("california", "united states")``.

    A place is ``state, country`` or a country alone; the country is required.
    """
    parts = [part.strip().lower() for part in (place or "").split(",") if part.strip()]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[-1]


def place_of(state: str, country: str) -> str:
    """A place in the casing the index matches — ``State, Country`` or ``Country``."""
    from openoutfind.discovery import as_place

    country, state = as_place(country), as_place(state)
    if not country:
        return ""
    return f"{state}, {country}" if state else country


# ── the search ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Terms:
    """One search's terms, one entry per field — the individual's genome.

    ``title`` is a set of words, ``tag`` one phrase as its word sequence, ``place`` the
    index's own spelling (``California, United States``), ``seniority`` one of
    ``discovery.LEAD_SENIORITIES``. An empty field is absent from the query.
    """

    title: frozenset[str] = field(default_factory=frozenset)
    tag: tuple[str, ...] = ()
    place: str = ""
    seniority: str = ""

    @classmethod
    def written(cls, title: str | list[str] = "", tag: str = "", place: str = "",
                seniority: str = "") -> "Terms":
        """Terms from what a person or a model wrote, run through the query analyzers."""
        title_text = " ".join(title) if isinstance(title, list) else title
        tag_words = phrase_words(tag)
        return cls(
            title=query_title_words(title_text),
            tag=() if is_stopword_phrase(tag_words) else tag_words,
            place=place.strip(),
            seniority=(seniority or "").strip().lower(),
        )

    def with_(self, **changes) -> "Terms":
        return replace(self, **changes)

    @property
    def is_legal(self) -> bool:
        """Whether this may be fired: a tag only with a title, and some term to search.

        A place alone matches the whole market, whose one 10,000-row window is the
        provider's famous-company head, so a search holds a title or a seniority too.
        """
        if self.tag and not self.title:
            return False
        return bool(self.title or self.seniority)

    @property
    def key(self) -> str:
        """sha256 of the canonical terms — a search's identity, whoever bred it."""
        canonical = json.dumps(
            [sorted(self.title), list(self.tag), self.place.lower(), self.seniority],
            separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def holds_all_of(self, other: "Terms") -> bool:
        """Whether this search is ``other`` or stricter: it holds every one of its terms.

        A stricter search matches a subset of the people, so a dead ``other`` convicts
        this one. Title words are a superset, the tag phrase sits inside this one's, the
        place is the same or a state of ``other``'s country, the seniority the same.
        """
        if not other.title <= self.title:
            return False
        if other.tag and not holds_phrase(self.tag, other.tag):
            return False
        if other.seniority and other.seniority != self.seniority:
            return False
        if other.place:
            state, country = split_place(self.place)
            other_state, other_country = split_place(other.place)
            if country != other_country or (other_state and other_state != state):
                return False
        return True

    def matches(self, source_fields: dict | None) -> bool:
        """Whether a person matches every term, each in its own field."""
        fields = source_fields or {}
        if self.title and not self.title <= title_words(fields.get("contact_job_title", "")):
            return False
        if self.tag and not any(holds_phrase(tag, self.tag)
                                for tag in company_tags(fields.get("company_keywords", ""))):
            return False
        if self.seniority and (
                str(fields.get("contact_seniority") or "").strip().lower() != self.seniority):
            return False
        if self.place:
            state, country = split_place(self.place)
            if str(fields.get("contact_location_country") or "").strip().lower() != country:
                return False
            if state and str(fields.get("contact_location_state") or "").strip().lower() != state:
                return False
        return True

    def to_filters(self, headcount: tuple[int, int] | None = None) -> dict:
        """These terms as one Lead Finder filter dict — the only thing the provider sees.

        Title words go in one string (every word required, any order), the tag phrase
        in one string (adjacent, in order), the place and the seniority as one value each.
        Free text is sent with ``exact_match`` off so a word matches inside a longer value.
        """
        filters: dict = {}
        if headcount is not None:
            filters["company_headcount_min"] = int(headcount[0])
            filters["company_headcount_max"] = int(headcount[1])
        if self.title:
            filters["lead_job_title"] = {"include": [" ".join(sorted(self.title))],
                                         "exact_match": False}
        if self.tag:
            filters["company_keywords"] = {"include": [" ".join(self.tag)],
                                           "exact_match": False}
        if self.place:
            filters["lead_location"] = {"include": [self.place]}
        if self.seniority:
            filters["lead_seniority"] = {"include": [self.seniority]}
        return filters

    def as_text(self) -> str:
        """``title head procurement · tag cannabis extraction · California, United States``."""
        parts = []
        if self.title:
            parts.append("title " + " ".join(sorted(self.title)))
        if self.tag:
            parts.append("tag " + " ".join(self.tag))
        if self.seniority:
            parts.append(self.seniority)
        if self.place:
            parts.append(self.place)
        return " · ".join(parts) or "(no terms)"

    def as_json(self) -> dict:
        return {"title": sorted(self.title), "tag": " ".join(self.tag),
                "place": self.place, "seniority": self.seniority}

    @classmethod
    def from_json(cls, data: dict) -> "Terms":
        return cls(title=frozenset(data.get("title") or ()),
                   tag=tuple(phrase_words(data.get("tag") or "")),
                   place=data.get("place") or "", seniority=data.get("seniority") or "")
