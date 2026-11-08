# openoutfind/core/pipeline/terms.py
"""A search's terms, one entry per field, and who they match.

Each field holds what its match type allows (the probe table above
``discovery.SEARCH_FIELDS``):

    title      words, ANDed (``match``)        any number
    tag        one phrase (``match_phrase``)   inside one company tag; never without a title
    place      ``state, country``              the market's country, or a state in it
    seniority  one closed-list value           at most one of the provider's 12

The size band and the country code ride every search unchanged, so they live on the
``Search`` row, not here. A person matches when every term matches its own field, read
through that field's analyzer (``analyzers.py``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

from openoutfind.core.pipeline.analyzers import (
    company_tags, holds_phrase, is_stopword_phrase, phrase_words, query_title_words,
    title_words,
)


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
        """sha256 of the canonical terms — a search's identity, whichever move reached it."""
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
        """Whether a person matches every term, each in its own field.

        A row with no state matches any state of its country: the provider filters on
        ``Bali, Indonesia`` (15 people against the country's 158) but only returns a state
        on US rows, so an empty state says nothing about where they are.
        """
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
            own_state = str(fields.get("contact_location_state") or "").strip().lower()
            if state and own_state and own_state != state:
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
