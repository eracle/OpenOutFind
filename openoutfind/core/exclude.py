"""State for ``find --exclude``: the profiles this run must never show or print.

A caller that has already handed somebody a set of leads — the hub, delivering orders to
one customer — passes those URLs back so a second delivery cannot repeat a person. An
excluded candidate is never put up for a verdict and never exported; nothing else about
it changes, so the same store serves the next customer unfiltered.

A contextvar rather than a parameter, for the reason ``agent_qualify`` is one: the set is
read at the two narrowest points — the qualification scan and the export query — and none
of ``run_job -> cycle -> top_up`` has any other reason to carry it.

**The normalisation is shared with the hub, character for character.** Both sides key on
the same string or an excluded profile slips through on a trailing slash: lowercase the
host, drop the query and the fragment, drop a trailing slash, keep everything else.
"""
from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_excluded: ContextVar[frozenset[str]] = ContextVar("excluded_profiles", default=frozenset())


def normalize_linkedin_url(url: str) -> str:
    """The key an exclude list and a stored ``profile_url`` are compared on."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def read_exclude_file(path: str | Path) -> frozenset[str]:
    """The normalised URLs in *path*, one per line; blank lines are ignored.

    Raises ``OSError`` when the file cannot be read — the caller decides how to say so.
    """
    text = Path(path).read_text(encoding="utf-8")
    return frozenset(normalize_linkedin_url(line) for line in text.splitlines() if line.strip())


def enable(urls: frozenset[str]) -> None:
    """Exclude *urls* (already normalised) for the rest of this process."""
    _excluded.set(urls)


def is_excluded(profile_url: str | None) -> bool:
    """Whether a stored ``profile_url`` is on this run's exclude list."""
    excluded = _excluded.get()
    if not excluded or not profile_url:
        return False
    return normalize_linkedin_url(profile_url) in excluded
