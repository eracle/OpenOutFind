"""Per-invocation discovery selection. No configured key opts a run into spending."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol


class Source(Protocol):
    required_field: str

    def discover(self, site_config) -> bool: ...

    def report(self) -> dict: ...


_current: ContextVar[Source | None] = ContextVar("discovery_source", default=None)


def current() -> Source | None:
    return _current.get()


@contextmanager
def selected(source: Source | None):
    token = _current.set(source)
    try:
        yield
    finally:
        _current.reset(token)
