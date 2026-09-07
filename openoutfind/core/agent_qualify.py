"""State for ``find --agent-qualify``, read where a call to ``AI_MODEL`` would fire.

A contextvar rather than a parameter threaded through ``run_job -> cycle -> top_up ->
qualify`` — at most one verdict is ever in flight, and none of those layers has any
other reason to know a calling agent is answering instead of a model.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    fit: bool
    reason: str


_active: ContextVar[bool] = ContextVar("agent_qualify_active", default=False)
_verdict: ContextVar[Verdict | None] = ContextVar("agent_qualify_verdict", default=None)


def enable(verdict: Verdict | None) -> None:
    """Turn agent-qualify mode on for this process, with an optional resume answer."""
    _active.set(True)
    _verdict.set(verdict)


def active() -> bool:
    return _active.get()


def take_verdict() -> Verdict | None:
    """Consume the answer once — a second read must never re-apply it to a new candidate."""
    verdict = _verdict.get()
    _verdict.set(None)
    return verdict
