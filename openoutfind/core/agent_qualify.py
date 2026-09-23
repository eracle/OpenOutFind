"""State for ``find --agent-qualify``, read where a call to ``AI_MODEL`` would fire.

A contextvar rather than a parameter threaded through ``run_job -> cycle -> top_up ->
qualify`` — at most one verdict is ever in flight, and none of those layers has any
other reason to know a calling agent is answering instead of a model.

Two things can be answered: a candidate's **verdict** (``--verdict``/``--reason``) and,
once per store, the **ICP** — the cold start's opening keywords and ideal profiles
(``--icp``), which are the only other ``AI_MODEL`` calls a run makes.
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
_icp: ContextVar[object | None] = ContextVar("agent_qualify_icp", default=None)


def enable(verdict: Verdict | None, icp=None) -> None:
    """Turn agent-qualify mode on for this process, with optional resume answers.

    ``icp`` is a validated ``icp.IcpAnswer``, or ``None`` when none was given.
    """
    _active.set(True)
    _verdict.set(verdict)
    _icp.set(icp)


def active() -> bool:
    return _active.get()


def take_verdict() -> Verdict | None:
    """Consume the answer once — a second read must never re-apply it to a new candidate."""
    verdict = _verdict.get()
    _verdict.set(None)
    return verdict


def icp_answer():
    """The ``--icp`` answer, if one was given. Not consumed: the seed and the anchors
    are written at different moments of one run and both read it."""
    return _icp.get()
