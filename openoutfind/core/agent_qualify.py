"""State for ``find --agent-qualify``, read where a call to ``AI_MODEL`` would fire.

A contextvar rather than a parameter threaded through ``run_job -> cycle -> top_up ->
qualify`` — at most one verdict is ever in flight, and none of those layers has any
other reason to know a calling agent is answering instead of a model.

Three things can be answered: a candidate's **verdict** (``--verdict``/``--reason``);
once per store, the **population** — the walk's initial searches and the ideal profiles
(``--population``); and the **keyword guesses** (``--keyword-guesses``), asked whenever
the market or the walk's vocabulary holds keywords with none. Those are every ``AI_MODEL`` call a run
makes.
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
_population: ContextVar[object | None] = ContextVar("agent_qualify_population",
                                                    default=None)


_keyword_guesses: ContextVar[object | None] = ContextVar("agent_qualify_keyword_guesses",
                                                         default=None)


def enable(verdict: Verdict | None, population=None, keyword_guesses=None) -> None:
    """Turn agent-qualify mode on for this process, with optional resume answers.

    ``population`` is a validated ``icp.PopulationAnswer`` and ``keyword_guesses`` a
    validated ``keyword_guess.KeywordGuesses``, or ``None`` when none was given.
    """
    _active.set(True)
    _verdict.set(verdict)
    _population.set(population)
    _keyword_guesses.set(keyword_guesses)


def active() -> bool:
    return _active.get()


def take_verdict() -> Verdict | None:
    """Consume the answer once — a second read must never re-apply it to a new candidate."""
    verdict = _verdict.get()
    _verdict.set(None)
    return verdict


def population_answer():
    """The ``--population`` answer, if one was given. Not consumed: the searches and the
    anchors are written at different moments of one run and both read it."""
    return _population.get()


def keyword_guesses_answer():
    """The ``--keyword-guesses`` answer, if one was given. Not consumed: storing it again
    is a no-op."""
    return _keyword_guesses.get()
