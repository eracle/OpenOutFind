"""``find --labels FILE`` — a customer's own marks, taken in as verdicts.

A store starts from a guess: opening keywords somebody wrote and three invented ideal
people. A customer who has already marked the leads they were sent has said who fits far
better than that guess can, and each of those marks names a real profile. So a fresh store
takes them in before its first search: every mark becomes a ``Lead`` with a ``Deal``
holding the verdict, and the ranking trains on it, the vocabulary grows from the fits'
words and the walk counts each mark against the keywords that match it — all through the
queries they already run, because a mark is stored exactly as a verdict is.

**The file is the hub's ``/ratings/`` as it comes**: a JSON array, or the same objects as
JSON Lines, each with ``linkedin_url``, ``value``, ``value_reason``, ``note`` and the
person as ``find --json`` printed them (``profile_text``, ``source_fields``, the size
band, ``full_name``, ``title``, ``company``). OpenOutFind never calls the hub: the marks
come in as a file, so anyone can start a store from their own.

**A marked lead is never judged again, never printed and never sent to the paid lookup.**
``Lead.marked`` says so, and the export, the promote gate and the counts read it. The
customer already has these people; what the store wants from them is the label.

**Loading is idempotent and happens once, on a fresh store.** The caller passes the same
flag on every call of an ``--agent-qualify`` loop, so a mark already in the store is
skipped; a mark not yet in it is refused once the store holds verdicts of its own,
because a store's walk has already been scored without it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from django.db import transaction

from openoutfind.core.errors import ErrorType, OpenOutFindError

logger = logging.getLogger(__name__)

# The hub's ``Label.Reason`` values, as the reason on a no-fit reads them.
_REASON_TEXT = {
    "wrong_company": "wrong kind of company",
    "too_big": "company too big",
    "wrong_role": "wrong person or role",
    "other": "another reason",
}
# *Already a customer or contact* is the right kind of profile, marked "not a fit" only
# because the customer will not pay for someone they have. *Gone* says nothing about fit.
_FIT_REASONS = frozenset({"already_known"})
_UNSAID_REASONS = frozenset({"gone"})


class LoadedLabels(NamedTuple):
    """What ``--labels`` did with the file — the run reports it."""

    fits: int
    no_fits: int
    left_out: int
    already_in: int
    no_profile: list[str]

    def as_dict(self) -> dict:
        return self._asdict()


# ── the file ─────────────────────────────────────────────────────


def read_labels_file(path: str | Path) -> list[dict]:
    """The marks in *path*: a JSON array of objects, or one object per line.

    Raises ``OSError`` when the file cannot be read and ``ValueError`` when it is not
    marks — the caller decides how to say so.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        marks = json.loads(text)
    else:
        marks = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(mark, dict) and mark.get("linkedin_url") for mark in marks):
        raise ValueError("every mark needs to be an object with a linkedin_url")
    return marks


# ── one mark ─────────────────────────────────────────────────────


def verdict_for(mark: dict) -> bool | None:
    """The mark as a verdict: ``True`` fit, ``False`` no-fit, ``None`` left out.

    *Not sure*, a blank mark and *gone* say nothing about who fits.
    """
    value, reason = mark.get("value"), mark.get("value_reason") or ""
    if value == "good":
        return True
    if value != "not_fit" or reason in _UNSAID_REASONS:
        return None
    return reason in _FIT_REASONS


def reason_for(mark: dict, fit: bool) -> str:
    """The verdict's reason, in the customer's terms, their note after it."""
    reason = mark.get("value_reason") or ""
    if fit:
        said = ("already a customer or contact" if reason in _FIT_REASONS else "good")
    elif reason in _REASON_TEXT:
        said = f"not a fit: {_REASON_TEXT[reason]}"
    else:
        said = "not a fit"
    note = (mark.get("note") or "").strip()
    text = f"The customer marked them {said}."
    return f'{text} Their note: "{note}"' if note else text


# ── loading ──────────────────────────────────────────────────────


def load(marks: list[dict]) -> LoadedLabels:
    """Store every mark that says who fits as a verdict on a marked lead.

    A URL marked twice (a lead shown again in a later batch) keeps its latest mark,
    since the hub lists them oldest first. A mark with no ``profile_text`` is listed in
    the result rather than stored: a verdict on a profile nobody can read trains on
    nothing.
    """
    from openoutfind.core.exclude import normalize_linkedin_url

    latest = {normalize_linkedin_url(mark["linkedin_url"]): mark for mark in marks}
    said = {url: (mark, verdict_for(mark)) for url, mark in latest.items()}
    counted = {url: pair for url, pair in said.items() if pair[1] is not None}
    left_out = len(said) - len(counted)

    no_profile = sorted(url for url, (mark, _) in counted.items()
                        if not (mark.get("profile_text") or "").strip())
    readable = {url: pair for url, pair in counted.items() if url not in no_profile}
    in_store = _urls_with_a_verdict(readable)
    to_load = {url: pair for url, pair in readable.items() if url not in in_store}

    if to_load:
        _refuse_a_used_store()
        _store(to_load)

    fits = sum(fit for _, fit in to_load.values())
    loaded = LoadedLabels(fits=fits, no_fits=len(to_load) - fits, left_out=left_out,
                          already_in=len(in_store), no_profile=no_profile)
    _log(loaded)
    return loaded


def _urls_with_a_verdict(marks: dict) -> set[str]:
    """Which of these URLs already carry a verdict in this store."""
    from openoutfind.crm.models import Deal

    return set(Deal.objects.filter(lead__profile_url__in=list(marks))
               .values_list("lead__profile_url", flat=True))


def _refuse_a_used_store() -> None:
    """Refuse to add marks to a store that has judged leads of its own."""
    from openoutfind.crm.models import Deal

    own = Deal.objects.filter(lead__marked=False).count()
    if own:
        raise OpenOutFindError(
            ErrorType.BAD_CONFIG,
            f"--labels starts a fresh store, and this one already holds {own} verdict(s) "
            "of its own — point --db at a new file",
        )


def _store(marks: dict) -> None:
    """Write each mark as a marked Lead and a Deal holding its verdict, all or none.

    The embeddings are computed first, outside the transaction: they are the slow part
    and touch no row.
    """
    from openoutfind.core.ml.embeddings import embed_texts

    urls = list(marks)
    vectors = embed_texts([marks[url][0]["profile_text"] for url in urls])
    with transaction.atomic():
        for url, vector in zip(urls, vectors):
            mark, fit = marks[url]
            _store_one(url, mark, fit, vector)


def _store_one(url: str, mark: dict, fit: bool, vector: np.ndarray) -> None:
    """One mark as a Lead (an existing un-judged one is taken over) and its Deal."""
    from openoutfind.crm.models import (
        Company, Deal, DealState, Lead, Outcome, PendingQualification,
    )

    company = Company.from_row(
        (mark.get("company") or "").strip() or None, None,
        mark.get("employees_min"), mark.get("employees_max"))
    lead, _ = Lead.objects.update_or_create(
        profile_url=url,
        defaults={
            "profile_text": mark["profile_text"],
            "source_fields": mark.get("source_fields") or {},
            "embedding": np.asarray(vector, dtype=np.float32).tobytes(),
            "full_name": (mark.get("full_name") or "").strip() or None,
            "job_title": (mark.get("title") or "").strip() or None,
            "company": company,
            "marked": True,
        },
    )
    # A lead an earlier call put up for a verdict is answered by the mark instead.
    PendingQualification.objects.filter(lead=lead).delete()
    Deal.objects.create(
        lead=lead,
        state=DealState.QUALIFIED if fit else DealState.FAILED,
        outcome="" if fit else Outcome.WRONG_FIT,
        reason=reason_for(mark, fit),
    )


def _log(loaded: LoadedLabels) -> None:
    """Say what the marks became, and name every one that could not be read."""
    logger.info("Customer's marks: %d fit · %d not a fit · %d left out (not sure, gone)"
                " · %d already in this store",
                loaded.fits, loaded.no_fits, loaded.left_out, loaded.already_in)
    for url in loaded.no_profile:
        logger.info("    no profile, not loaded: %s", url)
