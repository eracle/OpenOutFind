# openoutfind/core/pipeline/fitness.py
"""A page's fitness — how good the people on that page are, and nothing else.

A fired individual (a search at a page) is scored on **its own latest page alone**: the
people that page brought in who match all its terms (``query.Terms.matches``), each
counting their verdict if judged, else the GP's ``P(f > 0.5)``. The people of one company
are averaged first, then the companies — a person with no company is their own company.
Nothing comes from its parents or its earlier pages, and nothing is guessed for a search
that has not been fired.

- **Per company**, because pages arrive grouped by company (a median 1.57 people per
  company per page), people at one company share its tags and mostly its verdict, and on
  the held 500-lead store counting each company once ranked better than counting people.
- **A ranking, not a fit rate.** The GP orders people but is not calibrated (0.15 where
  the truth was 0.28), so a fitness is only ever compared with another fitness.
- **Recomputed every generation**, as the GP retrains on new verdicts: a calculation over
  the page–lead table, not a request.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def verdicts(lead_ids=None) -> dict[int, int]:
    """``lead_id → 1|0`` for every judged lead: 0 is an LLM rejection, any other deal 1."""
    from openoutfind.crm.models import Deal, DealState, Outcome

    deals = Deal.objects.filter(lead_id__isnull=False)
    if lead_ids is not None:
        deals = deals.filter(lead_id__in=lead_ids)
    return {
        lead_id: 0 if (state == DealState.FAILED and outcome == Outcome.WRONG_FIT) else 1
        for lead_id, state, outcome in deals.values_list("lead_id", "state", "outcome")
    }


def lead_scores(qualifier, lead_ids) -> dict[int, float]:
    """Each lead's verdict if judged, else the GP's ``P(f > 0.5)``.

    A lead the GP cannot score (no qualifier, or no embedding) is left out, so it counts
    on no page until it can.
    """
    from openoutfind.crm.models import Lead

    lead_ids = set(lead_ids)
    scores: dict[int, float] = {pk: float(v) for pk, v in verdicts(lead_ids).items()}
    unjudged = [(pk, emb) for pk, emb in
                Lead.objects.filter(pk__in=lead_ids - scores.keys(), embedding__isnull=False)
                .values_list("pk", "embedding")]
    if qualifier is None or not unjudged:
        return scores
    X = np.array([np.frombuffer(bytes(emb), dtype=np.float32) for _, emb in unjudged],
                 dtype=np.float64)
    probs = qualifier.predict_probs(X)
    if probs is not None:
        scores.update((pk, float(p)) for (pk, _), p in zip(unjudged, probs))
    return scores


def company_mean(people: list[tuple[int | None, float]]) -> float | None:
    """The mean over companies of each company's mean score. ``None`` with nobody.

    ``people`` is ``(company_id, score)``; a ``None`` company is a company of one.
    """
    by_company: dict = defaultdict(list)
    for index, (company_id, score) in enumerate(people):
        by_company[company_id if company_id is not None else ("alone", index)].append(score)
    if not by_company:
        return None
    return float(np.mean([np.mean(scores) for scores in by_company.values()]))


def page_fitness(pages, qualifier) -> dict[int, float]:
    """``page_id → fitness`` for each page — 0.0 for a page that holds nobody it matches.

    One GP prediction over every person on these pages.
    """
    from openoutfind.core.models import PageLead

    rows = list(
        PageLead.objects.filter(page__in=pages)
        .values_list("page_id", "lead_id", "lead__company_id", "lead__source_fields")
    )
    scores = lead_scores(qualifier, {lead_id for _, lead_id, _, _ in rows})
    terms = {page.pk: page.search.query for page in pages}

    people: dict[int, list] = defaultdict(list)
    for page_id, lead_id, company_id, fields in rows:
        if lead_id in scores and terms[page_id].matches(fields):
            people[page_id].append((company_id, scores[lead_id]))
    return {page.pk: company_mean(people[page.pk]) or 0.0 for page in pages}
