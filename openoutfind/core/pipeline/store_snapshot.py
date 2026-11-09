# openoutfind/core/pipeline/store_snapshot.py
"""The store as one round of the walk reads it, once: every person indexed the way the
provider matches them, each person's fit rate, the verdicts, and the fired searches.

**A person's fit rate** is their verdict where judged, else the GP's score through an
isotonic map fitted on the store's verdicts at their out-of-fold scores
(``fit_rate_map``): the GP ranks well but is not calibrated (0.15 where the truth was
0.28), and a fit rate is what can be weighed against a cost.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from openoutfind.core.pipeline.analyzers import (
    company_tags, is_stopword_phrase, standalone_title_words, sub_phrases, title_words,
)
from openoutfind.core.pipeline.belief import query_keywords
from openoutfind.core.pipeline.terms import Terms, split_place

NOBODY: frozenset = frozenset()

# What a market reads as with nobody scored: nothing is known either way.
NEUTRAL_RATE = 0.5


# ── the fit rates and the verdicts ───────────────────────────────────

# The fitted map, under the fingerprint of the verdicts it was fitted on.
FIT_RATE_CACHE: tuple[str, object] | None = None


def fit_rate_map():
    """``scores → fit rates``: an isotonic regression of the store's verdicts on their
    out-of-fold scores, refitted when a verdict lands.

    The verdicts are the marks and the run's own, which the run chose by score — and
    choosing by score does not bias the rate *at* a score. Scores outside the verdicts'
    range clip to its ends. With one class of verdicts there is nothing to fit, and the
    score stands in.
    """
    global FIT_RATE_CACHE
    from openoutlearn.calibration import fit_rate_curve

    from openoutfind.core.ml.calibration import FOLDS
    from openoutfind.core.ml.qualifier import build_qualifier, label_fingerprint
    from openoutfind.crm.models import Lead

    key = label_fingerprint()
    if FIT_RATE_CACHE is None or FIT_RATE_CACHE[0] != key:
        ids, X, y = Lead.get_labeled_rows()
        curve = None
        if len(np.unique(y)) == 2:
            oof = build_qualifier(X, y).out_of_fold_probs(np.arange(len(ids)), k=FOLDS)
            if oof is not None:
                curve = fit_rate_curve(oof, y)
        FIT_RATE_CACHE = (key, curve)
    curve = FIT_RATE_CACHE[1]
    if curve is None:
        return lambda scores: np.asarray(scores, dtype=np.float64)
    return curve.predict


def lead_fit_rates(qualifier, fit_rate) -> tuple[dict[int, float], dict[int, float]]:
    """``(lead_id → fit rate, lead_id → confidence)`` for every stored person: their
    verdict if judged, else the GP's score through ``fit_rate`` and the GP's confidence
    there (``BayesianQualifier.confidence``). A person the GP cannot score is left out."""
    from openoutfind.crm.models import Deal, DealState, Lead, Outcome

    rates: dict[int, float] = {
        lead_id: 0.0 if (state == DealState.FAILED and outcome == Outcome.WRONG_FIT) else 1.0
        for lead_id, state, outcome in Deal.objects.filter(lead_id__isnull=False)
        .values_list("lead_id", "state", "outcome")
    }
    unjudged = list(Lead.objects.filter(synthetic=False, embedding__isnull=False)
                    .exclude(pk__in=list(rates)).values_list("pk", "embedding"))
    if qualifier is None or not unjudged:
        return rates, {}
    X = np.array([np.frombuffer(bytes(emb), dtype=np.float32) for _, emb in unjudged],
                 dtype=np.float64)
    probs = qualifier.predict_probs(X)
    if probs is None:
        return rates, {}
    pks = [pk for pk, _ in unjudged]
    rates.update(zip(pks, map(float, fit_rate(probs))))
    return rates, dict(zip(pks, map(float, qualifier.confidence(X))))


def judged_leads() -> set[int]:
    """Every lead holding a verdict, marks included."""
    from openoutfind.crm.models import Deal

    return set(Deal.objects.filter(lead_id__isnull=False).values_list("lead_id", flat=True))


def judged_fits() -> set[int]:
    """Every lead judged a fit: any deal but the ``wrong_fit`` rejection, marks included."""
    from openoutfind.crm.models import Deal, DealState, Outcome

    return set(Deal.objects.filter(lead_id__isnull=False)
               .exclude(state=DealState.FAILED, outcome=Outcome.WRONG_FIT)
               .values_list("lead_id", flat=True))


# ── the store's people ───────────────────────────────────────────────

class People:
    """The store's people, indexed per field the way the provider matches them: a title
    word, every contiguous phrase inside a tag, a state, a country, a seniority."""

    def __init__(self, rows, rates: dict[int, float], confidence: dict[int, float] | None = None):
        self.rates = rates
        self.confidence = confidence or {}
        self.company: dict[int, object] = {}
        self.everyone: set[int] = set()
        self.by_term: dict[tuple, set[int]] = defaultdict(set)
        self.titles_of: dict[int, set[str]] = {}
        self.tags_of: dict[int, set[tuple[str, ...]]] = {}
        self.states_by_country: dict[str, set[str]] = defaultdict(set)
        for pk, company_id, fields in rows:
            self.everyone.add(pk)
            self.company[pk] = company_key(pk, company_id, fields)
            for term in index_terms(fields):
                self.by_term[term].add(pk)
            self.add_to_vocabulary(pk, fields)

    @classmethod
    def load(cls, rates: dict[int, float],
             confidence: dict[int, float] | None = None) -> "People":
        from openoutfind.crm.models import Lead

        return cls(Lead.objects.filter(synthetic=False).exclude(source_fields={})
                   .values_list("pk", "company_id", "source_fields"), rates, confidence)

    def add_to_vocabulary(self, pk: int, fields: dict) -> None:
        self.titles_of[pk] = standalone_title_words(fields.get("contact_job_title", ""))
        self.tags_of[pk] = set(company_tags(fields.get("company_keywords", "")))
        country = str(fields.get("contact_location_country") or "").strip().lower()
        state = str(fields.get("contact_location_state") or "").strip().lower()
        if country and state:
            self.states_by_country[country].add(state)

    # ── the vocabulary ──

    def title_vocabulary(self, among: set[int]) -> set[str]:
        """Every title word ``among`` hold on its own: never a piece of a hyphenated one,
        since `co` alone reaches co-owners and co-CEOs."""
        return {word for pk in among for word in self.titles_of.get(pk, ())}

    def tag_vocabulary(self, among: set[int]) -> set[tuple[str, ...]]:
        """Every whole tag ``among`` hold, never a phrase inside one (``pre`` from
        ``pre-rolls`` reached founders of pre-seed funds) and never stopwords alone."""
        return {tag for pk in among for tag in self.tags_of.get(pk, ())
                if not is_stopword_phrase(tag)}

    def states_in(self, country: str) -> set[str]:
        """The states the store's people live in inside ``country``."""
        return set(self.states_by_country.get(country.lower(), ()))

    # ── matching ──

    def match(self, terms: Terms) -> set[int]:
        """Every stored person matching every term of ``terms`` — what ``Terms.matches``
        says over everyone — by intersecting the per-term sets."""
        keys = [("title", word) for word in terms.title]
        if terms.tag:
            keys.append(("tag", terms.tag))
        if terms.seniority:
            keys.append(("seniority", terms.seniority))
        if terms.place:
            state, country = split_place(terms.place)
            keys.append(("country", country))
        if not keys:
            return set(self.everyone)
        sets = [self.by_term.get(key, NOBODY) for key in keys]
        if terms.place and state:
            sets.append(self.state_holders(state, country))
        sets.sort(key=len)
        return set(sets[0]).intersection(*sets[1:])

    def holding(self, keyword: tuple) -> set[int]:
        """Everyone holding ``keyword`` (``belief.query_keywords``)."""
        if keyword[0] == "state":
            _, state, country = keyword
            return set(self.state_holders(state, country))
        return set(self.by_term.get(keyword, NOBODY))

    def state_holders(self, state: str, country: str) -> set[int]:
        """A state's own rows and its country's rows with no state: the provider returns
        a state on US rows alone, so a stateless row may live in any state."""
        return self.by_term.get(("state", state), NOBODY) | self.by_term.get(
            ("no state", country), NOBODY)

    # ── the rates ──

    def confidence_in(self, pk: int) -> float:
        """How far a person's rate is to be believed: the GP's confidence at them, or 1
        for a rate given with none (a verdict, or a rate set by hand)."""
        return self.confidence.get(pk, 1.0)

    def base_rate(self) -> float:
        """The store's own rate: every scored person, per company, then over companies."""
        by_company: dict = defaultdict(list)
        for pk, rate in self.rates.items():
            if pk in self.company:
                by_company[self.company[pk]].append(rate)
        if not by_company:
            return NEUTRAL_RATE
        return statistics.fmean(statistics.fmean(rates) for rates in by_company.values())


def company_key(pk: int, company_id, fields: dict):
    """Who a person counts with: their company, or with no company the people sharing
    their tag list — the provider's row for "Self employed" staples one list onto
    hundreds of strangers — or alone."""
    if company_id is not None:
        return company_id
    keywords = str(fields.get("company_keywords") or "").strip().lower()
    return ("tags", keywords) if keywords else ("alone", pk)


def index_terms(fields: dict):
    """The ``(field, term)`` keys a person is indexed under."""
    for word in title_words(fields.get("contact_job_title", "")):
        yield ("title", word)
    for tag in company_tags(fields.get("company_keywords", "")):
        for phrase in sub_phrases(tag):
            yield ("tag", phrase)
    if seniority := str(fields.get("contact_seniority") or "").strip().lower():
        yield ("seniority", seniority)
    if country := str(fields.get("contact_location_country") or "").strip().lower():
        yield ("country", country)
    if state := str(fields.get("contact_location_state") or "").strip().lower():
        yield ("state", state)
    elif country:
        yield ("no state", country)


# ── the fired searches ───────────────────────────────────────────────

@dataclass
class FiredSearch:
    """A search with pages: its latest, everyone its pages returned, and who its latest
    page returned and was first to bring in."""

    search: object
    latest: object
    people: set[int]
    latest_people: set[int]
    fresh: set[int]

    @property
    def terms(self) -> Terms:
        return self.search.query

    @property
    def keywords(self) -> tuple:
        return query_keywords(self.terms)

    @property
    def found(self) -> int | None:
        """The provider's count."""
        return self.search.leads_found

    @property
    def drained(self) -> bool:
        """Whether its pages returned everything it counts."""
        from openoutfind.core.models import PAGE_SIZE

        return self.found is not None and self.latest.number * PAGE_SIZE >= self.found


def fired_searches() -> list[FiredSearch]:
    """Every search with pages, as of its latest."""
    from openoutfind.core.models import Page, PageLead

    latest: dict[int, object] = {}
    for page in Page.objects.select_related("search").order_by("number"):
        latest[page.search_id] = page
    people: dict[int, set] = defaultdict(set)
    on_page: dict[int, set] = defaultdict(set)
    first_page: dict[int, int] = {}
    for page_id, search_id, lead_id in (PageLead.objects.order_by("page_id")
                                        .values_list("page_id", "page__search_id", "lead_id")):
        people[search_id].add(lead_id)
        on_page[page_id].add(lead_id)
        first_page.setdefault(lead_id, page_id)
    return [FiredSearch(page.search, page, people[search_id], on_page[page.pk],
                        {pk for pk in on_page[page.pk] if first_page[pk] == page.pk})
            for search_id, page in latest.items()]
