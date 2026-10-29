# tests/test_fitness.py
"""A page's fitness — its own people only, verdict or GP, each company once."""
import numpy as np
import pytest

from openoutfind.core.models import Page, PageLead, Search
from openoutfind.core.pipeline import fitness
from openoutfind.core.pipeline.query import Terms
from openoutfind.crm.models import Company, Deal, DealState, Lead, Outcome


class _Scores:
    """A GP whose P is the embedding's first value."""

    def predict_probs(self, X):
        return X[:, 0]


def _page(title="purchasing", number=1):
    terms = Terms.written(title=title)
    search, _ = Search.objects.get_or_create(key=terms.key, defaults={"terms": terms.as_json()})
    return Page.objects.create(search=search, number=number, generation=0, rows=1)


def _person(page, p=None, title="Purchasing Manager", company=None, verdict=None):
    lead = Lead.objects.create(
        profile_url=f"https://x/{Lead.objects.count()}", company=company,
        source_fields={"contact_job_title": title},
        embedding=None if p is None else np.full(384, p, dtype=np.float32).tobytes())
    if verdict is not None:
        Deal.objects.create(lead=lead, state=DealState.QUALIFIED if verdict else DealState.FAILED,
                            outcome="" if verdict else Outcome.WRONG_FIT)
    PageLead.objects.create(page=page, lead=lead)
    return lead


class TestCompanyMean:
    def test_companies_are_averaged_first(self):
        """Three at one company that miss, one alone that fits: 0.5, not 0.25."""
        assert fitness.company_mean([(1, 0.0), (1, 0.0), (1, 0.0), (2, 1.0)]) == 0.5

    def test_a_person_with_no_company_is_their_own(self):
        assert fitness.company_mean([(None, 0.0), (None, 1.0)]) == 0.5

    def test_nobody_is_none(self):
        assert fitness.company_mean([]) is None


@pytest.mark.django_db
class TestPageFitness:
    def test_a_verdict_beats_the_gp(self):
        page = _page()
        _person(page, p=0.1, verdict=1)

        assert fitness.page_fitness([page], _Scores()) == {page.pk: 1.0}

    def test_the_unjudged_count_as_the_gp_scores_them(self):
        page = _page()
        _person(page, p=0.2)
        _person(page, p=0.4)

        assert fitness.page_fitness([page], _Scores())[page.pk] == pytest.approx(0.3)

    def test_people_who_do_not_match_the_search_are_left_out(self):
        """The provider's fuzzy match brings people the search did not ask for."""
        page = _page()
        _person(page, p=0.2)
        _person(page, p=0.9, title="Chocolatier")

        assert fitness.page_fitness([page], _Scores())[page.pk] == pytest.approx(0.2)

    def test_one_company_counts_once(self):
        acme = Company.objects.create(key="acme.com", name="Acme")
        page = _page()
        for _ in range(3):
            _person(page, p=0.0, company=acme)
        _person(page, p=1.0)

        assert fitness.page_fitness([page], _Scores())[page.pk] == pytest.approx(0.5)

    def test_only_the_pages_own_people(self):
        """A deeper page of the same search is scored on its own people, not page 1's."""
        first, second = _page(number=1), _page(number=2)
        _person(first, p=0.9)
        _person(second, p=0.1)

        scores = fitness.page_fitness([first, second], _Scores())

        assert scores == {first.pk: pytest.approx(0.9), second.pk: pytest.approx(0.1)}

    def test_a_person_on_two_pages_counts_on_each(self):
        a, b = _page("purchasing"), _page("purchasing manager")
        lead = _person(a, p=0.7)
        PageLead.objects.create(page=b, lead=lead)

        assert fitness.page_fitness([a, b], _Scores()) == {
            a.pk: pytest.approx(0.7), b.pk: pytest.approx(0.7)}

    def test_a_page_holding_nobody_it_matches_scores_zero(self):
        page = _page()
        _person(page, p=0.9, title="Chocolatier")

        assert fitness.page_fitness([page], _Scores()) == {page.pk: 0.0}

    def test_without_a_model_only_judged_people_count(self):
        page = _page()
        _person(page, p=0.9)
        _person(page, verdict=0)

        assert fitness.page_fitness([page], None) == {page.pk: 0.0}
