# tests/test_agent_qualify.py
"""``find --agent-qualify`` — the calling agent answers instead of AI_MODEL.

Covers the pipeline layer only (``core.agent_qualify`` + the branch in
``run_qualification``); the CLI's own flag parsing and error rendering are covered by
``tests/test_find.py``.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core import agent_qualify
from openoutfind.core.ml.qualifier import BayesianQualifier
from openoutfind.core.pipeline.qualify import QualifyPending, run_qualification


def _make_lead(profile_url="https://www.linkedin.com/in/alice/", profile_text="engineer at acme"):
    from openoutfind.crm.models import Lead

    return Lead.objects.create(
        profile_url=profile_url,
        profile_text=profile_text,
        embedding=np.ones(384, dtype=np.float32).tobytes(),
    )


@pytest.fixture(autouse=True)
def _reset_agent_qualify():
    """A contextvar carries state past a test's own scope otherwise."""
    yield
    agent_qualify._active.set(False)
    agent_qualify._verdict.set(None)
    agent_qualify._population.set(None)


def _searches(n=15):
    return [{"title": f"founder baker {i}" if i else "head of purchasing", "tag": "bakery",
             "state": "California" if i % 2 else "", "seniority": None} for i in range(n)]


def _population_answer():
    from openoutfind.core.pipeline.icp import ANCHOR_COUNT, PopulationAnswer

    return PopulationAnswer.model_validate({
        "searches": _searches(), "country": "United States", "country_code": "US",
        "headcount_min": 1, "headcount_max": 50,
        "anchors": [{"profile": f"founder of bakery chain {n} united states",
                     "job_title": "founder", "location_country": "united states"}
                    for n in range(ANCHOR_COUNT)],
    })


@pytest.mark.django_db
class TestWritingThePopulation:
    """The cold start's two LLM calls — the searches and the anchors — answered by the agent."""

    def test_a_fresh_store_stops_for_the_population_instead_of_asking_the_llm(self, site_config):
        from openoutfind.core.pipeline.icp import write_initial_population
        from openoutfind.core.pipeline.qualify import PopulationPending

        agent_qualify.enable(None)

        with patch("openoutfind.core.pipeline.icp._population_from_llm") as llm:
            with pytest.raises(PopulationPending) as raised:
                write_initial_population(site_config)

        llm.assert_not_called()
        assert raised.value.error_type == "population_pending"
        payload = raised.value.payload
        assert payload["product_docs"] == site_config.product_docs
        assert payload["population_size"] == 15
        assert set(payload["fields"]) == {"title", "tag", "state", "seniority"}
        assert "searches" in payload["schema"]["properties"]

    def test_the_answer_becomes_the_written_searches_and_their_shape(self, site_config):
        from openoutfind.core.models import Search
        from openoutfind.core.pipeline.icp import write_initial_population

        agent_qualify.enable(None, _population_answer())

        assert write_initial_population(site_config) == 15

        first = Search.objects.get(terms__title=["head", "purchasing"])
        assert first.origin == Search.Origin.WRITTEN
        assert first.query.tag == ("bakery",)
        assert first.query.place == "United States"
        assert (first.headcount, first.country_code) == ((1, 50), "us")
        assert Search.objects.filter(terms__place="California, United States").count() == 7

    def test_the_answer_becomes_the_anchors(self, site_config):
        from openoutfind.core.pipeline.icp import generate_anchors

        agent_qualify.enable(None, _population_answer())

        anchors = generate_anchors(site_config, count=2)

        assert [a.profile for a in anchors] == [
            "founder of bakery chain 0 united states", "founder of bakery chain 1 united states"]

    def test_a_missing_answer_is_not_swallowed_as_an_anchor_outage(self, site_config):
        """Anchor generation is best-effort and eats LLM failures; the question to the
        agent must not be eaten with them."""
        from openoutfind.core.pipeline.icp import generate_anchors
        from openoutfind.core.pipeline.qualify import PopulationPending

        agent_qualify.enable(None)

        with pytest.raises(PopulationPending):
            generate_anchors(site_config)

    def test_too_few_anchors_or_searches_is_not_an_answer(self):
        from pydantic import ValidationError

        from openoutfind.core.pipeline.icp import PopulationAnswer

        base = {"country": "United States", "searches": _searches(),
                "anchors": [{"profile": "one"}]}
        with pytest.raises(ValidationError):
            PopulationAnswer.model_validate(base)
        with pytest.raises(ValidationError):
            PopulationAnswer.model_validate({**base, "searches": _searches(14),
                                             "anchors": [{"profile": "a"}] * 3})

    def test_without_agent_mode_the_llm_writes_the_searches(self, site_config):
        from openoutfind.core.models import Search
        from openoutfind.core.pipeline.icp import InitialPopulation, write_initial_population

        population = InitialPopulation(searches=_searches(), country="United States")
        with patch("openoutfind.core.pipeline.icp._population_from_llm",
                   return_value=population) as llm:
            write_initial_population(site_config)

        llm.assert_called_once()
        assert Search.objects.count() == 15

    def test_an_illegal_written_search_is_dropped(self, site_config):
        """A tag with no title words left once stopwords go cannot be fired."""
        from openoutfind.core.models import Search
        from openoutfind.core.pipeline.icp import InitialPopulation, write_initial_population

        searches = _searches()
        searches[0] = {"title": "of the", "tag": "bakery"}
        population = InitialPopulation(searches=searches, country="United States")
        with patch("openoutfind.core.pipeline.icp._population_from_llm",
                   return_value=population):
            assert write_initial_population(site_config) == 14
        assert not Search.objects.filter(terms__title=[]).exists()


@pytest.mark.django_db
class TestStoppingForAVerdict:
    def test_a_fresh_candidate_is_handed_back_instead_of_asking_the_llm(self, site_config):
        _make_lead(profile_text="engineer at acme, hiring for growth")
        agent_qualify.enable(None)
        qualifier = BayesianQualifier(seed=42)

        with patch("openoutfind.core.ml.qualifier.qualify_with_llm") as mock_llm:
            with pytest.raises(QualifyPending) as raised:
                run_qualification(site_config, qualifier)

        mock_llm.assert_not_called()
        assert raised.value.error_type == "qualify_pending"
        assert raised.value.payload["profile_text"] == "engineer at acme, hiring for growth"

        from openoutfind.crm.models import PendingQualification
        assert PendingQualification.objects.count() == 1

    def test_re_running_with_no_answer_re_asks_about_the_same_candidate(self, site_config):
        lead = _make_lead()
        from openoutfind.crm.models import PendingQualification
        PendingQualification.objects.create(lead=lead)
        agent_qualify.enable(None)
        qualifier = BayesianQualifier(seed=42)

        with pytest.raises(QualifyPending) as raised:
            run_qualification(site_config, qualifier)

        assert raised.value.payload["profile_url"] == lead.profile_url
        assert PendingQualification.objects.count() == 1


@pytest.mark.django_db
class TestResumingWithAnAnswer:
    def test_a_fit_verdict_promotes_the_lead(self, site_config):
        lead = _make_lead()
        from openoutfind.crm.models import PendingQualification
        PendingQualification.objects.create(lead=lead)
        agent_qualify.enable(agent_qualify.Verdict(fit=True, reason="Matches the ICP"))
        qualifier = BayesianQualifier(seed=42)

        with (
            patch("openoutfind.core.ml.qualifier.qualify_with_llm") as mock_llm,
            patch("openoutfind.core.db.leads.promote_lead_to_deal") as mock_promote,
        ):
            result = run_qualification(site_config, qualifier)

        mock_llm.assert_not_called()
        mock_promote.assert_called_once_with(lead.profile_url, reason="Matches the ICP")
        assert result == lead.profile_url
        assert PendingQualification.objects.count() == 0

    def test_a_no_fit_verdict_disqualifies_the_lead(self, site_config):
        lead = _make_lead()
        from openoutfind.crm.models import PendingQualification
        PendingQualification.objects.create(lead=lead)
        agent_qualify.enable(agent_qualify.Verdict(fit=False, reason="Wrong market"))
        qualifier = BayesianQualifier(seed=42)

        with (
            patch("openoutfind.core.ml.qualifier.qualify_with_llm") as mock_llm,
            patch("openoutfind.core.db.deals.create_disqualified_deal") as mock_disqualify,
        ):
            result = run_qualification(site_config, qualifier)

        mock_llm.assert_not_called()
        mock_disqualify.assert_called_once_with(lead.profile_url, reason="Wrong market")
        assert result == lead.profile_url
        assert PendingQualification.objects.count() == 0

    def test_the_answer_is_consumed_once(self, site_config):
        """A second read must not re-apply the same verdict to a different candidate."""
        agent_qualify.enable(agent_qualify.Verdict(fit=True, reason="ok"))

        first = agent_qualify.take_verdict()
        second = agent_qualify.take_verdict()

        assert first is not None
        assert second is None
