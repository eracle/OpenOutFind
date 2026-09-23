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
    agent_qualify._icp.set(None)


def _icp_answer():
    from openoutfind.core.pipeline.icp import ANCHOR_COUNT, IcpAnswer

    return IcpAnswer.model_validate({
        "seed": {"role_keywords": ["founder"], "domain_keywords": ["bakery"],
                 "location": "united states", "headcount_min": 1, "headcount_max": 50},
        "anchors": [{"profile": f"founder of bakery chain {n} united states",
                     "job_title": "founder", "location_country": "united states"}
                    for n in range(ANCHOR_COUNT)],
    })


@pytest.mark.django_db
class TestWritingTheIcp:
    """The cold start's two LLM calls — the seed and the anchors — answered by the agent."""

    def test_a_fresh_store_stops_for_the_icp_instead_of_asking_the_llm(self, site_config):
        from openoutfind.core.pipeline.icp import generate_seed
        from openoutfind.core.pipeline.qualify import IcpPending

        agent_qualify.enable(None)

        with patch("openoutfind.core.pipeline.icp._spec_from_llm") as llm:
            with pytest.raises(IcpPending) as raised:
                generate_seed(site_config)

        llm.assert_not_called()
        assert raised.value.error_type == "icp_pending"
        assert raised.value.payload["product_docs"] == site_config.product_docs
        assert "seed" in raised.value.payload["schema"]["properties"]

    def test_the_answer_becomes_the_seed(self, site_config):
        from openoutfind.core.pipeline.icp import generate_seed

        agent_qualify.enable(None, _icp_answer())

        seed = generate_seed(site_config)

        assert ("lead_job_title", "bakery") in seed.keywords
        assert ("lead_location", "United States") in seed.keywords
        assert seed.headcount == (1, 50)

    def test_the_answer_becomes_the_anchors(self, site_config):
        from openoutfind.core.pipeline.icp import generate_anchors

        agent_qualify.enable(None, _icp_answer())

        anchors = generate_anchors(site_config, count=2)

        assert [a.profile for a in anchors] == [
            "founder of bakery chain 0 united states", "founder of bakery chain 1 united states"]

    def test_a_missing_answer_is_not_swallowed_as_an_anchor_outage(self, site_config):
        """Anchor generation is best-effort and eats LLM failures; the question to the
        agent must not be eaten with them."""
        from openoutfind.core.pipeline.icp import generate_anchors
        from openoutfind.core.pipeline.qualify import IcpPending

        agent_qualify.enable(None)

        with pytest.raises(IcpPending):
            generate_anchors(site_config)

    def test_too_few_anchors_is_not_an_answer(self):
        from pydantic import ValidationError

        from openoutfind.core.pipeline.icp import IcpAnswer

        with pytest.raises(ValidationError):
            IcpAnswer.model_validate({"seed": {}, "anchors": [{"profile": "one"}]})

    def test_without_agent_mode_the_llm_writes_the_seed(self, site_config):
        from openoutfind.core.pipeline.icp import ICPSpec, generate_seed

        with patch("openoutfind.core.pipeline.icp._spec_from_llm",
                   return_value=ICPSpec(role_keywords=["cto"])) as llm:
            seed = generate_seed(site_config)

        llm.assert_called_once()
        assert ("lead_job_title", "cto") in seed.keywords


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
