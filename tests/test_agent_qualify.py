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
