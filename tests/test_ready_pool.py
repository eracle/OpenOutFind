# tests/test_ready_pool.py
"""Find-email pool: the GP rank gate promoting QUALIFIED → READY_TO_FIND_EMAIL."""
import pytest
from unittest.mock import patch

import numpy as np

from openoutfind.core.db.deals import set_profile_state
from openoutfind.core.db.leads import promote_lead_to_deal
from openoutfind.core.ml.qualifier import BayesianQualifier
from openoutfind.core.pipeline.ready_pool import promote_to_ready, find_ready_candidate
from openoutfind.crm.models import DealState


def _fitted_qualifier():
    """A genuinely fitted qualifier that scores an all-ones embedding ~1.

    Two observations at the two poles of the space the test leads live in is the
    minimum the GP will fit on, but not enough to clear ``min_gp_confidence``: the
    posterior over a single positive is wide, and P(f>0.5) lands at 0.86. Two of each
    pole tightens it to 0.97 — still the same two poles, with the confidence the gate
    is calibrated for.
    """
    scorer = BayesianQualifier(seed=42)
    scorer.warm_start(
        np.array([np.ones(384), np.ones(384), np.zeros(384), np.zeros(384)], dtype=np.float64),
        np.array([1, 1, 0, 0]),
    )
    return scorer


def _make_qualified(site_config, slug="alice"):
    """Create an embedded Lead and a QUALIFIED Deal for it. Returns the profile_url."""
    from openoutfind.crm.models import Lead

    url = f"https://www.linkedin.com/in/{slug}/"
    Lead.objects.create(
        profile_url=url,
        profile_text="engineer at acme",
        embedding=np.ones(384, dtype=np.float32).tobytes(),
    )
    promote_lead_to_deal(url)
    return url


@pytest.mark.django_db
class TestPromoteToReady:
    def test_promotes_above_threshold(self, site_config):
        alice_url = _make_qualified(site_config, "alice")
        bob_url = _make_qualified(site_config, "bob")

        scorer = BayesianQualifier(seed=42)

        with patch.object(scorer, "predict_probs", return_value=np.array([0.95, 0.60])):
            count = promote_to_ready(scorer)

        assert count == 1

        from openoutfind.crm.models import Deal
        alice_deal = Deal.objects.get(lead__profile_url=alice_url)
        bob_deal = Deal.objects.get(lead__profile_url=bob_url)
        assert alice_deal.state == DealState.READY_TO_FIND_EMAIL
        assert bob_deal.state == DealState.QUALIFIED

    def test_returns_zero_on_cold_start(self, site_config):
        _make_qualified(site_config)

        scorer = BayesianQualifier(seed=42)

        with patch.object(scorer, "predict_probs", return_value=None):
            assert promote_to_ready(scorer) == 0

    def test_returns_zero_on_empty_pool(self, site_config):
        scorer = BayesianQualifier(seed=42)
        assert promote_to_ready(scorer) == 0

    def test_promotes_with_a_real_fitted_model(self, site_config):
        """Unmocked on purpose. Every other test here patches ``predict_probs``, which
        is exactly how the gate once shipped against a qualifier that did not have the
        method at all — so one test has to drive a genuinely fitted model end to end.

        This used to run a ``KitQualifier``, the pre-trained model the freemium promo
        site_config scored with. That site_config and its qualifier are gone; the fitted
        pipeline underneath is the same shape, so the coverage is unchanged.
        """
        url = _make_qualified(site_config, "alice")

        assert promote_to_ready(_fitted_qualifier()) == 1

        from openoutfind.crm.models import Deal
        assert Deal.objects.get(lead__profile_url=url).state == DealState.READY_TO_FIND_EMAIL


@pytest.mark.django_db
class TestFindReadyCandidate:
    def test_returns_none_when_empty(self, site_config):
        scorer = BayesianQualifier(seed=42)
        assert find_ready_candidate(scorer) is None

    def test_returns_top_ranked(self, site_config):
        url = _make_qualified(site_config, "alice")
        set_profile_state(url, DealState.READY_TO_FIND_EMAIL.value)

        scorer = BayesianQualifier(seed=42)
        scorer.rank_profiles = lambda profiles: profiles

        result = find_ready_candidate(scorer)
        assert result is not None
        assert result["profile_url"] == url
