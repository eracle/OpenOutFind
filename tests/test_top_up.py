# tests/test_top_up.py
"""`top_up` narrates which regime it is in — cold phase, explore, or exploit —
because it is exactly the question an operator watching the run asks."""
from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core.ml.qualifier import BayesianQualifier
from openoutfind.core.pipeline.top_up import top_up


@pytest.mark.django_db
def test_the_cold_phase_names_itself_and_the_anchor_progress(site_config, caplog):
    qualifier = BayesianQualifier(embedding_dim=8)
    qualifier.set_anchors(np.random.RandomState(0).rand(3, 8))

    with (
        patch("openoutfind.core.pipeline.top_up.discover", return_value=False),
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=[]),
        caplog.at_level("INFO"),
    ):
        top_up(site_config, qualifier)

    assert "cold phase" in caplog.text
    assert "0/3 real positive" in caplog.text


def _exploiting_qualifier():
    """A fitted, anchor-free qualifier whose negatives outnumber its positives."""
    qualifier = BayesianQualifier(embedding_dim=8)
    rng = np.random.RandomState(0)
    qualifier.warm_start(rng.rand(7, 8), np.array([1, 1, 1, 0, 0, 0, 0]))
    return qualifier


class _Candidate:
    def __init__(self):
        self.embedding_array = np.zeros(8, dtype=np.float64)
        self.profile_text = "Head of Growth at Acme"


@pytest.mark.django_db
def test_exploit_qualifies_only_a_lead_clearing_the_spend_gate(site_config):
    qualifier = _exploiting_qualifier()

    with (
        patch.object(qualifier, "predict_probs", return_value=np.array([0.95])),
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=[_Candidate()]),
        patch("openoutfind.core.pipeline.top_up.run_qualification",
              return_value="https://example.com/in/alice/") as qualify,
        patch("openoutfind.core.pipeline.top_up.discover") as discover,
    ):
        assert top_up(site_config, qualifier) is True

    assert qualify.called
    assert not discover.called


@pytest.mark.django_db
def test_below_the_bar_exploit_judges_the_whole_pool_when_judging_is_cheaper(site_config):
    """Below the bar is not "nothing to do": when a verdict on the pool buys a fit for
    less than a search would, the best lead by P is judged, from every candidate.

    This is the 14h33m failure turned round: with 3 real positives a live pool topped out
    at P=0.37, every pass discovered, and the posterior that would open the bar never
    moved."""
    qualifier = _exploiting_qualifier()
    pool = [_Candidate(), _Candidate()]

    with (
        patch.object(qualifier, "predict_probs", return_value=np.array([0.37, 0.2])),
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=pool),
        patch("openoutfind.core.pipeline.top_up.judging_pays", return_value=True),
        patch("openoutfind.core.pipeline.top_up.run_qualification",
              return_value="https://example.com/in/alice/") as qualify,
        patch("openoutfind.core.pipeline.top_up.discover", return_value=True) as discover,
    ):
        assert top_up(site_config, qualifier) is True

    assert qualify.call_args.kwargs["candidates"] == pool
    assert not discover.called


@pytest.mark.django_db
def test_below_the_bar_exploit_searches_when_searching_is_cheaper(site_config):
    """CaleafTech's batch 5: 42 verdicts, no fit, 6 queries in the whole store. A pool
    that stops fitting is left for a fresh page."""
    qualifier = _exploiting_qualifier()

    with (
        patch.object(qualifier, "predict_probs", return_value=np.array([0.5])),
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=[_Candidate()]),
        patch("openoutfind.core.pipeline.top_up.judging_pays", return_value=False),
        patch("openoutfind.core.pipeline.top_up.run_qualification") as qualify,
        patch("openoutfind.core.pipeline.top_up.discover", return_value=True) as discover,
    ):
        assert top_up(site_config, qualifier) is True

    assert not qualify.called
    assert discover.called


@pytest.mark.django_db
def test_with_nothing_left_to_search_the_pool_is_judged(site_config):
    """Searching was cheaper, but the frontier is spanned: the job must not end with
    candidates still waiting."""
    qualifier = _exploiting_qualifier()

    with (
        patch.object(qualifier, "predict_probs", return_value=np.array([0.5])),
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=[_Candidate()]),
        patch("openoutfind.core.pipeline.top_up.judging_pays", return_value=False),
        patch("openoutfind.core.pipeline.top_up.run_qualification",
              return_value="https://example.com/in/alice/") as qualify,
        patch("openoutfind.core.pipeline.top_up.discover", return_value=False),
    ):
        assert top_up(site_config, qualifier) is True

    assert qualify.called


@pytest.mark.django_db
def test_an_empty_pool_with_nothing_to_search_is_the_end(site_config):
    qualifier = _exploiting_qualifier()

    with (
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=[]),
        patch("openoutfind.core.pipeline.top_up.run_qualification") as qualify,
        patch("openoutfind.core.pipeline.top_up.discover", return_value=False),
    ):
        assert top_up(site_config, qualifier) is False

    assert not qualify.called


@pytest.mark.django_db
def test_explore_counts_a_page_of_familiar_profiles_as_work(site_config):
    """A fired page is the unit of work, whatever fraction of it was new.

    A live run ended `goal_unreached` on this shape: the page came back 100 rows all
    already ours, so it left no candidate to label — which is not the same fact as a
    spanned frontier, and must not stop the job with the walk one node in."""
    qualifier = BayesianQualifier(embedding_dim=8)
    rng = np.random.RandomState(0)
    qualifier.warm_start(rng.rand(7, 8), np.array([1, 1, 1, 1, 0, 0, 0]))
    assert qualifier.acquisition_mode() != "exploit (p)"

    with (
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=[]),
        patch("openoutfind.core.pipeline.top_up.run_qualification") as qualify,
        patch("openoutfind.core.pipeline.top_up.discover", return_value=True),
    ):
        assert top_up(site_config, qualifier) is True

    assert not qualify.called
