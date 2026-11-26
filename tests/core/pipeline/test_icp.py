# tests/core/pipeline/test_icp.py
"""Cold-phase anchors — the synthetic ideal profiles that let a GP fit before any real
lead has qualified.

The LLM call (``run_agent_sync``) and the embedder are stubbed, so these assert the
lifecycle rather than the model: invented once, written as ``Lead`` rows, reloaded
without a second LLM call, filled up to ``ANCHOR_COUNT`` — and never mistaken for somebody
to contact. What the anchors do to the GP is ``tests/core/ml/test_qualifier.py``.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openoutfind.core.config import SiteConfig
from openoutfind.core.pipeline.icp import (
    Anchor,
    _AnchorProfile,
    _AnchorProfiles,
    anchor_leads,
    ensure_anchors,
    generate_anchors,
)

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def _icp_text(configure):
    """The ICP text an anchor is invented from."""
    configure(product_docs="p", campaign_target="t")


def _site_config():
    return SiteConfig.load()


def _profiles() -> list[str]:
    """The anchors as the operator would read them back."""
    return [lead.profile_text for lead in anchor_leads()]


@contextmanager
def _llm_returns(profiles):
    """Stub the whole LLM boundary — model resolution, Agent, and the run.

    A plain string is the profile line with no fielded values, which is what most of these
    tests care about; pass an ``_AnchorProfile`` where the fields are the point.
    """
    written = [
        p if isinstance(p, _AnchorProfile) else _AnchorProfile(profile=p)
        for p in profiles
    ]
    with (
        patch("openoutfind.core.llm.run_agent_sync",
              return_value=MagicMock(output=_AnchorProfiles(profiles=written))),
        patch("openoutfind.core.llm.get_llm_model"),
        patch("pydantic_ai.Agent"),
    ):
        yield


def _stub_embed():
    return patch("openoutfind.discovery.embed_profile",
                 side_effect=lambda text, *a, **kw: np.full(384, len(text), dtype=np.float32))


class TestGenerateAnchors:
    def test_normalizes_the_llm_output(self):
        with _llm_returns(["  Head Of Sales ACME  ", "", "cto northwind"]):
            written = generate_anchors(_site_config())
        assert [a.profile for a in written] == ["head of sales acme", "cto northwind"]

    def test_the_fielded_values_become_a_lead_row(self):
        """An anchor is a synthetic *lead row*, so the walk can count it the way it counts
        a real acceptance — each value already under the field it is searchable in, with
        nothing split by guess."""
        with _llm_returns([_AnchorProfile(
            profile="head of revenue at northwind california united states",
            job_title="Head Of Revenue", location_state="California",
            location_country="United States", company_keywords="Freight, Logistics",
        )]):
            (anchor,) = generate_anchors(_site_config())

        assert anchor.source_fields == {
            "contact_job_title": "head of revenue",
            "contact_location_state": "california",
            "contact_location_country": "united states",
            "company_keywords": "freight, logistics",
        }

    def test_an_empty_field_is_dropped_rather_than_stored_blank(self):
        """A country with no region worth naming yields the country alone — a blank state
        would otherwise reach `keywords_for` and build a place nobody matches."""
        with _llm_returns([_AnchorProfile(
            profile="founder at acme singapore", job_title="founder",
            location_state="", location_country="singapore",
        )]):
            (anchor,) = generate_anchors(_site_config())

        assert anchor.source_fields == {
            "contact_job_title": "founder",
            "contact_location_country": "singapore",
        }

    def test_an_llm_outage_leaves_the_install_unanchored(self):
        """Best-effort: an unanchored install still runs, just without a fitted GP."""
        with (
            patch("openoutfind.core.llm.run_agent_sync", side_effect=RuntimeError("down")),
            patch("openoutfind.core.llm.get_llm_model"),
            patch("pydantic_ai.Agent"),
        ):
            assert generate_anchors(_site_config()) == []


class TestEnsureAnchors:
    def test_it_writes_them_as_leads_with_their_own_embeddings(self):
        with _llm_returns(["cmo acme", "cto northwind"]), _stub_embed():
            embeddings = ensure_anchors(_site_config())

        assert embeddings.shape == (2, 384)
        assert _profiles() == ["cmo acme", "cto northwind"]
        assert all(lead.embedding_array is not None for lead in anchor_leads())

    def test_an_anchor_is_never_a_lead_to_contact(self):
        """The one query that picks a Lead up without a deal is the qualification scan,
        and everything a person would receive is reached through the deal it refuses to
        create."""
        from openoutfind.core.pipeline.qualify import fetch_qualification_candidates

        with _llm_returns(["cmo acme", "cto northwind"]), _stub_embed():
            ensure_anchors(_site_config())

        assert fetch_qualification_candidates() == []

    def test_reuses_the_stored_set_without_a_second_llm_call(self):
        """Re-inventing them each boot would re-anchor the GP somewhere slightly else."""
        site_config = _site_config()
        with _llm_returns(["cmo acme", "cto northwind"]), _stub_embed():
            first = ensure_anchors(site_config)

        with patch("openoutfind.core.llm.run_agent_sync",
                   side_effect=AssertionError("must not regenerate")):
            second = ensure_anchors(site_config)

        assert np.array_equal(first, second)

    def test_returns_none_without_icp_text(self, configure):
        with patch("openoutfind.core.llm.run_agent_sync",
                   side_effect=AssertionError("nothing to generate from")):
            assert ensure_anchors(configure(product_docs="", campaign_target="")) is None

    def test_returns_none_when_the_llm_proposes_nothing(self):
        with _llm_returns([]):
            assert ensure_anchors(_site_config()) is None


class TestAnchorFillUp:
    """A short first round is filled up to ``ANCHOR_COUNT`` on the next call."""

    def test_fills_up_to_anchor_count(self):
        site_config = _site_config()
        with _llm_returns(["a one", "b two"]), _stub_embed():
            ensure_anchors(site_config)
        with _llm_returns(["c three"]), _stub_embed():
            embeddings = ensure_anchors(site_config)

        assert embeddings.shape == (3, 384)
        assert _profiles() == ["a one", "b two", "c three"]

    def test_asks_only_for_the_shortfall_and_shows_what_exists(self):
        """A second round must widen the ideal region, not restate it."""
        site_config = _site_config()
        with _llm_returns(["a one", "b two"]), _stub_embed():
            ensure_anchors(site_config)

        with (
            patch("openoutfind.core.pipeline.icp.generate_anchors",
                  return_value=[Anchor("c three", {})]) as gen,
            _stub_embed(),
        ):
            ensure_anchors(site_config)

        assert gen.call_args.kwargs == {"count": 1, "existing": ["a one", "b two"]}

    def test_drops_profiles_the_model_repeated(self):
        site_config = _site_config()
        with _llm_returns(["a one"]), _stub_embed():
            ensure_anchors(site_config)
        with _llm_returns(["a one", "b two"]), _stub_embed():
            ensure_anchors(site_config)

        assert _profiles() == ["a one", "b two"]

    def test_a_failed_fill_up_keeps_what_is_already_there(self):
        site_config = _site_config()
        with _llm_returns(["a one"]), _stub_embed():
            first = ensure_anchors(site_config)

        with _llm_returns([]), _stub_embed():
            still = ensure_anchors(site_config)

        assert np.array_equal(first, still)

    def test_no_call_when_the_set_is_already_full(self):
        site_config = _site_config()
        with _llm_returns(["a one", "b two", "c three"]), _stub_embed():
            ensure_anchors(site_config)

        with patch("openoutfind.core.pipeline.icp.generate_anchors",
                   side_effect=AssertionError("already full")):
            assert ensure_anchors(site_config).shape == (3, 384)
