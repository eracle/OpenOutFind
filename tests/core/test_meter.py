# tests/core/test_meter.py
"""The running means (``core/meter.py``), and where a verdict records its own.

Where a round measures itself is ``tests/core/pipeline/test_walk.py``.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pytest
from django.utils import timezone

from openoutfind.core import agent_qualify, meter
from openoutfind.core.ml.qualifier import BayesianQualifier
from openoutfind.core.pipeline.qualify import run_qualification
from openoutfind.crm.models import Lead, PendingQualification

pytestmark = pytest.mark.django_db


class TestMeter:
    def test_the_mean_is_the_sum_over_the_count(self):
        for value in (10, 20, 60):
            meter.record(meter.ROUND_SECONDS, value)

        assert meter.totals(meter.ROUND_SECONDS) == (90.0, 3)
        assert meter.mean(meter.ROUND_SECONDS, default=70) == pytest.approx(30)

    def test_the_default_stands_in_until_the_first_event(self):
        assert meter.totals(meter.ROUND_SECONDS) == (0.0, 0)
        assert meter.mean(meter.ROUND_SECONDS, default=70) == 70

    def test_a_reset_starts_the_mean_over(self):
        meter.record(meter.POOL_FIT, 1)
        meter.reset(meter.POOL_FIT)
        meter.record(meter.POOL_FIT, 0)

        assert meter.totals(meter.POOL_FIT) == (0.0, 1)

    def test_resetting_a_quantity_never_measured_is_harmless(self):
        meter.reset(meter.POOL_FIT)
        assert meter.totals(meter.POOL_FIT) == (0.0, 0)


def _candidate(url="https://www.linkedin.com/in/alice/"):
    return Lead.objects.create(profile_url=url, profile_text="extraction lead",
                               embedding=np.ones(384, dtype=np.float32).tobytes())


def _pending_for(seconds):
    """A candidate handed to the calling agent ``seconds`` ago, answered now."""
    pending = PendingQualification.objects.create(lead=_candidate())
    PendingQualification.objects.filter(pk=pending.pk).update(
        created=timezone.now() - timedelta(seconds=seconds))
    agent_qualify.enable(agent_qualify.Verdict(fit=False, reason="a supplier"))


class TestVerdictMeasurements:
    def test_a_model_verdict_is_timed_and_counted_toward_the_pool(self, site_config):
        lead = _candidate()
        with (
            patch("openoutfind.core.ml.qualifier.qualify_with_llm", return_value=(1, "fits")),
            patch("openoutfind.core.pipeline.brief.ensure_rules", return_value="rules"),
        ):
            run_qualification(site_config, BayesianQualifier(seed=42), candidates=[lead])

        assert meter.totals(meter.VERDICT_SECONDS_MODEL)[1] == 1
        assert meter.totals(meter.POOL_FIT) == (1.0, 1)

    def test_an_agent_verdict_takes_from_the_hand_over_to_the_answer(self, site_config):
        _pending_for(45)

        run_qualification(site_config, BayesianQualifier(seed=42))

        total, count = meter.totals(meter.VERDICT_SECONDS_AGENT)
        assert count == 1
        assert 45 <= total < 60
        assert meter.totals(meter.POOL_FIT) == (0.0, 1)

    def test_a_session_left_waiting_is_not_a_verdicts_time(self, site_config):
        """Batch 5's first measured agent verdict had waited 1h46m on a stopped session."""
        _pending_for(2 * 3600)

        run_qualification(site_config, BayesianQualifier(seed=42))

        assert meter.totals(meter.VERDICT_SECONDS_AGENT) == (0.0, 0)
        assert meter.totals(meter.POOL_FIT) == (0.0, 1)
