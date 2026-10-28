# tests/ml/test_calibration.py
"""The calibrate arm — draws, weights, the measured bar and where it is read."""
import random
from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core.conf import CAMPAIGN_CONFIG
from openoutfind.core.ml import calibration
from openoutfind.core.ml.qualifier import BayesianQualifier
from openoutfind.core.models import CalibrationBar, CalibrationDraw
from openoutfind.core.pipeline.top_up import top_up
from openoutfind.crm.models import DealState, Outcome
from tests.factories import DealFactory, LeadFactory


class _FixedScores:
    """A qualifier whose P for each lead is set by the test, keyed on the embedding."""

    def __init__(self, probs):
        self.probs = list(probs)

    def predict_probs(self, X):
        return np.array(self.probs[:len(X)], dtype=np.float64)


def _pool(n):
    return [LeadFactory(embedded=True, profile_text=f"lead {i}") for i in range(n)]


def _judged_draw(band, label, band_size=100):
    lead = LeadFactory(embedded=True, profile_text="judged")
    CalibrationDraw.objects.create(lead=lead, band=band, p_at_draw=band / 10 + 0.05,
                                   band_size=band_size)
    if label:
        DealFactory(lead=lead, state=DealState.QUALIFIED)
    else:
        DealFactory(lead=lead, state=DealState.FAILED, outcome=Outcome.WRONG_FIT)
    return lead


# ── Bands and size ──

@pytest.mark.parametrize("p, band", [(0.0, 0), (0.05, 0), (0.1, 1), (0.69, 6),
                                     (0.7, 7), (0.99, 7), (1.0, 7)])
def test_scores_fall_in_their_band(p, band):
    assert calibration.band_of(p) == band


def test_the_arm_switches_on_only_when_every_band_can_be_filled():
    assert not calibration.is_switched_on(1599)  # √N = 39 < 8 bands × 5
    assert calibration.is_switched_on(1600)


@pytest.mark.django_db
def test_a_small_pool_never_calibrates():
    assert not calibration.should_calibrate([object()] * 900)


@pytest.mark.django_db
def test_the_arm_stops_at_root_n_and_pool_growth_reopens_it():
    for i in range(40):
        _judged_draw(band=i % 8, label=i % 2)

    assert not calibration.should_calibrate([object()] * 1600)  # √N = 40, 40 in
    assert calibration.should_calibrate([object()] * 2500)      # √N = 50


# ── The draw ──

@pytest.mark.django_db
def test_the_draw_takes_the_thinnest_band_and_records_its_size():
    _judged_draw(band=0, label=0)
    pool = _pool(3)
    qualifier = _FixedScores([0.05, 0.05, 0.85])

    lead = calibration.next_lead(qualifier, pool, rng=random.Random(0))

    assert lead == pool[2]
    draw = CalibrationDraw.objects.get(lead=lead)
    assert (draw.band, draw.band_size) == (7, 1)
    assert draw.p_at_draw == pytest.approx(0.85)


@pytest.mark.django_db
def test_a_draw_waiting_on_its_verdict_is_judged_before_a_new_one():
    pool = _pool(3)
    CalibrationDraw.objects.create(lead=pool[1], band=3, p_at_draw=0.35, band_size=10)

    lead = calibration.next_lead(_FixedScores([0.9, 0.9, 0.9]), pool)

    assert lead == pool[1]
    assert CalibrationDraw.objects.count() == 1


@pytest.mark.django_db
def test_no_draw_without_a_fitted_model():
    class _Unfitted:
        def predict_probs(self, X):
            return None

    assert calibration.next_lead(_Unfitted(), _pool(2)) is None
    assert not CalibrationDraw.objects.exists()


def test_a_lead_with_no_profile_text_is_never_drawn():
    class _Lead:
        def __init__(self, text):
            self.profile_text = text

    readable = _Lead("Head of Growth")
    assert calibration.calibration_pool([_Lead(""), readable, _Lead(None)]) == [readable]


@pytest.mark.django_db
def test_a_resumed_draw_says_it_is_a_calibration_verdict(caplog):
    pool = _pool(2)
    CalibrationDraw.objects.create(lead=pool[0], band=3, p_at_draw=0.35, band_size=10)

    with caplog.at_level("INFO"):
        calibration.next_lead(_FixedScores([0.9, 0.9]), pool)

    assert "calibrate — verdict 1 of 1, resuming the draw from P 0.3–0.4" in caplog.text


# ── The costs ──

@pytest.mark.django_db
def test_leads_per_search_counts_pages_from_the_walk():
    from openoutfind.core.models import QueryNode

    QueryNode.objects.create(token_key="a", next_offset=300, state=QueryNode.State.FIRED)
    QueryNode.objects.create(token_key="b", next_offset=0, state=QueryNode.State.DEAD)
    _pool(20)
    LeadFactory(synthetic=True)

    # Three pages advanced plus the one empty page that retired the dead node.
    assert calibration.leads_per_search() == pytest.approx(20 / 4)


@pytest.mark.django_db
def test_no_search_yet_adds_no_leads():
    assert calibration.leads_per_search() == 0.0


@pytest.mark.django_db
def test_the_customers_marks_were_not_found_by_a_search():
    from openoutfind.core.models import QueryNode

    QueryNode.objects.create(token_key="a", next_offset=100, state=QueryNode.State.FIRED)
    _pool(10)
    for i in range(30):
        LeadFactory(marked=True, profile_text=f"marked {i}")

    assert calibration.leads_per_search() == pytest.approx(10)


@pytest.mark.django_db
def test_a_search_costs_its_measured_time_once_there_is_one():
    from openoutfind.core import meter

    meter.record(meter.SEARCH_SECONDS, 30)
    meter.record(meter.SEARCH_SECONDS, 50)
    with patch.dict(CAMPAIGN_CONFIG, {"run_hour_dollars": 36.0, "search_seconds": 70}):
        assert calibration.search_cost() == pytest.approx(0.4)


@pytest.mark.django_db
def test_an_agent_verdict_costs_its_measured_time_plus_its_fixed_bill():
    from openoutfind.core import meter

    meter.record(meter.VERDICT_SECONDS_AGENT, 20)
    meter.record(meter.VERDICT_SECONDS_MODEL, 1000)  # another judge's time is not this one's
    with (
        patch("openoutfind.core.agent_qualify.active", return_value=True),
        patch.dict(CAMPAIGN_CONFIG, {"run_hour_dollars": 36.0, "verdict_dollars_agent": 0.05}),
    ):
        assert calibration.verdict_cost() == pytest.approx(0.2 + 0.05)


@pytest.mark.django_db
def test_an_agent_verdict_costs_more_than_a_model_verdict():
    with patch("openoutfind.core.agent_qualify.active", return_value=False):
        model = calibration.verdict_cost()
    with patch("openoutfind.core.agent_qualify.active", return_value=True):
        agent = calibration.verdict_cost()

    assert agent > model > 0


@pytest.mark.django_db
def test_a_verdict_costs_its_time_and_its_llm_bill():
    with (
        patch("openoutfind.core.agent_qualify.active", return_value=False),
        patch.object(calibration, "model_verdict_bill", return_value=0.002),
        patch.dict(CAMPAIGN_CONFIG, {"run_hour_dollars": 36.0, "verdict_seconds_model": 10}),
    ):
        assert calibration.verdict_cost() == pytest.approx(0.1 + 0.002)


def test_a_model_verdict_is_billed_at_that_models_rates():
    with patch.dict(CAMPAIGN_CONFIG, {"verdict_input_tokens": 1_000_000,
                                      "verdict_output_tokens": 1_000_000}):
        assert calibration.model_verdict_bill("anthropic:claude-haiku-5-5") == pytest.approx(0.60)
        assert calibration.model_verdict_bill("anthropic:claude-sonnet-5-5") == pytest.approx(12.0)
        # A dated id bills as its family; the longest matching key wins.
        assert calibration.model_verdict_bill("claude-haiku-4-5-20251001") == pytest.approx(6.0)
        assert calibration.model_verdict_bill("anthropic:claude-sonnet-5") == pytest.approx(12.0)


def test_an_unpriced_model_is_billed_the_fallback():
    with patch.dict(CAMPAIGN_CONFIG, {"verdict_dollars_model": 0.004}):
        assert calibration.model_verdict_bill("openai:gpt-4o") == 0.004


@pytest.mark.django_db
def test_a_search_costs_only_its_time():
    with patch.dict(CAMPAIGN_CONFIG, {"run_hour_dollars": 36.0, "search_seconds": 70}):
        assert calibration.search_cost() == pytest.approx(0.7)


# ── The bar ──

@pytest.mark.django_db
def test_the_bar_falls_back_until_one_is_measured():
    assert calibration.current_threshold() == CAMPAIGN_CONFIG["min_gp_confidence"]

    CalibrationBar.objects.create(threshold=None, break_even=0.4, n_verdicts=40)
    assert calibration.current_threshold() == CAMPAIGN_CONFIG["min_gp_confidence"]

    CalibrationBar.objects.create(threshold=0.45, break_even=0.1, n_verdicts=50)
    assert calibration.current_threshold() == 0.45


@pytest.mark.django_db
def test_too_few_verdicts_are_not_measured():
    for i in range(39):
        _judged_draw(band=i % 8, label=i % 2)

    assert not calibration.needs_measuring()


def _measured(oof_for_label, pool_scores, *, leads_per_search=10.0, verdict=0.005,
              search=0.07):
    """Measure with stand-in out-of-fold scores (one per label) and pool scores."""
    from openoutfind.crm.models import Lead

    _, _, labels = Lead.get_labeled_rows()

    class _Model:
        def out_of_fold_probs(self, indices, k):
            return np.array([oof_for_label(i, y) for i, y in enumerate(labels)])

    with (
        patch("openoutfind.core.ml.qualifier.build_qualifier", return_value=_Model()),
        patch.object(calibration, "leads_per_search", return_value=leads_per_search),
        patch.object(calibration, "verdict_cost", return_value=verdict),
        patch.object(calibration, "search_cost", return_value=search),
    ):
        return calibration.measure(_FixedScores(pool_scores), _pool(len(pool_scores)))


@pytest.mark.django_db
def test_measuring_stores_the_lowest_score_where_a_verdict_pays(caplog):
    # Fits only in the upper half: out of fold they score high, the rejections low.
    for i in range(40):
        _judged_draw(band=i % 8, label=int(i % 8 >= 4))
    assert calibration.needs_measuring()

    with caplog.at_level("INFO"):
        bar = _measured(lambda i, y: 0.9 if y else 0.1, [0.2, 0.5, 0.95, 0.96],
                        leads_per_search=0.1)

    # A search that adds a tenth of a lead is worth little, so the break-even is small.
    # The rate is 0 up to 0.1 and 1 from 0.9; the isotonic curve interpolates between,
    # so the bar is the first score whose rate is above that break-even.
    assert 0.1 < bar.threshold < 0.2
    assert 0 < bar.break_even < 0.1
    assert bar.n_verdicts == 40
    assert calibration.current_threshold() == bar.threshold
    assert not calibration.needs_measuring()
    assert "judging pays above" in caplog.text
    assert "bar measured at" in caplog.text
    assert "opens 4 of 4 waiting" in caplog.text


@pytest.mark.django_db
def test_dearer_verdicts_raise_the_bar():
    for i in range(40):
        _judged_draw(band=i % 8, label=int(i % 8 >= 4))

    def graded(i, y):
        return 0.3 + 0.6 * (i % 8) / 7  # scores spread with the bands

    cheap = _measured(graded, [0.4, 0.6, 0.8], verdict=0.001).threshold
    CalibrationBar.objects.all().delete()
    dear = _measured(graded, [0.4, 0.6, 0.8], verdict=0.3).threshold

    assert dear > cheap


@pytest.mark.django_db
def test_a_sample_with_no_fits_stores_no_bar(caplog):
    for i in range(40):
        _judged_draw(band=i % 8, label=0)

    with caplog.at_level("INFO"):
        bar = _measured(lambda i, y: 0.1, [])

    assert bar.threshold is None
    assert calibration.current_threshold() == CAMPAIGN_CONFIG["min_gp_confidence"]
    assert "no score fits that often" in caplog.text


@pytest.mark.django_db
def test_a_failed_fit_is_stored_so_it_is_not_retried():
    for i in range(40):
        _judged_draw(band=i % 8, label=i % 2)

    class _Unfittable:
        def out_of_fold_probs(self, indices, k):
            return None

    with patch("openoutfind.core.ml.qualifier.build_qualifier", return_value=_Unfittable()):
        bar = calibration.measure(_FixedScores([0.5]), _pool(1))

    assert bar.threshold is None
    assert not calibration.needs_measuring()


# ── The arm in top_up ──

def _exploiting_qualifier():
    qualifier = BayesianQualifier(embedding_dim=8)
    qualifier.warm_start(np.random.RandomState(0).rand(7, 8), np.array([1, 1, 1, 0, 0, 0, 0]))
    return qualifier


class _Candidate:
    def __init__(self, profile_url="https://example.com/in/alice/"):
        self.pk = None
        self.profile_url = profile_url
        self.embedding_array = np.zeros(8, dtype=np.float64)
        self.profile_text = "Head of Growth at Acme"


@pytest.mark.django_db
def test_a_candidate_left_pending_earlier_does_not_count_as_the_draw(site_config, caplog):
    """Under --agent-qualify the pending row is answered first, whoever asks — on the
    held order's store that was a lead left over from an earlier agent run."""
    qualifier = _exploiting_qualifier()
    drawn = _Candidate("https://example.com/in/drawn/")

    with (
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=[drawn]),
        patch.object(calibration, "should_calibrate", return_value=True),
        patch.object(calibration, "next_lead", return_value=drawn),
        patch("openoutfind.core.pipeline.top_up.run_qualification",
              return_value="https://example.com/in/left-pending/"),
        caplog.at_level("INFO"),
    ):
        assert top_up(site_config, qualifier) is True

    assert "answered a candidate left pending earlier" in caplog.text


@pytest.mark.django_db
def test_the_arm_judges_a_calibration_lead_before_an_exploit_pick(site_config):
    qualifier = _exploiting_qualifier()
    drawn = _Candidate()

    with (
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=[_Candidate(), drawn]),
        patch.object(calibration, "should_calibrate", return_value=True),
        patch.object(calibration, "next_lead", return_value=drawn),
        patch("openoutfind.core.pipeline.top_up.run_qualification",
              return_value="https://example.com/in/alice/") as qualify,
        patch("openoutfind.core.pipeline.top_up.discover") as discover,
    ):
        assert top_up(site_config, qualifier) is True

    qualify.assert_called_once()
    assert qualify.call_args.kwargs["candidates"] == [drawn]
    assert qualify.call_args.kwargs["preselected"] is True
    assert not discover.called


@pytest.mark.django_db
def test_exploit_judges_above_a_measured_bar_instead_of_searching(site_config):
    """The held order: 0.7 cleared 2 leads of 21,134, so every pass searched."""
    CalibrationBar.objects.create(threshold=0.45, break_even=0.1, n_verdicts=145)
    qualifier = _exploiting_qualifier()

    with (
        patch.object(qualifier, "predict_probs", return_value=np.array([0.5])),
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=[_Candidate()]),
        patch("openoutfind.core.pipeline.top_up.run_qualification",
              return_value="https://example.com/in/alice/") as qualify,
        patch("openoutfind.core.pipeline.top_up.discover") as discover,
    ):
        assert top_up(site_config, qualifier) is True

    assert qualify.called
    assert qualify.call_args.kwargs.get("preselected") is not True
    assert not discover.called


@pytest.mark.django_db
def test_without_a_measured_bar_the_fallback_gates_exploit(site_config):
    qualifier = _exploiting_qualifier()

    with (
        patch.object(qualifier, "predict_probs", return_value=np.array([0.5])),
        patch("openoutfind.core.pipeline.top_up.judging_pays", return_value=False),
        patch("openoutfind.core.pipeline.top_up.fetch_qualification_candidates",
              return_value=[_Candidate()]),
        patch("openoutfind.core.pipeline.top_up.run_qualification") as qualify,
        patch("openoutfind.core.pipeline.top_up.discover", return_value=True) as discover,
    ):
        top_up(site_config, qualifier)

    assert not qualify.called
    assert discover.called


def test_a_calibration_verdict_does_not_move_the_mode():
    qualifier = BayesianQualifier(embedding_dim=8)
    # 3 fits, 5 rejections — but 3 of those rejections are calibration verdicts.
    qualifier.warm_start(np.random.RandomState(0).rand(8, 8), np.array([1, 1, 1, 0, 0, 0, 0, 0]))
    assert qualifier.acquisition_mode() == "exploit (p)"

    qualifier.exclude_from_balance(3, 0)

    assert qualifier.balance_counts == (2, 3)
    assert qualifier.acquisition_mode() == "explore (BALD)"


@pytest.mark.django_db
def test_the_promote_gate_reads_the_measured_bar():
    from openoutfind.core.pipeline.ready_pool import promote_to_ready

    DealFactory(lead=LeadFactory(embedded=True), state=DealState.QUALIFIED)
    CalibrationBar.objects.create(threshold=0.45, break_even=0.1, n_verdicts=145)

    assert promote_to_ready(_FixedScores([0.5])) == 1
