# tests/core/ml/test_qualifier.py
"""``BayesianQualifier`` — what it adds to ``openoutlearn.GPBaldQualifier``: the anchors,
the early-window balancing, the confidence, and ``qualifier_for``'s one fit per label set.

The engine itself (update, warm start, BALD, the kernel search) is ``openoutlearn``'s and
tested there.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core.ml.qualifier import BayesianQualifier


def _make_trained_qualifier(n_pos=10, n_neg=10, seed=42):
    """Create a qualifier with both classes so the GPC can fit."""
    qualifier = BayesianQualifier(seed=seed)
    rng = np.random.RandomState(seed)
    pos_emb = rng.randn(384).astype(np.float32) + 1.0
    neg_emb = rng.randn(384).astype(np.float32) - 1.0
    for _ in range(n_pos):
        qualifier.update(pos_emb + rng.randn(384).astype(np.float32) * 0.1, 1)
    for _ in range(n_neg):
        qualifier.update(neg_emb + rng.randn(384).astype(np.float32) * 0.1, 0)
    return qualifier, pos_emb, neg_emb


class TestBayesianQualifierPredict:
    def test_predict_returns_prob_entropy_and_std(self):
        qualifier, pos_emb, _ = _make_trained_qualifier()
        result = qualifier.predict(pos_emb)
        assert result is not None
        prob, entropy, std = result
        assert 0 <= prob <= 1
        assert entropy >= 0
        assert std >= 0

    def test_predict_returns_none_when_unfitted(self):
        qualifier = BayesianQualifier(seed=42)
        emb = np.random.randn(384).astype(np.float32)
        assert qualifier.predict(emb) is None

    def test_predict_returns_none_single_class(self):
        qualifier = BayesianQualifier(seed=42)
        for _ in range(5):
            qualifier.update(np.random.randn(384).astype(np.float32), 1)
        assert qualifier.predict(np.random.randn(384).astype(np.float32)) is None

    def test_predict_shifts_after_training(self):
        qualifier, pos_emb, _ = _make_trained_qualifier(n_pos=20, n_neg=5)
        result = qualifier.predict(pos_emb)
        assert result is not None
        prob, _, _ = result
        assert prob > 0.7


class TestConfidence:
    def test_it_is_high_at_a_verdict_and_near_zero_far_from_every_one(self):
        qualifier, pos_emb, _ = _make_trained_qualifier()
        far = np.random.RandomState(7).randn(384).astype(np.float32) * 10

        near, away = qualifier.confidence(np.stack([pos_emb, far]))

        assert near > 0.5
        assert away < 0.05

    def test_it_is_none_when_unfitted(self):
        qualifier = BayesianQualifier(seed=42)
        assert qualifier.confidence(np.random.randn(1, 384)) is None


class TestWarmStart:
    def test_the_one_fit_is_over_the_labels_and_the_anchors_together(self, caplog):
        """**The whole set, once.** The build order is load-then-anchor, and the fit
        that survives is the one after both — 20 real labels *plus* 3 anchors, never the
        20 alone. Waiting is what makes that true: an eager fit at load time could only
        ever see the smaller set, which is why the discarded fit was also the wrong one.
        """
        rng = np.random.RandomState(99)
        X = rng.randn(20, 384).astype(np.float32)
        y = np.array([i % 2 for i in range(20)], dtype=np.int32)

        qualifier = BayesianQualifier(seed=42)
        qualifier.warm_start(X, y)
        qualifier.set_anchors(rng.randn(3, 384).astype(np.float32))

        with caplog.at_level("INFO"):
            qualifier.predict(rng.randn(384).astype(np.float32))

        fits = [r.getMessage() for r in caplog.records
                if "training the ranking model" in r.getMessage()]
        assert len(fits) == 1
        assert "23 judged lead(s)" in fits[0] and "3 of them still synthetic" in fits[0]


class TestAnchors:
    """Synthetic positives that let a GP fit before any real lead has qualified.

    The cold phase is the state this exists for: every LLM verdict is a rejection until
    the ICP is right, and a single-class label set produces no posterior at all.
    """

    @staticmethod
    def _anchors(seed=0):
        rng = np.random.RandomState(seed)
        return rng.randn(3, 384).astype(np.float32) + 1.0

    def test_anchors_make_an_all_negative_qualifier_fit(self):
        qualifier = BayesianQualifier(seed=42)
        rng = np.random.RandomState(1)
        for _ in range(5):
            qualifier.update(rng.randn(384).astype(np.float32) - 1.0, 0)
        assert qualifier.acquisition_mode() is None  # single class — nothing to fit

        qualifier.set_anchors(self._anchors())

        assert qualifier.acquisition_mode() is not None
        assert qualifier.predict_probs(self._anchors()) is not None

    def test_anchors_count_as_positives_but_not_as_real_ones(self):
        qualifier = BayesianQualifier(seed=42)
        qualifier.update(np.zeros(384, dtype=np.float32), 0)
        qualifier.set_anchors(self._anchors())

        assert qualifier.class_counts == (1, 3)
        assert qualifier.n_obs == 4
        assert qualifier.has_real_positive is False

    def test_a_real_positive_leaves_every_anchor_standing(self):
        """The anchors are permanent — a real acceptance grows the positive class
        alongside them, it never displaces one."""
        qualifier = BayesianQualifier(seed=42)
        rng = np.random.RandomState(1)
        for _ in range(10):
            qualifier.update(rng.randn(384).astype(np.float32) - 1.0, 0)
        qualifier.set_anchors(self._anchors())

        qualifier.update(np.ones(384, dtype=np.float32), 1)

        assert qualifier.has_real_positive is True
        assert qualifier.is_cold is True          # 1 real positive < ANCHOR_COUNT (3)
        assert qualifier.class_counts == (10, 4)  # one real positive + all three anchors

    def test_the_cold_phase_ends_once_real_positives_reach_anchor_count(self):
        qualifier = BayesianQualifier(seed=42)
        rng = np.random.RandomState(1)
        for _ in range(2):
            qualifier.update(rng.randn(384).astype(np.float32) - 1.0, 0)
        qualifier.set_anchors(self._anchors())

        for _ in range(3):
            qualifier.update(np.ones(384, dtype=np.float32), 1)

        assert qualifier.is_cold is False
        assert qualifier.n_anchors == 3  # still standing — only the phase clock moved
        assert qualifier.class_counts == (2, 6)  # three real positives, three anchors
        assert qualifier.n_obs == 8

    def test_a_rejection_keeps_the_anchors(self):
        """Only real positives move the phase clock — rejections are what it is made of."""
        qualifier = BayesianQualifier(seed=42)
        qualifier.set_anchors(self._anchors())

        qualifier.update(np.zeros(384, dtype=np.float32), 0)

        assert qualifier.class_counts == (1, 3)
        assert qualifier.has_real_positive is False

    def test_anchoring_is_not_trimmed_by_prior_real_positives(self):
        """Safe to call on every daemon boot — restoring the stored set must not read the
        real positives already on the qualifier as a reason to drop any of it."""
        qualifier = BayesianQualifier(seed=42)
        qualifier.update(np.ones(384, dtype=np.float32), 1)
        qualifier.update(np.zeros(384, dtype=np.float32), 0)

        qualifier.set_anchors(self._anchors())

        assert qualifier.class_counts == (1, 4)  # one real positive + all three anchors
        assert qualifier.is_cold is True

    def test_anchoring_twice_does_not_stack(self):
        qualifier = BayesianQualifier(seed=42)
        qualifier.set_anchors(self._anchors())
        qualifier.set_anchors(self._anchors(seed=7))

        assert qualifier.class_counts == (0, 3)

    def test_real_negatives_are_not_subsampled_away_by_the_anchors(self):
        """``_balance`` caps the majority at 2x the minority. With 3 synthetic positives
        that would throw away all but 6 of the real rejections — the opposite of its job,
        so balancing is skipped until a real positive exists."""
        qualifier = BayesianQualifier(seed=42)
        rng = np.random.RandomState(2)
        for _ in range(50):
            qualifier.update(rng.randn(384).astype(np.float32) - 1.0, 0)
        qualifier.set_anchors(self._anchors())

        with patch.object(BayesianQualifier, "_balance",
                          side_effect=AssertionError("must not balance while anchored")):
            assert qualifier.acquisition_mode() is not None

    @staticmethod
    def _real(n_pos: int, n_neg: int) -> BayesianQualifier:
        qualifier = BayesianQualifier(seed=42)
        rng = np.random.RandomState(3)
        for label in [1] * n_pos + [0] * n_neg:
            qualifier.update(rng.randn(384).astype(np.float32), label)
        return qualifier

    def test_a_few_real_fits_are_balanced(self):
        qualifier = self._real(n_pos=4, n_neg=40)

        assert qualifier.balances
        _, y_fit = qualifier._balance(*qualifier._training_arrays())
        assert len(y_fit) == 12

    def test_past_the_early_window_every_verdict_trains(self):
        """Subsampling there drops paid-for verdicts: a fresh no-fit was usually left
        out, and its lookalike topped the pool."""
        qualifier = self._real(n_pos=BayesianQualifier._BALANCE_UNTIL_POSITIVES, n_neg=60)

        assert not qualifier.balances
        with patch.object(BayesianQualifier, "_balance",
                          side_effect=AssertionError("must not balance past the window")):
            assert qualifier._fit_if_needed()

    def test_warm_start_leaves_anchors_intact(self):
        """Boot order is warm_start then anchor, but neither may clobber the other."""
        qualifier = BayesianQualifier(seed=42)
        qualifier.set_anchors(self._anchors())

        qualifier.warm_start(np.zeros((2, 384), dtype=np.float32), np.array([0, 0]))

        assert qualifier.class_counts == (2, 3)
        assert qualifier.has_real_positive is False


class TestColdPhaseAcquisition:
    """While the only positives are invented, the axis is exploit — see acquisition_mode."""

    @staticmethod
    def _cold(n_rejections: int, n_anchors: int):
        rng = np.random.RandomState(0)
        q = BayesianQualifier(embedding_dim=8)
        q.set_anchors(rng.rand(n_anchors, 8))
        for _ in range(n_rejections):
            q.update(rng.rand(8), 0)
        return q

    def test_cold_phase_exploits_however_the_classes_balance(self):
        # The live bug: while the anchors were held at the rejection count, `n_neg > n_pos`
        # was never true and the axis was pinned to BALD for the whole cold phase.
        for rejections, anchors in ((6, 7), (9, 11), (50, 3)):
            q = self._cold(rejections, anchors)
            assert q.acquisition_mode() == "exploit (p)", (rejections, anchors)

    def test_the_axis_returns_to_the_balance_once_real_positives_reach_anchor_count(self):
        """A first acceptance does not end the phase — the anchors stay standing and the
        axis keeps exploiting until real positives themselves reach ANCHOR_COUNT."""
        q = self._cold(n_rejections=3, n_anchors=3)
        q.update(np.random.RandomState(1).rand(8), 1)

        assert q.has_real_positive and q.is_cold
        assert q.acquisition_mode() == "exploit (p)"

        for _ in range(2):
            q.update(np.random.RandomState(2).rand(8), 1)

        assert q.is_cold is False
        assert q.n_anchors == 3  # the anchors never left
        assert q.class_counts == (3, 6)  # three real positives plus the three anchors
        assert q.acquisition_mode() == "explore (BALD)"  # real positives caught up

    def test_unfitted_model_still_reports_no_axis(self):
        assert BayesianQualifier(embedding_dim=8).acquisition_mode() is None


@pytest.mark.django_db
class TestBuildingOneCostsOneFit:
    """A construction fits **once**, over the labels *and* the anchors together.

    It used to fit twice: `warm_start` fitted the real labels eagerly, `set_anchors`
    marked the model dirty, and the first prediction fitted again with the anchors in.
    The first fit was never asked anything — pure O(n³) waste on every build, 7s at 230
    labels on the production install where it was caught.
    """

    def _labelled_deals_with_both_classes(self):
        from openoutfind.crm.models import Deal, DealState, Lead, Outcome

        rng = np.random.RandomState(7)
        for i in range(6):
            lead = Lead.objects.create(
                profile_url=f"https://linkedin.com/in/p{i}/",
                embedding=rng.randn(384).astype(np.float32).tobytes(),
            )
            fit = i % 2 == 0
            Deal.objects.create(
                lead=lead,
                state=DealState.QUALIFIED if fit else DealState.FAILED,
                **({} if fit else {"outcome": Outcome.WRONG_FIT}),
            )

    def _fits(self, caplog) -> list[str]:
        return [r.getMessage() for r in caplog.records
                if "training the ranking model" in r.getMessage()]

    def test_a_build_and_its_first_prediction_train_once(self, site_config, caplog):
        from openoutfind.core.ml.qualifier import qualifier_for

        self._labelled_deals_with_both_classes()

        with caplog.at_level("INFO"):
            qualifier = qualifier_for()
            qualifier.predict(np.random.RandomState(0).randn(384).astype(np.float32))

        assert len(self._fits(caplog)) == 1

    def test_building_alone_trains_nothing(self, site_config, caplog):
        """Nothing is fitted until something is asked — `_score_qualified` builds one
        and may then find an empty pool to score."""
        from openoutfind.core.ml.qualifier import qualifier_for

        self._labelled_deals_with_both_classes()

        with caplog.at_level("INFO"):
            qualifier_for()

        assert self._fits(caplog) == []


@pytest.mark.django_db
class TestOneFitPerLabelSet:
    """`qualifier_for` keeps the fitted model under a fingerprint of its evidence.

    The fit is O(n³) — 15s at 310 labels on the production install — and the loop that
    calls it runs once per *action*. Almost no action writes a label: discovery adds
    unlabelled leads, buying an address and checking a lookup move a deal's state. Each
    used to be followed by a full refit to a posterior that could not have changed.
    """

    def _labelled(self, fit: bool):
        from openoutfind.crm.models import Deal, DealState, Lead, Outcome

        lead = Lead.objects.create(
            profile_url=f"https://linkedin.com/in/p{Lead.objects.count()}/",
            embedding=np.random.RandomState(Lead.objects.count()).randn(384)
            .astype(np.float32).tobytes(),
        )
        Deal.objects.create(
            lead=lead,
            state=DealState.QUALIFIED if fit else DealState.FAILED,
            outcome="" if fit else Outcome.WRONG_FIT)
        return lead

    def test_the_same_labels_hand_back_the_same_fitted_model(self, site_config):
        from openoutfind.core.ml.qualifier import qualifier_for

        for fit in (True, False, True, False):
            self._labelled(fit)

        first = qualifier_for()
        first.predict(np.random.RandomState(0).randn(384).astype(np.float32))
        assert qualifier_for() is first

    def test_a_new_verdict_builds_a_new_model(self, site_config):
        from openoutfind.core.ml.qualifier import qualifier_for

        for fit in (True, False):
            self._labelled(fit)
        first = qualifier_for()

        self._labelled(True)
        assert qualifier_for() is not first

    def test_moving_a_deal_along_the_pipeline_keeps_the_fit(self, site_config):
        """A promotion, an address on order and a resolution all change a deal's state
        and none of them changes its fit verdict — which is the only thing fitted."""
        from openoutfind.core.ml.qualifier import qualifier_for
        from openoutfind.crm.models import Deal, DealState

        for fit in (True, False):
            self._labelled(fit)
        first = qualifier_for()

        promoted = Deal.objects.filter(state=DealState.QUALIFIED).first()
        promoted.state = DealState.READY_TO_FIND_EMAIL
        promoted.save(update_fields=["state"])

        assert qualifier_for() is first

    def test_a_first_anchor_set_is_a_new_model(self, site_config):
        """Anchors are fitted alongside the real labels, so writing them is new evidence."""
        from openoutfind.core.ml.qualifier import qualifier_for
        from openoutfind.crm.models import Lead

        self._labelled(False)
        first = qualifier_for()

        Lead.objects.create(synthetic=True, profile_text="a founder at a saas company")

        assert qualifier_for() is not first
