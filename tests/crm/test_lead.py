# tests/crm/test_lead.py
"""``Lead``: the stored vector, and the labels the GP trains on."""
from __future__ import annotations

import numpy as np
import pytest

from openoutfind.crm.models import Deal, DealState, Lead, Outcome

pytestmark = pytest.mark.django_db


def test_the_embedding_round_trips_through_the_store():
    vector = np.random.randn(384).astype(np.float32)
    lead = Lead(profile_url="https://linkedin.com/in/alice/")
    lead.embedding_array = vector
    lead.save()

    np.testing.assert_array_almost_equal(Lead.objects.get().embedding_array, vector)
    assert Lead(profile_url="https://linkedin.com/in/bob/").embedding_array is None


@pytest.mark.parametrize("state,outcome,label", [
    (DealState.QUALIFIED, "", 1),
    # The LLM qualified it; only enrichment failed — still a fit.
    (DealState.NO_EMAIL_FOUND, "", 1),
    (DealState.FAILED, Outcome.WRONG_FIT, 0),
    # A FAILED deal that is not the LLM's rejection is no verdict at all.
    (DealState.FAILED, Outcome.UNKNOWN, None),
])
def test_the_label_is_the_llms_fit_verdict(state, outcome, label):
    lead = Lead.objects.create(profile_url="https://linkedin.com/in/alice/",
                               embedding=np.ones(384, dtype=np.float32).tobytes())
    Deal.objects.create(lead=lead, state=state, outcome=outcome)

    X, y = Lead.get_labeled_arrays()

    assert list(y) == ([] if label is None else [label])
    assert X.shape == (len(y), 384)
