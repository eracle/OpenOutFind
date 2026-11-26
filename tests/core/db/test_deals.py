# tests/core/db/test_deals.py
"""State-transition logging in `core/db/deals.set_profile_state`.

A `no email` enrichment miss is a benign, expected terminal outcome: it logs as a muted
`NO EMAIL`, never as the red `FAILED` reserved for genuine failures — the operator reads
this line while the run works.
"""
import logging

import numpy as np
import pytest

from openoutfind.core.db.deals import set_profile_state
from openoutfind.core.db.leads import promote_lead_to_deal
from openoutfind.crm.models import DealState, Lead


@pytest.mark.django_db
@pytest.mark.parametrize("state,shown,hidden", [
    (DealState.NO_EMAIL_FOUND, "NO EMAIL", "FAILED"),
    (DealState.FAILED, "FAILED", "NO EMAIL"),
])
def test_a_miss_and_a_failure_log_apart(site_config, caplog, state, shown, hidden):
    url = "https://www.linkedin.com/in/alice/"
    Lead.objects.create(profile_url=url, profile_text="engineer at acme",
                        embedding=np.ones(384, dtype=np.float32).tobytes())
    promote_lead_to_deal(url)

    with caplog.at_level(logging.INFO, logger="openoutfind.core.db.deals"):
        set_profile_state(url, state.value, reason="wrong fit")

    assert shown in caplog.text
    assert hidden not in caplog.text
