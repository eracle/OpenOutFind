# tests/conftest.py
from unittest.mock import patch

import numpy as np
import pytest
import requests

from openoutfind.core.management.setup_crm import setup_crm
from tests.factories import UserFactory


@pytest.fixture(autouse=True)
def _ensure_crm_data(db):
    """
    Ensure CRM bootstrap data exists before every test.
    Uses `db` fixture (not transactional_db) for compatibility.
    Since transaction=True tests rollback, we re-create data each time.
    """
    setup_crm()


@pytest.fixture(autouse=True)
def _no_live_writes_to_our_own_services():
    """No test may write to the real hub or the real mailing list.

    Both are reached by *completing onboarding*, which many tests do incidentally on
    their way to something else: `_finalize_account` mints the operator's hub token
    and, on a yes, subscribes them to the newsletter. Unguarded, anyone's `make test`
    POSTs a fabricated operator into **production** — a service holding other
    people's contributions — and signs a fake address up to the list.

    Both callers are best-effort by design, so a refused connection is exactly the
    no-op they already handle. Tests that exercise either client patch the same
    target themselves and win, because their patch is applied inside this one.
    """
    refuse = requests.ConnectionError("no network in tests")
    with patch("openoutfind.contacts.service.requests.post", side_effect=refuse), \
         patch("openoutfind.core.newsletter.requests.post", side_effect=refuse):
        yield


@pytest.fixture(autouse=True)
def _mock_embeddings(request):
    """Stub fastembed so tests don't need the ONNX model."""
    if "no_embed_mock" in request.keywords:
        yield
    else:
        with patch("openoutfind.core.ml.embeddings.embed_text", return_value=np.ones(384)):
            yield


@pytest.fixture(autouse=True)
def _no_fit_survives_a_test():
    """``qualifier_for`` keeps the fitted model, keyed on the labels.

    That key cannot go stale inside a run, but a test database reuses primary keys, so
    two tests can share one label set and different intent. Each test starts from an
    empty cache.
    """
    import openoutfind.core.ml.qualifier as qualifier_module
    import openoutfind.core.pipeline.vocabulary as vocabulary_module
    import openoutfind.core.cycle as cycle_module

    qualifier_module._FITTED = None
    vocabulary_module._refreshed_at = None
    cycle_module._scored_at = None
    yield
    qualifier_module._FITTED = None
    vocabulary_module._refreshed_at = None
    cycle_module._scored_at = None


@pytest.fixture
def operator(db):
    """The onboarded operator — what ``core.operator.get_active_user()`` will find."""
    return UserFactory(username="testuser", email="testuser@example.com")


@pytest.fixture
def site_config(db, operator):
    """The ``SiteConfig`` singleton under test.

    Holds exactly the fields the old ``Campaign`` model held, folded onto the config
    singleton (2026-08-30) since this install has never run more than one campaign.
    Steps and pipeline functions take it directly; the operator is looked up
    (``core/operator.py``) rather than threaded through, so nothing carries a session
    object either.
    """
    from openoutfind.core.models import SiteConfig

    return SiteConfig.load()
