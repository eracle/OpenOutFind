# tests/conftest.py
import os
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


@pytest.fixture(autouse=True)
def _no_live_model_ping(request):
    """No test may reach a real LLM provider to check a key.

    The check is unconditional now — nothing is stored, so every run verifies what it was
    given — which means any test that reaches a readiness check turns into a live 401
    unless this stands. Tests *about* verification carry ``no_llm_mock`` and get the real
    function; tests that only pass through it patch the same target themselves and win,
    because their patch is applied inside this one.
    """
    if "no_llm_mock" in request.keywords:
        yield
    else:
        with patch("openoutfind.core.llm.verify_llm_credentials", return_value=None):
            yield


@pytest.fixture
def configure(monkeypatch):
    """Set this run's configuration, the only way an install is configured: the environment.

    Returns a callable taking field names — ``configure(llm_api_key="k")`` — so a test
    names the values it depends on and inherits blanks for the rest. Every
    ``OPENOUTFIND_*`` variable is cleared first, so a developer's own exported keys can
    never make a test pass (or spend).
    """
    from openoutfind.core.config import ENV_PREFIX, SiteConfig, variable_for

    for name in [n for n in os.environ if n.startswith(ENV_PREFIX)]:
        monkeypatch.delenv(name)

    def _configure(**values) -> SiteConfig:
        for field, value in values.items():
            monkeypatch.setenv(variable_for(field), value)
        return SiteConfig.load()

    return _configure


@pytest.fixture
def site_config(db, operator, configure):
    """The configuration under test — blank, as an install that was told nothing.

    An install runs exactly one ICP, so there is nothing to name or select between.
    Pipeline functions take this value directly; the operator is looked up
    (``core/operator.py``) rather than threaded through, so nothing carries a session
    object either.
    """
    return configure()
