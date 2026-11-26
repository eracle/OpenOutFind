# tests/conftest.py
"""What every test stands on: no network, no model download, no state left behind.

The guards below are autouse because the code they stub is reached *incidentally* — a
readiness check, a stored lead, a prompt — by tests about something else. A test *of*
one of the stubbed functions does not opt out with a marker: it imports the function by
name at module level, which binds the real one before any patch here is applied.
"""
import os
from unittest.mock import patch

import numpy as np
import pytest
import requests

from tests.factories import UserFactory


# ── no network, no model ─────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_live_writes_to_our_own_services():
    """No test may write to the real hub.

    It is reached by *completing onboarding*, which many tests do on their way to
    something else: `_ensure_operator` mints the operator's hub token. Unguarded, anyone's
    `make test` POSTs a fabricated operator into **production**. The caller is
    best-effort, so a refused connection is the no-op it already handles. Tests of the
    client patch the same target themselves and win, because their patch is inside this.
    """
    refuse = requests.ConnectionError("no network in tests")
    with patch("openoutfind.contacts.service.requests.post", side_effect=refuse):
        yield


@pytest.fixture(autouse=True)
def _no_embedding_model():
    """Stub fastembed so tests don't need the ONNX model."""
    with patch("openoutfind.core.ml.embeddings.embed_text", return_value=np.ones(384)), \
            patch("openoutfind.core.ml.embeddings.embed_texts",
                  side_effect=lambda texts: np.ones((len(texts), 384))):
        yield


@pytest.fixture(autouse=True)
def _no_live_model_ping():
    """No test may reach a real LLM provider to check a key.

    The check is unconditional — nothing is stored, so every run verifies what it was
    given — so any test that reaches a readiness check would otherwise be a live 401.
    """
    with patch("openoutfind.core.llm.verify_llm_credentials", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _no_live_brief_writer():
    """No test may have a model write a brief.

    Every prompt reads the brief first (``pipeline/brief.py``), so a test that patches one
    prompt's ``Agent`` would otherwise feed that fake to the brief's writer too.
    """
    with patch("openoutfind.core.pipeline.brief._write_with_llm", return_value=""):
        yield


@pytest.fixture(autouse=True)
def _no_live_keyword_guesses():
    """No test may have a model guess the walk's market or keywords: the stub guesses a
    market and no keyword, so every keyword reads the store's share and the market's rate."""
    from openoutfind.core.pipeline.keyword_guess import FieldAnswer, MarketAnswer

    target = "openoutfind.core.pipeline.keyword_guess"
    with patch(f"{target}.guess_market_with_llm",
               return_value=MarketAnswer(people=10_000_000, fits=10_000)), \
            patch(f"{target}.guess_field_with_llm", return_value=FieldAnswer(keywords=[])):
        yield


# ── no state survives a test ─────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_fit_survives_a_test():
    """``qualifier_for`` keeps the fitted model, keyed on the labels, and ``belief`` its
    posterior, keyed on the pages and verdicts.

    That key cannot go stale inside a run, but a test database reuses primary keys, so two
    tests can share one label set and different intent.
    """
    import openoutfind.core.cycle as cycle_module
    import openoutfind.core.ml.qualifier as qualifier_module
    import openoutfind.core.pipeline.belief as belief_module
    import openoutfind.core.pipeline.store_snapshot as store_snapshot_module

    def clear():
        qualifier_module._FITTED = None
        cycle_module._scored_at = None
        store_snapshot_module.FIT_RATE_CACHE = None
        belief_module.FITTED = None

    clear()
    yield
    clear()


@pytest.fixture(autouse=True)
def _no_run_mode_survives_a_test():
    """``--agent-qualify`` and ``--exclude`` live in contextvars, which a test's thread
    carries into the next test unless they are put back."""
    from openoutfind.core import agent_qualify, exclude

    yield
    agent_qualify._active.set(False)
    agent_qualify._verdict.set(None)
    agent_qualify._population.set(None)
    agent_qualify._keyword_guesses.set(None)
    exclude._excluded.set(frozenset())


# ── the install under test ───────────────────────────────────────


@pytest.fixture
def operator(db):
    """The onboarded operator — what ``core.operator.get_active_user()`` will find."""
    return UserFactory(username="testuser", email="testuser@example.com")


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
    """The configuration under test — blank, as an install that was told nothing."""
    return configure()
