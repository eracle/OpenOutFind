# tests/test_paths.py
"""Where the tool writes when it is installed rather than checked out.

Installed from a wheel, ``ROOT_DIR`` is inside site-packages: the CRM was moved out of
it when the package was built, but the fastembed cache — a model download, not a few
kilobytes — was still pointed there.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.test import override_settings

from openoutfind.defaults import state_dir


# ── the checkout / installed split ───────────────────────────────

def test_checkout_keeps_its_own_state(tmp_path):
    (tmp_path / "manage.py").touch()
    assert state_dir(tmp_path) == tmp_path


def test_installed_state_leaves_the_package(tmp_path):
    """No ``manage.py`` beside the package means a wheel — nothing may be written there."""
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()

    resolved = state_dir(site_packages)

    assert resolved == Path.home() / ".openoutfind"
    assert site_packages not in resolved.parents


# ── the model cache follows the state dir, not the database ──────

def test_cache_dir_is_not_derived_from_the_database_path(tmp_path):
    """``--db /tmp/scratch.sqlite3`` must not send fastembed off to re-download weights."""
    with override_settings(DATABASE_PATH=tmp_path / "scratch.sqlite3"):
        assert tmp_path not in settings.FASTEMBED_CACHE_DIR.parents


def test_model_loads_from_the_configured_cache_dir(tmp_path):
    cache_dir = tmp_path / ".cache" / "fastembed"
    text_embedding = MagicMock()

    with override_settings(FASTEMBED_CACHE_DIR=cache_dir), \
            patch("openoutfind.core.ml.embeddings._model", None), \
            patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=text_embedding)}):
        from openoutfind.core.ml.embeddings import _get_model

        _get_model()

    assert cache_dir.is_dir()
    assert text_embedding.call_args.kwargs["cache_dir"] == str(cache_dir)
