# tests/test_defaults.py
"""Where the tool writes when it is installed rather than checked out.

Installed from a wheel, ``ROOT_DIR`` is inside site-packages: neither a CRM nor a model
download belongs there.
"""
from __future__ import annotations

from pathlib import Path

from openoutfind.defaults import app_settings, state_dir


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


def test_the_model_cache_follows_the_state_dir_not_the_database(tmp_path):
    """``--db /tmp/scratch.sqlite3`` must not send fastembed off to re-download weights."""
    scratch = tmp_path / "scratch" / "db.sqlite3"

    names = app_settings(tmp_path / "state", db_path=scratch)

    assert names["DATABASE_PATH"] == scratch
    assert names["FASTEMBED_CACHE_DIR"] == tmp_path / "state" / ".cache" / "fastembed"
