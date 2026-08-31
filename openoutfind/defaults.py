# openoutfind/defaults.py
"""What a host project must put in its settings for `openoutfind`'s apps to run.

This repo's own `settings.py` is one host; OpenOutreach, which installs these apps
in its registry next to OpenOutSend's, is another. Both splat `app_settings()`
rather than each spelling the names out, so a new requirement lands in one place
instead of drifting between two settings modules.

Only what the *apps* read belongs here. `MEDIA_ROOT`/`MEDIA_URL` do not: they are
read by `openoutfind/urls.py`, which is this project's URLconf, and a host that
brings its own URLconf owns them itself.
"""
from __future__ import annotations

import os
from pathlib import Path

#: The apps, in dependency order, for a host's `INSTALLED_APPS`.
APPS = [
    "openoutfind.crm.apps.CrmConfig",
    "openoutfind.core.apps.CoreConfig",
]


def state_dir(root: Path) -> Path:
    """Where the operator's own files live: the checkout, or `~/.openoutfind` installed.

    Installed from a wheel, `root` is inside site-packages — no place for an operator's
    CRM or a model cache, and possibly not writable. A checkout keeps its own `data/`
    and `.cache/` only because it already has them.
    """
    return root if (root / "manage.py").exists() else Path.home() / ".openoutfind"


def database_path(state: Path) -> Path:
    """The SQLite file. `OPENOUTFIND_DB` (what `--db PATH` sets) names another one."""
    override = os.environ.get("OPENOUTFIND_DB")
    return Path(override or state / "data" / "db.sqlite3").expanduser()


def allow_async_unsafe() -> None:
    """Relax Django's async-safety guard, which a host must do before `django.setup()`.

    The agents drive async pydantic-ai from a sync boundary (`core/llm.py`), so an event
    loop can be live on the thread when the ORM is touched. We only use the ORM
    synchronously, so the guard has nothing to protect here.
    """
    os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")


def app_settings(state: Path, db_path: Path | None = None) -> dict:
    """The settings names `openoutfind`'s apps read, for a host to splat into its module.

    `db_path` lets a host that owns the database — the OpenOutreach orchestrator, whose
    one file also holds OpenOutSend's tables — name it; alone, `state` decides.
    """
    return {
        "DATABASE_PATH": db_path or database_path(state),
        # Deliberately *not* derived from the database path: `--db /tmp/scratch.sqlite3`
        # must not send fastembed off to re-download its weights beside a throwaway DB.
        "FASTEMBED_CACHE_DIR": state / ".cache" / "fastembed",
    }
