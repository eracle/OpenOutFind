# openoutfind/core/management/base.py
"""The base command that implements the CLI's output contract.

Three rules, and they exist because the reader is as often a program as a person:

  * **stdout is result-only** — the thing you would pipe into ``jq`` or a file. Logs
    and progress go to stderr (``core/logging.py``), so redirecting stdout yields
    data and nothing else.
  * **errors are one line with a stable type** — ``error: <type>: <message>`` on
    stderr, from the vocabulary in ``core/errors.py``, and a non-zero exit. Under
    ``--json`` that line becomes ``{"error": {"type", "message"}}``, still on stderr:
    a caller that asked for JSON is parsing rather than reading, and the same
    vocabulary answers both.
  * **no traceback for an expected failure.** A rejected API key is not a bug; it is
    an answer, and it should read like one.

A freshly installed tool is the fourth case, and it used to be a raw Django traceback:
the database file exists (``settings.py`` creates its directory) but has no schema until
``run`` migrates, so asking anything else first hit ``no such table: core_siteconfig``.
"""
from __future__ import annotations

import json
import sys

from django.core.management.base import BaseCommand

from openoutfind.core.errors import ErrorType, OpenOutFindError


class OpenOutFindCommand(BaseCommand):
    """A management command whose expected failures obey the error contract."""

    requires_database = True
    """Set ``False`` on a verb that creates the schema rather than reading it."""

    def execute(self, *args, **options):
        """Guard the schema after argument parsing, so ``--help`` still answers."""
        if self.requires_database:
            require_initialized_database()
        return super().execute(*args, **options)

    def run_from_argv(self, argv):
        """Render ``OpenOutFindError`` in the caller's own format, then exit non-zero.

        Anything else keeps Django's behaviour — an unexpected exception is a bug and
        deserves its traceback.
        """
        try:
            super().run_from_argv(argv)
        except OpenOutFindError as exc:
            sys.stderr.write(format_failure(exc, as_json="--json" in argv))
            sys.exit(1)


def format_failure(exc: OpenOutFindError, *, as_json: bool) -> str:
    """The failure as the caller asked to be spoken to — one line, or one object.

    **Both go to stderr.** A caller that passed ``--json`` is parsing, not reading, so
    an error it cannot parse is barely better than none; but stdout stays result-only
    either way, or ``find 10 --json > leads.json`` would write an error object into the
    file the operator is keeping.
    """
    if as_json:
        return json.dumps({"error": {"type": exc.error_type, "message": exc.message}}) + "\n"
    return f"{exc}\n"


def require_initialized_database() -> None:
    """Refuse to read a database that is missing or behind on migrations.

    Answering with zero leads instead would be the empty-result failure the error
    vocabulary exists to prevent: nothing was found because nothing has ever run. A
    schema that exists but is behind (a ``.venv``/checkout refreshed against a DB file
    from before some migration) hits the same class of failure — a raw
    ``OperationalError: no such column`` — so it gets the same gentle message rather
    than a traceback.
    """
    from django.conf import settings
    from django.db import connection
    from django.db.migrations.executor import MigrationExecutor

    from openoutfind.core.models import SiteConfig

    # Asked of the model rather than spelled out: the app label is namespaced when
    # OpenOutreach hosts these apps, and a literal table name would go stale silently.
    if SiteConfig._meta.db_table not in connection.introspection.table_names():
        raise OpenOutFindError(
            ErrorType.NOT_INITIALIZED,
            f"no pipeline yet at {settings.DATABASE_PATH} — run `outfind init` to create it",
        )

    executor = MigrationExecutor(connection)
    if executor.migration_plan(executor.loader.graph.leaf_nodes()):
        raise OpenOutFindError(
            ErrorType.NOT_INITIALIZED,
            f"{settings.DATABASE_PATH} is behind on migrations — run `outfind find 0` to bring it current",
        )
