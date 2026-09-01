# openoutfind/core/management/bootstrap.py
"""The things that must be true before any finding can start.

These used to be private methods on the `find` command, which made *getting ready* and
*finding leads* one verb with one exit code. They are shared now because `check` exists to
do exactly this and nothing else — and a phase worth its own verb is a phase worth naming
in one place rather than two.

The order matters and is not arbitrary: there is no schema to write an operator into
until the migrations run.
"""
from __future__ import annotations

import logging

from django.core.management import call_command

logger = logging.getLogger(__name__)


def ensure_database(stderr) -> None:
    """Migrate to the current schema and make sure the CRM's fixtures exist.

    ``stderr`` is where Django's migration narration goes. That is not a style choice:
    stdout carries the CSV, and a stray "Applying core.0001_initial… OK" in a redirected
    file is exactly what the output contract exists to prevent.
    """
    call_command("migrate", "--no-input", stdout=stderr)

    from openoutfind.core.management.setup_crm import setup_crm
    setup_crm()
