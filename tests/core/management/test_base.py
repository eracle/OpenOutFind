# tests/core/management/test_base.py
"""The CLI's error contract — the part a program depends on.

**An expected failure is one typed line** on stderr with a non-zero exit, so an agent
branches instead of parsing prose; a caller that asked for JSON gets it as JSON; and stdout
stays result-only either way.
"""
import json
from unittest.mock import patch

import pytest

from openoutfind.core.errors import ErrorType, OpenOutFindError, format_error
from openoutfind.core.management.base import OpenOutFindCommand

pytestmark = pytest.mark.django_db


def _failing(error, with_json=False):
    class Failing(OpenOutFindCommand):
        def add_arguments(self, parser):
            if with_json:
                parser.add_argument("--json", action="store_true", dest="as_json")

        def handle(self, *args, **options):
            raise error

    return Failing()


def test_error_line_has_the_contract_shape():
    assert format_error("provider_auth", "boom") == "error: provider_auth: boom"
    assert str(OpenOutFindError(ErrorType.PROVIDER_AUTH, "boom")) == \
        "error: provider_auth: boom"


def test_command_renders_the_error_line_and_exits_non_zero(capsys):
    """No traceback: a rejected key is an answer, not a bug."""
    command = _failing(OpenOutFindError(ErrorType.PROVIDER_AUTH, "the key was rejected"))

    with pytest.raises(SystemExit) as exc:
        command.run_from_argv(["openoutfind", "failing"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "error: provider_auth: the key was rejected"
    assert captured.out == ""


def test_json_callers_get_the_failure_as_json(capsys):
    """Same vocabulary, in the shape a parser consumes — and still on stderr, or
    `find --json > leads.json` would collect an error object into the operator's file."""
    command = _failing(OpenOutFindError(ErrorType.BAD_CONFIG, "no key"), with_json=True)

    with pytest.raises(SystemExit) as exc:
        command.run_from_argv(["openoutfind", "failing", "--json"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"error": {"type": "bad_config", "message": "no key"}}
    assert captured.out == ""


def test_a_payload_rides_under_json_and_stays_off_the_plain_line(capsys):
    """`qualify_pending`'s candidate — the one error that needs more than a type and a
    message to be acted on."""
    error = OpenOutFindError(ErrorType.QUALIFY_PENDING, "Ada Lovelace needs a verdict",
                             payload={"profile_text": "engineer at acme", "company": "Acme"})

    with pytest.raises(SystemExit):
        _failing(error, with_json=True).run_from_argv(["openoutfind", "failing", "--json"])

    document = json.loads(capsys.readouterr().err)
    assert document["error"]["type"] == "qualify_pending"
    assert document["error"]["company"] == "Acme"
    assert str(error) == "error: qualify_pending: Ada Lovelace needs a verdict"


def test_an_unexpected_exception_still_raises():
    """Only *expected* failures are flattened; a bug keeps its traceback."""
    with pytest.raises(ZeroDivisionError):
        _failing(ZeroDivisionError("a real bug")).run_from_argv(["openoutfind", "buggy"])


# ── a fresh install is an answer, not a traceback ────────────────

def test_reading_an_unmigrated_database_is_a_typed_error(capsys):
    """`outfind status` before the first run used to raise `no such table`."""
    command = _failing(AssertionError("the guard should have stopped this"))

    with patch("django.db.connection.introspection.table_names", return_value=[]):
        with pytest.raises(SystemExit) as exc:
            command.run_from_argv(["openoutfind", "reader"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith(f"error: {ErrorType.NOT_INITIALIZED}: ")
    assert captured.out == ""


def test_the_verb_that_migrates_is_not_guarded(capsys):
    """`find` creates the schema, so it must be allowed to find none."""
    class Migrating(OpenOutFindCommand):
        requires_database = False

        def handle(self, *args, **options):
            self.stdout.write("ran")

    with patch("django.db.connection.introspection.table_names", return_value=[]):
        Migrating().run_from_argv(["openoutfind", "migrating"])

    assert capsys.readouterr().out.strip() == "ran"
