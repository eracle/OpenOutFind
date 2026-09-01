# tests/test_find.py
"""`find` — the one verb that does work, and the only one that boots the install.

Two things are pinned here. **Boot: environment first, wizard only if a human is there to
answer** — the regression an agent-driven install hits first, where the tool used to die
on a missing TTY with a message that named a mailbox (gone with the sending leg) and never
said which variables to set. And **the command's own contract**: which site_config it acts
on, what it prints, and the fact that exit 0 means the goal was met and nothing else.
"""
import contextlib
import csv
import io
import json
import logging
import webbrowser
from unittest.mock import patch

import pytest
from django.core.management import call_command

from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.core.management.commands.find import Command
from openoutfind.core.readiness import check_ready
from openoutfind.enrichment import bettercontact

FULL_ENV = {
    "OPENOUTFIND_PRODUCT_DOCS": "A self-hosted CI dashboard for small dev teams",
    "OPENOUTFIND_CAMPAIGN_TARGET": "book demos with CTOs at Series-A SaaS",
    "OPENOUTFIND_AI_MODEL": "anthropic:claude-sonnet-4-5-20250929",
    "OPENOUTFIND_LLM_API_KEY": "sk-test",
    "OPENOUTFIND_BETTERCONTACT_API_KEY": "bc-test",
    "OPENOUTFIND_OPERATOR_EMAIL": "me@posteo.eu",
    "OPENOUTFIND_COUNTRY": "US",
    "OPENOUTFIND_ACCEPT_LEGAL_NOTICE": "true",
}


@pytest.fixture
def headless(monkeypatch, configure):
    """Nobody at a terminal, and none of the developer's own variables."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)


@pytest.fixture
def command():
    cmd = Command()
    cmd.stderr = io.StringIO()
    return cmd


@pytest.fixture
def booted(site_config):
    """Skip the preamble. Migrating and the readiness check have their own tests;
    neither is what the command contract asserts.

    Patched where `find` looks them up rather than where they are defined — the command
    imports both by name, so patching the defining module would rebind an attribute
    nothing reads."""
    with patch.object(Command, "_configure_logging"), \
            patch("openoutfind.core.management.commands.find.ensure_database"), \
            patch("openoutfind.core.management.commands.find.check_ready"):
        yield


@pytest.mark.django_db
def test_an_unconfigured_run_names_the_variables(headless):
    with pytest.raises(OpenOutFindError) as exc:
        check_ready()

    assert exc.value.error_type == ErrorType.ONBOARDING_INCOMPLETE
    message = str(exc.value)
    assert message.startswith("error: onboarding_incomplete: ")
    assert "OPENOUTFIND_PRODUCT_DOCS" in message
    assert "OPENOUTFIND_BETTERCONTACT_API_KEY" in message
    assert "OPENOUTFIND_ACCEPT_LEGAL_NOTICE" in message
    assert "mailbox" not in message.lower()


@pytest.mark.django_db
def test_a_fully_configured_environment_is_ready_with_nothing_asked(headless, monkeypatch):
    """There is no prompt to fall back to: what the environment does not carry, the run
    does not have."""
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)

    with patch("openoutfind.core.newsletter.subscribe_to_newsletter"):
        check_ready()

    from openoutfind.core.readiness import missing_variables
    assert missing_variables() == {}


# ── the command's contract ───────────────────────────────────────


@pytest.mark.django_db
class TestTheCommandContract:
    """What `find` prints, and what its exit code means.

    `call_command` raises whatever `handle` raises, so `OpenOutFindError` here is the
    non-zero exit the base command turns it into — see `tests/test_output_contract.py`.
    """

    def test_a_met_goal_prints_the_campaign_and_does_not_raise(self, site_config, booted):
        _exportable(site_config, "ada@acme.com")

        rows = _run("0")

        assert [row["email"] for row in rows] == ["ada@acme.com"]

    def test_zero_does_no_work_at_all(self, site_config, booted):
        with patch("openoutfind.core.cycle.run_one_action") as action:
            _run("0")

        action.assert_not_called()

    def test_an_unreached_goal_still_prints_its_rows_then_exits_non_zero(self, site_config, booted):
        """Seven leads are seven leads. The rows go to stdout either way, and the error
        line carries the type an agent branches on."""
        _exportable(site_config, "ada@acme.com")
        out = io.StringIO()

        with patch("openoutfind.core.cycle.run_one_action", return_value=False):
            with pytest.raises(OpenOutFindError) as exc:
                call_command("find", "5", stdout=out)

        assert exc.value.error_type == ErrorType.GOAL_UNREACHED
        assert "0 of 5 leads" in str(exc.value)
        assert len(list(csv.DictReader(io.StringIO(out.getvalue())))) == 1

    def test_the_whole_campaign_prints_not_just_this_run(self, site_config, booted):
        """What makes `> leads.csv` correct by construction: the newest file supersedes
        every earlier one."""
        _exportable(site_config, "old@acme.com")

        with patch("openoutfind.core.cycle.run_one_action",
                   side_effect=lambda c, buy_addresses=True, max_new_lookups=None: bool(
                       _exportable(c, "new@acme.com"))):
            rows = _run("1")

        assert {row["email"] for row in rows} == {"old@acme.com", "new@acme.com"}

    def test_new_narrows_to_what_this_run_produced(self, site_config, booted):
        _exportable(site_config, "old@acme.com")

        with patch("openoutfind.core.cycle.run_one_action",
                   side_effect=lambda c, buy_addresses=True, max_new_lookups=None: bool(
                       _exportable(c, "new@acme.com"))):
            rows = _run("1", "--new")

        assert [row["email"] for row in rows] == ["new@acme.com"]

    def test_json_puts_the_records_on_stdout_one_per_line(self, site_config, booted):
        """JSON Lines, so a stream truncated mid-run has still delivered every complete
        record before the break — and the full record, profile text included, which is
        the field the CSV projection drops."""
        _exportable(site_config, "ada@acme.com")
        out = io.StringIO()

        call_command("find", "0", "--json", stdout=out)

        lines = out.getvalue().splitlines()
        assert [json.loads(line)["email"] for line in lines] == ["ada@acme.com"]
        assert "profile_text" in json.loads(lines[0])

    def test_json_puts_the_run_metadata_on_stderr_and_nothing_else(self, site_config, booted, capsys):
        """Otherwise a `2>` capture is prose with an object somewhere in it, and every
        caller writes the same fragile `tail -1`."""
        _exportable(site_config, "ada@acme.com")

        call_command("find", "0", "--json", stdout=io.StringIO())

        document = json.loads(capsys.readouterr().err)  # a banner or a log line would raise
        assert document["reached"] is True and document["stopped_because"] is None
        assert document["goal"] == {"count": 0, "unit": "leads"}
        assert document["rows"] == 1

    def test_by_default_what_is_already_there_prints_before_any_work(self, site_config, booted):
        """The opening bulk: even a run that produces nothing yet has already printed
        the site_config as it stood at minute zero — unlike `--batch`, which waits."""
        _exportable(site_config, "old@acme.com")
        out = io.StringIO()

        with patch("openoutfind.core.cycle.run_one_action", return_value=False):
            with pytest.raises(OpenOutFindError):
                call_command("find", "5", stdout=out)

        assert [row["email"] for row in csv.DictReader(io.StringIO(out.getvalue()))] == \
            ["old@acme.com"]

    def test_by_default_a_new_lead_streams_after_the_opening_bulk(self, site_config, booted):
        """The point of streaming: a caller piping into a live sender sees each lead as
        it lands, not collected behind the whole job."""
        _exportable(site_config, "old@acme.com")
        out = io.StringIO()

        with patch("openoutfind.core.cycle.run_one_action",
                   side_effect=lambda c, buy_addresses=True, max_new_lookups=None: bool(
                       _exportable(c, "new@acme.com"))):
            call_command("find", "1", "--json", stdout=out)

        lines = out.getvalue().splitlines()
        assert [json.loads(line)["email"] for line in lines] == ["old@acme.com", "new@acme.com"]

    def test_new_skips_the_bulk_but_still_streams_the_new_lead(self, site_config, booted):
        """`--new` narrows what prints, not whether streaming happens."""
        _exportable(site_config, "old@acme.com")
        out = io.StringIO()

        with patch("openoutfind.core.cycle.run_one_action",
                   side_effect=lambda c, buy_addresses=True, max_new_lookups=None: bool(
                       _exportable(c, "new@acme.com"))):
            call_command("find", "1", "--new", "--json", stdout=out)

        lines = out.getvalue().splitlines()
        assert [json.loads(line)["email"] for line in lines] == ["new@acme.com"]

    def test_streaming_puts_the_run_metadata_on_stderr_and_nothing_else(self, site_config, booted,
                                                                        capsys):
        _exportable(site_config, "old@acme.com")

        with patch("openoutfind.core.cycle.run_one_action",
                   side_effect=lambda c, buy_addresses=True, max_new_lookups=None: bool(
                       _exportable(c, "new@acme.com"))):
            call_command("find", "1", "--json", stdout=io.StringIO())

        document = json.loads(capsys.readouterr().err)
        assert document["reached"] is True
        assert document["rows"] == 2  # the opening bulk's one row, plus the one produced

    def test_batch_restores_the_old_atomic_shape(self, site_config, booted):
        """`--batch` is the escape hatch: still the whole site_config, still one write, just
        held until the job ends instead of streamed as it goes."""
        _exportable(site_config, "old@acme.com")
        out = io.StringIO()

        with patch("openoutfind.core.cycle.run_one_action",
                   side_effect=lambda c, buy_addresses=True, max_new_lookups=None: bool(
                       _exportable(c, "new@acme.com"))):
            call_command("find", "1", "--batch", stdout=out)

        rows = list(csv.DictReader(io.StringIO(out.getvalue())))
        assert {row["email"] for row in rows} == {"old@acme.com", "new@acme.com"}

    def test_batch_never_writes_before_the_job_ends(self, site_config, booted):
        """The property `--batch` exists for: a consumer that cannot take a partial
        write gets nothing until `_report` runs, once, after the job."""
        _exportable(site_config, "old@acme.com")

        class _NoWriteBeforeReport(io.StringIO):
            def write(self, value):
                assert False, "wrote to stdout before the job ended"

        out = _NoWriteBeforeReport()
        real_report = Command._report

        def patched_report(self, result, options, writer):
            _NoWriteBeforeReport.write = io.StringIO.write  # writing is fine from here
            return real_report(self, result, options, writer)

        with patch("openoutfind.core.cycle.run_one_action", return_value=False), \
                patch.object(Command, "_report", patched_report):
            with pytest.raises(OpenOutFindError):
                call_command("find", "5", "--batch", stdout=out)

    def test_a_negative_count_is_refused(self, site_config, booted):
        with pytest.raises(OpenOutFindError) as exc:
            call_command("find", "-1", stdout=io.StringIO())

        assert exc.value.error_type == ErrorType.BAD_CONFIG

    def test_buying_is_off_by_default(self, site_config, booted):
        """A bare `find` cannot spend, however many deals are queued past the gate.

        This is the inversion of 2026-08-21: buying used to be on unless `--no-emails`
        turned it off, so a run counting *leads* quietly bought addresses. A flag you
        forget should cost a feature, never money.
        """
        _exportable(site_config, "ada@acme.com")

        with patch("openoutfind.core.cycle.run_one_action",
                   return_value=False) as action:
            with pytest.raises(OpenOutFindError):
                call_command("find", "1", stdout=io.StringIO())

        assert action.call_args.kwargs["buy_addresses"] is False

    def test_emails_flag_reaches_the_cycle(self, site_config, booted):
        """The flag is only worth having if it arrives where the spending happens."""
        with patch("openoutfind.core.cycle.run_one_action",
                   return_value=False) as action:
            with pytest.raises(OpenOutFindError):
                call_command("find", "1", "--emails", stdout=io.StringIO())

        assert action.call_args.kwargs["buy_addresses"] is True

    def test_an_emails_goal_implies_the_flag(self, site_config, booted):
        """The noun says what to count and the flag says what may be paid for — but a
        goal counted in addresses cannot be met without buying them."""
        with patch("openoutfind.core.cycle.run_one_action",
                   return_value=False) as action:
            with pytest.raises(OpenOutFindError):
                call_command("find", "5", "emails", stdout=io.StringIO())

        assert action.call_args.kwargs["buy_addresses"] is True

    def test_open_without_a_browser_fails_before_any_work(self, site_config, booted):
        """A flag that silently does nothing is the bug you find at 2am."""
        with patch("webbrowser.get", side_effect=webbrowser.Error), \
                patch("openoutfind.core.cycle.run_one_action") as action:
            with pytest.raises(OpenOutFindError) as exc:
                call_command("find", "1", "--open", stdout=io.StringIO())

        assert exc.value.error_type == ErrorType.BAD_CONFIG
        action.assert_not_called()

    def test_minute_zero_states_the_goal_and_whether_it_can_spend(self, site_config, booted, caplog):
        """Spending is opt-in at every layer, which is a good default and an invisible
        one. An operator who expected addresses should learn it in the first line, not
        from an empty column at the end."""
        with caplog.at_level(logging.INFO):
            call_command("find", "0", stdout=io.StringIO())

        assert "finding only, no addresses bought" in caplog.text

    def test_asking_to_buy_says_so_before_any_work(self, site_config, booted, caplog):
        with patch("openoutfind.core.cycle.run_one_action", return_value=False), \
                caplog.at_level(logging.INFO):
            with pytest.raises(OpenOutFindError):
                call_command("find", "1", "--emails", stdout=io.StringIO())

        assert "buying addresses, one credit each" in caplog.text

    def test_the_icp_echo_names_who_it_is_looking_for(self, site_config, booted, caplog):
        """The earliest possible proof the product description was understood — and the
        earliest chance to correct it, which is the loop the README sells."""
        from openoutfind.crm.models import Lead

        Lead.objects.create(
            synthetic=True,
            profile_text="vp of engineering saas acme senior california united states")

        with caplog.at_level(logging.INFO):
            call_command("find", "0", stdout=io.StringIO())

        assert "Looking for people like:" in caplog.text
        assert "vp of engineering saas acme" in caplog.text

    def test_an_unanchored_campaign_echoes_nothing(self, site_config, booted, caplog):
        """A first run has no anchors yet — they are written during the job, and print
        themselves there. Silence beats a heading with nothing under it."""
        with caplog.at_level(logging.INFO):
            call_command("find", "0", stdout=io.StringIO())

        assert "Looking for people like:" not in caplog.text

    def test_the_run_ends_with_the_ask_and_the_csv_stays_a_csv(self, site_config, booted, caplog):
        """A run that leaves ranked leads behind and an empty wallet has to say so.

        The sentence is `status`'s, rendered here — the run derives nothing. It goes to
        stderr with everything else that is not a row: a stray line in a CSV is not a
        CSV, and this one carries a URL.
        """
        _exportable(site_config, "ada@acme.com")
        _ranked(site_config)
        out = io.StringIO()

        with _wallet(balance=0), caplog.at_level(logging.INFO):
            call_command("find", "0", stdout=out)

        assert "0 credits left" in caplog.text
        assert bettercontact.SIGNUP_URL in caplog.text
        # Both rows export — the ranked one with a blank address, which is the whole
        # reason it is still waiting.
        rows = list(csv.DictReader(io.StringIO(out.getvalue())))
        assert sorted(row["email"] for row in rows) == ["", "ada@acme.com"]

    def test_json_carries_the_next_action_for_the_agent_to_relay(self, site_config, booted,
                                                                 caplog, capsys):
        """An agent reads the object, not the log, so the ask has to be in it."""
        _ranked(site_config)

        with _wallet(balance=0), caplog.at_level(logging.INFO):
            call_command("find", "0", "--json", stdout=io.StringIO())

        document = json.loads(capsys.readouterr().err)
        assert document["next_action"]["type"] == "add_credits"
        assert document["next_action"]["leads"] == 1
        assert "Next:" not in caplog.text  # the object is the whole answer

    def test_debug_is_the_shorthand_for_log_level_debug(self, site_config, booted):
        """Both flags write the same dest, so they cannot disagree."""
        from openoutfind.core.management.commands.find import Command

        with patch("openoutfind.core.cycle.run_one_action", return_value=False), \
                patch.object(Command, "_configure_logging") as configure:
            call_command("find", "0", "--debug", stdout=io.StringIO())

        assert configure.call_args.args[0] == "debug"


# ── helpers ──────────────────────────────────────────────────────


def _exportable(site_config, email):
    """One lead the export would write: judged, accepted, and carrying an address."""
    from openoutfind.crm.models import DealState
    from tests.factories import DealFactory, LeadFactory

    return DealFactory(lead=LeadFactory(email=email),
                       state=DealState.RESOLVED, reason="fits the ICP")


def _ranked(site_config):
    """One lead that cannot advance without a credit."""
    from openoutfind.crm.models import DealState
    from tests.factories import DealFactory, LeadFactory

    return DealFactory(lead=LeadFactory(email=None),
                       state=DealState.READY_TO_FIND_EMAIL, reason="fits the ICP")


@contextlib.contextmanager
def _wallet(balance):
    """A configured provider with a known balance, and the configuration out of the way —
    the two inputs the next action is derived from."""
    with patch("openoutfind.core.readiness.missing_variables", return_value={}), \
            patch("openoutfind.enrichment.bettercontact.is_configured", return_value=True), \
            patch("openoutfind.enrichment.bettercontact.credit_balance", return_value=balance):
        yield


def _run(*args):
    """Run `find` and parse the CSV it printed."""
    out = io.StringIO()
    call_command("find", *args, stdout=out)
    return list(csv.DictReader(io.StringIO(out.getvalue())))
