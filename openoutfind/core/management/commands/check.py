"""Say whether this install can find anything, print what it was told, and stop.

    outfind check                                   # the config this run would use
    outfind check --product-docs p.md --target t.md # the two long fields, from files
    outfind check --json                            # the same config, for a program

**This phase already happened; it just never had a name.** `find` migrates the database
and checks its configuration before doing any of the work it is named after — and
announces the config with a single INFO line between the migration narration and the
discovery walk. Getting ready and finding leads are different jobs with different failure
modes, so they are different verbs.

`find` still does all of it, because a fully configured environment must keep running end
to end in one command. What changes is that there is a way to see what you actually
configured, and only then spend anything.

**Nothing is asked and nothing is written down.** Configuration is read from
`OPENOUTFIND_*` on every run (`core/config.py`); this verb reports what that environment
says, checks the model answers to it, and creates the operator row on a first run. An
install that would rather be asked runs the wizard in OpenOutreach.

**The two long fields come from files, not from flags.** A product description is a page
of markdown with newlines and apostrophes in it; shell-quoting that on a command line is a
way to corrupt it quietly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from openoutfind.core.config import variable_for
from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.core.management.bootstrap import ensure_database
from openoutfind.core.management.base import OpenOutFindCommand
from openoutfind.core.readiness import check_ready

# Which configuration field each file flag stands in for. The flags are a second way to
# supply two of the values, never a second place that knows what the values *are*.
_FILE_FLAGS = {
    "product_docs": "product_docs",
    "target": "campaign_target",
}


class Command(OpenOutFindCommand):
    help = "Check this install is ready to find, then print what it was told."

    # The verb that migrates — it is the first thing anyone runs.
    requires_database = False

    def add_arguments(self, parser):
        parser.add_argument("--product-docs", metavar="FILE",
                            help="File holding the product description (markdown).")
        parser.add_argument("--target", metavar="FILE",
                            help="File holding the target market description (markdown).")
        parser.add_argument("--json", action="store_true", dest="as_json",
                            help="Emit the config as one JSON object.")
        parser.add_argument(
            "--log-level",
            choices=("debug", "info", "warning", "error"),
            help="Log verbosity (default: info).",
        )
        # Same dest, so the two cannot disagree: whichever comes last on the command
        # line wins. `--debug` is the one an operator reaches for mid-run.
        parser.add_argument("--debug", action="store_const", const="debug",
                            dest="log_level", help="Shorthand for --log-level debug.")

    def handle(self, *args, **options):
        from openoutfind.core.logging import configure_logging, print_banner, resolve_log_level

        configure_logging(level=resolve_log_level(options.get("log_level"), options["verbosity"]))
        print_banner()

        ensure_database(self.stderr)
        _seed_environment(options)
        check_ready()

        self._report(_describe(), options)

    def _report(self, described: dict, options) -> None:
        if options["as_json"]:
            self.stdout.write(json.dumps(described, indent=2))
            return

        self.stdout.write(f"  country     {described['country_code'] or '—'}")
        self.stdout.write(f"  product     {described['product_docs_chars']} chars")
        self.stdout.write(f"  target      {described['campaign_target_chars']} chars")
        self.stdout.write(f"  model       {described['ai_model']}")
        self.stdout.write(f"  leads       {described['leads_seen']} seen, "
                          f"{described['exportable']} exportable")
        self.stdout.write("")
        self.stdout.write("Next: outfind find 1")


# ── the two long fields ──────────────────────────────────────────

def _seed_environment(options) -> None:
    """Put the contents of ``--product-docs`` / ``--target`` where the config is read.

    Writing to ``os.environ`` rather than passing them down is deliberate:
    ``core/config.py`` stays the one place that knows where a value comes from. A flag
    that already has a value in the environment loses to it, so an explicit export is
    never silently overridden by a stale file.
    """
    for option_name, field in _FILE_FLAGS.items():
        path = options.get(option_name)
        if not path:
            continue
        os.environ.setdefault(variable_for(field), _read(path, option_name))


def _read(path: str, option_name: str) -> str:
    """Read a flag's file, or say which flag could not be satisfied and why."""
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OpenOutFindError(
            ErrorType.BAD_CONFIG, f"--{option_name.replace('_', '-')}: {exc}") from None

    if not text:
        raise OpenOutFindError(
            ErrorType.BAD_CONFIG, f"--{option_name.replace('_', '-')}: {path} is empty")
    return text


# ── what this run was told ───────────────────────────────────────

def _describe() -> dict:
    """The config as both readers want it: counts and sizes, never the whole markdown."""
    from openoutfind.core.config import SiteConfig
    from openoutfind.core.export import export_counts
    from openoutfind.crm.models import Deal

    site_config = SiteConfig.load()
    exportable, _ = export_counts()
    return {
        "country_code": site_config.country_code,
        "product_docs_chars": len(site_config.product_docs),
        "campaign_target_chars": len(site_config.campaign_target),
        "ai_model": site_config.ai_model,
        "leads_seen": Deal.objects.count(),
        "exportable": exportable,
    }
