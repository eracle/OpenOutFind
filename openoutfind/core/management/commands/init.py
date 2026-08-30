"""Create the pipeline and the campaign, print what was created, and stop.

    outfind init                                  # wizard on a TTY, environment otherwise
    outfind init --product-docs p.md --target t.md # the two long fields, from files
    outfind init --json                           # the same campaign, for a program

**This phase already happened; it just never had a name.** `find` migrated the database,
onboarded from the environment, created the campaign and validated the operator before
doing any of the work it is named after — and announced the campaign with a single INFO
line between the migration narration and the discovery walk. Setting up and finding leads
are different jobs with different failure modes, so they are different verbs.

`find` still does all of it, because a fully configured environment must keep running end
to end in one command with no TTY. What changes is that there is now a way to do the setup
deliberately, see the campaign you actually created, and only then spend anything.

**The two long fields come from files, not from flags.** A product description is a page
of markdown with newlines and apostrophes in it; shell-quoting that on a command line is a
way to corrupt it quietly, and pasting it into a wizard prompt is worse.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.core.management.bootstrap import (
    ensure_database,
    ensure_onboarded,
    validate_operator,
)
from openoutfind.core.management.base import OpenOutFindCommand

# Which onboarding variable each file flag stands in for. The flags are a second way to
# supply two of the fields, never a second place that knows what the fields *are*.
_FILE_FLAGS = {
    "product_docs": "PRODUCT_DESCRIPTION",
    "target": "CAMPAIGN_TARGET",
}


class Command(OpenOutFindCommand):
    help = "Create the pipeline and the campaign, then print what was created."

    # The verb that migrates — that is the whole point of it.
    requires_database = False

    def add_arguments(self, parser):
        parser.add_argument("--product-docs", metavar="FILE",
                            help="File holding the product description (markdown).")
        parser.add_argument("--target", metavar="FILE",
                            help="File holding the target market description (markdown).")
        parser.add_argument("--name", help="Campaign name. Defaults to the built-in name.")
        parser.add_argument("--json", action="store_true", dest="as_json",
                            help="Emit the campaign as one JSON object.")
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
        ensure_onboarded()
        validate_operator()

        self._report(_describe(_only_campaign()), options)

    def _report(self, described: dict, options) -> None:
        if options["as_json"]:
            self.stdout.write(json.dumps(described, indent=2))
            return

        self.stdout.write(f"Campaign: {described['name']}")
        self.stdout.write(f"  country     {described['country_code'] or '—'}")
        self.stdout.write(f"  headcount   {described['headcount_min']}–{described['headcount_max']}")
        self.stdout.write(f"  product     {described['product_docs_chars']} chars")
        self.stdout.write(f"  target      {described['campaign_target_chars']} chars")
        self.stdout.write(f"  model       {described['ai_model']}")
        self.stdout.write(f"  leads       {described['leads_seen']} seen, "
                          f"{described['exportable']} exportable")
        self.stdout.write("")
        self.stdout.write("Next: outfind find 1")


# ── the two long fields ──────────────────────────────────────────

def _seed_environment(options) -> None:
    """Put the contents of ``--product-docs`` / ``--target`` where onboarding looks.

    Writing to ``os.environ`` rather than creating the campaign here is deliberate:
    ``onboarding.STEPS`` stays the one place that knows which fields a campaign needs and
    how to validate them. A flag that already has a value in the environment loses to it,
    so an explicit export is never silently overridden by a stale file.
    """
    from openoutfind.core.onboarding import ENV_PREFIX

    for option_name, variable in _FILE_FLAGS.items():
        path = options.get(option_name)
        if not path:
            continue
        os.environ.setdefault(ENV_PREFIX + variable, _read(path, option_name))

    if options.get("name"):
        os.environ.setdefault(ENV_PREFIX + "CAMPAIGN_NAME", options["name"])


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


# ── what was created ─────────────────────────────────────────────

def _only_campaign():
    """The operator's campaign — the one onboarding just made, or the one already there."""
    from openoutfind.core.operator import campaigns

    known = campaigns()
    if len(known) > 1:
        # Initialising twice against different names is a real thing to do; picking one
        # silently is not. `find --campaign` is how you choose between them.
        return sorted(known, key=lambda c: c.pk)[-1]
    return known[0]


def _describe(campaign) -> dict:
    """The campaign as both readers want it: counts and sizes, never the whole markdown."""
    from openoutfind.core.export import export_counts
    from openoutfind.core.models import SiteConfig
    from openoutfind.crm.models import Deal

    exportable, _ = export_counts(campaign)
    return {
        "name": campaign.name,
        "country_code": campaign.country_code,
        "headcount_min": campaign.headcount_min,
        "headcount_max": campaign.headcount_max,
        "product_docs_chars": len(campaign.product_docs),
        "campaign_target_chars": len(campaign.campaign_target),
        "ai_model": SiteConfig.load().ai_model,
        "leads_seen": Deal.objects.filter(campaign=campaign).count(),
        "exportable": exportable,
    }
