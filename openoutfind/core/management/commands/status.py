# openoutfind/core/management/commands/status.py
"""What is in the database, without running a job.

    outfind status            # human summary
    outfind status --json     # the whole document, for a program

`find` reports what it did; this reports what stands. It touches nothing and never
blocks: a provider that will not answer is an unknown balance, not an exception. SQLite
is in WAL mode, so it still answers while another job holds a write lock.

Output contract: the summary is the **result**, so it goes to stdout; logs go to
stderr. ``--json`` prints one object and nothing else, so it pipes into ``jq``.
"""
from __future__ import annotations

import json

from openoutfind.core.management.base import OpenOutFindCommand
from openoutfind.core.status import build_status, render_next_action


class Command(OpenOutFindCommand):
    help = "Report what is configured, what is blocked, the counts, and the next action."

    def add_arguments(self, parser):
        parser.add_argument(
            "--json",
            action="store_true",
            dest="as_json",
            help="Emit the full status document as JSON on stdout.",
        )

    def handle(self, *args, **options):
        status = build_status()
        if options["as_json"]:
            self.stdout.write(json.dumps(status, indent=2))
            return
        self.stdout.write(render(status))


# ── the human summary ────────────────────────────────────────────

def render(status: dict) -> str:
    """Render the status document as a short human summary."""
    sections = (
        _render_config(status["config"]),
        _render_pipeline(status["totals"]),
        _render_credits(status["credits"]),
        _render_hub(status["hub"]),
        _render_blocked(status["blocked"]),
        render_next_action(status["next_action"]),
    )
    return "\n".join(section for section in sections if section)


def _render_config(config: dict) -> str:
    if config["complete"]:
        return "Configuration: complete."
    lines = ["Configuration: incomplete."]
    for group, variables in config["missing"].items():
        lines.append(f"  {group}: set {', '.join(variables)}")
    return "\n".join(lines)


def _render_pipeline(totals: dict) -> str:
    if not totals["leads_seen"]:
        return "Pipeline: no leads yet."

    return "\n".join((
        "Pipeline:",
        f"  {totals['leads_seen']} lead(s) seen, {totals['rejected']} rejected by the qualifier",
        f"  {totals['exportable']} exportable — {totals['exportable_with_email']} with an email, "
        f"{totals['exportable_without_email']} without (a row exports either way)",
        f"  {totals['ranked_for_lookup']} ranked for a paid lookup, "
        f"{totals['lookup_in_flight']} in flight",
    ))


def _render_credits(credits: dict) -> str:
    if credits["balance"] is not None:
        return f"Credits: {credits['balance']} left."
    return f"Credits: unknown ({credits['error']})."


def _render_hub(hub: dict) -> str:
    if not hub["known"]:
        return "Hub store: no balance on record — a hub outage, or not registered yet."
    if hub["balance"] > 0:
        return f"Hub store: {hub['balance']} free read(s) — earned by contributing addresses."
    return "Hub store: no balance — contribute an address to earn a read."


def _render_blocked(blocked: list[dict]) -> str:
    if not blocked:
        return ""
    lines = ["Blocked:"]
    lines += [f"  {item['type']}: {item['message']}" for item in blocked]
    return "\n".join(lines)


