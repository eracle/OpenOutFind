"""Enrichment — the finder's one paid step: a profile URL in, a work email out.

The split is between *the providers*, *the seam* and *the pipeline step*:

- ``provider`` is the seam: one interface, and ``active()`` names which vendor an
  install resolves with. Everything above this package talks to a provider, never to
  a vendor. ``provider.Lookup`` carries the sync/async transport difference so the
  pipeline does not have to know it.
- ``bettercontact`` is the client — a submit-and-poll waterfall.
- **Discovery is not behind the seam.** ``discovery.py`` pages BetterContact's free
  Lead Finder index through ``submit_and_poll`` on the same key that pays for its
  enrichment.
- ``lookup`` is the step the cycle drives: ``buy_address`` resolves the free sources
  first and runs the finder only if they miss, ``check_lookup`` polls anything left
  in flight.

**This is deliberately not part of discovery, and no longer part of a mail package.**
It used to live under ``emails/`` because a resolved address existed to be written to;
that is no longer true, and the coupling it implied — resolve only what there is send
headroom for — was the single line that made a mailbox-less install produce nothing.
An address is now just a column in the export: nice to have, never a precondition, and
a lead with none still exports with its ``reason``.
"""
