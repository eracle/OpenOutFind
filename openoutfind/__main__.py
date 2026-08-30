"""The `outfind` console script — the single entry point for both readers.

The verbs, in the order a reader meets them:

    outfind find 10 [emails]                  # find that many more, print the campaign, exit
    outfind status [--json]                   # what is configured, blocked and counted

`find` is the only verb that does work, and it is bounded: it returns when the goal is
met, exits 0 only if it was, and prints the campaign as CSV on stdout — so the CSV is not
a verb either, it is what redirecting the command gives you.

There is no daemon and no default verb. A bare `outfind` prints `OVERVIEW` — the
three verbs and nothing else; the first run is `outfind find 10`, which also creates
the database.

**It deliberately does not hand a bare invocation to Django.** `execute_from_command_line`
answers with every management command it can find, which is fifty lines of
`squashmigrations`, `startproject`, `sendtestemail` and `createcachetable` with the three
verbs buried in a `[core]` block halfway down. That is the first thing anyone sees after
`pip install openoutfind`, and it reads as a Django project shipped by accident.

Django's own commands remain available and reachable — `migrate`, `runserver` (the Admin
at http://localhost:8000/admin/), `createsuperuser` — they are just not the answer to
*what does this tool do*. `outfind help <command>` still goes straight to Django.

Any command accepts `--db PATH` (or `--db=PATH`) to work against a SQLite file
other than the default `~/.openoutfind/data/db.sqlite3`; the `OPENOUTFIND_DB`
env var does the same.

`manage.py` is a thin shim over this module, kept for work inside a checkout.
"""

import os
import sys

OVERVIEW = """\
OpenOutFind — find B2B leads that fit, with the reason written out.

  outfind init             onboard and create the database
  outfind find 10          ten more qualified leads → CSV on stdout
  outfind find 10 emails   ...with a verified work email (1 credit each)
  outfind status           what is configured, blocked and counted

  outfind help <command>   details for one command

Django's own commands (migrate, createsuperuser, runserver) still work.
"""


def wants_the_overview(argv) -> bool:
    """Whether this invocation asks *what can I do*, rather than naming a command."""
    return len(argv) <= 1 or (len(argv) == 2 and argv[1] in ("-h", "--help", "help"))


def extract_db_path(argv):
    """Strip `--db PATH` / `--db=PATH` out of argv, returning (rest, path_or_None).

    Django parses arguments per-command, so the flag has to come off before
    execute_from_command_line ever sees argv.
    """
    rest, db_path, i = [], None, 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--db":
            if i + 1 >= len(argv):
                sys.exit("outfind: --db requires a path")
            db_path = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--db="):
            db_path = arg.split("=", 1)[1]
        else:
            rest.append(arg)
        i += 1
    return rest, db_path


def main(argv=None):
    """Run a management command.

    A bare invocation used to default to `run`, the daemon. With the work verb bounded by
    a goal there is nothing sensible to default *to* — `find` needs a number, and picking
    one for the operator would be spending their credits on a guess — so a bare
    invocation prints `OVERVIEW`.

    The overview is answered before Django is imported at all, which is why the empty
    invocation is instant: the settings module, the ORM and the command registry are the
    cost of doing work, not of asking what the work is.
    """
    argv, db_path = extract_db_path(list(sys.argv if argv is None else argv))
    if wants_the_overview(argv):
        print(OVERVIEW, end="")
        return

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "openoutfind.settings")

    from django.core.management import execute_from_command_line

    if db_path:
        os.environ["OPENOUTFIND_DB"] = db_path

    execute_from_command_line(argv)


if __name__ == "__main__":
    main()
