# Architecture

Detailed module documentation for OpenOutFind. See `CLAUDE.md` for rules and quick reference.

OpenOutFind is a browserless **lead finder**: it learns your ICP and runs the
funnel — **define ICP → discover → qualify → rank → (optionally) resolve an address →
export** — off licensed data, with no social-network account and no scraping.

**It does not send.** The product's output is a CSV of qualified leads, each carrying the
reason the LLM chose it. Everything downstream of that file — the message, the mailbox, the
sequence, the opt-out — belongs to whatever tool the operator already sends with.

## Project Layout

All source lives in the single `openoutfind/` package; Django apps are nested inside it
(dotted `AppConfig.name`, labels namespaced `outfind_*` so OpenOutreach can host these apps
in one registry beside OpenOutSend's). Two apps, and that is the whole of it:

```
manage.py
tests/
openoutfind/
  settings.py        # Django settings (SQLite at data/db.sqlite3) — one host of the apps
  defaults.py        # what any host must splat: APPS, state_dir(), app_settings()
  urls.py
  discovery.py       # Lead Finder client (ICP search + row embedding) — the top of the funnel
  core/              # engine app (label: outfind_core) — the cycle, operator lookup,
                     #   config.py (the environment) + readiness.py, llm.py, conf.py,
                     #   ML (qualifier/embeddings), discovery+qualify pipeline,
                     #   the lead export, db/ helpers, geo, management commands
  enrichment/        # the one paid step (not an app, no models): the finder clients
                     #   (Enrow, BetterContact, FullEnrich, and LeadMagic, the
                     #   last synchronous), the price-aware selection seam
                     #   (provider.py), and the buy/check lookup
  crm/               # app (label: outfind_crm) — Lead, Company, Deal
  contacts/          # central contacts-store client (service.py only — no models, not an app)
```

Layering: `core` owns orchestration, the ML/discovery/qualify pipeline and the models;
`enrichment` owns the paid provider call and its pipeline step. `core` imports it only at
wiring points (the cycle's hierarchy).

**What used to be here.** An `emails` app carried Mailbox, SMTP/IMAP, the mail log, the sender,
the three send guards and two pipeline steps; `core` carried the outreach agent and the sending
window; `chat` and `legacy` were model-less apps holding migration history. All of it is gone.
The sending half was **ported to [OpenOutSend](https://github.com/eracle/OpenOutSend)**
(the `openoutsend` package in that repo); the model-less
anchors went with the migration history they anchored, because the cut was clean rather than
upgradeable — see *Migrations* below.

## Entry Flow

`openoutfind/__main__.py:main` — the `outfind` console script, and the entry point
(`manage.py` is a shim over it for a checkout). **There is no default verb**: a bare invocation prints
the command list, because the work verb needs a number and picking one would spend the operator's
credits on a guess. A global `--db PATH` (or `--db=PATH`) is stripped from argv before Django
parses it and exported as `OPENOUTFIND_DB`, which `settings.py` reads for the SQLite file (default
`~/.openoutfind/data/db.sqlite3` installed, `data/db.sqlite3` in a checkout); the parent directory
is created if missing.

### The verb surface

Three verbs, chosen rather than accumulated — they are what the two readers actually need:

| verb | |
|:-----|:--|
| `init [--product-docs F] [--target F] [--json]` | create the pipeline, print what it made, spend nothing. Safe to run twice. |
| `find N [emails] [--emails] [--new] [--json] [--batch] [--open] [--debug]` | find that many more, print every lead as CSV, exit. |
| `status [--json]` | what is configured, blocked, counted, and the next action — including the command that prints the CSV. |

**There is no `--campaign` flag, and no campaign to name.** This install has never run more than one,
so `Campaign` was folded away and every `campaign=` scoping FK went with it — the ICP text is two
environment variables and the walk is the `Search`/`Page` rows. A second ICP is a second store —
`--db PATH`, or `OPENOUTFIND_DB`.

**The CSV is not a verb and not a file the tool writes** — it is the stdout of `find`, so redirecting
the command is what produces it. `export_leads` was deleted as a second way to do one thing, and the
one nobody found. See *The Lead Export* below.

**`--json` belongs on every verb with a result** — `init`, `find` and `status`, which is all of them.
A failure answers in the caller's format too: `--json` turns the `error:` line into
`{"error": {"type", "message"}}` on stderr, so a program never has to parse prose to find out why.

**There is no `reset` verb, by decision.** A `reset` would have to earn its keep against something
the state directory already does: everything an install writes hangs off one root
(`settings.state_dir()`), so **`rm -rf ~/.openoutfind` followed by `outfind find` is the
reset**, and configuration comes back from the environment. What that costs is real and worth
naming: the ICP seed only runs when there are **no** query nodes yet, so there is now no way to
re-seed a walk *while keeping* its leads and verdicts. Changing the ICP text means starting the
evidence over too.

Others were removed rather than renamed. **`setup_crm`** was an implementation detail with a verb
around it — `find` calls the function directly, so nothing is lost. **`reset_data`** deleted every
lead and deal in the store, which `reset` already did with a backup first; both are gone now.

### `find` management command (`management/commands/find.py`)

    outfind find 10 [emails] [--emails] [--new] [--json] [--batch] [--open] [--debug]

**Finding is free, and the free thing is the default.** A bare `find 10` cannot spend a credit,
however many deals have queued up past the confidence gate; `--emails` permits the lookup and the
`emails` unit implies it. A flag you forget should cost a feature, never
money. The default is `buy_addresses=False` in `run_job` and `run_one_action` too, not just at the
CLI, so no caller can spend by omission.

**The unit is a noun, not a flag**, and that is a budget decision: the provider bills one credit per
verified hit, so `find 10 emails` is capped at ten credits by construction — the number typed is the
budget, in the same unit as the invoice. A `--with-email` modifier would have hidden the money, and
`--max-credits` would be a second ceiling saying the same thing. The noun says what to **count**;
`--emails` says what may be **paid for**. They meet in exactly one place: a goal counted in addresses
cannot be met without buying them, so the unit implies the flag.

**What enforces that ceiling is a cap on addresses _on order_, not on lookups _made_**
(`job._lookup_budget`: `goal.count − produced − _on_order()`, passed down as
`run_one_action(max_new_lookups=…)`). The distinction is the whole of it. A submission almost never
resolves synchronously, so without a cap independent of `produced` a goal of 1 would submit a lookup
for every lead past the confidence gate; but a lookup that **misses** produced no address and must
not spend the goal either, or `N emails` tops out at the provider's hit rate — URL-only resolves
~42%, so counting misses put a ceiling of ~`0.42 × N` on every `emails` goal and left the run
spinning on discovery once the last submission was spent, unable to buy and unable to stop. A miss
releases its slot and the run buys again; only a hit spends one, so at most `goal.count` addresses
ever resolve and the invoice still matches the number typed.

`--debug` is the shorthand for `--log-level debug`; both write the same dest, so they cannot disagree. `find 0` does no work and prints
what is already there; it is not a special case, it is `produced >= count` being true before the loop.

Startup sequence, inherited whole from the deleted `run`:
1. **Configure logging** — level from `--verbosity`, banner, noisy third-party loggers silenced (`core/logging.py`).
2. **Ensure DB** — `migrate --no-input` (narrated to stderr; the custom migrate, see below) + the idempotent CRM bootstrap (`core/management/setup_crm.py`, a function — the verb around it is gone).
3. **Check this run was given what it needs** — `readiness.check_ready()`: one error naming every missing `OPENOUTFIND_*` variable, one live ping proving the model answers, and the operator `User` written from the environment if this is the first run. Nothing is prompted and nothing is stored. See *Configuration and the readiness check* below.
4. **Run the job** — `core/job.py:run_job`, then print.

**stdout carries every lead in the store, not just this run's rows.** That is what makes `> leads.csv`
correct by construction: the newest file supersedes every earlier one, and a lead whose address
resolved since last time comes back with it filled in. It is one file you overwrite, not a batch per
run. `--new` narrows to what this run produced — the escape hatch for a caller reading stdout into a
context window rather than into a file — and `--json` emits the rows as JSON Lines, one record per
line, with the goal, the outcome and the next action as one JSON object on stderr.

**The run ends with the one thing to do next, and it derives none of it.** `_report` reads
`build_status()["next_action"]` and renders it with `status.render_next_action` — on stderr beside the
counts, or as the `next_action` key when `--json` is asked for. That matters most on a run that stops
with ranked leads and an empty wallet: the moment to offer a top-up is the moment the run ends, and
nothing else prints it. **Two earlier attempts put the arithmetic in the wrong place** — the balance
read and a deal count inside `core/job.py` gave the bounded-goal loop an HTTP call to a payment
provider, and inside `enrichment/lookup.py` it went per-deal, so *read once* and *ask once* both
needed module-level mutable state. The ask is about the state the run **left behind**, so it is read
once after the work rather than carried through it, and there is one derivation with two callers.

**Exit 0 means the goal was met, and nothing else.** Short of it, the rows still print and one
`error: <type>: <message>` line goes to stderr: *the code says how much you got, the type says why it
stopped*. That is the property that lets a caller branch without parsing anything, and the cost —
`set -e` dying on a good partial result — is real and accepted.

`--open` hands each new lead's `profile_url` to the operator's own browser as it lands. **It does not
spend the browserless claim**: nothing is fetched, automated or authenticated, and `profile_url` stays
*stored, never fetched* by us. No browser available is a `bad_config` error at argument time, because
a flag that silently does nothing is the bug you find at 2am.

Docker's `start` script `exec`s `outfind find "$@"` — one job, then the container exits.

### The other two verbs

- `init` — **the phase that already happened, given a name.** Migrate, bootstrap the CRM, onboard, write the config, validate the operator — then stop and print what it made. It cannot spend, so it is the one verb always safe to run.
- `status` — **what is in the database, without running a job.** Human summary by default, `--json` for a program. See *Status* below.

## The Output Contract (`core/management/base.py`, `core/errors.py`)

What a program depends on, and the reason both exist as one place rather than a habit.

- **stdout is result-only.** Logs, the startup banner, and Django's `migrate` narration all go to
  **stderr** — `configure_logging` attaches its handler there, `print_banner` writes there, and
  `run` passes `stdout=self.stderr` into `call_command("migrate")`. Verified end-to-end: a
  headless `run` that cannot start writes **zero bytes** to stdout.
- **Colour is gated on `stderr.isatty()`**, not stdout, or piping a result would strip the colour out
  of an interactive run.
- **An expected failure is one line**: `error: <type>: <message>` on stderr, exit 1, no traceback.
  `OpenOutFindCommand.run_from_argv` catches `OpenOutFindError` and writes it verbatim; anything
  else keeps Django's behaviour, because a bug deserves its traceback. `OpenOutFindError` is
  deliberately **not** a `CommandError` — Django prefixes that with the exception's class name, which
  would put noise in front of the line an agent parses.
- **A fresh install is an answer too.** `settings.py` creates the data *directory*, but only `run`
  migrates, so on a wheel every other verb used to die on a raw `no such table:
  outfind_core_siteconfig`.
  `OpenOutFindCommand.requires_database` (default `True`; `run` sets it `False`) checks for the
  schema in `execute()` — **after** argument parsing, so `--help` still answers — and raises
  `not_initialized`, naming the database path. Reporting zero leads instead would be precisely
  the empty-result failure `core/errors.py` exists to prevent: nothing was found because nothing has
  ever run.
- **A failure answers in the caller's format.** `run_from_argv` renders `OpenOutFindError` through
  `format_failure`: the `error: <type>: <message>` line, or `{"error": {"type", "message"}}` when
  `--json` is in argv. **Both on stderr** — a caller that asked for JSON is parsing rather than
  reading, but stdout stays result-only either way, or `find 10 --json > leads.json` would write an
  error object into the file the operator is keeping.
- **Only what a configuration can cause is a credential answer.** `verify_llm_credentials` catches
  the provider's own refusals (`ModelAPIError`), an unusable model id (`UserError`/`ValueError`) —
  and nothing else, because its caller reports whatever it returns as
  `bad_config: OPENOUTFIND_LLM_API_KEY`. It used to catch `Exception`, so when `anthropic` 1.0.0
  dropped `temperature` the resulting `TypeError` in our own process read as a rejected key and sent
  the operator after credentials that were fine. A broken install is a bug and gets its traceback.
- **A provider's refusals are three different things**, typed at the HTTP boundary in each adapter's
  `_request` (`bettercontact`, `enrow`, `fullenrich` and `leadmagic`): 401 → `provider_auth`, 402 →
  `provider_out_of_credits`, an exhausted 429 backoff → `provider_rate_limited`. Everything else
  stays `provider_unavailable`, which is also what a finder that cannot run at all raises (each
  adapter's own `BetterContactUnavailable` / `EnrowUnavailable` / `FullEnrichUnavailable` /
  `LeadMagicUnavailable`, all subclasses of `provider.ProviderUnavailable`, carrying the vendor's
  message). Every one of them is a
  *couldn't-run*, and that is the distinction `lookup.buy_address` acts on: it steps to the next
  finder in the chain, where a terminal miss ends the lookup rather than spending a second credit.
- **A 429 is backed off, never retried at speed** — their docs warn that a client which keeps firing
  can get the *account* blocked. The backoff is `urllib3.Retry` mounted on the session
  (`_RETRY`: 5 attempts, `backoff_factor=5` doubling, `backoff_max=120`, `respect_retry_after_header`,
  forcelist `(429,)` only). It lives in the transport rather than in our code because urllib3 already
  implements exactly this, `Retry-After` included; `tenacity` retries on *exceptions*, so using it
  here would mean raising one just to make it fire.

## The Run's Narrative (`core/logging.py`, `core/logblock.py`, and the log lines)

What `find` prints on stderr while it works **is** the product surface: it is what a first-time
operator watches and what an agent summarises for its human. stdout is the CSV and stays result-only.
The script it follows is written down in `openoutreach-docs/docs/first-run-design.md` §6 — the UX
definition came first and the code implements it, rather than the other way round.

**Two levels, and nothing is deleted between them.** INFO is addressed to the operator: every line
answers a question they would ask. DEBUG (`--debug`) is addressed to the maintainer and carries
**everything INFO ever said plus the engine's own reasoning** — the posterior behind the spend gate
(`ready_pool.py`), each round's search with how many fits it holds or that it was drawn, its rate and expected new people (`search_arms.py`) and its dead searches (`walk.py`), the initial population and the anchor top-up (`icp.py`), and the cycle's per-row
decisions and timings
(`cycle.py`). A line that stops being operator-facing is **demoted, never removed**: the redesign
changed who a line is addressed to, not what the tool is willing to say. **The GP's acquisition
strategy and class counts moved the other way** (`qualify.py:run_qualification`, `top_up.py:_advance`)
— *"is it exploring or exploiting right now, and why"* is a question an operator watching the run
genuinely asks, not just the engine reasoning about itself, so it now prints at INFO: which regime
(`cold phase`, `exploit (p)`, `explore (BALD)`) picked the candidate, and the neg/pos count (including
anchors) that put it there. Every path through candidate selection says something — a single
candidate, a degraded no-posterior fallback, and cold phase (with its anchor progress, `N/ANCHOR_COUNT`)
each get their own line, so silence never reads as "nothing decided" when a decision was made.

The beats, in order:

- **Minute 0 states the deal** — who you sell to, the goal the operator typed, and whether this run may
  spend (`find._announce_the_run`). Spending is opt-in at every layer, which is a good default and an
  invisible one; an operator who expected addresses learns it in the first line, not from an empty
  column at the end.
- **The ICP echo** (`icp.log_icp_echo`) — the synthetic ideal profiles, printed as *"Looking for
  people like:"*. They were already computed, already one line each in `profile_text`'s shape, and
  until now only their *count* was shown. It is the earliest proof the product description was
  understood, and therefore the earliest chance to correct it. A first run has none yet — they are
  written during the job and print themselves there.
- **Both verdicts** (`qualify._verdict_line`) — a rejection carries its reason exactly as an
  acceptance does, `✗` against `✓`. *Watching it turn people down is what makes the acceptances
  credible*; a log of nothing but hits reads like a row dump. `logblock.step_line` is not the
  primitive here: its fixed-width label column aligns short plumbing labels, and a person's name
  overflows it on every row.
- **Progress counts toward the goal** (`job._log_progress`) — `2 of 10 leads · 58 seen · 1m34s`, in
  the unit the operator typed, with `★ first lead · 52s` as its own milestone because *how long until
  anything at all happens* is what a first run is really asking. `JobResult.elapsed` is **reported,
  never enforced** — there is still no timeout.
- **The gate that is actually holding is named** (`cycle.pipeline_summary`) — and **every consequence
  it has**. The finder key is one key doing two jobs, so losing it stops the walk finding anybody
  *and* stops the paid lookup; naming only the second sent the operator after the wrong thing. It
  does not stop the free address sources. `buy_addresses` is passed in because the run holds that
  permission and the summary cannot see it: *addresses not requested* is the gate most likely to be
  holding on a bare `find`.
- **The run ends with the next action** — see the `find` section above.

`logblock.py`'s grammar (a `▶` header naming one action, aligned step lines under it) is the
foundation and was not replaced; the redesign happened inside it.

## Status (`core/status.py`)

`build_status()` assembles one dict and reads nothing else; the command renders it. SQLite runs in
WAL, so it still answers while a job holds a write lock (verified against a held `BEGIN IMMEDIATE`).
One renderer lives in `core/status.py` rather than in the command — `render_next_action` — because
the end of a `find` run prints the same ask, and the sentence must have one spelling.

`walk` lists every fired page: `expected_count` and `expected_rate`, the `N(q)` and `θ(q)` at the
mean belief the walk chose it on (`core.Page`, `None` on a written search), beside the provider's
`leads_found`, and the page's `people`, `verdicts`, `fits` and `observed_rate`. It reads from the
store because an `--agent-qualify` run is many processes and a page's verdicts land after it, and
`--json` keeps the walk's debug lines off stderr.

**It is smaller than it was, on purpose.** It began as the verb an agent asked *instead of tailing a
log*, because a daemon could not answer for itself — `next_action` existed so a caller had something
to interrogate at all. With `find` bounded by a goal the work verb returns its own result, and this
one is back to being what it says: the standing state of the database, for a reader who did not run
the job.

| key | what it carries |
|:----|:----------------|
| `config` | `complete`, the satisfied groups, and per-group the variables that would satisfy the rest |
| `totals` | the pipeline counts |
| `credits` | `balance` + `error` for the finder `active()` names, with `provider` beside them — Enrow's read is the free `GET /account/info` → `credits`, BetterContact's is `GET /api/v2/account` → `credits_left`, LeadMagic's is the free `GET /v1/credits`; FullEnrich has none, so it is *unknown* |
| `hub` | `balance` + `known` — the give-to-get counter (`contacts.service.hub_balance`), a different number on a different service than `credits` |
| `blocked` | what stands between now and more qualified rows, typed from `core/errors.py` |
| `next_action` | the one thing to do next, with what it unlocks and the command or URL that does it |

Three decisions inside it:

- **A balance that could not be read is not a balance of zero.** `no_credential`,
  `provider_auth` and `provider_unavailable` are reported as *why*, because a rejected key must
  never render like a run that has simply found nothing yet.
- **`exportable` is not `mailable`.** The export excludes only the two rejections, so a `QUALIFIED`
  lead exports with a blank `email` column — an address is an enrichment on top, never a
  precondition. The counts therefore split `exportable_with_email` / `exportable_without_email`, and
  they are counted **from the records** rather than from `RESOLVED` standing in for them.
- **`next_action` is ordered by what blocks progress**, so `add_credits` sits above `print_leads`:
  a ranked lead cannot advance without credits, while printing what exists costs nothing. This does
  not break the *never before value* rule — `ranked_for_lookup > 0` is itself the proof that
  qualified leads with written reasons exist, and an install that has qualified nobody is asked for
  no money, only told to go and find some (`find_leads`).
- **The hub balance is not the `Credits:` line.** `credits` is whichever finder `provider.active()`
  names, which is the cheapest configured one, reported with a `provider` key beside its balance
  (`GET /account/info` for Enrow, `GET /api/v2/account` for BetterContact, `GET /v1/credits` for
  LeadMagic); `hub` is the give-to-get counter on the contacts store, a different number on a
  different service, so it gets its own key rather than being folded into or shown beside `credits`
  under one label. Read via `hub_balance()`, which piggybacks on `register` (a
  record-less call is idempotent server-side and already returns `credits`), so checking it spends
  nothing. `known: False` (no token yet, or a hub outage) must render as *unknown*, never as zero —
  the same rule the provider balance already follows, and a FullEnrich install lives under that rule
  permanently because FullEnrich exposes no balance read, so `credit_balance()` raises rather than
  guess.

## Configuration (`core/config.py`) and the readiness check (`core/readiness.py`)

**Configuration is the environment, and nothing a human answered is stored.** `SiteConfig` is a
frozen dataclass read from `OPENOUTFIND_*` on every run — no row, no singleton, no Admin page, and
no wizard. The line is **who produced the value**: an answer somebody gave is a variable, something
the pipeline produced or measured is a row, and identity is a row written once.

That is what this program is. A library, a pipe stage (`outfind find --json | outsend`) and a
scripted command cannot answer a prompt, and an agent supplies its environment on every invocation
and has nothing to remember. The wizard is not gone — it belongs to
[OpenOutreach](https://github.com/eracle/OpenOutreach), which owns the human half, holds the answers
in its own `SiteConfig` and exports these names to both children.

```
campaign        OPENOUTFIND_PRODUCT_DOCS, OPENOUTFIND_CAMPAIGN_TARGET  (+ MAX_PER_COMPANY, optional)
llm             OPENOUTFIND_AI_MODEL, OPENOUTFIND_LLM_API_KEY  (+ LLM_API_BASE, required for openai_compatible:*)
bettercontact   OPENOUTFIND_BETTERCONTACT_API_KEY
fullenrich      OPENOUTFIND_FULLENRICH_API_KEY
enrow           OPENOUTFIND_ENROW_API_KEY
leadmagic       OPENOUTFIND_LEADMAGIC_API_KEY
account         OPENOUTFIND_OPERATOR_EMAIL, OPENOUTFIND_OPERATOR_COUNTRY  (optional)
hub             OPENOUTFIND_CONTACTS_API_TOKEN  (optional)
```

`check_ready()` runs before any work — from `outfind check`, and from `find` itself, so a scripted
install never discovers a setup step it did not know to run. It raises **one** error naming every
variable that would have satisfied the run, rather than three in a row; `missing_variables()` is the
same reading without the raise, which is what `status` renders.

Four rules, each answering a way this could go quietly wrong:

- **What makes reading fresh safe is the check, not the row.** The defect that first pushed the LLM
  key into the database was a key that sat unread until an agent asked for a model, mid-pass, with a
  lead already chosen. `_check_llm` pings the model before any work starts, so a key rotated out
  from under a timer fails at second one. It costs one provider round trip per run, and that is the
  price of not storing the answer — deliberately paid.
- **A bad value stops; an absent one is named.** `error: bad_config: <VAR>: <problem>` — falling
  through to "missing" would print a variable the operator has already set.
- **The operator is a row, written once** (`_ensure_operator` → `User`, seeded from the optional
  `OPENOUTFIND_OPERATOR_EMAIL`). Identity is not an answer to be re-read: a renamed variable must
  not rename the person a campaign belongs to, or re-key the contacts store. Without an email the
  row is `operator` and the install has no hub identity — no store reads, no give-back; an email
  given later fills that blank once and registers.
- **The hub token is not written down either.** `contacts.service.token_in_hand` returns the one the
  environment gave, or registers for one and keeps it for the life of the process; `register` is
  idempotent server-side, so a run that mints its own loses nothing by forgetting it. A read-only
  caller (`hub_balance`, for `status`) passes `mint=False` — reporting a balance must not register
  an install as a side effect.

## Deal State Machine

`crm/models/deal.py:DealState` (an `openoutfind`-owned `TextChoices`) is the whole funnel — a lead
is discovered and qualified **without** an email in hand (Lead Finder returns firmographics, not
addresses), so the funnel qualifies first and *optionally* resolves an address after:

```
QUALIFIED ─(GP rank gate)─▶ READY_TO_FIND_EMAIL ──(buy_address)──▶ FINDING_EMAIL ──(check_lookup)──▶ hit:  RESOLVED
 discovered + qualified      ranked, awaiting the      provider job in flight;      miss: NO_EMAIL_FOUND
 exportable from here        paid lookup               request_id on the deal
                            (free hub hit → RESOLVED directly, no job)
```

**Every state below `RESOLVED` is terminal, and that is the shape of the product.** The output is
a row in a file, not a conversation: a deal reaches its verdict, optionally gains an address, and
stops. `core/export.py` reads it from there.

- **`QUALIFIED`** — the LLM judged this lead a fit and wrote `reason`. **Exportable from this moment**; an address is an enrichment on top, never a precondition.
- **`READY_TO_FIND_EMAIL`** — passed the **GP confidence gate** (`ready_pool.promote_to_ready` at or above `calibration.current_threshold()`); queued for the *paid* lookup (one credit per verified hit). The gate is the paid-lookup **spend** gate and nothing else — it is not a quality score, and is deliberately absent from the export.
- **`FINDING_EMAIL`** — a provider job is in flight; the deal is excluded from the candidate pool (so the next cycle can't re-select it and double-charge) while `check_lookup` polls to termination. The job handle (`lookup_request_id`) and the poll backoff (`lookup_attempt` + `not_before`) live **on the deal**, so an in-flight lookup survives a restart and its wait gates that one row and nothing else.
- **`RESOLVED`** — an address is in hand. Where a fully-enriched deal comes to rest, at no cost, since nothing iterates it.

**The paid lookup runs behind a provider seam** (`enrichment/provider.py`): `active()` names the cheapest finder by `UNIT_COST_USD` among those holding a key, and `chain()` is that one followed by every other configured finder, cheapest first. Nothing above the package knows the vendor. `buy_address` resolves free-hub-first (hit → `RESOLVED` with no job/credit), then walks the chain. **The transport is each provider's, and the interface spans both**: a synchronous finder (LeadMagic's one-call finder, or a match-style API a fork adds) comes back already terminated and lands on `RESOLVED`/`NO_EMAIL_FOUND` without ever entering `FINDING_EMAIL`; an async one (Enrow's single search, BetterContact's waterfall, FullEnrich's bulk enrich) returns a
handle and parks there. A finder that cannot run — every `ProviderUnavailable`, whether a typed refusal or a missing key or an API that is down — steps to the next finder in the chain, and only when the whole chain has refused does the deal stay `READY_TO_FIND_EMAIL`. A finder that *does* run is authoritative: a terminal miss ends the lookup there rather than spending a second credit. The handle records **which** finder minted it (`Deal.lookup_provider`), so swapping keys mid-flight cannot hand one vendor's `request_id` to another and `check_lookup` dispatches by name. The poll is then **tri-state**: hit → `RESOLVED` (address given back to the hub); **miss** (job terminated, no address) → `NO_EMAIL_FOUND` — its own terminal, with **no reason and blank outcome**, critically not `FAILED+wrong_fit`, which the ML labeler reads as a negative: a lead we simply couldn't reach was still an LLM fit positive and stays label=1; **still running** → chains the next poll with doubled backoff on the same `request_id`, with no deadline and no attempt limit.

`crm/models/deal.py:Outcome` is down to **two** values: `wrong_fit` and `unknown`. The reply
outcomes it used to carry — converted, not_interested, no_budget, has_solution, bad_timing,
unresponsive — described how a *negotiation* went, and a negotiation is the sender's. They left
with `emails/*`, and that retired a live mislabelling: the reply step wrote the real outcome and
moved the deal to `COMPLETED`, not `FAILED`, so `get_labeled_arrays` (which labels every
non-`FAILED` deal **1**) was training a "not interested" reply as a **positive**.

`Lead.disqualified=True` = permanent account-level exclusion (never given a new deal), and it is a
**different column** from `FAILED` — the export filters both, and the first shipped version
filtered only the second. Pre-Deal Lead states are implicit: url-only (a `Lead` row with a null
`embedding`) vs embedded (has an `embedding` + `profile_text`, awaiting qualification).

*(Two legs have been removed over the project's life. The connect leg — `READY_TO_CONNECT`/
`PENDING`/`CONNECTED` — went with the browser channel. The send leg — `READY_TO_EMAIL`,
`EMAILED`, `COMPLETED`, `UNSUBSCRIBED` — went with `emails/*` to OpenOutSend.)*

## The Cycle

**A queue is a status, not a table.** Work is found by asking the deals what they need
(`Deal.objects.filter(state=…)`), so a deal is available because of its own row. Nothing is created
in advance, so nothing can drift, be lost, or need reconciling — and no row's timestamp can gate
anything but itself.

The loop is `core/job.py`, and it is four lines:

```python
while produced(site_config) < goal.count:
    if not run_one_action(site_config):
        break          # every remaining deal is waiting on its own not_before
```

**There is no timeout, deliberately.** Every unit of work already carries its own bound —
`deal.not_before` is the one retry mechanism there is, the lookup poll doubles its own backoff, and
`urllib3.Retry` bounds the 429s. A job-level clock would be a second timeout answering a question the
first one already answers, and answering it worse: a clock knows nothing about *why* it is waiting.
The terminal condition is derived from real state instead — the job ends when **nothing can advance
right now**, which is what `run_one_action` returning `False` already meant.

**Progress is a set, not a subtraction.** A goal counts the leads that *entered* it during this run,
so a lead rejected mid-run cannot silently cancel out one that was found — and `--new` stays exact
for both units, which matters because an `emails` goal is usually satisfied by leads that were
already exportable and merely gained an address, something no timestamp on the row would identify.

**What ends a job short, in the vocabulary of `core/errors.py`:** `goal_unreached` (nothing left to
do, or `Ctrl-C` — the operator's own deadline, since the one case with no natural bound is a store
whose leads are all rejected), `bad_config` when a `HALTING_ERRORS` exception says the model
rejected the request, or `qualify_pending` when `find --agent-qualify` opted a candidate out of the
`AI_MODEL` call — see the `qualify.py` entry under `core/pipeline/` below. Provider refusals keep
their own types. A `qualify_pending` stop in plain mode also narrates the candidate's
`profile_url` and `profile_text` on stderr (`find._narrate_pending_candidate`), since the error line
stays one line and carries only the name; under `--json` the same fields ride the error object.
`population_pending` is the same stop one step earlier — a fresh store under `--agent-qualify`
needs the walk's 4 initial searches and its anchors written, and the agent answers with
`--population`. `keyword_guess_pending` stops before a composed round whose market or
keywords have no guess yet, answered with `--keyword-guesses`.

*(Two lines are gone from the shape above along with the sending leg: `read_mail_if_due()` and
`refresh_capacities_if_due()`, the only two periodic side-effects the loop ever had.)*

### What replaced, and why

`core/scheduler.py` wrote `Task` rows that were **permission tokens** stamped with a time — "at
14:32 someone may send one email for campaign A" — without saying to whom; the loop took the
earliest-due token and the handler then went and found its own target. When no token was due the
daemon slept **until the earliest token's timestamp**. On 2026-08-05 that put a live install to
sleep for 34 hours: two BetterContact polls had never terminated, their uncapped backoff had
reached 45h30m, they were the only rows in the table, and 55 ready deals with 70 sends of headroom
had no token and therefore were not work. The shape was inherited from the Playwright era, when a
token queue was how access to one browser got serialised; the browser went, the queue outlived it.

Deleted with it: `Task`/`TaskQuerySet`, `core/scheduler.py` (558 lines), `core/quota.py` (172) and
`Campaign.action_fraction`, `core/session.py`, `add_business_hours`, and `emails/tasks/` (598).

### The hierarchy

`run_one_action(site_config)` walks one ordered list and stops at the first thing it can do, so
priority is exactly the order these are written in:

| # | State | Step | Condition |
|---|---|---|---|
| 1 | `FINDING_EMAIL` | `check_lookup` / `reclaim_lookup` | `not_before` elapsed |
| 2 | `QUALIFIED` | `promote_to_ready` | — |
| 3 | `READY_TO_FIND_EMAIL` | `buy_address` | a provider is configured |
| 4 | *(nothing queued)* | `top_up` | — |

A state that is not listed is terminal, and terminal costs nothing: `RESOLVED`,
`NO_EMAIL_FOUND`, `FAILED`.

**There is no spend gate on rows 2 and 4, and that is the shape of the finder.** Both rows used to
share one (`cycle.room_to_send_today`, now deleted): *never resolve an address, and never spend an
LLM call qualifying, for someone there is no room to email today.* That was right while every lead
ended in a send. It was also the single line that made a mailbox-less install produce **nothing** —
with no `Mailbox` rows the pool headroom was 0, the comparison was never true, and discovery and
qualification both stopped, with no error and no log line saying why.

Nothing replaced it, because nothing is left to ration. Discovery is free, qualification costs one
LLM call against a key the operator pays for directly, and the loop is bounded the only way that
matters: **one unit of work per cycle, forever**. Row 3 is the single paid step and asks only
whether there is a provider to pay.

**Two rows are gone with the sending leg** — answering a reply (`EMAILED` + an unanswered reply)
and sending a first email (`READY_TO_EMAIL`, a mailbox free) — along with both periodic
side-effects the loop used to run before every action: the mail pass (IMAP into each box every five
minutes) and the daily warmth re-measure. They live in OpenOutSend now.

Row 2 is the only step that fits a model: building a qualifier dominates the cost of using it, so it
scores the whole `QUALIFIED` pool in one pass and drops the model (`qualifier_for`). There is no
`Lead.is_ranked` column — "worth paying for" is what `READY_TO_FIND_EMAIL` already means.
`promote_to_ready` logs the promotion itself, carrying the score that
justified it (`P(f>0.5)=0.997 ≥ 0.90`), and passes `log=False` so the transition is not printed
twice; the score cannot ride in `reason`, which holds the LLM's qualification rationale.

**The walk is also the job's time accounting.** `ROWS` pairs each row with a name, so every action
logs which row fired and how long it took (`[Email Outreach] buy an email address — 2.3s`), and at
`debug` every row logs its decision time even when it declines. The steps log what they *did*;
without this a row that spends twenty seconds deciding it has nothing to do says so nowhere.

When no row fires, `pipeline_summary` prints the counts — and that same line becomes the *reason* the
job stopped short, because the counts separate a drained search from three addresses still on order,
which are a dead end and a reason to run again in an hour. It used to be throttled to one line a
minute (`_log_idle`, `IDLE_LOG_INTERVAL_S`): idle was the normal state of a process that never ended,
so a line per cycle buried everything else. A bounded job stops at the first idle, so it prints once
and the throttle is gone.

**Log vocabulary is the operator's, not the schema's.** A row is named for what happens to a lead
(`find & qualify new leads`, not `top_up`), the idle counts say what each group is waiting *on*
(`60 waiting to be ranked`, not `Qualified=60`), and the one remaining gate is printed as its
consequence (`no finder key, so not buying addresses`, not a boolean). Function and state names
belong in the code and the diagrams; a log line is read by someone asking what is happening to their
pipeline.

**Campaigns no longer take turns, and then there stopped being more than one.** `_rotate`
round-robined them because a process that never ends has to decide, forever, whose work to do next;
a bounded job left the fairness question with nobody to be fair between, and the scheduler that
answered it went. The `Campaign` model itself followed — one install, one implicit campaign, and
what it held is now two environment variables and the walk's own rows.

### Steps

Each takes one entity and returns the next `DealState` or `None`; `cycle._apply` then does **one**
`deal.save()`, so a transition and the fields that justify it commit together. Steps are **total**:
they catch the failures they can actually meet and return an explicit state, so the cycle's
`try/except` is a bug backstop, not a retry policy. A step that wants to wait writes
`deal.not_before` — that is the only retry mechanism there is, and it is also what lets a bounded job
end honestly: a row that waits stops being due, so `run_one_action` can return `False` and mean it.
`HALTING_ERRORS` (today `ModelHTTPError`) is the exception: a misconfigured LLM key ends the job with
a typed `bad_config` line rather than being retried forever behind an `alive` log line.

1. **`buy_address`** (`enrichment/lookup.py`) — resolves cheapest-first: an address already on the lead → `RESOLVED`; the free hub cache (`contacts.resolve`) → `RESOLVED`; else the chain's leading finder (cheapest configured first) fires a job through its `start`, `Deal.lookup_provider` records which one, and the handle lands in `lookup_request_id` as the deal parks at `FINDING_EMAIL`. Couldn't-submit (no finder key, API down) stays in `READY_TO_FIND_EMAIL` — no credit spent, no handle to poll — but **backs the deal off first** (`_back_off`, the same doubling the poll uses). Without that the row is still due on the next pass, so the same deal is re-picked immediately and forever: noise every few seconds under the daemon, an endless job now that a run stops only when nothing can advance. *Queued* and *due* are different things.
2. **`check_lookup`** (same module) — polls `lookup_request_id` once: hit → `RESOLVED` + hub give-back; miss → `NO_EMAIL_FOUND` (no reason, blank outcome); still-running → double `not_before`. **The only terminal outcomes are the provider's own** — no deadline, no attempt limit. Both were tried: past the deadline the leg abandoned the job and reverted the deal, where the buy step bought a *second* job for the same lead, so a provider outage became a hot resubmit loop (418 submits and 4,512 polls in a week for ~40 leads, none terminating). Doubling makes waiting nearly free (a week costs 17 polls) and refuses to mislabel — a timeout is evidence about the provider, not about the lead. The interval rails at `COLLECT_BACKOFF_MAX_S` (a month) only so `datetime` can still express it. A deal parked here with an **empty** `lookup_request_id` has no job and never cost a credit, so row 1 routes it to **`reclaim_lookup`** → `READY_TO_FIND_EMAIL` instead. The row used to `.exclude(lookup_request_id="")` and skip it, which stranded it in a state no other row claims — measured on a live install: two deals stuck for 206 hours.

A third and fourth step stood here — `send_first_email` and `answer_reply` — and both left with
`emails/*`. Note what did **not** follow them: `_store_identity` in `check_lookup` still writes the
`first_name`/`last_name` the provider echoes back with the address, because those are export
columns (a sequencer's `{{first_name}}` merge tag) rather than send machinery.
3. **`top_up`** (`core/pipeline/top_up.py`) — the one step with no queue to walk, because a lead nobody has discovered yet has no row to find. One acquisition move per call, chosen by the qualifier's own cold/explore/exploit strategy (unchanged — see **Qualification ML Pipeline**).

### Pacing and capacity — removed with the sending leg

Three guards lived here — **hours** (Mon–Fri 08:00–20:00 in the operator's own timezone),
**rate** (a 3.5–4.5 minute gap between two first emails from one box) and **volume** (a per-box
daily ceiling *measured* from the box's own Sent folder rather than configured). They exist now in
OpenOutSend, and the reasoning is preserved with them, because it was hard-won: receivers
punish rate and volume separately and a recipient reads the *hour*, so no one guard covers the
others.

Nothing replaced them here. A finder emits nothing to pace.

## Opt-out and suppression — it left with the sender, and that is a legal position

This project **has no opt-out mechanism, and needs none**, because it contacts nobody.

What was here: a `List-Unsubscribe` header pointing at a `+unsub` alias of the operator's own
sending address, a visible reply-line in every body, a mailbox scan that caught client-generated
unsubscribes the threaded reader could never see, and an outreach agent with a `suppress` action
for worded requests — all enforced permanently on `Lead.disqualified`. Every one of
those is a **sending** mechanism, so all of them moved to OpenOutSend with `emails/*`, and
`core.db.leads.suppress_email` went with them.

A finder that never contacts anyone is not the sender under CAN-SPAM / GDPR / CASL, so the duty is
not inherited — it belongs to whatever tool makes contact. Instantly and Smartlead both block a
suppressed address at import.

**Two things survive, and they are not the same thing.**

- **`Lead.disqualified` stays.** It is the permanent, account-level exclusion that eleven candidate queries already filter and that `core/export.py` filters too. What is gone is the *inbound* path that used to set it from mail we received; nothing writes it automatically any more.
- **The hub store's own suppression is untouched and unrelated.** A person objecting to the shared contacts store is removed store-wide through the hub's endpoint. That obligation arises from contributing vectors and addresses, runs between the data subject and the hub, and involves no sequencer at any point.

**The one duty the split hands to the operator**, which no code here can discharge: turn on the
receiving sequencer's **import deduplication**. Re-exporting a lead who was contacted but never
opted out can otherwise contact them twice. It is opt-in on Smartlead and undocumented on
Instantly, so every adapter's docs must say so.

**What is knowingly given up.** Bounces never come back, so the enrichment leg has no signal that
it is emitting dead addresses — and there is no other source for that. It is the one cost of the
one-way boundary with no substitute, and the one thing that would justify reopening it (as a single
inbound event, not a reply vocabulary). `lead_id` rides in the export as the join key, so the door
stays open at no cost.

## Qualification ML Pipeline

GPR (sklearn, `ConstantKernel * RBF` inside `Pipeline(StandardScaler, GPR)`) with BALD active
learning, over 384-dim FastEmbed embeddings (`BAAI/bge-small-en-v1.5`) stored on `Lead.embedding`.

**The fit is never persisted.** `qualifier_for` holds it in memory keyed on `_label_fingerprint()` —
a digest of exactly what the fit reads, the per-lead verdicts plus the anchor set — so it is reused
while the evidence is unchanged and refit the moment a verdict lands. A `model_blob` column used to
be written on every fit and read back by nothing at all, which is what it was: a cache of a value
already derived from rows that are still there.

1. **Discovery** feeds the pool as **a best-first search over every search the store's keywords compose** (`core/pipeline/walk.py` — the card `p1-e3-the-walk-explores-the-vocabulary-best-first`). An **arm** is a search at a page; the **frontier** (`core/pipeline/search_arms.py`) holds every arm next to what was fired, and after the first round each round fires the one that one draw of the walk's belief about the market (`core/pipeline/belief.py`) puts first among the ten worth the most at the mean.

   **How the next search is chosen: Thompson sampling over the ten best at the mean, on a belief measured by what searches brought.** The walk infers how many people hold each keyword and how many of them fit, keeps the ten candidates whose page brings the most fits at the mean of that belief, values them on one draw of it, and fires the one worth the most on the draw. The written searches go first, whole. The candidates are every fired search's next page, every unfired written search, and the 200 searches composed from the store's keywords worth the most at the mean. Each fired search then gets five random verdicts on what it brought (`core/pipeline/probe.py`). The paragraphs after this one give the detail.

   What follows from that rule:

   - **It leaves an area that has gone dry.** A candidate is valued on the fits still out there, so one with nobody new values at 0, and so does an area whose fits the store already holds.
   - **It spreads its picks over the good ideas.** A search among the ten with a wide belief draws high on some rounds and gets fired; its probe verdicts then narrow it, and a page that came back without fits falls out of the ten. The greedy pick it replaced (mean plus one σ over 200 draws) fired one idea round after round: 46 of batch 6's rounds 16–66 were `ceo`/`chief`/`executive`/`officer` in another combination, and 34 of the 51 rounds from 17 brought nobody new.
   - **A search is measured on the market, not the store.** A fired search's next page reads its own effect in the fit-rate model, fitted on the people its pages brought. The store's other people holding a keyword — the customer's marks above all — move no lift; they reach the walk as its vocabulary instead: among Bill's marks `wholesale` meant cannabis wholesalers, and counted as evidence they read founder · `wholesale` across the US at 0.35 ± 0.08 after its first page brought 170 people and no fit. The market's own rate `π` is learned from the searched people too: read off the store, it put every unknown keyword at the marks' 0.21.
   - **The GP reaches the choice in two places**: an unjudged person counts at their fit rate times a quarter times the GP's **confidence** at them (`BayesianQualifier.confidence`, the share of its prior variance the verdicts removed there — 0.90 on the judged, about 0.05 on the 780 strangers batch 6's walk brought, who read the base rate whoever they were); and it ranks the pool the probe does not touch. The embeddings are not used to propose terms.

   **A search** (`core/pipeline/terms.py:Terms`, stored as `core.Search`) is a set of `field:term` pairs, each field holding what the provider's match type for it allows (*What the provider can express*, the probe table above `discovery.SEARCH_FIELDS`): **title** words, any number, every one required in any order (`match`); at most one **tag** phrase, adjacent and in order inside one of the company's tags (`match_phrase`) — a word, or a phrase a company's tags hold — and **never without a title**, since a tag alone pages one employer's whole staff; one **place**, the market's country or a state in it (`state, country`, parsed); at most one **seniority** from the closed list of 12. No field takes query-string syntax, two include-list entries OR on every field, so **no query can AND two tags**. A search holds a title or a seniority: a place alone is the market, whose one 10,000-row window is the famous-company head. `Terms.to_filters` is the only place a search becomes provider JSON. The size band and the market's country code are the **shape** the customer fixed: every `Search` carries the initial population's and they never vary. Identity is `Search.key` (sha256 of the canonical terms), so a search two moves reach is one row.

   **A person matches a search when every term matches its own field** (`Terms.matches`), read from `Lead.source_fields` (`discovery.SOURCE_FIELDS` — `contact_job_title`, `company_keywords`, `contact_location_state/country`, `contact_seniority`) with each field's analyzer copied from the provider's (`core/pipeline/analyzers.py`): words are split by Unicode's word rules (ICU, via `icu4py`) and lowercased by `str.lower`, accents kept (`gérant` 35,377 ≠ `gerant` 3,972) and Turkish `İ` keeping its dot (`İşletme` ≠ `işletme`); a straight apostrophe or a dot joins a word (`d'affaires` = `daffaires`, `c.e.o` = `ceo`) where a curly apostrophe splits it, an underscore joins (`home_and_garden` ≠ `home and garden`), a Thai run is one word and a Chinese character one each (`经理` = `理经`); nothing is stemmed on either free-text field (`manager` ≠ `managers`, `vape` ≠ `vapes`); a title drops Elasticsearch's English stopwords and a tag keeps them as words of its phrase (`food and beverage` ≠ `food beverage`); a hyphen splits a title word and also matches the joined form (`co-founder` = `co founder` = `cofounder`), while in a tag it only splits (`e-commerce` = `e commerce` ≠ `ecommerce`); `&` and other punctuation are stripped. A tag phrase made only of stopwords is never a term. **A row with no state matches any state of its own country**: the provider filters on a state outside the US (`Bali, Indonesia` counted 15 where `Indonesia` counted 158) but returns one on US rows alone, so an empty state says nothing about where a person is.

   **The initial population is written whole by the LLM**, not drawn at random (`icp.write_initial_population`, `initial_population.j2`, one call): 4 searches, each a person worth finding — "a head of procurement at a cannabis extraction company in California" is `{title: head, procurement; tag: cannabis extraction; place: California, United States}`. The model knows which words go together, and a search keeps them together, where the old seed shredded `head of procurement` into one-word openers and the walk had to rebuild by chance what the model already knew. The answer carries the market and the size band too. Under `--agent-qualify` the calling agent writes the same answer with the anchors (`population_pending` → `find --population`). The customer's marks are never fired as searches: their terms join the vocabulary searches are composed from, and they train the GP.

   **A round** (`walk.run_round`) is one blocking step of the run, taken where a search used to be: **read the frontier** as the store stands, after the run has judged what the last round brought in; **fire the search one draw of the belief puts first among the ten worth the most at the mean**, a worker thread making the API call and nothing else; **wait for it**, then in the main thread record its page and take its people in — one embedding pass, shortest profile first (`openoutlearn.embed_texts`, batches of 16, vectors handed back in order), since a batch pads to its longest text and embedding is most of a round's time (15 pages page by page took 14m35s, 10 in one sorted pass 2m42s; `profile_text` is never capped to save time, since it is what the GP reads). Round 0 is the written population, fired whole and unvalued. Nothing is ranked on landing and nothing is ever cut: an arm not fired stays on the frontier and is recounted every round. **Every store is anchored** (`walk._anchor`): the anchors are written with the initial population (`icp.ensure_anchors` right after `write_initial_population`, from the one `--population` answer under `--agent-qualify`), so a store seeded with `--labels` — which starts with fits and so never ran cold — has them too, and a store from before them gets them once before its next round. When none can be written the round raises `unanchored` and fires nothing. The set is still never topped up once it exists.

   **The arms** (`search_arms.Frontier.arms`) are every fired search's next page (`deepen`), every unfired written search, and the searches **composed from the qualified profiles' keywords** (`core/pipeline/compose.py`) — the `COMPOSED` (200) worth the most at the mean belief. **No term is cut before it is valued**, the way an LLM scores its whole vocabulary at every step: the vocabulary (`Frontier.vocabulary`) is every title word the qualified profiles (`judged_fits`: the customer's good marks and the run's fits) hold on its own (never a piece of a hyphenated word: `co` alone finds 283,068, co-owners and co-CEOs beyond the 179,644 co-founders), every whole tag they hold (never a phrase inside one: `pre` from `pre-rolls` found 6,512 founders of pre-seed funds, pre-k schools and pre-owned car dealers against the 71 tagged `pre rolls`, and `health` from `health care` 11–15k founders of anything; `analyzers.sub_phrases` builds the phrases a tag search matches, for matching only), the written searches' terms, all 12 seniorities, and the market's places — the country and the states its people live in, or only the written states where every written search names one. **The titles and tags come from the fits alone**: with every lift centred on the market's rate, a keyword no search has held reads that rate, and the store's 10,000 would be ranked on size alone — the fits are how the customer's marks reach the walk before a search is fired. At the mean each keyword brings a **lift** (`logit θ_t − logit π`) and a **share** (`log N_t / N₀`), both summing over a search's keywords (`Beliefs.lifts`, `Beliefs.shares`), so the best searches are found without listing them: at one size — a 0.1-wide bin of the summed share — only the highest summed lifts can be worth the most, so the fields are added one at a time keeping the `PER_BIN` (4) best per bin — every title word alone, then each word that lifts added where it pays, then a seniority or none (never one a title word already names: title and seniority `founder` are near one keyword, and 19 of batch 6's 23 searches held both), then the place, then every tag — and each finished search is valued at the fits a page of it brings (`Frontier.page_worth`, with nobody seen) — per dollar it ranked on the rate alone, since a verdict a person is most of a page's cost, and its best were searches of 1 to 6 people. A search below one person is never composed. The frontier then values the composed searches exactly — the store's people they match, the counts of fired searches they narrow. **A composed search always holds a title and a tag**, since a tag-less search is the market's head for its title (batch 5's two brought 334 of a round's 426 new people, none in cannabis). A store with no written search has no market and composes nothing. No LLM is called. **Never an arm**: a search already fired or dead, one holding all of a dead search's terms, or one holding all of a **drained** search's terms (its pages returned everything it counts, so a narrower one can only re-find people the store holds — batch 5 spent all 5 stricter slots of a GA generation that way), or, where every written search names a state, a search of the whole country (`Frontier.in_market`): the market is the states the customer asked for, so neither the cover nor the fill leaves them. A store written partly on the country has no such limit. **Searches are composed at the mean only**: composed on each draw under the GP prior this belief replaced, only 9 of 50 draws stayed in CaleafTech's market — across 10,248 tag phrases one always drew high (`military science`, the "Self employed" list's `finacial services`).

   **The belief** (`core/pipeline/belief.py`, the card `p1-e3-the-belief-is-one-bayesian-model`) holds two numbers per keyword, a title word, a whole tag, a seniority or a place: `N_t`, the people holding it over everyone a search runs on (all of LinkedIn in the market's country, the customer's trade and every other), and `F_t`, how many of them fit, so its rate is `F_t / N_t`. **A fired search is one sample of the AND of its keywords**, taken as independent, and both halves are **one hierarchical model sampled whole by NUTS** — PyMC, compiled by nutpie (numba, so `uvx` needs no C++ compiler), 4 chains of 500 warm-up and 500 draws at `target_accept` 0.99:

   ```
   log N(q)   = log N₀ + Σ_{t∈q} ( μ_field(t) + γ · x_t + u_t )     x_t = log(people_t / market people)
   logit θ(q) = α + Σ_{t∈q} ( p_t + w_t − u_t ) + e_q              p_t = logit(fits_t / people_t) − logit(market's)
   (u_t, w_t) ~ MVNormal(0, diag(τ, s_f) · R(ρ) · diag(τ, s_f))    non-centred
   count(q)   ~ NegativeBinomial(N(q), φ)
   ```

   A place is one keyword, the state where the query names one and the country otherwise. **A keyword's lift is its fits less its size**: `u_t` is how far its log people stray from the guess and `w_t` how far its log fits do, so a count far above the guess lowers the rate before any verdict, the fits held where they were guessed. On batch 6, `processing` was guessed at cannabis processors' 2,000 people and its first search counted 61× that; with the lift free of the size, it took 14 no-fits to come down. Verdicts move `w_t`. **The spreads are learned** — `τ`, `s_f`, `s_e`, `φ` and the correlation `ρ` between a keyword's size and its fits. **The count is a negative binomial**: inside this model it covered 16 of batch 6's 20 held-out counts in its 80% interval, against 14 for a log-normal on `log(y + ½)`. **The prior is the LLM's guess, in plain counts** (`keyword_guess.py`, `guess_market.j2`, `guess_keywords.j2`): counted over what a search runs on (`keyword_guess.COUNTS_MEAN`, in both prompts and the `keyword_guess_pending` payload): the market is everyone on LinkedIn in the customer's country, not the customer's buyers — handed a payload that did not say so, batch 6's session guessed the customer's market, 60,000 people. First the market, once (`core.MarketGuess`: its people and how many of them fit), then per keyword how many people in the market hold it, in the customer's trade or any other, and how many of those fit (`core.KeywordGuess`), every keyword against the stored market so all share its scale, each field's keywords in one structured answer beside the ones of that field already guessed. A count that breaks the market (more people than it holds, more fits than it has) or a keyword not asked for is refused and the keyword asked again; fits above people do not parse. `belief.Guesses` turns the counts into the model's terms: `x_t = log(people / market people)`, evidence whose weight `γ` the counts learn, and `p_t = logit(fits / people) − logit(market fits / market people)`, half a count on each side (`count_log_odds`, not `clip_rate`, whose 0.001 floor sits above a guessed market's rate), the lift's centre with no weight on it: a wrong guess is corrected by the keyword's own `u_t` as its counts come back and its `w_t` as its verdicts do, and how far keywords stray from their guesses is the learned `τ` and `s_f`. A keyword with no guess, or a store with no market guess, reads its log share of the store and `p_t` 0, the market's rate. `next_round` has the market and every keyword of the vocabulary and the fired searches guessed first (`Frontier.guess_keywords`): all of them before the first composed round, then the ones new fits bring, one `AI_MODEL` call for the market and one per field, or a `keyword_guess_pending` stop under `--agent-qualify` answered with `--keyword-guesses` (its payload names what was refused, and why). A keyword is guessed once; a replaced brief clears every guess. On batch 6 the walk's searches moved from `retail` and California to the customer's market the round a prior on the lifts went in. `log N₀` is anchored broadly at the guessed market's people (`Guesses.market_people`, else `MARKET_PEOPLE`, 10M), since no search counts the market itself — fitted to the counts alone it took the size of round 0's searches of 4 to 21 people. A keyword no fired search holds reads `p_t` at the mean and on a draw `p_t` plus a pair from that draw's own `MVNormal(0, Σ)`, seeded by the draw's seed and the keyword so every read agrees. **Sampling has no mode to reach**: the Newton fit with a Laplace posterior it replaced fell into a 2-cycle on batch 6 and stopped at `α` = +17.86, every arm reading `θ(q)` = 1, so the walk fired ever larger searches. **It refits only when a page or a verdict lands** (`belief.fitted`, keyed on the fired pages and the judged set and held for the process, as `qualifier_for` keys the GP), and the model is compiled once per size (`belief.COMPILED`: searches and keywords padded to a power of two, the data swapped in), since compiling is 2.5–5 s and sampling 1–2. A `Draw` is one stored posterior draw chosen by its seed (`Beliefs.draw`, `draw_sizes`), both halves from the same draw, so `F` is joint. **The fit rate's evidence**: each fired search's people add `evidence` — a judged person their verdict, an unjudged one their fit rate at `SCORE_WEIGHT` (a quarter) of a verdict times the GP's confidence at them, since every score comes from the same few verdicts and a score far from all of them is the GP's prior mean; a company the mean of its people once — as a binomial sample of its `θ(q)`. **`α` is the searched market's rate**, centred on the store's (`People.base_rate`) at `MARKET_RATE_SPREAD` and learned from the searched people only: on batch 6's 23 searches it reads 0.06 against the store's 0.13. **`e_q` is a fired search's own effect**, so a page that came back bad says so about the search that brought it; its next page is valued on `α + Σ (p_t + w_t − u_t) + e_q`.

   **The search fired** (`search_arms.Frontier.pick`) is **Thompson sampling over the top `k`**: the `TOP_K` (10) arms whose page brings the most fits at the mean (`search_arms.top_at_mean`, ties broken in one order) are valued on one draw of the belief — one joint draw of `α`, the fitted lifts and every fired search's own effect (`Beliefs.draw`, `Beliefs.draw_searches`), each other arm keyword's lift about its prior, and one joint draw of the sizes (`Beliefs.draw_sizes`) — seeded `(round, 0)` so a rerun repeats, and the one with the most fits on that draw is fired. The top `k` keeps the pick to arms that make sense at the mean; the draw spreads the picks among them by how sure the belief is. Measured offline at batch 6's round 68: over 50 seeds one draw over all the arms fired 40 distinct searches where greedy fired 1, and at `k` = 10 the pick moved from `processing` synonym stacks to `ceo executive · cannabis`. Composing on the draw (every keyword drawn, then composed) is not done: maximising over 504 drawn lifts overvalued its own pick, a median of about 75 fits on a page where real pages bring 10–20%. A next page keeps the provider's count on any draw. **The probe settles what the draw tried** (`core/pipeline/probe.py`): past the cold phase, each `top_up` pass first judges one person drawn at random from the latest search whose people hold fewer than `PROBE_VERDICTS` (5) verdicts. No other move reaches those people — exploit takes the best-scored, BALD the least certain, calibrate one per score band — so on batch 6 founder · `food` brought 186 and none was judged; five random verdicts with no fit take a search from 0.22 ± 0.21 to about 0.08 ± 0.09. `judge_or_search` weighs the arm worth the most at the mean belief (`search_arms.best_expected`) against judging. **The walk stops** when no arm is expected to bring a new fit (`FITS_WORTH_A_SEARCH`, one on the page), and the pool is judged to its end; the cold phase has nothing yet to stop on, so it fires while an arm brings somebody new (`search_arms.worth_firing`).

   **An arm's fits** are `new · (F − held) / (N − seen)` (`search_arms.Arm.density` times `new`), and `judge_or_search` divides them by its cost. **N** is the provider's count where the search fired, otherwise the belief's estimate, capped by the count of any fired search it narrows and kept `SEEN_MARGIN` (a tenth) above the store's people it matches, since an estimate below them is disproved and must not close the search for good. **F** is `N · θ(q)`. **seen** and **held** are the store's people it matches (`store_snapshot.People.match`, a per-field inverted index that answers what `Terms.matches` would over everyone) and the fits expected among them, each person at their verdict or fit rate, or the market's rate where the GP cannot score them: counting only the judged would leave an unjudged person's share of `F` on those still out there. **new** is the people the page would bring that the store does not hold: for a next page, as many as the latest page was first to bring in (the **page–lead table**, `core.PageLead`, says which page holds a person first); for an unfired search, a page of those still out there, times the share of rows a page keeps. A person with no company counts with the others sharing their tag list. **The fit-rate map** (`store_snapshot.fit_rate_map`) is an isotonic regression of the store's verdicts on their out-of-fold scores, fitted once per label set: the GP ranks well but is not calibrated (0.15 where the truth was 0.28), and only a fit rate can be weighed against a cost. Choosing verdicts by score does not bias the rate *at* a score; with one class of verdicts the score stands in. **The costs** are dollars: a search's measured ready time, and for each new person their measured ingest time (`meter.INGEST_SECONDS_PER_LEAD`) and a verdict, so a page of 200 strangers pays for every one.

   **The page–lead table records every person a page returned** (`discover.harvest`), whether new to the store or not; rows dropped for no headline (`is_live_profile`) or a company at its cap are not on it, and neither is a marked lead or an anchor, so the marks never inflate a search. It replaces `Lead.discovered_by` and the query terms, which kept only the first search to find a person. Rows are stored in one batch (`db.leads.store_rows`, one embedding call for every new row), and a lead's embedding is its `profile_text` alone — no query terms ride it, since a search valued on its people's scores must not be valued on its own words.

   **What ends a search is a corpus fact, never a model fact.** A search whose **page 1 is empty with a count of 0** is dead (`Search.dead`), and so is every search holding all its terms (`Terms.holds_all_of`: a stricter search matches a subset of the people). A page past the provider's count, at page 50 (`LAST_PAGE`, where the 10,000-row reach ends) or empty is **at the bottom** (`Page.at_bottom`): its search is not paged again, but a narrower or sibling search can still be composed, since it opens a fresh window — unless it is **drained**, its pages holding everything it counts, when no search holding all its terms is ever an arm. Nothing is pruned for scoring badly: a poor arm is only valued low, and revalued next round. **An empty page 1 with a positive count is a transport artifact** (a burst once answered a 71-million-lead query with nothing in 0.0s) and leaves no trace, nor does an outage; a refusal (401, 402, an exhausted 429) is re-raised, since every later search would repeat it. The count is read only at offset 0, because past the end of any result set the API reports 0.

   **Firing within the rate limit.** A search is a submit plus its polls (`bettercontact.search_and_poll`), and the first poll goes at the store's measured typical ready time (`meter.SEARCH_READY_SECONDS`, 1.5s until measured), then one a second; a page holds 200 (`limit`/`offset` — never `max_leads`, which reports the rows returned as the query's size). Every request the key sends, polls and enrichment included, passes one per-process budget (`bettercontact.RequestBudget`: 15 in any 60 seconds, `GET /account` exempt as the provider exempts it), a share of the key's 100/min since a process cannot see another's requests; a round spends about 12, and the run judges for longer than a minute between rounds, so it rarely waits. The limit is never probed by going over it — a client firing through 429s can get the account blocked — and the fulfil skill runs at most six `find`s on the key at once, 90 a minute between them.

   Every search and fired page is inspectable in Django Admin (`SearchAdmin`, `PageAdmin`: terms, origin and move, count, dead, pages, the round each page was fired in, and each page's people).

   Each unit of work (`top_up._advance`) picks a lead to label by the qualifier's own **explore/exploit** split (`acquisition_mode`, driven by class balance), and that split *is* the whole steering. **Cold phase** (`qualifier.is_cold` — `n_real_positives < ANCHOR_COUNT`, a clock that ends once real acceptances reach `ANCHOR_COUNT` even though the anchors themselves never leave): judge the strongest lead every pass, and fire a round of the walk only when the pool is empty. A round is 4 pages, up to 800 people, so the first one fills the pool for many verdicts; rankings here still lean on the anchors' *guess* at the ICP (below), so nothing observed says a second round beats judging what the first brought. A round adding nobody is still a move: only a frontier with nobody new left ends the pass. **The cold phase always exploits**: while real positives are still short of `ANCHOR_COUNT`, the campaign's one goal is *more real positives* — reaching that count ends the phase and makes the downstream balance real, though the anchors stay in every fit after that too — and the highest-P lead is the one most like the ideal profile. BALD does the opposite, spending each call on the lead the model is most *confused* about, which with invented positives is the lead least like the ICP; a live run picked four in a row at P≈0.25–0.42 and got veterinary services, cybersecurity education, K-12 tutoring and a metaverse PM against a health-and-wellness ICP. (The balance could not have chosen it anyway: while `is_cold` held, `n_neg > n_pos` was false by construction, so the axis was pinned to BALD for the whole phase.) **Explore** (`neg ≤ pos`, past the cold phase): label the most *informative* lead in the pool (max BALD) with **no gate** — a low-confidence lead is exactly the label that teaches the GP the most, so filtering by confidence here would discard the point of exploring. The GP now ranks on real positives (plus the permanent anchors), so labelling *is* the better move; run a round only when the pool is empty. **Exploit** (`neg > pos`): spend the LLM call on a lead that will actually convert — the strongest lead clearing `calibration.current_threshold()` (`consumable_candidates`); if none clears it, **judge or search, whichever is cheaper per fit** (`core/pipeline/judge_or_search.py:decide`). Judging the best lead waiting costs `c_verdict / rate(best waiting)`; firing the arm worth the most at the mean belief costs `cost(arm) / expected fits(arm)`, the inverse of the value the walk ranks arms on, with a verdict on each of its people in the cost. **Both rates are fit rates on one scale** — the GP's score through the isotonic map fitted on the store's verdicts at their out-of-fold scores (`store_snapshot.fit_rate_map`), so a waiting lead and a search's people are weighed alike, never a raw score against a mean of raw scores. Before the first round the pool is judged, and after a round that added leads it is judged at least once before the next (no verdict in `meter.POOL_FIT` since its reset): on batch 5 a GP scoring everyone alike fired six rounds with no verdict between them. `meter.ROUND_LEADS` is a running mean of each round's lead count after its harvest less before, and `meter.POOL_FIT` — how often the verdicts since a round last added leads were fits — starts over with each such round and is logged beside the decision. The times in `c_verdict` and `c_arm` are running means measured on the store (`core/meter.py`, one `RunningMean` row each — sum and count, read by every process of an `--agent-qualify` run); the bills are a verdict's tokens at the model's rates or `verdict_dollars_agent`. When the walk has nothing left to fire the pool is judged to its end. A fixed BALD floor (`min_bald_gain`) used to stand here and judged the leads the model was least sure of — on a store with dozens of real positives, the off-industry ones. **Calibrate** runs ahead of both, past the cold phase (`core/ml/calibration.py`): while the store holds fewer than **√N calibration verdicts** (N = the unjudged pool, until the first bar is measured), the pass judges one lead drawn at random from the score band holding the fewest (`CalibrationDraw` keeps the band, the score and the band's size at draw time). Once √N are in, `measure` scores every calibration lead out of fold on today's labels (`GPBaldQualifier.out_of_fold_probs`, k=5, deterministic per fold), fits the **fit rate at each score** as an isotonic regression of the verdicts on those scores (`openoutlearn.calibration.fit_rate_curve` — no weights: stratifying on the score does not bias the rate *at* a score), and derives the bar from **costs**: every lead is judged before it ships, so the bar decides where the run's time goes, not the list's quality. Costs are in dollars: run time priced at `run_hour_dollars` (the smallest machine a run needs on its own — runs are independent, so its hours are all time costs: a 2 GiB DigitalOcean droplet) plus what the step is billed. A verdict on a lead fitting at rate p buys p fits for `verdict_cost()` — `verdict_seconds_*` of run time plus the LLM's bill — for a model's call, `model_verdict_bill(AI_MODEL)`: its `MODEL_PRICES` rates times `verdict_input_tokens`/`verdict_output_tokens` (`verdict_dollars_model` for a model the table lacks), so a dearer judge raises the bar; for a calling agent's turn under `--agent-qualify`, `verdict_dollars_agent`; a search — one round — buys `leads_per_search()` new leads (`meter.ROUND_LEADS`) for `search_cost()` — its measured run time, `round_seconds` until there is one, since Lead Finder bills nothing — plus the verdicts those are worth. Judging pays while p beats the break-even rate — the search route's best fits per second times a verdict's cost, the marginal value theorem's rule (`break_even_rate`) — and the bar is the lowest score whose rate is above it (`lowest_bar`), stored as a `CalibrationBar` with its break-even. `current_threshold()` returns it, or `min_gp_confidence` until there is one. Once a bar is measured the arm is closed: pool growth no longer reopens it, since on batch 5 every round of mostly off-industry people raised √N (40 → 53) and bought another stranger judged at random while the bar held at 0.19–0.21. The arm switches on only when √N reaches 8 bands × `calibration_min_band_verdicts` (N ≥ 1,600), so a small order never spends verdicts on a sample it cannot fill. Calibration verdicts train the GP and rate the walk's arms, but not in the mode balance (`BayesianQualifier.balance_counts`): they are mostly low-band rejections, and counted there they would end explore after the first calibration. Why it exists: a 500-lead order at the fixed 0.7 judged 71 people in 17h24m while 21,134 waited, because 2 of them cleared it.

   The gate is the **same constant the promote gate uses**, and it belongs to exploit alone: it is a *spend* gate — "will this LLM call buy an email, or just park at QUALIFIED?" — not an "is this pool promising?" judgment. Explore wants labels, not emails, so it never consults the gate. (The earlier design applied the gate in **both** states and so ran BALD over the confidence-*filtered* set — picking the most-uncertain lead from a bucket it had just stripped of uncertain leads; that incoherence is what the explore/exploit split removes.) Two other bars that *were* judgments both failed earlier and are not to be reintroduced (see `top_up.py`'s module docstring): each compared an **out-of-sample** candidate score against a bar drawn from **in-sample** ones, and a fitted GP never puts those two populations on the same scale. **Measured 2026-07-17**: the pool tops out at 0.327 against a 0.9 gate, so exploit rarely fires until many more labels exist — a lead the LLM accepts meanwhile parks at QUALIFIED unemailed; it did its job by contributing a label.

   **The GP ranks people, never query words.** One model decides *which lead to label next* (`qualifier.acquisition_scores`), gates the paid lookup (`calibration.current_threshold()`, read by `promote_to_ready` and by `top_up`'s exploit branch — the same getter in both, so they cannot drift), and scores the people on each page the walk ranks. A model over query words was tried and lost: over bare keyword strings the GP's posterior collapses toward its prior mean (median +0.080 against +0.797 for real profiles, because a keyword string sits far from every profile embedding), and word counts lost to the GP over people. `min_gp_confidence` is *only* the spend gate on the paid lookup.
2. **Balance-driven selection** — `n_negatives > n_positives` → exploit (highest P); else → explore (highest BALD). Anchors count as positives here, but the balance decides nothing while any of them stand — the cold phase is pinned to exploit (above); both run against a real posterior from the first pass. If `acquisition_scores` still returns None the campaign is *unanchored* (LLM outage, no ICP text) — the degraded path, where selection falls back to `creation_date` order because nothing can rank.
3. **LLM decision** — every qualify decision is an LLM call (`qualify_lead.j2` reading the lead as `qualify.verdict_fields` hands it over — title, company, the company's LinkedIn size band (`Company.size_text`: *51–200 employees (LinkedIn)*, or *not reported*), location and the company's LinkedIn tags as labelled fields, then the stored `profile_text` with a note on its field order — beside the customer's two texts and the store's **brief**). The labels are there because `profile_text` is the provider's fields run together, lowercased and unlabelled: read bare, a model glued the industry label to the seniority bucket ("manufacturing … president/vice president" → "VP of Manufacturing" for a VP of finance), took "alternative medicine" for a company name, and read a first-person headline as the seller's text. The company tags carry what the company makes or sells (a dispensary, an emulsion house) and are marked as possibly the wrong company, since Lead Finder fuzzy-matches the employer. The size band is there because a customer's "small" is a size: without it the judge guessed size from the company's name, and a 500-lead order for small vape brands came back with 229 of 258 misses marked *company too big*. It is a labelled field and **not** part of `profile_text`, which is embedded — adding it there would move new leads off the vector space every cached `Lead.embedding` was built in. The `--agent-qualify` payload carries the same fields, so an agent and a model judge from one input. The GP is used only for candidate selection and the confidence gate. The prompt makes the brief binding (an exclusion wins over seniority, a role missing from the fit list is out), asks whether the company buys what the customer sells or sells it, tells the model to use what it knows about the company, and asks for a one-sentence, customer-facing reason. No temperature — the current Claude models reject one that isn't their default. **The brief** (`core/pipeline/brief.py`, one `Brief` row in `core/models.py`) is our reading of the customer's text — who fits, who does not, which companies look like buyers and are not — so it is produced, not configured, and lives in the store. `ensure_rules` has `AI_MODEL` write it (`write_brief.j2`) the first time a prompt needs it; `find --brief PATH` replaces it, which is how a stronger model than the judge (the agent running an order) writes it once. The seed and anchor prompts read it too, so a misreading the brief settles never reaches the walk. Why: on the first 500-lead order the model judged from the customer's two paragraphs and a generic checklist (role, industry, *decision-making authority*, size), and about a third of its fits broke rules the target already implied — hardware suppliers accepted as buyers because their executives had "authority over hardware", edibles-only brands, finance and sales roles.
4. **Rank gate** — `ready_pool.promote_to_ready` promotes `QUALIFIED → READY_TO_FIND_EMAIL` when `P(f>0.5)` reaches `calibration.current_threshold()` — the measured bar, or `min_gp_confidence` (0.7) until there is one — so a paid credit is only ever spent on a ranked lead.

The GP needs ≥2 labels of **both** classes to fit, and `qualifier_for` warm-starts each campaign's
from `Lead.get_labeled_arrays` where it is needed rather than holding one resident. Every qualifier
is now a `BayesianQualifier` fitted on the operator's own verdicts; the pre-trained `KitQualifier`
(a HuggingFace kit) existed only for the promo campaign, which had no labels of its own.

**The GP trains on the LLM's fit verdict and on nothing else** (`get_labeled_arrays`: label 1 = any
non-`FAILED` deal, label 0 = `FAILED` + `wrong_fit`). No market signal was ever in that loop, which
is why handing sending away cost the model nothing — there was never a reply, an open or a bounce in
its training data. The correction loop is the human: the operator reads `reason` and edits the
product description and the ICP.

**Anchors — the cold-phase positives.** A first run has no positives at all: the LLM rejects
everything until the ICP is right, so the label set is single-class and *nothing* fits. That is
not a degraded model but an absent one — BALD, `P(f>0.5)`, the promote gate and the query
selector all go dark together, and the engine walks its whole cold phase blind. So a campaign
with no real positive is seeded with a few synthetic ones: `icp.generate_anchors` has the LLM
invent `ANCHOR_COUNT` (3) ideal-lead profiles from `product_docs + campaign_target`, written in
the shape `discovery.profile_text_for` produces, embedded (`ensure_anchors`) and handed to the GP
via `BayesianQualifier.set_anchors`. Several rather than one so the positive region is outlined
rather than pinned to a single hallucination.

- **Profiles, not the product text.** The space they must land in is one of *lead* embeddings;
  marketing prose about the product embeds nowhere near a row of firmographics and would anchor
  the model where no candidate lives. Like a discovered lead, an anchor is embedded from its
  profile alone, with no search's words folded in.
- **They never become leads.** They exist only as GP observations plus an Admin window
  (`CampaignAdmin.phase`) onto what the model currently believes; no `Lead` or `Deal` row is
  created and nobody is emailed. Paid spend still needs a real acceptance: `promote_to_ready`
  only ever reads `QUALIFIED` deals, and a deal is only `QUALIFIED` because the LLM accepted a
  real lead — an anchor can raise a real lead's score, it can never be the lead.
- **They are permanent — `set_anchors`/`update` never trim them.** The anchors are a guess at the
  ICP; the campaign's own accepted leads are additional evidence, not a replacement for it, so
  they stand alongside real positives for the campaign's whole life rather than being retired as
  ground truth accumulates. `ensure_anchors` still only ever *fills up to* `ANCHOR_COUNT` and never
  invents more once a real positive exists; `stored_anchors` restores the same fixed set on every
  later boot.
- **`is_cold` is a separate clock, not "any anchor still standing."** It is
  `n_real_positives < ANCHOR_COUNT` — not `has_real_positive`, not "is it fitted?": with anchors
  the GP fits from the first pass, so fittedness says nothing about whether the campaign knows
  anything or has only been told what to hope for. The threshold is the same `ANCHOR_COUNT` the
  anchors were sized to, but reading it off the real positives rather than off the anchor count is
  what lets the anchors stay forever without freezing the engine in its cold-phase acquisition
  strategy forever too.
- **Balancing runs in the early window only** (`BayesianQualifier.balances`). `_balance` caps the
  majority at 2× the minority, drawn at random. It is skipped while `is_cold` holds: with the
  positive class still mostly invented, subsampling would throw away real rejections to match it.
  Once real positives reach `ANCHOR_COUNT` it takes over — the anchors keep contributing to every
  fit, but there is now enough real evidence for the balance to mean something, and 3 fixed
  synthetic positives are a small, constant addition next to a growing real count; their pull
  stays local to their own neighbourhood (RBF kernel), which is the shape wanted. **At
  `_BALANCE_UNTIL_POSITIVES` (10) real fits it stops**, and every verdict trains: past a handful
  of fits, subsampling only drops verdicts the run paid for — CaleafTech's batch 5 at 54 fits kept
  108 of 375 no-fits, a fresh no-fit was usually left out, and its lookalike topped the pool on
  the next fit — while the scores need only rank, since `store_snapshot.fit_rate_map` turns them into
  fit rates. Every verdict at 12% fits ranks only because the GP centres its labels
  (`normalize_y` in `openoutlearn.qualifier.gpr_for`): uncentred, batch 5's 445 verdicts fitted
  `RBF(length_scale=3.2e4)`, a flat score, and the fit-rate map gave everyone the base rate. 10 is
  a placeholder until a replay measures where balancing stops helping.
- **They are on no page.** An anchor was never returned by a search, so it scores none; it reaches
  the walk only through the GP, which rates the real people on each page.

## Django Apps

- **`core`** — Engine: the `Search` / `Page` / `PageLead` walk models; the cycle, operator lookup, LLM factory, the environment config + readiness check, the ML/discovery/qualify pipeline, the lead export, geo.
- **`crm`** — `Lead` (identity + embedding + email), `Company` (the shared employer row) and `Deal` (`crm/models/lead.py`, `crm/models/company.py`, `crm/models/deal.py`); also defines `DealState` and `Outcome`.
- **`enrichment`** — **not an app** (no models). `bettercontact.py` (paid finder: the two-leg `submit(query)→request_id` + `poll_once(request_id)→PollOutcome`, the blocking `search_and_poll` transport used by discovery, the shared `RequestBudget`, `is_configured`, `BetterContactQuery`/`Result`/`PollOutcome`/`Unavailable`); `fullenrich.py` (the second finder, and a different shape of one: the bare `professional_network_url` is a complete query, `submit` → `enrichment_id` and `poll_once` → `PollOutcome` once `FINISHED`, the vendor's statuses mapped through, its own `RequestBudget`, and no balance read because the vendor exposes none); `leadmagic.py` (the third finder and the only synchronous one: `start` is the whole lookup and returns a finished `PollOutcome`, so it has no `poll_once` and leaves no handle to strand, one POST to `/v1/people/email-finder` with nothing but the profile URL, a `valid` status or a miss, and a real balance read from the free `GET /v1/credits`); `enrow.py` (the fourth finder added, and the cheapest: `start(profile_url)` POSTs one `linkedin_url` and returns the vendor's `id`, `poll_once(id)` → `PollOutcome` while it runs at HTTP 202, `valid`/`invalid` mapped through, a `lookup_failed` LinkedIn status deferred rather than read as a miss because the vendor bills nothing for it, a `RequestBudget` of 120 a minute with the free `GET /account/info` exempt, and a real balance read from that same endpoint); `provider.py` (`configured`/`active`/`chain`/`by_name` — which finder is cheapest among those holding a key, and the order the rest are tried in; no HTTP of its own); `lookup.py` (`buy_address`/`check_lookup`/`reclaim_lookup` — one entity in, the next `DealState` or `None` out). It sat under `emails/` while a resolved address existed to be written to; that coupling is exactly what made a mailbox-less install produce nothing.
- **`contacts`** — the central contacts-store client (`service.py`, no models, **not** an installed app) — "the hub" (`hub.openoutreach.app`), logged under the `hub:` prefix. `resolve(lead)` (free read-back before the paid finder) `contribute(lead, emails, origin)` (give-back, non-EEA only, registers on first use), `share_profiles(rows, country_code)` (every discovery page, non-EEA only, no credit) and `share_companies(rows, country_code)` (that page's distinct employers, same gates, no person attached). All best-effort; an outage or missing token degrades to a no-op.

**Three apps are gone.** `emails` (Mailbox, the mail log, sender, the send guards, two steps) went
to OpenOutSend. `chat` and `legacy` were model-less anchors holding migration history for
pre-pivot installs; the cut was clean rather than upgradeable, so the history they anchored went
with them — see *Migrations*.

## The Mail Log — moved to OpenOutSend

Every message a mailbox emitted or received was a `Message` row keyed on `(mailbox, message_id)`,
written *before* anything decided what it was, with `kind` + `classifier_version` as the reading and
`processed_at IS NULL` as the third state. Three ordered jobs — **sync** (IMAP → rows, the only
network step), **classify** (pure and versioned over stored bytes, so bumping the version *repairs*
history rather than only affecting future mail) and **project** (bounces → `DeliveryEvent`,
opt-outs → suppression) — plus union-find threading over Message-IDs, so a reply carrying only
`In-Reply-To` still reached its deal.

It exists because the record and the interpretation used to be the same object: deciding a message
was uninteresting *was* deleting it. That cost one install two UIDs for good, the only human reply
it ever received, two apologies to a dead address, and a `SendVerdict` table at 0 rows against 590
sends. The separation is the fix, and it moved intact.

`Deal.thread` and `Deal.mailbox` went with it, and so did `Deal.chat_summary` — a conversation
summary is only worth keeping if something is holding a conversation.

## Migrations

**One `0001_initial` per app, and no history before it.** Removing the sending leg took a large slice
of schema with it — the whole `emails` app, two FKs on `Deal`, four send-era states, the freemium
flag — and the cut was made **clean rather than upgradeable**: the migration history was deleted and
regenerated, and with it went `core/migration_compat.py` and the `migrate` command override that
existed only to reconcile the pre-pivot `linkedin`→`legacy` app rename for existing installs.

The consequence is deliberate and is the reason the decision was recorded: **a database created
before this cut cannot upgrade past it.** A pre-cut install has to stay on the `pre-finder-cut` tag
(the last commit where sending works) — pulling `main` onto it leaves a database that cannot migrate
onto the new `0001_initial`, so the process will not start. There is no upgrade path and none is
planned: export what you need and start fresh. A fresh install migrates from nothing and is
unaffected.

## The Lead Export (`core/export.py`)

The finder's **public output** — what leaves OpenOutFind and reaches whatever the operator
actually sends with. Tier 0 of the integration surface described by the boundary card
(`roadmap/p1-e3-leadfinder-sequencer-boundary.md` in `openoutreach-docs`), whose governing rule
is that **our own sender gets no privileged path**: a sequencer, a CRM and a spreadsheet all
read the same rows.

**The column names are other people's, not ours.** Instantly and Smartlead both *require*
`email`, `first_name`, `last_name`, and both recognise `company`, `title`, `website` and
`linkedin_url` as standard fields, mapping anything else to a custom variable. So the record
uses those names exactly — `company`, not `company_name`; `title`, not `job_title` — and an
exported file imports without column mapping. The internal model keeps its own names
(`profile_url` is provider-agnostic on purpose; a `domain` is not a `website`), and
`lead_record()` is the one translation between the two. That is deliberate: one schema cannot
match N importers, so the mapping is a function, not a migration.

```
email, first_name, last_name, company, title, website, linkedin_url, reason, lead_id, qualified_at, full_name
```

**`find --exclude PATH` narrows it** (`core/exclude.py`): a file of LinkedIn URLs, one per line,
loaded into a contextvar and read at exactly two points — `lead_records()` (so the print, `find 0`
and the goal count all drop the listed leads) and `qualify.fetch_qualification_candidates()` plus
the `--agent-qualify` resume (so a listed lead is never put up for a verdict; a pending row naming
one is deleted rather than resumed). `normalize_linkedin_url` is the hub's function verbatim —
host lowercased, query, fragment and trailing slash dropped — so both sides key on one string.

**A marked lead never prints either.** `find --labels PATH` (`core/pipeline/labels.py`) takes a
customer's marks — the hub's `/ratings/`, a JSON array or JSON Lines — in as verdicts, so a fresh
store starts from what the customer said instead of from the cold start's guess alone. Each mark
that says who fits becomes a `Lead` with `marked=True` (its `profile_text`, `source_fields`,
`Company` and size band from the mark, embedded from `profile_text`) and a `Deal` holding the
verdict: `good` and *already a customer or contact* as `QUALIFIED`, any other *not a fit* but
*gone* as `FAILED` + `wrong_fit` with the customer's reason and note; *not sure*, blank and
*gone* are left out, and a mark with no `profile_text` is listed rather than stored. Stored as a
verdict, a mark reaches the GP (`get_labeled_rows`) and the walk's term pools
(`moves.TermPools`) through the queries they already run, and is written to no page, so it never
scores a search. `lead_records()` (so the
print and the goal), `deals._deals_at_state` (so the promote gate and the paid lookup),
`pipeline_summary` and `status` leave marked leads out — the customer already has them —
and `status` counts them on their own as `marked`. The company cap still counts a marked fit,
since the customer has it. Loading is idempotent: a URL already holding a deal is skipped,
so the flag rides every call of an `--agent-qualify` loop, and new marks on a store holding
verdicts of its own are refused (`bad_config`) — its walk was scored without them. The marks
are a file and never a hub call: the finder is open source and the orders are not.

`reason` lands as a custom variable and is the reason the product exists. `full_name` is the one
name discovery reports, verbatim: `first_name`/`last_name` stay null until the paid lookup returns
the provider's own split, so without it a leads-only file names nobody. It is last because the
record only ever adds. Two columns are there for
us: `lead_id`, a stable join key that survives an address changing under us, and `qualified_at`
(`Deal.creation_date`, ISO 8601), so a file carries its own provenance — a caller tells which rows
its own call produced by comparing against the time it started, and a sequencer imports only what is
newer than last time. **A `new` flag would have been the obvious alternative and is wrong**:
invocation-relative state written into a file that outlives the invocation means two files disagree
about the same lead, and the column is a lie the second time anyone reads it. A timestamp is true
forever.

**The boundary is one-way — nothing comes back** (decided 2026-08-19 on the same card). Reply
outcomes are conversation states that depend on the message and the sender's skill, which is the
half being handed away; ingesting them would infer "was this a good lead" from "did that email
work". Suppression stays with the sender too: a finder that never contacts anyone is not the sender
under CAN-SPAM/GDPR, and the mainstream sequencers block a suppressed address at import
([Instantly](https://help.instantly.ai/en/articles/6192983-global-blocklist),
[Smartlead](https://helpcenter.smartlead.ai/en/articles/139-what-is-global-block-list-your-comprehensive-cold-outreach-guide)).
So there is **no inbound endpoint and no event vocabulary**, and the GP keeps training on the LLM's
fit verdict alone — `get_labeled_arrays` has never seen a market signal. `lead_id` keeps the door
open at no cost if that is ever revisited; the one candidate is `bounced`, which grades the row we
emitted rather than the message. **Note the operator-side duty this creates**: import dedupe is
opt-in on Smartlead and undocumented on Instantly, so a re-exported lead who never opted out can be
contacted twice unless the operator enables it — say so in every adapter's docs.

- **One record, two serialisations.** `JSON_FIELDS` is the record — the columns above plus
  `profile_text`, the raw firmographic string the qualifier judged on, `source_fields` (the
  lead's `Lead.source_fields`: title, state, country and the company's LinkedIn tags, kept apart
  under their Lead Finder names, as the walk reads them) and `employees_min`/`employees_max` (the
  employer's LinkedIn band, null when unreported or there is no company). Those three are there
  so a record carries the whole person a store holds: a lead the customer marked can be taken
  back in as a verdict on a fresh store, without the store it was found in. `RECORD_FIELDS` is the
  **importer-shaped projection** of it, and `write_csv` projects (`extrasaction="ignore"`) rather
  than defining a second schema, so a field added for a sender never becomes a column an importer
  has to map. `None` writes as an empty cell, which is what an importer expects for a field we
  were never told; `profile_text` is `""` and `source_fields` is `{}`, never absent, so a
  receiver keying on them need not tell *no text* from *no such key*.

  ```
  outfind find 50 --json | outsend       # the full record, profile text included
  outfind find 50 > leads.csv            # the importer-shaped projection of it
  ```

  `--json` is **JSON Lines** (`write_json_lines`): one record per line and nothing else on stdout,
  with the run's metadata (`campaign`, `goal`, `produced`, `reached`, `stopped_because`,
  `next_action`, `rows`) as one JSON object on **stderr** — so one rule serves both formats,
  *stdout is records, stderr is narration*. Line-delimited because **a truncated stream stays
  usable**: one big object that stops halfway is a parse error and the whole batch is lost, where
  a line-delimited one has already delivered every complete record before the break, and the rest
  is a re-run. Under `--json` stderr carries JSON and nothing else — no banner, no log lines (the
  run object, plus the `{"error": …}` object after it if the run fell short), or every caller ends
  up writing the same fragile `tail -1` over a stream of prose. **An explicit `--debug` or
  `--log-level` keeps the log lines** beside the objects (the banner stays off), each object still
  on a line of its own: an agent judging under `--agent-qualify` needs both the candidate's payload
  and the walk's reasoning, the belief's refit and the judge-or-search call.

  **Output is progressive by default, in either format.** `find` writes what the campaign already
  has immediately, before touching the job (`handle` reads `lead_records(campaign)` and calls
  `IncrementalWriter.write` on each, right after `_announce_the_run` and before `run_job`), then
  writes each new lead's record the moment its deal settles — `_streamer` rides the same
  `on_new_lead` hook `--open` uses (`_combine` in `find.py` chains the two, so neither has to know
  the other exists):

  ```
  outfind find 20 emails | outsend        # feeds the sender within seconds of the first
                                            # resolved address, not after the whole run
  ```

  This is safe in a way a live-updated *file* was not: the daemon-managed CSV described above was
  deleted because rows kept changing state under a process that ran for days, which is the naming/
  collision/atomic-rewrite bug class that killed it. A single `find` run's deals settle once and
  stay settled for the rest of that run, so writing a record the moment it terminates has nothing
  left to contradict — and the total written is still the whole campaign, exactly what `> leads.csv`
  always promised, just delivered progressively rather than atomically. `--new` skips the opening
  bulk and only streams what this run itself produces. `--batch` is the escape hatch back to the
  old shape: `writer` stays `None`, nothing is written until `_report` runs after the job, and it
  materialises the whole thing (or just `--new`'s narrowed rows) in one call, exactly as `find` did
  before this existed — for a consumer that genuinely cannot take a partial stream. `_report` reads
  `writer.count` for the closing `rows` figure when streaming, since that is what actually went out
  (opening bulk plus whatever this run produced, `--new`-narrowed already if that flag was set),
  rather than re-querying the campaign a second time.
- **`reason` is operator-facing.** It is the `QualificationDecision`'s justification for a yes/no —
  third-person, evaluative, about the act of selecting someone. It is evidence for the person
  running this, **never text for the person receiving the mail**; a sender writes the message from
  `profile_text` instead.
- **The compatibility rule is the substitute for the shared record-schema package we are not
  building**: a receiver ignores keys it does not know, and this side never renames a key or
  repurposes one — it only ever adds. The docs are the contract, so a third party reading JSON
  Lines needs no package from us; two repos and one record cannot stay in step on good intentions.
- **There is no score column, and the export is a pure database read.** An earlier version
  exported the GP's `P(f>0.5)`; it was removed as a category error. `core/pipeline/ready_pool.py`
  defines `min_gp_confidence` as "the paid-lookup spend gate **and nothing else**" — the GP decides
  whether to spend a credit resolving an address, not whether a lead fits. The fit verdict is the
  LLM's and it is already in the file as `reason`, in language a person reads; and since every
  exported lead has a Deal, it has already passed the qualifier, so the number separated nothing.
  It was also expensive and unsafe: scoring meant `qualifier_for`, an O(n³) fit over every label
  (**minutes** on the live install's 2,538-deal campaign, against a docstring assuming "tens to low
  hundreds"), which also calls `ensure_anchors` — so a cold campaign would have made **LLM calls and
  mutated campaign state from a read-only export**. `lead_records` now streams one indexed query
  straight to the writer.
- **The Deal is the unit, not the Lead** — the `reason` is a verdict on this person *against this
  install's ICP*, which is what the export carries. One install, one implicit campaign, so it is
  one deal per lead; a second ICP is a second store, and its verdicts never share a row with these.
- **A Deal is not an endorsement**, and this is the trap the live install exposed. There are *two*
  rejections and they live in different columns: `DealState.FAILED` (+ `wrong_fit`) is the LLM's
  own campaign-scoped rejection, and `Lead.disqualified` is the permanent account-level exclusion
  (an opt-out). The first shipped version filtered only on `disqualified`, so it exported **1,944
  rows** from a campaign where most deals were rejections — rows whose `reason` read *"does not
  align well with the target market"*. Both are now excluded, always; there is no flag to include
  them.
- **No command, no file, and no options — it is the stdout of `find`.** This one took two attempts on
  one day and the history is the argument. `export_leads --campaign N > leads.csv` was a command
  nobody discovered, so it became a CSV the daemon wrote and kept current under `<data dir>/leads/`;
  that needed a naming rule, a collision tie-break, rewrite-vs-append, and an atomic
  temp-file-and-rename. **Every one of those problems existed only because a background process can
  change a row after it was written.** Bounding the run to a goal deleted all of them at once: the
  rows are final when the job ends, so they are simply printed, and the shell does files. What is
  printed is the **whole campaign**, which is what makes `> leads.csv` correct by construction —
  the newest file supersedes every earlier one, and a lead whose address resolved since last time
  comes back with it filled in. It is one file to overwrite, not a batch per run. `find 0` prints
  without working, which covers *give me that again* at zero cost. *(Also deleted with the command:
  the format switch, output path, state filter and rejected-leads escape hatch it never grew.)*

## CRM Data Model

**There is no config model.** What a human answered is read from `OPENOUTFIND_*` on every run
(`core/config.py`), and what the pipeline produced is in the tables below. There is no `Campaign`
model and no campaign identity either: this install has never run more than one, so a separate row
bought nothing but a selection step nobody used, and a second ICP is a second store (`--db`).

- **Search** (`core/models.py`) — one search the walk has written or fired: `terms` (`terms.Terms` as JSON — title words, one tag phrase, one place, one seniority), `key` (sha256 of the canonical terms, unique, so a search two moves reach is one row), `origin` (`written` — the initial population — or `drawn`, composed by the frontier), `move` (`drawn`; a store walked on the old lattice keeps `stricter`, `wider`, `swap` or `exchange`), `leads_found` (the provider's count, read from page 1, the only offset whose count is an answer), `dead` (page 1 empty with a count of 0 — and every search holding all its terms is dead with it), and **the shape** every search rides: `headcount_min`/`headcount_max` (the customer's size band, never a search axis, since loosening it queries off-ICP) and `country_code` (what a lead its pages bring in is tagged with — Lead Finder rows carry no ISO code; **not** the operator's own country, which is jurisdiction and lives in the environment). The initial population writes the shape and every later search copies it. `query` returns the `Terms`, `to_filters()` the provider JSON.
- **Page** (`core/models.py`) — one fired page of a search, the walk's arm once fired: `search`, `number` (1-based, at `PAGE_SIZE` = 200), `round`, `rows` (what the provider returned, before any were dropped). `at_bottom` — empty, past the count, or page `LAST_PAGE` (50, where the 10,000-row reach ends). Unique on `(search, number)`. **No value column**: an arm's value is recomputed every round from the store's people on the GP as it stands (`search_arms.Frontier`), so a stored number would only be a stale one.
- **PageLead** (`core/models.py`) — one person a page returned, for every row the page brought back whether new to the store or not (no-headline and capped-company rows, marked leads and anchors excluded). `Page.leads` and `Lead.pages` read it. It replaces `Lead.discovered_by`, which kept only the first search to find a person.
- **Company** (`crm/models/company.py`) — the employer, stored once and shared by every `Lead` at that firm. Identity is `key` (unique): the lowercased `domain`, or `name:<lowercased name>` when the provider reported no domain — a single computed column, because no constraint can express "the domain when there is one, the name otherwise". `name`/`domain` are nullable; `employees_min`/`employees_max` hold LinkedIn's employee band (`discovery.COMPANY_SIZE_FIELDS`, from `company_employees_range_start`/`_end`), for the verdict to read — a later row that reports a band replaces it, one that reports none leaves it. `from_row(name, domain, employees_min, employees_max)` get-or-creates and returns `None` when the row named no company — **or named a placeholder** (`is_placeholder`: "Self employed", "Stealth startup", "None at this time", …, listed in `PLACEHOLDER_NAMES`). The provider fuzzy-matches those to a real domain like any other name, and one live store had 340 self-employed leads sharing `selfemployedhq.com` and 114 stealth founders sharing `harmonic.ai`. `export.lead_record` drops the same placeholders from stores written before discovery stopped creating them. **What the provider said, not verified truth**: Lead Finder fuzzy-matches this record (a boutique law firm's founder comes back as Meta — see `discovery.TEXT_FIELDS`), so anything treating a Company as an *account* inherits that error. Known limitation of the simple key: a firm seen once with a domain and once without produces two rows (`acme.com` and `name:acme`); reconciling them is a later pass, not merge logic at write time.
- **Lead** (`crm/models/lead.py`) — Keyed on `profile_url` (unique, nullable — the discovery provider's per-person URL, the opaque identity/lookup key, **stored, never fetched**; `NULL` where there is no such profile, which SQL lets a unique column hold any number of). `synthetic` marks the **anchors**: invented ideal leads the LLM wrote from the ICP (`icp.generate_anchors`), carrying the same `profile_text`, `source_fields` and embedding a real lead does, which is why they are rows here rather than three parallel arrays on a config singleton — the GP reaches them with the same queries it reaches real leads with. Permanent once written, and **never contacted**: `qualify.fetch_qualification_candidates` is the one scan that picks a Lead up without a deal and it excludes them, so nothing downstream — enrichment, the export, a message — can reach one. `country_code` (stamped from the discovery ICP; drives the contacts-store geo-gate; blank → never contributed). `embedding` (384-dim float32 BinaryField, built at discovery). `profile_text` (the firmographic text — headline/location/industry/title/company/company-description, plus seniority, company-industry, location state+country, and company-keywords folded in *when the row carries them* — built from the Lead Finder row at discovery, the LLM qualifier's input; no re-scrape). `email` (the finder result; null = not found/unresolved — populated by the two-leg buy/check lookup or a free hub-cache hit, never on the model itself). `disqualified` (the permanent, account-level exclusion the export filters — nothing sets it automatically now that the inbound opt-out path is gone). **Identity fields** — `full_name`, `first_name`, `last_name`, `job_title` (all nullable, `NULL` = the provider never told us) and `company` (FK). They exist for the **lead export** and for the record the product keeps, and are deliberately kept out of `profile_text` and the embedding: a name carries no ICP signal and would only give the GP noise to learn on. **The two name sources are distinct and neither is a guess** — discovery reports one `contact_full_name`; the paid enrichment response reports the real `contact_first_name`/`contact_last_name`, which `enrichment/lookup.py` writes on a hit. A lead resolved from the free hub cache never reaches that provider, so its name parts stay `NULL` rather than being split in-house, because they feed a sequencer's `{{first_name}}` merge tag where a wrong guess lands in someone's cold email. `to_profile_dict()` → `{lead_id, profile_url}`; `embedding_array` for numpy; `get_labeled_arrays()` → (X, y) for GP warm start (non-FAILED → 1, FAILED+wrong_fit → 0, other FAILED → skipped). `source_fields` keeps the row's searchable fields apart (`discovery.SOURCE_FIELDS`, `contact_seniority` among them), which is what a search's terms are matched against. Created browserless via `core/db/leads.store_rows(rows, country_code)`, a page at a time with one embedding call — there are no scrape accessors.
- **Deal** (`crm/models/deal.py`) — one per lead (`unique(lead)` — with one implicit campaign there is one verdict per person). `state` (`DealState`), `outcome` (`Outcome` — now only `wrong_fit`/`unknown`), `reason` (**the product**: why the LLM chose or rejected this lead, in its own words, and the only fit signal that leaves in the export). `not_before` (**the only schedule a deal carries** — "do not touch this row before this time", written by the lookup backoff, null = always eligible), `lookup_request_id`/`lookup_attempt` (the in-flight paid job and its backoff exponent), `creation_date`, `update_date`.

  *Dropped with the sending leg:* `mailbox` and `thread` (FKs into the `emails` app), `email_subject`, `email_sent_at`, and `chat_summary` — every one of them a fact about a conversation.

**The mail-log models — `Thread`, `Message`, `DeliveryEvent`, `FolderCoverage` — and `Mailbox` are
gone** with the `emails` app. See *The Mail Log* above for what they did and where they went.

## Key Modules

Paths relative to `openoutfind/`.

- **`core/job.py`** — the bounded run: `Goal` (count + unit, a **delta** not a total), `JobResult`, `run_job(campaign, goal, on_new_lead)`, `_unit_ids` (progress as a set, per unit). No timeout, by decision — see **The Cycle**.
- **`core/cycle.py`** — the hierarchy, and no loop at all: `run_one_action` (rows 1–4, first match wins), `_apply` (one save per transition), `pipeline_summary` (also the reason a job stopped short), `HALTING_ERRORS`. *(Gone with the daemon: `run_daemon`, `_rotate`, `CYCLE_SECONDS`, `IDLE_LOG_INTERVAL_S`. Gone with the sending leg: `unanswered_replies` — the follow-up trigger — `room_to_send_today` — the spend gate — and `read_mail_if_due`/`refresh_capacities_if_due`, the two periodic side-effects.)*
- **`core/operator.py`** — who is running this install: `get_active_user()`, `self_profile()`, `seller_name()`/`seller_full_name()`. Nothing is cached across calls — the read is one indexed row, and a cache would only let a renamed operator keep signing with the old name until restart. Replaces the browser era's `OperatorSession`, which by the end held nothing session-like: just the Django `User` and whichever campaign the handler was on.
- **`discovery.py`** — Lead Finder client and the provider contract. `search(api_key, filters, limit, offset, first_poll_s)` → `Fetched(Page(leads, leads_found), ready_seconds)`: the rows plus the corpus count from `summary.leads_found`, surfaced **only at offset 0** (past the end of *any* result set the API reports 0), and how long the search took to be ready. Thread-safe and quiet — the walk calls it from worker threads, so it touches no database and logs nothing. `SEARCH_FIELDS` is the four fields a search holds terms in — `lead_job_title`, `company_keywords` (the employer's LinkedIn tags, only ever fired with a title, because alone it pages one company's whole staff at a time), `lead_location` and `lead_seniority`; what each holds is `core/pipeline/terms.py`'s, and how its text becomes words `core/pipeline/analyzers.py`'s. `company_industry` is absent because it adds little precision over `company_keywords` and loses most of the recall, `company_description` is `company_keywords` under a second name, `company_name` is ignored, `lead_industry` is absent because it is **inert** (a nonsense value returns the identical count to no filter), `lead_function` because it and `lead_department` are one field under two names whose values are ORed (naming both *widens* the query), and `lead_department` because no lead row carries a department, so no vocabulary could ever grow for it. **How each field matches** (measured; the probe table is the comment above `SEARCH_FIELDS`): no field takes Elasticsearch's query-string syntax, so `AND`, `OR`, `+`, `*` and quotes are never operators. `lead_job_title` is a `match` with every word required in any order (`production manager` = `manager production` = `production OR manager`); `company_keywords` is a `match_phrase` inside one tag — adjacent, in order, case-insensitive (`cannabis vapes` 76, `vapes cannabis` 0); `lead_location` is a parsed `state, country` with the country required; `lead_seniority` and `company_industry` take a whole value from a closed list (industry case-sensitive). **Nothing is stemmed** (`manager` 1,140,870, `managers` 231; tag `vape` 2,282, `vapes` 1,505), so every word form is its own term, and the two free-text fields analyse differently: a title drops stopwords (`head of procurement` = `procurement head`; a stopword-only title matches nobody) and treats `co-founder`, `co founder` and `cofounder` as one, while a tag keeps stopwords as words of its phrase (`food and beverage` 187,794, `food beverage` 162,754; `and` alone matches 7.4M) and splits a hyphen without joining it (`e-commerce` = `e commerce`, `ecommerce` apart). Two include-list entries OR on every field, and so do `company_keywords` + `company_description` together, so **no query ANDs two company tags** — one tag word, or one phrase a company's tag list actually holds, is all the provider can be asked for. **What a search costs**: no credits, and a few seconds — a search is ready 1.1–3.6s after the submit, and a count-only `limit=1` call costs what a full page costs, so a query worth sizing is worth paging (`limit` goes up to 200, 250 is refused; `max_leads` reports `leads_found` as the rows returned, so pagination is `limit`/`offset` only). The rate limit counts every request, polls included, per API key across all endpoints: 100/min on Starter and free accounts, 200/min from 1,000 monthly credits, `GET /account` exempt. `company_headcount_min/max` is converted server-side into the provider's overlapping size ranges (`1-10`, `11-50`, …), not applied as exact bounds, and only `lead_job_title`'s `exclude` is applied by the provider — every other `exclude` is a filter BetterContact runs after the search (doc.bettercontact.rocks, *API Rate Limits*, *Request statuses*, *New Lead Finder Search*). `SOURCE_FIELDS` maps each search field to the row fields it is matched against, and `source_fields_for(row)` stores exactly those on the Lead (a tag list joined with `", "`). `profile_text_for(row)` builds the qualifier's text from `TEXT_FIELDS`, and `embed_profile` embeds it alone. A field earns its `TEXT_FIELDS` slot by **varying between leads**: the GP ranks the pool's candidates against each other, so a field constant across them adds nothing however accurate. That test excludes the `company_*` free text — Lead Finder staples a fuzzy-matched company record onto every row (a law firm's founder comes back as Meta, mission statement and all; 1–4 distinct records per 100-row page), so `company_description` (59% of the old text) and `company_keywords` (21%) were 80% of every vector at ~zero bits; `contact_location` is absent from every response. **Changing `TEXT_FIELDS` moves the vector space — every `Lead` must be re-embedded.** The raw rows are not persisted locally, but every page goes to the hub with its fields kept apart (`SHARED_FIELDS`, `contacts.service.share_profiles`), so the hub's profile lake can rebuild `profile_text` under a new `TEXT_FIELDS` without paging Lead Finder again. Its transport is `enrichment/bettercontact.py`'s `search_and_poll`.
- **`core/pipeline/`** — `icp.py` (the two cold-start priors, same inputs, two shapes — under `find --agent-qualify` both are written by the calling agent instead: `agent_population_answer` returns the `--population` answer (`PopulationAnswer`: the `InitialPopulation` — 4 `WrittenSearch`es, the market `country`/`country_code` and the size band — plus `ANCHOR_COUNT` anchor profiles) or raises `population_pending` with the texts, the brief, `FIELD_TYPES` and the answer's schema — `write_initial_population`: one LLM pass (`initial_population.j2`) → the walk's first round as `Search` rows, `as_terms` placing each written state inside the market; it is the *only* LLM call the walk makes; `generate_anchors`/`ensure_anchors`: the ICP as synthetic ideal *profiles*, embedded as the GP's positives so a campaign whose every verdict is a rejection can fit at all, kept permanently once written), `analyzers.py` (the provider's analyzers: `words` — ICU's Unicode word split with the measured rules on top; `title_words`/`query_title_words`/`standalone_title_words` — a profile's title, a title search, and the title words a composed search may hold; `phrase_words`/`company_tags`/`holds_phrase`), `terms.py` (`Terms` — a search's terms per field, `is_legal`, `key`, `holds_all_of` (the dead-search conviction), `matches` (every term in its own field, through `analyzers.py`), `to_filters`; `split_place`/`place_of`), `compose.py` (`Vocabulary` — every keyword a search may hold, per field; `compose` — the `limit` searches worth the most on one draw's lifts and shares, the fields added one at a time keeping the `PER_BIN` best per size bin, each holding a title and a tag; `as_terms`), `store_snapshot.py` (the store as one round reads it: `fit_rate_map` — the GP's score → fit rate, isotonic on the store's verdicts at their out-of-fold scores, cached under the label fingerprint; `lead_fit_rates` — each stored person's verdict or fit rate, and the GP's confidence there; `judged_leads`, `judged_fits`; `People` — a per-field inverted index over the store, `match` equal to `Terms.matches` over everyone, `holding` the people of one keyword; `FiredSearch` — a fired search with its pages and whether it is drained; `fired_searches`), `search_arms.py` (`Arm` — a search at a page with its `count`, `seen`, `held_fits`, `new`, `rate`, `cost` and the `score` (fits on the draw) and `rank` (among the `k`) it was picked on; `Frontier` — `arms` (every fired search's next page, every unfired written search, the searches composed at the mean, none fired, dead-convicted or holding all of a drained search's terms; `vocabulary`, the keywords they are composed from) and `pick` (Thompson sampling over `top_at_mean`, the `TOP_K` arms with the most fits on their page at the mean: the one with the most fits on one draw of the belief seeded by the round, a next page keeping the provider's count); `next_round(qualifier, round_number)`, `best_expected(qualifier)` for `judge_or_search`, `written_arms()` for round 0), `belief.py` (`Beliefs` — the posterior of the sizes and fit rate, read at the mean (`count`, `shares`, `lifts`, `theta_at_mean`, `mean_sizes`) or on one seeded `Draw` of `α`, each keyword's lift, each fired search's own `θ` and the sizes (`draw`, `draw_sizes`, `theta_on`); `evidence` — what people add to a keyword; `joint_model` and `sample_posterior` — the PyMC model and its nutpie draws; `fitted` — refit only on a new page or verdict; `query_keywords`), `judge_or_search.py` (`decide` → whether to judge, dollars per fit for the best lead waiting against the best arm), `walk.py` (`run_round(site_config, qualifier)` — round 0 fires the written population whole, later rounds `search_arms.next_round`; `_fire_all` in a thread pool, `_take_in` each page on the main thread, record the round's time, new leads and ingest time per lead), `discover.py` (`discover(site_config, qualifier)`: the two gates, then one round; `harvest(pages, country_code)`, a round's `(page, rows)` pairs at once — drops any row failing `discovery.is_live_profile` (no headline, which in practice means a forgotten profile, ~40% of what Lead Finder returns) or from a company at its cap, stores the rest in one `store_rows` call (one embedding pass), writes each page's page–lead rows, and gives each page back to the hub), `qualify.py` (`run_qualification` / `fetch_qualification_candidates` — reads `Lead.profile_text`, no scrape; under `find --agent-qualify` — read via the `core/agent_qualify.py` contextvar rather than a parameter, since none of `job`/`cycle`/`top_up` needs to know — it skips `qualify_with_llm` and instead resumes the one `crm.PendingQualification` row if one exists, or creates it and raises `QualifyPending` naming the fresh candidate), `brief.py` (the store's judging rules: `stored`, `adopt` — `--brief`'s replace, `ensure_rules` — written by `AI_MODEL` on first need, `""` with no ICP text or under `--agent-qualify`, whose `qualify_pending` and `population_pending` payloads carry the rules instead, `log_brief_echo` — printed beside the ICP echo at the start of every run), `labels.py` (`find --labels`: `read_labels_file`, `verdict_for`/`reason_for` — one mark as a verdict, `load` — the marks as marked leads with their deals, all or none, skipping any already in the store), `ready_pool.py` (GP gate: `promote_to_ready`, `find_ready_candidate`; the bar it reads is the spend gate **and nothing else**), `probe.py` (`next_lead` — one person drawn at random from the latest fired search whose people hold fewer than `PROBE_VERDICTS` verdicts, seeded by the search and its count so a stopped run draws the same one), `top_up.py` (`top_up` — **one** acquisition move per call: cold phase, then probe, then calibrate, then explore/exploit; `_consumable_candidates` is the exploit gate — see its module docstring. The `while True` that used to wrap it is gone: the cycle is the loop).
- **`core/ml/`** — `qualifier.py` (`Qualifier` protocol, `BayesianQualifier`, `build_qualifier`, `qualify_with_llm`, `format_prediction`), `calibration.py` (the calibrate arm: `should_calibrate`, `next_lead`, `needs_measuring`/`measure`, `current_threshold`; models `CalibrationDraw`/`CalibrationBar` in `core/models.py`), `embeddings.py` (`embed_text`/`embed_texts`, cached FastEmbed model). *(`KitQualifier` and `hub.py` — the HuggingFace campaign kit — went with the promo campaign, which had no labels of its own to fit on.)*
- **`core/db/leads.py`** — `create_lead(row, country_code)` (persist one Lead Finder row as an embedded Lead, idempotent), `promote_lead_to_deal`, `disqualify_lead`. *(`suppress_email` left with the sending leg — see* Opt-out and suppression.*)*
- **`core/db/deals.py`** — Deal state ops: `set_profile_state`, the state-pool queries (`get_qualified_profiles`, `get_ready_to_find_email_profiles`), `create_disqualified_deal`. `_STATE_LOG_STYLE` colors the funnel transitions in the log.
*(`core/db/summaries.py` and the vendored mem0 prompt under `core/vendor/` are gone. They built a `Deal.profile_summary` — an LLM fact-extraction over the lead's `profile_text` — for the outreach agent, and had **no consumer left in the finder** once the sending leg went: the qualifier reads `profile_text` directly. **Summarising for a message is the sender's job**, so `profile_text` crosses the boundary raw and the receiver extracts what an opener needs — tuned for the reader who wants it, paid for only for people actually written to, and derived once rather than twice with no rule for which wins.)*
- **`core/llm.py`** — `get_llm_model()` factory (reads the run's `SiteConfig`, `split_model_id` parses the provider out of `ai_model`, dispatches to the per-provider builder), `build_llm_model` (from explicit creds), `verify_llm_credentials` (one live ping, tenacity-retried, run before every pass by `readiness.check_ready` — it returns only the failures a *configuration* causes and lets anything else propagate, since its caller blames `LLM_API_KEY` for whatever comes back), and `run_agent_sync(coro)` — the sync boundary that drives async pydantic-ai on a dedicated long-lived worker-thread loop (never `Agent.run_sync`, whose anyio portal poisons the caller thread's loop slot; never per-call `asyncio.run`, which closes loops the SDK HTTP clients still reference).
- **`core/geo.py`** — jurisdiction sets + predicates: `is_gdpr_protected` (broad opt-in set, read only by OpenOutreach's wizard for its newsletter default) and `is_eea_located` / `EEA_UK_CH` (narrow EEA/UK/CH collection-regime set — the client-side pre-gate for contacts-store contribution; the server re-gates authoritatively). The operator's country is `OPENOUTFIND_OPERATOR_COUNTRY`; a lead's is the one its query node searched. Neither ever comes from a scrape.
- **`enrichment/bettercontact.py`** — the provider client. The paid finder is the two-leg `submit(query) → request_id` + `poll_once(request_id) → PollOutcome`, so a run never blocks on a poll; the free Lead Finder index uses the blocking `search_and_poll` transport from the same module, since `discovery.search` genuinely wants a page back: its first poll goes at the measured ready time, the rest a second apart (enrichment keeps `_POLL_INTERVAL_S`, 5s). Every request but `GET /account` first takes a place in `_BUDGET` (`RequestBudget`, `REQUESTS_PER_MINUTE` = 15 in any 60 seconds, thread-safe), one process's share of the key's 100/min. `is_configured()` reads `OPENOUTFIND_BETTERCONTACT_API_KEY` — one key, two endpoints, and only one of them bills.
- **`enrichment/fullenrich.py`** — the second finder, and a different shape of one: a bare `professional_network_url` is a complete query, so `submit(url) → enrichment_id` + `poll_once(enrichment_id) → PollOutcome` mirror the first finder's two legs without the name/company fields. The vendor's job statuses are mapped through (`CREDITS_INSUFFICIENT` → out of credits, `RATE_LIMIT` → rate limited, `CANCELED` → a miss, anything unfinished → still running), only `DELIVERABLE` / `HIGH_PROBABILITY` count as a usable address, and `credit_balance()` raises instead of guessing because the vendor exposes no balance read at all (so a FullEnrich install renders its balance as *unknown*). It carries its own `_BUDGET` with no exempt account read, since there is no account read to exempt.
- **`enrichment/leadmagic.py`** — the third finder and the only synchronous one: `start(profile_url)` is the whole lookup, a single POST to `/v1/people/email-finder` carrying nothing but the URL, so the answer comes back already finished (`valid` and an address, or a free miss) and there is no `poll_once` here at all, no handle for a vendor to strand. It keeps its own `RequestBudget` (60 a minute, a courtesy fifth of the 300 the entry plan publishes per organisation) with the free `GET /v1/credits` exempt, and unlike FullEnrich it reads a real balance, floor-don't-round, since `status` shows it.
- **`enrichment/enrow.py`** — the fourth finder added, and the cheapest: `start(profile_url)` POSTs a single `linkedin_url` to `/email/find/single` and returns the vendor's handle (`id`), `poll_once(id)` reads it back as a `PollOutcome` while the search runs at HTTP 202, and `valid`/`invalid` are the only two verdicts that settle a lookup (`valid` carries the address). A `lookup_failed` LinkedIn status is transient and free, so it defers rather than recording a miss, and an unknown or expired handle comes back as a 500 rather than a 404, which also defers. It keeps its own `RequestBudget` (120 a minute, a courtesy fifth of the documented 10 a second) with the free `GET /account/info` exempt, and unlike FullEnrich it reads a real balance, floor-don't-round, since `status` shows it.
- **`enrichment/provider.py`** — the selection seam and no HTTP of its own: `configured()` (the finders holding a key), `active()` (the cheapest of them by `UNIT_COST_USD`), `chain()` (that one first, then the rest cheapest-first), and `by_name()` (dispatch for a parked lookup, from `Deal.lookup_provider`). Also home to `ProviderUnavailable` (the couldn't-run base every adapter subclasses), `PollOutcome`, and `Lookup`.
- **`enrichment/lookup.py`** — the two pipeline steps, `buy_address` / `check_lookup` / `reclaim_lookup`, plus `_store_identity` (the name parts the provider echoes back with the address) and the backoff helpers. The enrichment query is **URL-only by decision** — the provider accepts name and company and resolves better with them, but the less of a lead's record leaves for a third party the better, and URL-only measures ~42% usable. The docstring says not to widen it without a decision to widen it.
- **`core/business_time.py`** — `business_days_between(start, end)`: whole Mon–Fri days elapsed. It was the agent's only sense of a thread's age; with no agent it is now unused by the pipeline and kept as a small, correct utility. Public holidays are not modelled (per-country data we don't carry).
- **`core/logging.py`** — `configure_logging` + `print_banner`; `SILENCED_LOGGERS` quiets urllib3/httpx/pydantic_ai/openai/fastembed/etc.; `format_elapsed` (`52s` / `4m09s` / `1h04m`) for the milestones.
- **`contacts/service.py`** — the hub client: `resolve(lead)` (free read before the paid finder; `/resolve` returns an `emails[]` list plus `credits` on both the hit and the miss, first email taken — a miss logs *no balance* when `credits <= 0` and *no stored email* otherwise, since a permanent zero must not read like an ordinary miss), `contribute(lead, emails, origin)` (give-back at a fresh paid hit, non-EEA only, registers + mints the token on first use; **no vector** — the person's vector reaches the hub with their discovery page, and `Contribution.embedding` stays nullable only for older clients), `share_profiles(rows, country_code)` (called by `discover.harvest` on each **whole** page after the round's `store_rows` — headline-less, capped and already-known rows included, since a new fetch of a known person is still news. One `POST /api/v2/profiles/` per page: each row's `discovery.SHARED_FIELDS` kept apart — `TEXT_FIELDS`, `PERSON_FIELDS`, `COMPANY_FIELDS`, never `company_description`/`company_keywords` — plus the searched country and the **profile-only** vector (a stored person's `Lead.embedding` as it is, since that is `profile_text` alone, and anything else embedded once a round, `service._vectors_for`, so nothing is embedded twice) tagged with `embedding_model`. Gated like `contribute`: an EEA/UK/CH operator, or a page searched in the EEA/UK/CH or in no named country, sends nothing. Best-effort through `_send` like every other hub call — a down hub is logged and the walk carries on; no credit is earned), `share_companies(rows, country_code)` (called right after it on the same page: one `POST /api/v2/companies/` carrying one record per distinct `company_linkedin_url` — `discovery.company_records`, the `COMPANY_RECORD_FIELDS` with description, tags and LinkedIn size band, a placeholder employer skipped — each with the vector of `company_text_for` and its model; no person rides along. The page's gates are one helper, `_page_token`, so the two calls cannot drift apart. The hub builds a private company index from them), `hub_balance()` (the give-to-get counter for `status`, read via a record-less `register` call — idempotent server-side, so checking spends nothing; `known: False` on no token or an outage, never a balance of zero). The token comes from `OPENOUTFIND_CONTACTS_API_TOKEN`, or from `token_in_hand()` registering once and holding it for the life of the process — nothing is written down, and `hub_balance` passes `mint=False` so reporting a balance never registers an install as a side effect.

## Configuration

- **`core/config.py`** — the whole configuration surface: a frozen `SiteConfig` dataclass read from `OPENOUTFIND_*` on every run, with `variable_for()` naming where each field comes from and `missing()` reporting what a run was not given. Nothing writes it. See *Configuration and the readiness check*.
- **`conf.py` lookup backoff** — `COLLECT_BACKOFF_BASE_S` (5), `COLLECT_BACKOFF_MAX_S` (30 days): the poll doubles its delay on every still-running attempt and **never gives up** — MAX rails the *interval* so `datetime` can still express it, and is not a deadline. **There is no spend cap.** Paid spend used to be gated by mailbox send-headroom (never resolve an address there is no room to email today); with no sending leg that gate is gone and nothing replaced it, because what bounds the spend is the operator's own prepaid credit balance, which the provider enforces and we cannot see. `COLLECT_TODAY_HORIZON_S` went with the gate it served.
- **`conf.py` — the three send guards are gone.** `WARM_*` (the measured per-box daily ceiling), `MIN_SEND_INTERVAL_SECONDS`/`SEND_INTERVAL_JITTER_*` (the 3.5–4.5 minute gap between first emails) and `SEND_WINDOW_*` (Mon–Fri 08:00–20:00, operator-local) were most of this file. They moved with the code they governed; the reasoning is preserved in `openoutsend/core/conf.py` on the OpenOutSend side, because it is worth not re-deriving: receivers punish *rate* and *volume* separately and a recipient reads the *hour*, so no one guard covers the others.
- **`conf.py:CAMPAIGN_CONFIG`** — `min_gp_confidence` (the GP rank gate's fallback until the calibrate arm has measured one — **only** a spend gate on the paid lookup; it is not a steering signal and never a quality score), `run_hour_dollars` (0.01786, a 1 vCPU / 2 GiB DigitalOcean droplet — a run peaks at ~0.8 GB), `round_seconds` (60, one round of the walk: 4 pages fired at once and taken in), `ingest_seconds_per_lead` (0.2, embedding, storing and giving back one new person — what a page of strangers costs on the frontier), `verdict_seconds_model` / `verdict_seconds_agent` (5 / 40 — these four times are starting values, each replaced by the store's measured running mean, `core/meter.py`, once it has one), `verdict_input_tokens` / `verdict_output_tokens` (2,800 / 300, measured on the held order; priced by `conf.MODEL_PRICES` — dollars per million tokens per model family, longest-prefix match) and `verdict_dollars_model` / `verdict_dollars_agent` (0.005 for an unpriced model / 0.05 for an agent's turn) — the costs the calibrated bar weighs, `calibration_min_band_verdicts` (5 per band, which also switches the arm on at N ≥ 1,600), `qualification_n_mc_samples` (100), `embedding_model` (`BAAI/bge-small-en-v1.5`). **There is no discovery cadence knob**: when to run a round is `judge_or_search`'s measured choice. `conf.py:WALK_CONFIG` holds the walk's constant `population` (4 written searches).
- **Prompt templates** (`core/templates/prompts/`) — `write_brief.j2` (the customer's two texts → the brief's judging rules), `initial_population.j2` (the cold-start ICP → the walk's 4 written searches, the market and the size band), `anchor_profiles.j2`, `qualify_lead.j2`; the last three render the brief as `rules`. *(`outreach_agent.j2` went with the agent; `mint_clauses.j2` with LLM clause minting.)*
- **`pyproject.toml`** — package metadata, dependencies, dev extras, and the `outfind` console
  script. Replaces the `requirements/*.txt` files, which are gone.

## Install

**The tool is CLI-first and ships on PyPI**: `uvx --from openoutfind outfind`, or `pip install openoutfind`. The
console script is `openoutfind/__main__.py:main`; `manage.py` is a shim over it for a checkout.
Everything the install writes hangs off one root — `settings.state_dir()`, the checkout when
`manage.py` sits beside the package and `~/.openoutfind` otherwise, because a wheel's `ROOT_DIR` is
site-packages and no operator's data belongs there. Under it: `data/db.sqlite3` and
`.cache/fastembed`. The model cache is deliberately **not** derived from the database path, so
`--db /tmp/scratch.sqlite3` does not send fastembed off to re-download 65 MB beside a throwaway DB.

**Measured on a cold machine** (2026-08-20, from a local wheel): `uv` resolves and installs in **30 s**
into a **550 MB** venv, and the first embedding fetches the ONNX weights in a further **11 s** / 65 MB
— about **40 s of one-time cost** before any lead work, which is small enough that no pre-warm verb is
needed. The venv's largest pieces are scipy (81 MB), onnxruntime (61 MB), pandas (45 MB) and Django
(41 MB); `botocore` + `fastavro` + `pillow` (~45 MB together) arrive only through the
`pydantic-ai-slim[…,cohere,bedrock]` extras, and are the one obvious lever if that number ever matters.

See the `openoutreach-docs` card `p1-e2-cli-entry-points`.

## Docker — the server deploy, not the install path

Two-stage build from `python:3.12-slim-bookworm`: stage one installs the package into a venv at
`/opt/venv` with `uv`, stage two copies that one directory and carries neither `git` nor `uv`.
No browser, no VNC. `compose/openoutfind/Dockerfile`. It exists for running jobs on a server —
**development and tests run natively**, there is no
`BUILD_ENV` and no dev extras in the image. `OPENOUTFIND_DB=/app/data/db.sqlite3` names the CRM path
explicitly, since the code no longer sits beside it; `local.yml` mounts `./data` there and nothing else.

## CI/CD

- `tests.yml` — native pytest on push / PRs (Python 3.12, `uv pip install -e ".[dev]"`).
- `deploy.yml` — **on every push to `main`**, and on `v*` tags. Runs the tests, then builds
  + pushes `ghcr.io/eracle/openoutfind`, then fires a `repository-dispatch` (`image-updated`) at
  `eracle/hub.openoutreach.app`. Image tags: `latest` (default branch only), `sha-<commit>`, and
  semver (`v*` tags only).

  **There is no release gate, on either artifact.** Merging to `main` republishes `:latest` *and*
  publishes a new version to PyPI — so code and **schema migrations reach anyone pulling `latest`, or
  installing the package, on merge**. `sha-<commit>` tags only the **pushed tip**: commits buried
  inside a multi-commit push never get their own image, so a migration can go from unpublished to
  live in one push.

- **Every green push to `main` is a PyPI release** (`publish-pypi`): sdist + wheel through trusted
  publishing (OIDC, environment `pypi` — no API token in the repo), with `skip-existing` so a re-run
  is a no-op rather than a red build. **`needs: test` is the entire gate**, which makes `main` the
  release branch in the literal sense.

  **The version is derived, never committed.** `pyproject.toml`'s `version` is the *base*: major and
  minor are declared there by hand, and the patch is `git rev-list --count v0.1.0..HEAD` at publish
  time — monotonic across a base bump (`0.1.20` → `0.2.21`), unique per commit, and requiring no
  bump commit. That is why the job checks out with `fetch-depth: 0`; a shallow clone cannot count.

  Deriving the version from the commit count rather than committing it means a release nobody has to
  remember cannot drift out of sync with what actually shipped. The trade is made knowingly:
  every commit that passes CI is public and permanent, and a bad merge reaches installers in
  minutes. **Tags do not publish** — they stay useful as human markers and the image job still
  reads them for semver tags, but a tag on an already-published commit would collide with the
  version that commit already shipped. The `pypi` environment must **not** carry a required
  reviewer, or every push waits on a click.

## Dependencies

`pyproject.toml`; `uv pip install` for fast installs. No browser/Playwright, no DjangoCRM.

Core: `Django`, `pydantic`, `pydantic-ai-slim` (with `openai`/`anthropic`/`google`/`groq`/`mistral`/`cohere`/`bedrock` extras; `griffe` pinned `<2`), `jinja2`, `pandas`, `termcolor`, `tenacity`, `questionary`, `tendo`, `pyyaml`, `jsonpath-ng`
ML: `scikit-learn`, `fastembed`, `huggingface_hub`, `numpy`/`joblib` (transitive)
