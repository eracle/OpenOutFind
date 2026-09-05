# Configuration

**Configuration is the environment.** Every value a person answers is read from `OPENOUTFIND_*` on
every run (`core/config.py`) and nothing is written down — there is no config row, no wizard and no
Admin page for it. An install runs exactly one ICP, so there is nothing to name and nothing to
select between. There are no social-network credentials either; OpenOutFind is browserless and uses
no such account.

The reason is what this program is: a library, a pipe stage and a scripted command, none of which
can answer a prompt, and an agent supplies its environment on every invocation and has nothing to
remember. An operator who wants to be *asked* runs the wizard in
[OpenOutreach](https://github.com/eracle/OpenOutreach), which owns the human half and exports these
names.

## The variables

Set these and run `outfind check` — or go straight to `outfind find 10`, which checks the same
things before it starts working.

| Variable | Group | Notes |
|:---------|:------|:------|
| `OPENOUTFIND_PRODUCT_DOCS` | campaign | what it does, who it's for, the problem it solves |
| `OPENOUTFIND_CAMPAIGN_TARGET` | campaign | who you're going after and the outcome you want |
| `OPENOUTFIND_AI_MODEL` | llm | `provider:model`, e.g. `anthropic:claude-sonnet-4-5-20250929`; bare `gpt-*`/`claude-*`/`gemini-*` are auto-prefixed. Providers: openai/anthropic/google/groq/mistral/cohere/openai_compatible |
| `OPENOUTFIND_LLM_API_KEY` | llm | **verified by one live ping before every run** — a key rotated out from under a timer fails before a lead is chosen, not mid-pass |
| `OPENOUTFIND_LLM_API_BASE` | llm | required for `openai_compatible:*`, ignored otherwise |
| `OPENOUTFIND_BETTERCONTACT_API_KEY` | bettercontact | [free account, 40 credits, no card](https://bettercontact.rocks?fpr=openoutreach) (affiliate link, no markup to you). Powers **both** Lead Finder discovery (billed nothing) **and** work-email enrichment (one credit per verified address, only with `--emails`) |
| `OPENOUTFIND_APOLLO_API_KEY` | bettercontact | optional second resolver; discovery still needs the key above |
| `OPENOUTFIND_EMAIL_FINDER` | bettercontact | optional — `bettercontact` \| `apollo`, only needed when both keys are set |
| `OPENOUTFIND_OPERATOR_EMAIL` | account | your own inbox — the contacts-store key and the newsletter target. Read **once**, to create the operator row; after that the row is the identity |
| `OPENOUTFIND_OPERATOR_COUNTRY` | account | ISO 3166 alpha-2, e.g. `US` — **your jurisdiction**, not your target market |
| `OPENOUTFIND_ACCEPT_LEGAL_NOTICE` | account | must be `true` — records that you accept the [Legal Notice](../LEGAL_NOTICE.md), and is asked on every run so an install cannot inherit somebody else's agreement with their database |
| `OPENOUTFIND_NEWSLETTER` | account | optional, **defaults off** — set `true` to subscribe, acted on once when the operator row is created |
| `OPENOUTFIND_CONTACTS_API_TOKEN` / `_URL` | hub | optional. Without a token a run registers for one and keeps it for the length of the process; `register` is idempotent, so nothing is lost by not storing it |

A missing value is never a prompt: the run stops with **one** error naming every variable that would
have satisfied it. `OPENOUTFIND_DB` (or `--db PATH`) points any command at a different SQLite file.

## What the database holds instead

Only what the pipeline produced or measured, which is the line: *who made this value*.

| Where | What |
|:------|:-----|
| `Keyword` / `QueryNode` | the walk — which keyword sets have been fired, how far each was paged, and the size band and target country each node searches (written by `icp.generate_seed` onto the nodes it opens, inherited by their children) |
| `Lead` with `synthetic=True` | the anchors — invented ideal leads the LLM wrote from the ICP, with the same `profile_text`, `source_fields` and embedding a real lead carries. Permanent once written, never contacted, never exported |
| `Lead` / `Company` / `Deal` | the leads themselves and the LLM's verdict on each |
| Django `User` | the operator — identity, written once, because a renamed variable must not rename the person a campaign belongs to |

The fitted GP is **not** in there. It is refit from the label rows whenever the evidence changes and
held in memory for the life of the process (`ml/qualifier.qualifier_for`), so a stored copy was a
cache of a value already derived.

## Sending mailboxes — there are none

The `Mailbox` model, the SMTP/IMAP credentials, the per-box signature, the measured daily cap and the
send-spacing clock all moved to [OpenOutSend](https://github.com/eracle/OpenOutSend) with
the sending leg. **Nothing here needs a mailbox**, and nothing asks for one.

## Newsletter consent

`OPENOUTFIND_NEWSLETTER` is off unless it says yes, in every jurisdiction: silence in a config file
is not consent anywhere, and there is nobody here to ask. (The wizard in OpenOutreach still offers
the jurisdiction-aware default, because that is a suggestion to a human. The rule it reads is
`core/geo.is_gdpr_protected`.)

Your `operator_country_code` is your own jurisdiction and nothing else — it decides whether this install
contributes to the contacts store (`geo.is_eea_located`). The country a *lead* is tagged with comes
from the query that found them, and lives on the query node.

## Hardcoded Defaults (`core/conf.py`)

Not user-configurable; edit the source to change.

| Key | Value | Description |
|:----|:------|:------------|
| `COLLECT_BACKOFF_BASE_S` / `COLLECT_BACKOFF_MAX_S` | `5` / `30d` | The lookup poll doubles its delay on every still-running attempt and **never gives up** — an unterminated job is queued, not lost, so the leg keeps the same `request_id` rather than abandoning the deal and paying for a second job. MAX rails the interval only, so the schedule stays representable. |
| `CAMPAIGN_CONFIG.min_gp_confidence` | `0.7` | GP probability threshold for promoting `QUALIFIED → READY_TO_FIND_EMAIL`. **A spend gate on the paid lookup and nothing else** — not a quality score, and deliberately absent from the export. |
| `CAMPAIGN_CONFIG.qualification_n_mc_samples` | `100` | Monte Carlo samples for BALD. |
| `CAMPAIGN_CONFIG.embedding_model` | `BAAI/bge-small-en-v1.5` | FastEmbed model for 384-dim embeddings. |

**There is no spend cap setting, because the command line is the cap.** A run cannot spend at all
unless you ask it to: `--emails` permits the paid lookup, and the `emails` unit implies it, so a bare
`find 10` is free however many deals have queued up past the confidence gate. When you do ask, the
number you type is the budget — one credit is one verified address, so `outfind find 10 emails`
cannot cost more than ten. (Beyond that, your
own prepaid balance at the provider, which the provider enforces and this software cannot see.)
Discovery and qualification are ungated entirely: searching is free and qualifying costs one call
against your own LLM key.

**There is no timeout setting either**, and there should not be one. A run ends when its goal is met or
when nothing can advance — and every wait that matters is already written on the row that is waiting
(`Deal.not_before`, the doubling lookup backoff, `urllib3`'s 429 retry). A clock over the top of those
would be a second answer to a question they already answer, and a worse one: it knows nothing about
*why* the run is waiting. If you want a deadline, `Ctrl-C` (or your agent's own timeout) prints the
rows found so far and exits non-zero.

*(Gone with the sending leg: `SEND_WINDOW_*`, `MIN_SEND_INTERVAL_SECONDS`, `SEND_INTERVAL_JITTER_*`,
`WARM_*`, `COLLECT_TODAY_HORIZON_S`, `MAIL_PASS_INTERVAL_S`.)*

## Working-day arithmetic

`core/business_time.py` only *measures*: whole Mon–Fri days between two dates
(`business_days_between`). It existed to tell the outreach agent how old a thread was; with no agent,
nothing in the pipeline calls it. Public holidays are not modelled — that data is per-country and
per-year.
