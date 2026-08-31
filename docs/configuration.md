# Configuration

Configuration lives in one place, the **`SiteConfig`** DB singleton (managed via interactive
onboarding or Django Admin), plus a few hardcoded defaults in **`core/conf.py`**. One row holds the
keys, the operator's country, and the product/target text: an install runs exactly one ICP, so
there is nothing to name and nothing to select between. There are no social-network credentials —
OpenOutFind is browserless and uses no such account.

## Configure without a terminal

Every onboarding field is also an environment variable, so an install with no TTY (an agent, a
container, CI) never needs the wizard. Set these and run `outfind init` — or go straight to
`outfind find 10`, which does the same setup before it starts working:

| Variable | Step | Notes |
|:---------|:-----|:------|
| `OPENOUTFIND_PRODUCT_DESCRIPTION` | campaign | what it does, who it's for, the problem it solves |
| `OPENOUTFIND_CAMPAIGN_TARGET` | campaign | who you're going after and the outcome you want |
| `OPENOUTFIND_AI_MODEL` | llm | `provider:model`, e.g. `anthropic:claude-sonnet-4-5-20250929` |
| `OPENOUTFIND_LLM_API_KEY` | llm | live-verified at boot; a bad key stops the run |
| `OPENOUTFIND_LLM_API_BASE` | llm | required for `openai_compatible:*`, ignored otherwise |
| `OPENOUTFIND_BETTERCONTACT_API_KEY` | bettercontact | powers discovery (free) and enrichment (paid) |
| `OPENOUTFIND_OPERATOR_EMAIL` | account | your own inbox — contacts key and newsletter target |
| `OPENOUTFIND_COUNTRY` | account | ISO 3166 alpha-2, e.g. `US` |
| `OPENOUTFIND_ACCEPT_LEGAL_NOTICE` | account | must be `true` — records that you accept the [Legal Notice](../LEGAL_NOTICE.md) |
| `OPENOUTFIND_NEWSLETTER` | account | optional, **defaults off** — set `true` to subscribe |

A step takes effect only when *all* of its variables are set; anything left over goes to the wizard on
a terminal, or exits listing exactly what is missing. `OPENOUTFIND_DB` (or `--db PATH`) points any
command at a different SQLite file.

## The `SiteConfig` singleton (pk=1)

Set during onboarding, editable in Django Admin. `SiteConfig` is the single source of truth for
keys, the operator's country, and the product/target text — one row, since an install runs exactly
one ICP.

| Field | Description | Default |
|:------|:------------|:--------|
| `ai_model` | pydantic-ai `provider:model` id (e.g. `anthropic:claude-sonnet-4-5-...`); bare `gpt-*`/`claude-*`/`gemini-*` are auto-prefixed. Providers: openai/anthropic/google/groq/mistral/cohere/openai_compatible. | (required) |
| `llm_api_key` | API key for the chosen provider. Live-verified at onboarding. | (required) |
| `llm_api_base` | Base URL — **only** for `openai_compatible:*`. | (none) |
| `bettercontact_api_key` | [BetterContact](https://bettercontact.rocks?fpr=openoutreach) key — **free account, 40 credits, no card** (affiliate link, no markup to you). Powers **both** Lead Finder discovery (billed nothing) **and** work-email enrichment (one credit per verified address, only with `--emails`). **Blank disables discovery + enrichment.** | (empty) |
| `contacts_api_token` / `contacts_api_url` | Cross-operator contacts-store token (earned on first contribution) and URL (blank → default hub). | (empty) |
| `country_code` | ISO-3166 alpha-2. Decides the newsletter opt-in default (`geo.is_gdpr_protected`), whether this install contributes to the contacts store (`geo.is_eea_located`), and the target country stamped on every discovered lead. | (from onboarding) |
| `product_docs` | Product/service description. Feeds ICP generation and qualification — **the whole input**. | (required) |
| `campaign_target` | Who you're going after + the outcome. Feeds the same. | (required) |
| `headcount_min` / `headcount_max` | Company-size band, applied to every discovery query. | `1` / `10000` |
| `anchor_profiles` / `anchor_embeddings` | Synthetic ideal profiles (JSON / binary) standing in for positives before any real acceptance exists. Once a lead qualifies the set is permanent — real acceptances outnumber it rather than retiring it. | (empty) |
| `model_blob` | The trained GP model (joblib, binary). | (empty) |

The operator's own email and name live on the Django `User` (created at onboarding), not on
`SiteConfig`. Discovery keeps no filter spec or page cursor here either: the keyword sets it has
fired, and how far each was paged, live in their own `Keyword` / `QueryNode` rows.

## Sending mailboxes — there are none

The `Mailbox` model, the SMTP/IMAP credentials, the per-box signature, the measured daily cap and the
send-spacing clock all moved to [OpenOutSend](https://github.com/eracle/OpenOutSend) with
the sending leg. **Nothing here needs a mailbox**, and onboarding no longer asks for one.

## Newsletter jurisdiction default

At onboarding you enter your `country_code`. If it is **not** an opt-in jurisdiction (EU/EEA, UK, Switzerland, Canada, Brazil, Australia, Japan, South Korea, New Zealand), the newsletter default is on; otherwise it is off. An explicit yes always subscribes. The check reads `core/geo.is_gdpr_protected` — country comes from onboarding, never from any account lookup.

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
