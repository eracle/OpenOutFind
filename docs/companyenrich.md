# Optional CompanyEnrich discovery

[CompanyEnrich](https://companyenrich.com/) is a B2B company and people data API.
It supports company enrichment, account search, similar companies, professional
profiles and workforce research. OpenOutFind uses its people-search endpoint to
bring a deliberately chosen, bounded page of profiles into local qualification.

**CompanyEnrich discovery is paid.** Merely setting its key does nothing. The
existing BetterContact free discovery remains the default. To authorize one paid
page, supply all three flags and the key:

```bash
export OPENOUTFIND_COMPANYENRICH_API_KEY='your-api-key'
outfind find 5 --companyenrich-query people.json \
  --companyenrich-request-id founders-page-1 --discovery-credit-budget 20
```

`people.json`:

```json
{
  "domains": ["example.com"],
  "positionQuery": ["Founder", "CEO"],
  "countries": ["US"],
  "page": 1,
  "pageSize": 10
}
```

The normal product/target and LLM configuration still apply. `--agent-qualify`
continues to work, including the existing `population_pending` cold-start and
`qualify_pending` verdict handoff. Keep the same query, request ID and budget when
answering those handoffs. A CompanyEnrich-only discovery run does not require a
BetterContact key; the separate `--emails` option still needs an email finder.
The `check` and standalone `status` verbs describe the default BetterContact path.

## What can spend

People search currently costs **2 CompanyEnrich credits per returned profile**.
The local reservation is conservative: `2 × pageSize` credits, even if the API
returns fewer profiles. `pageSize` defaults to 10 and is limited to 1–100; the
budget must cover that entire page before any request. The integration makes at
most **one** request, even with a larger budget. It never spends the unused budget
on another page. Pricing was checked on 2026-11-02 against the
[provider pricing](https://companyenrich.com/pricing); the provider controls actual
billing and this local cap assumes that published rate remains in effect.

The goal (`find 5`) counts qualified leads, not paid results. Ten purchased profiles
may produce fewer than five fits. If the pool cannot advance, the command prints
what it has and exits `goal_unreached`. A fresh run may satisfy its goal from an
existing pool without buying the selected page at all. `find 0` never purchases it.
No automatic free-source expansion runs while the CompanyEnrich source is selected.

Email purchase is a separate authorization: the discovery flags do not imply
`--emails`, and `--emails` does not authorize CompanyEnrich discovery. People search
returns **no email addresses or phone numbers**. Existing email lookup remains a
separate provider workflow; CompanyEnrich-sourced identities bypass the shared hub.

## Resuming without duplicate purchases

The request ID is an operator-chosen 1–64 character identifier using letters,
digits, `_`, `.` or `-`. It identifies one paid request in this database:

- The reservation is committed before HTTP. A repeated or concurrent invocation
  with the same ID cannot submit it again.
- The full successful response is stored before embedding or qualification. A
  local interruption can resume using that response without another purchase.
- A reused ID with different query parameters fails. Choose a **new ID** to
  explicitly buy a different page; keep `pageSize` fixed while incrementing `page`.
- HTTP failures, rate limits, redirects, invalid responses and timeouts are never
  retried. If the outcome is unknown, the existing ID remains blocked. Check
  provider usage before authorizing a retry using a new ID: it could charge again.

The request ID is local replay protection, **not provider-side idempotency**. A new
ID or different database creates a separate authorization. The stored
`core.CompanyEnrichPage` contains query, maximum credits, raw response and import
state. It contains no API key. Under `--json`, run metadata adds a `discovery`
object with source, ID, page, size, maximum credits, selected budget and hub policy.
Provider refusal uses existing `provider_auth`, `provider_out_of_credits`,
`provider_rate_limited` or `provider_unavailable` errors; configuration uses
`bad_config`/`onboarding_incomplete`. Raw error bodies and credentials are not echoed.

## Filters and profile fields

Supported request fields follow the checked-in official OpenAPI subset:
`domains`, `query` (company name/domain text), `positionQuery`, `countries`,
`seniority`, `department`, current-company/current-position UTC date bounds,
`education`, `exclude`, `page` and `pageSize`. At least one targeting filter is
required. Unknown fields, invalid enum values, page bounds and malformed UTC dates
fail before spending. See the [API reference](https://docs.companyenrich.com/reference).

Current title, reported names, country, current employer/domain/industry and
seniority feed the existing local lead/qualification fields. Names are copied
only when supplied, never split in-house. Historical employment is not treated
as current. Profiles without a usable LinkedIn URL or current title are retained
in the raw page but skipped as local qualification candidates; company caps also
apply. The BetterContact-specific missing-headline heuristic is not applied to a
provider that does not supply that field. Nested history and page metadata remain
in the stored response; they are not represented as new export columns.

## Data handling and scope

CompanyEnrich pages are never sent to the shared contacts hub: neither profiles,
companies nor embeddings. `Lead.discovery_source` persists across runs and known
profile deduplication. A later email resolution cannot send that identity or email
to the hub, and hub lookups are skipped too. Other sources cannot clear that marker.
An existing row's profile is not overwritten when the same person appears again.
Later free-provider pages also filter identities already marked CompanyEnrich before
sharing raw profiles or employers; alternate host/trailing-slash/query spellings are
matched conservatively. A new source cannot undo the persisted local-only choice.


“Local” here means excluded from the shared hub. The existing qualification path
still sends profile text to the operator's configured LLM, and stdout/CSV/JSON
still exports qualified leads to the operator's chosen tools. Operator registration
and aggregate account-balance reads retain their existing behavior. CompanyEnrich
is a data source, not permission to redistribute its records to a shared data pool.

Company search/lookalikes/enrichment, workforce, person reverse lookup, compound
`companyFilter` searches, async jobs, scroll, exports and beta email/mobile
operations are deliberately outside this finder contribution. They need separate
inputs and spend contracts; none runs behind a people lookup. In particular,
`companyFilter` could add a company-search step, so it is rejected by the bounded
people-only schema. This source does not add sending capabilities.

Tests use synthetic fixtures only; no paid API calls or real credentials are needed.
