# Testing

The suite is pytest, mirroring the package structure under `tests/`: the tests of `openoutfind/core/pipeline/walk.py` are `tests/core/pipeline/test_walk.py`. Mock at the **boundaries** — the BetterContact client, the hub API and the LLM — never inside business logic.

## Running

```bash
make test                 # full suite (native — tests do not run in Docker)

.venv/bin/pytest tests/core/pipeline/test_qualify.py   # a single file
.venv/bin/pytest -k test_name                          # a single test by name
```

## Layout

```
tests/
├── conftest.py            # autouse guards (no hub write, no model ping, no ONNX model, no brief
│                          #   or keyword guess written by a model, no fit or run mode left over)
│                          #   and the operator / configure / site_config fixtures
├── factories.py           # factory-boy factories, and exportable_deal()
├── store_builders.py      # people, companies, verdicts and fired searches for a walk's store
├── test_main.py           # the entry point: --db, and the overview a bare call prints
├── test_defaults.py       # the state dir and what a host must splat
├── test_discovery.py      # Lead Finder search, rows into leads and companies
├── contacts/              # the hub client: resolve, contribute, share, register, balance
├── crm/                   # Lead: the stored vector and the labels the GP trains on
├── enrichment/            # BetterContact transport, the finder seam, the paid lookup
└── core/
    ├── test_*.py          # cycle, job, export, status, readiness, exclude, agent_qualify,
    │                      #   company_cap, meter, llm, logging, geo, version
    ├── db/                # deal state transitions
    ├── ml/                # the qualifier, the embedding wrapper, the calibrate arm
    ├── pipeline/          # the walk and everything it reads: terms, analyzers, compose,
    │                      #   belief, search_arms, store_snapshot, keyword_guess, probe,
    │                      #   judge_or_search, icp (anchors), brief, labels, qualify,
    │                      #   ready_pool, top_up
    └── management/        # the error contract (test_base) and the verbs (commands/)
```

## Conventions

- **Mock at the boundary.** Patch the BetterContact HTTP client, the hub client and the pydantic-ai model — not the pipeline functions that call them.
- **A test of a function `conftest.py` stubs** (`verify_llm_credentials`, `embed_text(s)`) imports it by name at module level, which binds the real one before the stub is applied. There are no opt-out markers.
- **The database is opt-in**: a test that touches it takes `db`, `site_config` or `@pytest.mark.django_db`.
- **CRM objects** come from `factories.py` (factory-boy) or direct model creation.
- **No browser, no network.** There is nothing to launch and no live API to hit; the tool is browserless and every external call is stubbed.

## What the suite does not cover

**A `find` that actually finds.** Every external call is stubbed, so no test — and no manual
run — has yet taken the happy path end to end with live `OPENOUTFIND_LLM_API_KEY` and
`OPENOUTFIND_BETTERCONTACT_API_KEY`. That includes the Claude Code plugin: the skill is
inventoried and fires correctly on the refusal paths (it runs `status`, reads
`onboarding_incomplete`, names the right variables and declines to accept the legal notice
for the user), but a session that returns leads has never been observed. Exercising it costs
one BetterContact credit per verified hit.
