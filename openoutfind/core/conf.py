# openoutfind/core/conf.py
from __future__ import annotations

from pathlib import Path


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROMPTS_DIR = Path(__file__).parent / "templates" / "prompts"

# ----------------------------------------------------------------------
# The three send guards used to live here — warm capacity (the per-box daily
# ceiling, measured from the box's own Sent folder rather than declared), send
# pacing (a 3.5–4.5 minute gap between two first emails), and the sending window
# (Mon–Fri, 08:00–20:00 in the operator's own time). Together they were most of
# this file. They went to OpenOutSend with the code they governed.
#
# They are worth knowing about if anything here ever grows an outbound leg again,
# because the reasoning was hard-won and is recorded in `openoutsend/core/conf.py`
# on that side: receivers punish *rate* and *volume* separately and a recipient
# reads the *hour*, so one guard could not have covered the other two.
#
# Nothing replaced them, because a finder emits nothing to pace.
# ----------------------------------------------------------------------
# collect_email poll backoff — the bound leg that polls an in-flight paid
# lookup. Each still-running poll doubles the delay (BASE·2^attempt, capped at
# MAX); past DEADLINE the collect leg gives up and reverts the deal to
# READY_TO_FIND_EMAIL for a fresh submit. A provider job resolves in
# seconds-to-minutes, so these are short (unlike the retired channel's
# connect-accept poll, which backed off in hours). A future provider would carry its
# own triple.
#
#
# **The backoff is uncapped and there is no deadline.** Both used to exist, and
# together they made the failure worse than the outage: past the deadline the leg
# abandoned the job and reverted the deal to READY_TO_FIND_EMAIL, where the submit
# leg picked it up and paid for a *new* job — a hot resubmit loop against a
# provider already struggling. Measured on a live install during what the provider
# later confirmed was a multi-day infrastructure incident: 418 submits and 4,512
# polls in a week for ~40 leads, none of which ever terminated.
#
# Doubling without a rail fixes both halves. An unterminated job is *queued*, not
# lost, so the right move is to keep the same request_id and ask later — no second
# submit, and the deal stays at FINDING_EMAIL where nothing re-selects it. And
# doubling reaches long waits cheaply: 5s → a week in 17 polls, ~30 for a month.
# A capped backoff would poll a week-long outage 10,000 times to learn the same
# thing. Nothing is ever written off either, which matters because a timeout is
# evidence about the provider, not about the lead — when the queue drains, the
# next poll simply lands and the deal proceeds with no intervention.
# ----------------------------------------------------------------------
COLLECT_BACKOFF_BASE_S = 5

# The rail is on the *interval*, not on the number of attempts: polling stretches
# to a month and then stays there forever. That is not a give-up in disguise —
# nothing is abandoned or relabelled, the job is simply checked monthly instead of
# ever more rarely. It exists because unbounded doubling stops being representable:
# 5s·2^41 is ~348,000 years, `datetime` raises OverflowError, the handler dies
# before minting its successor, and the deal is stranded at FINDING_EMAIL with no
# pending task — the one outcome the whole design is meant to prevent. Past a
# month the difference between "later" and "much later" has no practical meaning
# anyway, while the difference between "later" and "never polled again" is total.
COLLECT_BACKOFF_MAX_S = 30 * 24 * 3600

# ``COLLECT_TODAY_HORIZON_S`` stood here — past it, a lookup in a long backoff
# stopped counting against today's send headroom, so a handful of stalled jobs
# could not wedge the pipeline shut. It has no meaning without send headroom to
# count against: nothing gates buying an address on anything but having a provider.

# ----------------------------------------------------------------------
# Campaign config (timing + ML defaults — hardcoded, no YAML)
# ----------------------------------------------------------------------
CAMPAIGN_CONFIG = {
    "qualification_n_mc_samples": 100,
    # GP confidence gate: P(f>0.5) above this promotes QUALIFIED → READY_TO_FIND_EMAIL,
    # rationing the paid lookup to leads the model is confident about. It is also the
    # *first* arm of the qualification gate in pipeline/top_up.py — a lead worth buying an
    # address for is certainly worth a verdict — and below it `judge_or_search` decides
    # on measured costs whether the best lead is still worth judging or a search is cheaper.
    #
    # 0.7, down from 0.9. The bar is what the exploit state has to clear to do anything at
    # all, and at 0.9 a live campaign cleared it with nothing: 7 leads waited to be ranked
    # while every pass fell through to discovery, so the pool widened and the class balance
    # never moved. A gate the pool cannot reach rations the pipeline shut rather than
    # rationing the spend.
    "min_gp_confidence": 0.7,
    # The calibrate arm (core/ml/calibration.py) derives the bar above instead of taking it
    # as given. While the store holds fewer than √N calibration verdicts (N = the unjudged
    # pool), each pass judges one lead drawn at random from the thinnest score band; once
    # √N are in, the fit rate at each score is measured from them, and the bar is the
    # lowest score where a verdict pays more than a search would. `min_gp_confidence`
    # stays the fallback until then.
    #
    # Why: a 500-lead order held at 0.7 judged 71 people in 17h24m while 21,134 sat in its
    # pool — 2 of them cleared 0.7, though leads at 0.4–0.7 fit 54–77% of the time out of
    # fold. See the roadmap card `p1-e1-the-bar-is-measured-on-a-stratified-sample`.
    #
    # The two costs the rule weighs, in dollars: run time priced at `run_hour_dollars`,
    # plus what the step itself is billed. The same two costs drive `judge_or_search`,
    # which decides below the bar. How many new leads a search adds is not a setting — it
    # is counted from the store.
    #
    # **The times below are starting values only.** Each store measures its own as a
    # running mean (`core/meter.py`: every page fetched, every verdict given), and a
    # setting stands in only until its first measurement.
    #
    # A search is one Lead Finder page: ~70 s on the held order's VM run (878 searches in
    # 17h24m, polling included), and free — Lead Finder bills nothing.
    #
    # A verdict is one candidate judged: a few seconds and an LLM call for a model, and
    # ~40 s for a calling agent under --agent-qualify (178 verdicts in 1h49m on the same
    # store), whose turn re-reads its context: a few cents. The bill is not measured: it
    # is the tokens below at the model's rates, or `verdict_dollars_agent`.
    #
    # A model's call is priced from the model that makes it (`MODEL_PRICES` below) times
    # the tokens one verdict reads and writes, measured over 20 verdicts on the held
    # order with its ~450-word brief: ~2,760 read (the customer's two texts, the brief,
    # the profile, the instructions and the answer's schema), and 112 written on Sonnet
    # 5.5 against 389 on Haiku 5.5, which thinks before it answers. So Haiku 5.5 pays
    # ~$0.0004 a verdict, Sonnet 5.5 ~$0.006, Opus 5.5 ~$0.012, and the bar rises with
    # the judge's price — at $0.50–$2 a lead, even Opus's per-fit bill is a few cents. A
    # model missing from the table is charged `verdict_dollars_model`.
    #
    # `run_hour_dollars` prices an hour of run time as the smallest machine a run needs on
    # its own — runs are independent, so the machine's hours are the only cost time has: a
    # DigitalOcean Basic droplet, 1 vCPU / 2 GiB, $12/month = $0.01786/h
    # (digitalocean.com/pricing/droplets, 2026-10-16). The heaviest pass measured —
    # re-measuring the bar on a 21k-lead store, five GP fits and the pool scored — peaked at
    # 824 MB resident, so 1 GiB is too tight once an OS sits underneath and 2 GiB is not. At
    # this price time is nearly free next to an LLM's bill, so the bill decides the bar.
    "run_hour_dollars": 0.01786,
    "search_seconds": 70,
    "verdict_seconds_model": 5,
    "verdict_seconds_agent": 40,
    # An agent's verdict longer than this is not measured: the candidate waited on a
    # session that had stopped, not on a verdict. Batch 5's first measured one had waited
    # 1h46m, and the plain running mean would have carried it for the rest of the store.
    "verdict_seconds_agent_max": 600,
    "verdict_input_tokens": 2800,
    "verdict_output_tokens": 300,
    "verdict_dollars_model": 0.005,
    "verdict_dollars_agent": 0.05,
    # Every band must hold this many calibration verdicts before the bar is believed. It
    # also switches the arm on: √N must reach bands × this (N ≥ 1,600 at 8 bands × 5), so
    # a small order's pool never spends verdicts on a sample it cannot fill.
    "calibration_min_band_verdicts": 5,
    # There is no discovery cadence knob. Growing the vocabulary used to be an LLM call
    # worth rationing ("mint_every_n_qualified"); it is now a tokenize-and-count over a
    # few hundred profiles (pipeline/vocabulary.py), so it simply runs every pass. The
    # walk's other constants are the df≥2 admission floor and the ten-person prior on a
    # node's score (`select.PRIOR_WEIGHT`), each beside the code that measured it.
    # Discovery steering ranks each fired node by the GP's view of the people it found —
    # no threshold and no confidence gate. See pipeline/select.py.
    "embedding_model": "BAAI/bge-small-en-v1.5",
}

# What a verdict's model is billed, in dollars per million tokens (input, output), keyed by
# the model name without its provider prefix. A dated id (`claude-haiku-4-5-20251001`)
# matches its family by the longest key it starts with. Anthropic's first-party rates,
# 2026-10-06; Haiku 5.5's apply to prompts under 100k tokens, which a verdict always is.
MODEL_PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
}


