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
    # address for is certainly worth a verdict — but no longer the only one: see
    # `min_bald_gain` below for why an unfitted posterior can never open this one.
    #
    # 0.7, down from 0.9. The bar is what the exploit state has to clear to do anything at
    # all, and at 0.9 a live campaign cleared it with nothing: 7 leads waited to be ranked
    # while every pass fell through to discovery, so the pool widened and the class balance
    # never moved. A gate the pool cannot reach rations the pipeline shut rather than
    # rationing the spend.
    "min_gp_confidence": 0.7,
    # The second arm of the qualification gate, in nats of expected information.
    #
    # ``min_gp_confidence`` alone cannot open on a cold campaign, and the reason is
    # arithmetic rather than tuning: ``P(f>0.5)`` is ``norm.sf(0.5, mean, std)``, so a wide
    # posterior pulls *every* candidate toward 0.5. Measured on a live campaign with 3 real
    # positives, the highest-scoring lead of 26,737 reached **0.37** — no bar above 0.5
    # admits anything, and discovery cannot help because it adds unlabelled leads, which
    # move no posterior. The run spent 14h33m on 295 discovery pages and 19 LLM calls.
    #
    # So a lead is worth judging if the model is either confident about it *or* uncertain
    # enough to learn from it. The two arms are anti-correlated by construction (BALD peaks
    # where P is near 0.5): on the same live data, leads clearing P>=0.7 and leads in the
    # top BALD decile were **disjoint sets**, in both campaigns.
    #
    # 0.04 measured: it admits 92.9% of the 3-positive campaign's pool and 1.1% of the
    # 122-positive one's, so the arm carries a cold start and then retires itself as the
    # posterior tightens. An absolute floor is safe *here* where it was not on P, because
    # the failure closes a loop instead of an absorbing state: gate shut -> discover ->
    # novel leads -> BALD rises -> gate reopens.
    "min_bald_gain": 0.04,
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
    # plus what the step itself is billed. How many new leads a search adds is not a
    # setting — it is counted from the store.
    #
    # A search is one Lead Finder page: ~70 s on the held order's VM run (878 searches in
    # 17h24m, polling included), and free — Lead Finder bills nothing.
    #
    # A verdict is one candidate judged: a few seconds and an LLM call for a model (a
    # Haiku-class call reads ~1k tokens and writes ~100: about $0.002), and ~40 s for a
    # calling agent under --agent-qualify (178 verdicts in 1h49m on the same store), whose
    # turn re-reads its context: a few cents.
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
    "verdict_dollars_model": 0.002,
    "verdict_dollars_agent": 0.05,
    # Every band must hold this many calibration verdicts before the bar is believed. It
    # also switches the arm on: √N must reach bands × this (N ≥ 1,600 at 8 bands × 5), so
    # a small order's pool never spends verdicts on a sample it cannot fill.
    "calibration_min_band_verdicts": 5,
    # There is no discovery cadence knob. Growing the vocabulary used to be an LLM call
    # worth rationing ("mint_every_n_qualified"); it is now a tokenize-and-count over a
    # few hundred profiles (pipeline/vocabulary.py), so it simply runs every pass. The
    # walk's only other constant is the df≥2 admission floor, which lives with the code
    # that measured it. Discovery steering is arithmetic over labels — no threshold, no
    # confidence gate, no model. See pipeline/select.py.
    "embedding_model": "BAAI/bge-small-en-v1.5",
}


