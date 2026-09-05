# openoutfind/core/config.py
"""What this install was told, read from the environment on every run.

**These are answers a human gives, and this program is not the one asking.** Remembering
what somebody typed is a convenience for somebody who types; an agent supplies its
environment on every invocation and has nothing to remember. So there is no row and no
singleton here — `OPENOUTFIND_*` is the whole configuration surface, and it is read fresh
each run. The wizard that asks these questions lives in OpenOutreach, which owns the
human half and exports these names.

**What the pipeline produced still lives in the store**, and the line is who produced the
value rather than whether it is a secret:

  * the walk's nodes and vocabulary — measured, `core/models.py`
  * the invented ideal leads — LLM-written once and permanent, and they are *leads*, so
    they are `Lead` rows carrying `synthetic=True` (`core/pipeline/icp.py`)
  * the ICP size band — written onto the query nodes it rides, where it is part of the
    query that was actually fired
  * the operator — identity, a `User` row written once, because a renamed variable must
    not rename the person a campaign belongs to

Nothing else was ever produced. The fitted GP used to be dumped into a `model_blob`
column that nothing ever read back: the fit is reproduced from the label rows every time
the evidence changes (`core/ml/qualifier.py:qualifier_for`), so the blob was a cache of a
value already derived, and it is gone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields

ENV_PREFIX = "OPENOUTFIND_"

# What each field is read from, where the variable is not simply the field's own name
# upper-cased. `operator_country_code` would otherwise upper-case to
# `OPERATOR_COUNTRY_CODE` — one word longer than the name OpenOutSend already uses
# (`OUTSEND_OPERATOR_COUNTRY`) for the same concept.
_ALIASES = {"operator_country_code": "OPERATOR_COUNTRY"}

# The model needs both halves or it cannot be built; `llm_api_base` is not among them
# because only the `openai_compatible` provider reads it, and that builder raises its own
# error naming it. Requiring it here would stop an Anthropic install over a value it
# never looks at.
REQUIRED_LLM_FIELDS = ("ai_model", "llm_api_key")

# What the ICP is written from. Without these there is no seed, no anchor and no verdict
# — every LLM call this tool makes renders one of them into its prompt.
REQUIRED_ICP_FIELDS = ("product_docs", "campaign_target")

# Discovery itself, which is free, runs on the BetterContact key; enrichment spends it.
# An install without one cannot search at all, so it is required rather than optional.
REQUIRED_DISCOVERY_FIELDS = ("bettercontact_api_key",)


@dataclass(frozen=True)
class SiteConfig:
    """The configuration this run was given.

    Frozen because nothing may write it: a value edited at runtime would live until the
    process ended and then be gone, which is the silent half-persistence this replaced.
    Empty string rather than None throughout — "unset" and "deliberately blank" are the
    same state to a prompt template, and the template renders every one of them as text.
    """

    # A pydantic-ai model identifier in `provider:model` form (e.g.
    # `anthropic:claude-sonnet-4-5-20250929`, `openai:gpt-4o`, `groq:llama-3.3-70b`). The
    # provider lives inside this single string — there is no separate provider field to
    # drift out of sync. A bare model name whose prefix is unambiguous (`gpt`/`o1`/`o3`
    # →openai, `claude`→anthropic, `gemini`→google) is also accepted; everything else
    # must carry an explicit prefix. See core/llm.py:split_model_id.
    ai_model: str = ""
    llm_api_key: str = ""
    # Only consulted for the openai_compatible provider (OpenRouter / Together / Ollama / vLLM).
    llm_api_base: str = ""

    # Email-finder keys — one per supported vendor, and a key is all it takes to select
    # one (see enrichment/provider.py:active). BetterContact's key additionally powers
    # Lead Finder *discovery*, which is billed nothing; Apollo's does not, so an
    # Apollo-only install still needs the other key for discovery. They are not
    # interchangeable at that leg, only at enrichment.
    bettercontact_api_key: str = ""
    apollo_api_key: str = ""

    # Which finder resolves addresses when *both* keys are set. Blank means "decide from
    # whichever key exists", which is the whole answer for a one-vendor install; it only
    # has to be set to move an install that holds both.
    email_finder: str = ""

    # The operator's own ISO-3166 alpha-2 jurisdiction — not to be confused with
    # `Lead.country_code`, the per-lead target country an ICP search surfaced someone
    # under. Drives the email-jurisdiction rules (core/geo.py): whether we contribute to
    # the contacts store (derived, `not is_eea_located` — never a stored toggle).
    operator_country_code: str = ""

    # The campaign content: what this install sells, and to whom. The two things every
    # prompt in the tool is written from.
    product_docs: str = ""
    campaign_target: str = ""

    # Central contacts store (see openoutfind/contacts/). The token names this install to
    # the hub; blank means the run registers for one itself and keeps it for the length of
    # the process. The URL is blank by default (falls back to DEFAULT_API_URL).
    contacts_api_token: str = ""
    contacts_api_url: str = ""

    @classmethod
    def load(cls) -> "SiteConfig":
        """This run's configuration, read from `OPENOUTFIND_*`."""
        return cls(**{field.name: _from_environment(variable_for(field.name))
                      for field in fields(cls)})


def variable_for(field_name: str) -> str:
    """The environment variable one field is read from."""
    return ENV_PREFIX + _ALIASES.get(field_name, field_name.upper())


def missing(config: SiteConfig, field_names) -> list[str]:
    """The variables that would have filled the named fields this run lacks."""
    return [variable_for(name) for name in field_names if not getattr(config, name)]


def _from_environment(variable: str) -> str:
    """One `OPENOUTFIND_*` value, stripped — "" whether it was unset or blank."""
    return (os.environ.get(variable) or "").strip()
