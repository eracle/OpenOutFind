# tests/test_exclude.py
"""``find --exclude`` — profiles a run must never put up for a verdict or print.

Covers ``core.exclude`` and the two places it is read: the qualification scan (plus the
``--agent-qualify`` resume) and the export. The CLI's flag and its error are covered by
``tests/test_find.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from openoutfind.core import agent_qualify, exclude
from openoutfind.core.ml.qualifier import BayesianQualifier
from openoutfind.core.pipeline.qualify import (
    QualifyPending, fetch_qualification_candidates, run_qualification,
)
from openoutfind.crm.models import DealState
from tests.factories import DealFactory, LeadFactory


@pytest.fixture(autouse=True)
def _reset_contextvars():
    """A contextvar carries state past a test's own scope otherwise."""
    yield
    exclude._excluded.set(frozenset())
    agent_qualify._active.set(False)
    agent_qualify._verdict.set(None)


def _candidate(profile_url):
    from openoutfind.crm.models import Lead

    return Lead.objects.create(
        profile_url=profile_url,
        profile_text="engineer at acme",
        embedding=np.ones(384, dtype=np.float32).tobytes(),
    )


# ── normalisation ────────────────────────────────────────────────


class TestNormalizeLinkedinUrl:
    def test_lowercases_the_host_and_drops_query_fragment_and_trailing_slash(self):
        url = " https://WWW.LinkedIn.com/in/Ada-Lovelace/?trk=abc#top \n"

        assert exclude.normalize_linkedin_url(url) == "https://www.linkedin.com/in/Ada-Lovelace"

    def test_keeps_the_scheme_and_the_path_case(self):
        """The hub keys on the same string — anything more aggressive here would make
        two URLs equal on one side and different on the other."""
        assert exclude.normalize_linkedin_url("http://linkedin.com/in/AdaL") == \
            "http://linkedin.com/in/AdaL"


# ── the file ─────────────────────────────────────────────────────


class TestReadExcludeFile:
    def test_reads_one_url_per_line_and_skips_blank_lines(self, tmp_path):
        path = tmp_path / "exclude.txt"
        path.write_text("https://www.linkedin.com/in/ada/\n\n   \nhttps://WWW.linkedin.com/in/bob?x=1\n")

        assert exclude.read_exclude_file(path) == frozenset({
            "https://www.linkedin.com/in/ada",
            "https://www.linkedin.com/in/bob",
        })

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            exclude.read_exclude_file(tmp_path / "nope.txt")


class TestIsExcluded:
    def test_matches_a_stored_url_after_normalisation(self):
        exclude.enable(frozenset({"https://www.linkedin.com/in/ada"}))

        assert exclude.is_excluded("https://www.linkedin.com/in/ada/")
        assert not exclude.is_excluded("https://www.linkedin.com/in/bob/")

    def test_nothing_is_excluded_without_a_list_or_a_url(self):
        assert not exclude.is_excluded("https://www.linkedin.com/in/ada/")
        exclude.enable(frozenset({"https://www.linkedin.com/in/ada"}))
        assert not exclude.is_excluded(None)


# ── where it is read ─────────────────────────────────────────────


@pytest.mark.django_db
class TestNeverPutUpForAVerdict:
    def test_an_excluded_lead_is_not_a_qualification_candidate(self, site_config):
        _candidate("https://www.linkedin.com/in/ada/")
        kept = _candidate("https://www.linkedin.com/in/bob/")
        exclude.enable(frozenset({"https://www.linkedin.com/in/ada"}))

        assert fetch_qualification_candidates() == [kept]

    def test_agent_qualify_never_stops_on_an_excluded_lead(self, site_config):
        _candidate("https://www.linkedin.com/in/ada/")
        exclude.enable(frozenset({"https://www.linkedin.com/in/ada"}))
        agent_qualify.enable(None)

        assert run_qualification(site_config, BayesianQualifier(seed=42)) is None

    def test_a_pending_candidate_on_the_list_is_dropped_not_resumed(self, site_config):
        """A stop an earlier run left behind would otherwise show the agent a profile
        this run was told to keep out."""
        from openoutfind.crm.models import PendingQualification

        excluded = _candidate("https://www.linkedin.com/in/ada/")
        _candidate("https://www.linkedin.com/in/bob/")
        PendingQualification.objects.create(lead=excluded)
        exclude.enable(frozenset({"https://www.linkedin.com/in/ada"}))
        agent_qualify.enable(None)

        with pytest.raises(QualifyPending) as raised:
            run_qualification(site_config, BayesianQualifier(seed=42))

        assert raised.value.payload["profile_url"] == "https://www.linkedin.com/in/bob/"
        assert list(PendingQualification.objects.values_list("lead__profile_url", flat=True)) == \
            ["https://www.linkedin.com/in/bob/"]


@pytest.mark.django_db
class TestNeverExported:
    def test_an_excluded_lead_is_left_out_of_the_export(self, site_config):
        from openoutfind.core.export import lead_records

        for name in ("ada", "bob"):
            DealFactory(lead=LeadFactory(profile_url=f"https://www.linkedin.com/in/{name}/"),
                        state=DealState.QUALIFIED, reason="fits")
        exclude.enable(frozenset({"https://www.linkedin.com/in/ada"}))

        assert [r["linkedin_url"] for r in lead_records()] == ["https://www.linkedin.com/in/bob/"]
