# tests/test_labels.py
"""``find --labels`` — a customer's marks taken in as verdicts on a fresh store.

Covers ``core/pipeline/labels.py`` and the places that must leave a marked lead out: the
export, the promote pool, the pipeline counts. The flag itself is covered by
``tests/test_find.py``.
"""
from __future__ import annotations

import json

import pytest

from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.core.export import lead_records
from openoutfind.core.pipeline import labels
from openoutfind.crm.models import Deal, DealState, Lead, Outcome
from tests.factories import DealFactory, LeadFactory


def _mark(handle, value="good", value_reason="", note="", profile_text=None, **extra):
    """One row as the hub's ``/ratings/`` returns it."""
    return {
        "linkedin_url": f"https://www.linkedin.com/in/{handle}/",
        "full_name": handle.title(),
        "title": "Head of Extraction",
        "company": "Green Labs",
        "reason": "makes its own oil",
        "value": value,
        "value_reason": value_reason,
        "note": note,
        "profile_text": f"{handle} head of extraction at green labs"
        if profile_text is None else profile_text,
        "source_fields": {"contact_job_title": "Head of Extraction",
                          "company_keywords": "cannabis, extraction"},
        "employees_min": 11,
        "employees_max": 50,
        **extra,
    }


def _url(handle):
    return f"https://www.linkedin.com/in/{handle}"


# ── the file ─────────────────────────────────────────────────────


class TestReadLabelsFile:
    def test_reads_a_json_array_as_the_hub_returns_it(self, tmp_path):
        path = tmp_path / "ratings.json"
        path.write_text(json.dumps([_mark("ada"), _mark("bob")]))

        assert [m["full_name"] for m in labels.read_labels_file(path)] == ["Ada", "Bob"]

    def test_reads_json_lines(self, tmp_path):
        path = tmp_path / "ratings.jsonl"
        path.write_text(json.dumps(_mark("ada")) + "\n\n" + json.dumps(_mark("bob")) + "\n")

        assert len(labels.read_labels_file(path)) == 2

    def test_a_row_without_a_url_is_refused(self, tmp_path):
        path = tmp_path / "ratings.json"
        path.write_text(json.dumps([{"value": "good"}]))

        with pytest.raises(ValueError):
            labels.read_labels_file(path)


# ── one mark ─────────────────────────────────────────────────────


class TestVerdictFor:
    @pytest.mark.parametrize("value, reason, verdict", [
        ("good", "", True),
        ("not_fit", "already_known", True),
        ("not_fit", "wrong_company", False),
        ("not_fit", "too_big", False),
        ("not_fit", "other", False),
        ("not_fit", "", False),
        ("not_fit", "gone", None),
        ("unsure", "", None),
        ("", "", None),
    ])
    def test_maps_each_mark(self, value, reason, verdict):
        assert labels.verdict_for({"value": value, "value_reason": reason}) is verdict


class TestReasonFor:
    def test_a_no_fit_names_the_customers_reason_and_note(self):
        reason = labels.reason_for(
            {"value": "not_fit", "value_reason": "too_big", "note": "they have 400 staff"},
            fit=False)

        assert reason == ('The customer marked them not a fit: company too big. '
                          'Their note: "they have 400 staff"')

    def test_a_contact_reads_as_one(self):
        reason = labels.reason_for({"value": "not_fit", "value_reason": "already_known"}, True)

        assert reason == "The customer marked them already a customer or contact."

    def test_a_no_fit_without_a_reason_says_only_that(self):
        assert labels.reason_for({"value": "not_fit"}, False) == \
            "The customer marked them not a fit."


# ── loading ──────────────────────────────────────────────────────


@pytest.mark.django_db
class TestLoad:
    def test_each_mark_becomes_a_marked_lead_with_its_verdict(self):
        loaded = labels.load([
            _mark("ada"),
            _mark("bob", "not_fit", "already_known"),
            _mark("cy", "not_fit", "wrong_role", note="an accountant"),
            _mark("di", "unsure"),
            _mark("ed", "not_fit", "gone"),
        ])

        assert (loaded.fits, loaded.no_fits, loaded.left_out) == (2, 1, 2)
        deals = {d.lead.profile_url: d for d in Deal.objects.select_related("lead")}
        assert set(deals) == {_url("ada"), _url("bob"), _url("cy")}
        assert deals[_url("ada")].state == DealState.QUALIFIED
        assert deals[_url("cy")].state == DealState.FAILED
        assert deals[_url("cy")].outcome == Outcome.WRONG_FIT
        assert "an accountant" in deals[_url("cy")].reason

    def test_the_lead_carries_the_whole_person(self):
        labels.load([_mark("ada")])

        lead = Lead.objects.get(profile_url=_url("ada"))
        assert lead.marked and not lead.synthetic
        assert lead.profile_text == "ada head of extraction at green labs"
        assert lead.source_fields["company_keywords"] == "cannabis, extraction"
        assert lead.job_title == "Head of Extraction" and lead.full_name == "Ada"
        assert (lead.company.name, lead.company.employees_max) == ("Green Labs", 50)
        assert lead.embedding_array is not None

    def test_the_latest_mark_on_a_url_wins(self):
        labels.load([_mark("ada", "good"), _mark("ada", "not_fit", "too_big")])

        assert Deal.objects.get().state == DealState.FAILED

    def test_a_mark_without_a_profile_is_listed_not_loaded(self):
        loaded = labels.load([_mark("ada", profile_text="  ")])

        assert loaded.no_profile == [_url("ada")]
        assert not Lead.objects.exists()

    def test_a_placeholder_employer_is_no_company(self):
        labels.load([_mark("ada", company="Self-employed")])

        assert Lead.objects.get().company is None

    def test_loading_twice_changes_nothing(self):
        labels.load([_mark("ada"), _mark("bob", "not_fit", "too_big")])

        again = labels.load([_mark("ada"), _mark("bob", "not_fit", "too_big")])

        assert (again.fits, again.no_fits, again.already_in) == (0, 0, 2)
        assert Deal.objects.count() == 2

    def test_a_store_with_verdicts_of_its_own_refuses_new_marks(self):
        DealFactory(lead=LeadFactory(), state=DealState.QUALIFIED)

        with pytest.raises(OpenOutFindError) as exc:
            labels.load([_mark("ada")])

        assert exc.value.error_type == ErrorType.BAD_CONFIG
        assert not Lead.objects.filter(marked=True).exists()

    def test_marks_already_in_still_pass_once_the_store_has_judged(self):
        """The loop repeats the flag on every call, after verdicts have been given."""
        labels.load([_mark("ada")])
        DealFactory(lead=LeadFactory(), state=DealState.QUALIFIED)

        assert labels.load([_mark("ada")]).already_in == 1

    def test_a_discovered_lead_awaiting_a_verdict_is_taken_over(self):
        from openoutfind.crm.models import PendingQualification

        lead = LeadFactory(profile_url=_url("ada"))
        PendingQualification.objects.create(lead=lead)

        labels.load([_mark("ada")])

        lead.refresh_from_db()
        assert lead.marked and Deal.objects.get().lead == lead
        assert not PendingQualification.objects.exists()


# ── left out downstream ──────────────────────────────────────────


@pytest.mark.django_db
class TestAMarkedLeadIsLeftOut:
    def test_of_the_export(self):
        labels.load([_mark("ada")])

        assert list(lead_records()) == []

    def test_of_the_promote_pool(self):
        from openoutfind.core.db.deals import get_qualified_profiles

        labels.load([_mark("ada")])

        assert get_qualified_profiles() == []

    def test_but_not_of_the_training_labels(self):
        labels.load([_mark("ada"), _mark("bob", "not_fit", "too_big")])

        _, _, y = Lead.get_labeled_rows()
        assert sorted(y) == [0, 1]
