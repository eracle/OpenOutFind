# tests/test_store_snapshot.py
"""The store as a round reads it: the people index, the fit rates."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core.pipeline import store_snapshot
from openoutfind.core.pipeline.store_snapshot import People
from openoutfind.core.pipeline.terms import Terms
from openoutfind.crm.models import Lead
from tests.store_builders import US, make_lead


@pytest.mark.django_db
class TestPeople:
    def test_match_is_what_terms_matches_says_over_everyone(self):
        make_lead("Purchasing Manager", "cannabis edibles, thc vape", "Ohio")
        make_lead("Head of Procurement", "cannabis extraction", "California", seniority="head")
        make_lead("Co-Founder", "e-commerce, food & beverage", "Ohio")
        make_lead("Production Manager", "vapes", "Texas")
        searches = [
            Terms.written(title="manager"),
            Terms.written(title="purchasing", tag="cannabis"),
            Terms.written(title="procurement head", tag="cannabis extraction"),
            Terms.written(title="cofounder", tag="food beverage"),
            Terms.written(title="founder", tag="e commerce", place=f"Ohio, {US}"),
            Terms.written(title="manager", place=f"Texas, {US}"),
            Terms.written(title="procurement", seniority="head", place=US),
            Terms.written(title="manager", tag="edibles cannabis"),
        ]
        people = People.load({})
        leads = list(Lead.objects.all())

        for terms in searches:
            assert people.match(terms) == {
                lead.pk for lead in leads if terms.matches(lead.source_fields)}, terms

    def test_a_row_with_no_state_is_matched_by_a_state_of_its_own_country(self):
        stateless = make_lead(state="")
        Lead.objects.filter(pk=stateless.pk).update(source_fields={
            "contact_job_title": "Owner", "company_keywords": "restaurant",
            "contact_location_country": "indonesia"})
        ohio = make_lead(state="Ohio")
        people = People.load({})

        assert people.match(Terms.written(place="Bali, Indonesia")) == {stateless.pk}
        assert people.match(Terms.written(place=f"Ohio, {US}")) == {ohio.pk}
        assert people.match(Terms.written(place=f"Texas, {US}")) == set()

    def test_a_state_is_held_by_its_rows_and_its_countrys_rows_with_no_state(self):
        ohio, texas, stateless = (make_lead(state="Ohio"), make_lead(state="Texas"),
                                  make_lead(state=""))
        people = People.load({})

        assert people.holding(("state", "ohio", "united states")) == {ohio.pk, stateless.pk}
        assert texas.pk not in people.holding(("state", "ohio", "united states"))

    def test_a_hyphenated_title_offers_none_of_its_pieces(self):
        """`co` alone reaches co-owners and co-CEOs: 283,068 against 179,644 co-founders."""
        both = {make_lead("Co-Founder & CEO", "cannabis edibles, food & beverage", "Ohio").pk,
                make_lead("Founder", "cannabis", "Ohio").pk}

        people = People.load({})

        assert people.title_vocabulary(both) == {"founder", "ceo"}
        assert {("cannabis",), ("cannabis", "edibles"), ("food", "beverage")} <= (
            people.tag_vocabulary(both))
        assert people.states_in("United States") == {"ohio"}

    def test_the_vocabulary_holds_whole_tags_and_a_phrase_inside_one_still_matches(self):
        """`pre` from `pre-rolls` brought 6,512 founders of pre-seed funds and pre-k schools."""
        held = make_lead("Founder", "pre-rolls, health care, cannabis extraction", "Ohio")

        people = People.load({})

        assert people.tag_vocabulary({held.pk}) == {
            ("pre", "rolls"), ("health", "care"), ("cannabis", "extraction")}
        assert people.match(Terms.written(title="founder", tag="cannabis")) == {held.pk}
        assert people.match(Terms.written(title="founder", tag="pre")) == {held.pk}

    def test_the_vocabulary_is_of_the_people_asked_for(self):
        fit = make_lead("Extraction Lead", "live resin", "Ohio")
        make_lead("Clerk", "insurance", "Ohio")

        people = People.load({})

        assert people.title_vocabulary({fit.pk}) == {"extraction", "lead"}
        assert people.tag_vocabulary({fit.pk}) == {("live", "resin")}

    def test_a_store_with_nobody_scored_reads_as_neutral(self):
        make_lead()
        assert People.load({}).base_rate() == store_snapshot.NEUTRAL_RATE


@pytest.mark.django_db
class TestTheFitRate:
    def test_one_class_of_verdicts_leaves_the_score_as_it_is(self):
        make_lead(verdict=1)
        assert list(store_snapshot.fit_rate_map()([0.2, 0.7])) == [0.2, 0.7]

    def test_the_score_is_mapped_through_the_verdicts_at_their_out_of_fold_scores(self):
        for verdict in (0, 0, 1, 1):
            lead = make_lead(verdict=verdict)
            lead.embedding_array = np.full(384, 0.1, dtype=np.float32)
            lead.save()

        class Folds:
            def out_of_fold_probs(self, indices, k):
                return np.array([0.1, 0.2, 0.8, 0.9])

        with patch("openoutfind.core.ml.qualifier.build_qualifier", return_value=Folds()):
            rate = store_snapshot.fit_rate_map()

        assert rate([0.05])[0] == 0.0 and rate([0.95])[0] == 1.0

    def test_it_is_fitted_once_per_set_of_verdicts(self):
        for verdict in (0, 1):
            lead = make_lead(verdict=verdict)
            lead.embedding_array = np.full(384, 0.1, dtype=np.float32)
            lead.save()

        class Folds:
            def out_of_fold_probs(self, indices, k):
                return np.array([0.2, 0.8])

        with patch("openoutfind.core.ml.qualifier.build_qualifier",
                   return_value=Folds()) as build:
            store_snapshot.fit_rate_map()
            store_snapshot.fit_rate_map()
        assert build.call_count == 1
