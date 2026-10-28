# tests/test_select.py
"""Query selection — the add-only frontier walk.

A node is a keyword set. Its value is what the people it found look like
(``P̂ = (s + 10·P̂(parent)) / (n + 10)``, a judged lead counting its verdict and any other
the GP's score); the label store decides which children exist. These tests assert the
estimate, the expansion and the retirement rules — what the walk's correctness rests on.
"""
import numpy as np
import pytest

from openoutfind.core.models import Keyword, QueryNode
from openoutfind.core.pipeline import select
from openoutfind.core.pipeline.icp import Seed
from openoutfind.core.pipeline.select import REACH_CAP, LabelStore, token_key
from openoutfind.crm.models import Deal, DealState, Lead, Outcome


def _campaign(anchor_profiles=()):
    """This install, with its invented ideal leads written as the rows they are."""
    for profile in anchor_profiles:
        Lead.objects.create(synthetic=True, profile_text=profile)


def _node(_site_config, pairs, parent=None, **kw):
    node = QueryNode.objects.create(token_key=token_key(pairs), parent=parent, **kw)
    node.keywords.set(Keyword.rows_for(pairs))
    return node


def _labelled(_site_config, profile_text, qualified, found_by=None):
    """A lead with a verdict — what expansion counts, and, found by a node, its harvest."""
    lead = Lead.objects.create(
        profile_url=f"https://x/{Lead.objects.count()}", profile_text=profile_text,
        discovered_by=found_by)
    Deal.objects.create(
        lead=lead,
        state=DealState.QUALIFIED if qualified else DealState.FAILED,
        outcome="" if qualified else Outcome.WRONG_FIT)
    return lead


def _found(node, fits, misses):
    """``fits`` + ``misses`` judged people the node's pages brought in."""
    for qualified, count in ((True, fits), (False, misses)):
        for _ in range(count):
            _labelled(None, "someone", qualified, found_by=node)


def _scores(qualifier=None):
    return select.HarvestScores.load(qualifier, LabelStore.load().base_rate)


class TestTokenKey:
    def test_is_order_independent(self):
        # Add-only over three fields means most nodes are reachable several ways, so
        # identity has to be the set, not the path that reached it.
        a = [("lead_job_title", "founder"), ("lead_job_title", "cto")]
        assert token_key(a) == token_key(list(reversed(a)))

    def test_distinguishes_the_field(self):
        # `marketing` counts 3.9M in job_title and 5.5M in department — different sets.
        assert token_key([("lead_job_title", "x")]) != token_key([("lead_location", "x")])


class TestLabelStore:
    def test_counts_are_containment_over_all_tokens(self, db):
        c = _campaign()
        _labelled(c, "founder cto stealth ai startup", qualified=True)
        _labelled(c, "founder cto fintech", qualified=True)
        _labelled(c, "founder marketing agency", qualified=False)
        store = LabelStore.load()

        assert store.counts([("lead_job_title", "founder")]) == (2, 1)
        assert store.counts([("lead_job_title", "founder"), ("lead_job_title", "cto")]) == (2, 0)
        assert store.counts([("lead_job_title", "nobody")]) == (0, 0)

    def test_base_rate_is_the_level_a_depth_one_node_inherits(self, db):
        c = _campaign()
        _labelled(c, "alpha", qualified=True)
        _labelled(c, "beta", qualified=True)
        _labelled(c, "gamma", qualified=False)
        # Laplace: (2 + 1) / (3 + 2)
        assert LabelStore.load().base_rate == pytest.approx(3 / 5)

    def test_empty_store_is_an_even_prior(self, db):
        _campaign()
        assert LabelStore.load().base_rate == 0.5

    def test_an_all_rejection_campaign_still_has_a_usable_level(self, db):
        # The state the anchors exist for, so a common one. A raw rate of 0 here makes
        # every unlabelled node's Beta degenerate (α = a + 2·0 = 0).
        c = _campaign()
        for _ in range(20):
            _labelled(c, "nope", qualified=False)
        assert 0 < LabelStore.load().base_rate < 0.5

    def test_an_all_qualified_campaign_stays_below_one(self, db):
        c = _campaign()
        for _ in range(20):
            _labelled(c, "yes", qualified=True)
        assert 0.5 < LabelStore.load().base_rate < 1

    def test_cooccurring_only_offers_tokens_seen_with_a_qualified_lead(self, db):
        c = _campaign()
        _labelled(c, "founder cto ai", qualified=True)
        _labelled(c, "founder plumber", qualified=False)
        store = LabelStore.load()
        candidates = [("lead_job_title", t) for t in ("cto", "ai", "plumber", "unseen")]

        offered = store.cooccurring([("lead_job_title", "founder")], candidates)

        # `plumber` shares a profile with founder but not a *qualified* one; `unseen`
        # shares none. Both would enter with a=b=0 and be indistinguishable noise.
        assert sorted(offered) == [("lead_job_title", "ai"), ("lead_job_title", "cto")]

    def test_cooccurring_never_offers_a_token_the_node_already_has(self, db):
        c = _campaign()
        _labelled(c, "founder cto", qualified=True)
        store = LabelStore.load()
        pairs = [("lead_job_title", "founder")]
        assert ("lead_job_title", "founder") not in store.cooccurring(pairs, pairs)


class TestAnchorsAsPositives:
    """The cold phase's only positives — see ``LabelStore.load``."""

    def test_anchors_count_as_qualified_profiles(self, db):
        c = _campaign(anchor_profiles=["founder cto stealth ai health startup"])
        _labelled(c, "utilities telecom manager", qualified=False)
        store = LabelStore.load()

        assert store.qualified_count == 1
        assert store.counts([("lead_job_title", "founder")]) == (1, 0)

    def test_they_unblock_expansion_when_nothing_has_qualified(self, db):
        # Without them this is the closed loop: no qualified profile → nothing
        # co-occurs → the frontier never grows past its one-token seed nodes → the
        # queries stay too broad to qualify anybody → still nothing co-occurs.
        c = _campaign(anchor_profiles=["founder cto health supplements startup"])
        for _ in range(5):
            _labelled(c, "utilities telecom manager", qualified=False)
        store = LabelStore.load()
        seed = [("lead_job_title", t) for t in ("founder", "cto", "health", "manager")]
        node = _node(c, [("lead_job_title", "founder")])

        assert select.expand(node, store, seed) == 2  # cto, health — not manager
        assert QueryNode.objects.filter(parent=node).count() == 2

    def test_without_anchors_a_cold_campaign_cannot_expand(self, db):
        c = _campaign(anchor_profiles=[])
        for _ in range(5):
            _labelled(c, "founder utilities telecom", qualified=False)
        store = LabelStore.load()
        node = _node(c, [("lead_job_title", "founder")])

        assert select.expand(node, store, [("lead_job_title", "telecom")]) == 0

    def test_an_anchor_keeps_counting_after_a_real_positive_arrives(self, db):
        # The anchors are permanent: a real acceptance joins them rather than replacing
        # them, so what changes is that real evidence starts to outweigh the invented
        # kind, not that the invented kind disappears.
        c = _campaign(anchor_profiles=["founder cto invented"])
        assert LabelStore.load().qualified_count == 1

        _labelled(c, "founder cto real", qualified=True)
        store = LabelStore.load()

        assert store.qualified_count == 2
        assert store.counts([("lead_job_title", "invented")]) == (1, 0)


class TestHarvestScores:
    def test_a_judged_lead_counts_its_verdict(self, db):
        node = _node(None, [("lead_job_title", "founder")])
        _found(node, fits=3, misses=1)

        assert _scores().counts(node) == (3.0, 4)

    def test_an_unjudged_lead_counts_as_the_gps_score(self, db):
        class _Model:
            def predict_probs(self, X):
                return np.full(len(X), 0.25)

        node = _node(None, [("lead_job_title", "founder")])
        _found(node, fits=1, misses=0)
        Lead.objects.create(profile_url="https://x/unjudged", discovered_by=node,
                            embedding=np.ones(4, dtype=np.float32).tobytes())

        assert _scores(_Model()).counts(node) == (pytest.approx(1.25), 2)

    def test_without_a_model_only_judged_leads_count(self, db):
        node = _node(None, [("lead_job_title", "founder")])
        Lead.objects.create(profile_url="https://x/unjudged", discovered_by=node,
                            embedding=np.ones(4, dtype=np.float32).tobytes())

        assert _scores().counts(node) == (0.0, 0)

    def test_the_customers_marks_and_the_anchors_are_no_nodes_harvest(self, db):
        node = _node(None, [("lead_job_title", "founder")])
        mark = _labelled(None, "marked", qualified=True, found_by=node)
        Lead.objects.filter(pk=mark.pk).update(marked=True)
        Lead.objects.create(profile_url="https://x/anchor", synthetic=True, discovered_by=node)

        assert _scores().counts(node) == (0.0, 0)

    def test_the_root_is_every_searched_lead_smoothed(self, db):
        node = _node(None, [("lead_job_title", "founder")])
        _found(node, fits=1, misses=3)

        assert _scores().root == pytest.approx((1 + 1) / (4 + 2))

    def test_with_nothing_searched_the_root_is_the_fallback(self, db):
        assert select.HarvestScores.load(None, fallback_level=0.3).root == 0.3


class TestEstimate:
    def test_an_unfired_node_sits_at_the_inherited_level(self, db):
        found = _node(None, [("lead_job_title", "founder")])
        _found(found, fits=2, misses=6)
        scores = _scores()
        node = _node(None, [("lead_job_title", "unseen")])

        assert select.estimate(node, scores) == pytest.approx(scores.root)

    def test_few_people_stay_near_the_parent(self, db):
        """The sample size: one good person cannot drag a child far from its parent."""
        parent = _node(None, [("lead_job_title", "founder")])
        _found(parent, fits=10, misses=30)
        child = _node(None, [("lead_job_title", "founder"), ("lead_job_title", "rare")],
                      parent=parent)
        _found(child, fits=1, misses=0)
        scores = _scores()

        parent_p = select.estimate(parent, scores)
        assert select.estimate(child, scores) - parent_p < 0.1

    def test_many_people_are_mostly_themselves(self, db):
        parent = _node(None, [("lead_job_title", "founder")])
        _found(parent, fits=10, misses=30)
        child = _node(None, [("lead_job_title", "founder"), ("lead_job_title", "cannabis")],
                      parent=parent)
        _found(child, fits=80, misses=20)
        scores = _scores()

        assert select.estimate(child, scores) > 0.7

    def test_poor_people_pull_a_child_below_its_parent(self, db):
        parent = _node(None, [("lead_job_title", "founder")])
        _found(parent, fits=6, misses=6)
        bad = _node(None, [("lead_job_title", "founder"), ("lead_job_title", "sales")],
                    parent=parent)
        _found(bad, fits=0, misses=12)
        scores = _scores()

        assert select.estimate(bad, scores) < select.estimate(parent, scores)


class TestFrontier:
    def test_holds_unfired_children_and_fired_veins_together(self, db):
        # One pool: deepening a vein and opening a fresh node are two rows scored the
        # same way, not two policies needing an alternation rule.
        c = _campaign()
        fresh = _node(c, [("lead_job_title", "a")])
        vein = _node(c, [("lead_job_title", "b")], state=QueryNode.State.FIRED,
                     next_offset=100)
        _node(c, [("lead_job_title", "c")], state=QueryNode.State.DEAD)
        _node(c, [("lead_job_title", "d")], state=QueryNode.State.DRAINED)

        assert {n.pk for n in select.frontier()} == {fresh.pk, vein.pk}

    def test_next_node_is_none_when_nothing_is_fireable(self, db):
        c = _campaign()
        _node(c, [("lead_job_title", "a")], state=QueryNode.State.DEAD)
        assert select.next_node(_scores()) is None

    @pytest.mark.parametrize("qualified", [True, False])
    def test_a_single_class_store_can_still_be_drawn_from(self, db, qualified):
        # Reproduces a live crash: a campaign whose every verdict was a rejection gave
        # a level of 0, so `rng.beta(0, ...)` raised `ValueError: a <= 0` and killed the
        # find_email task. Both saturated directions must stay drawable.
        seen = _node(None, [("lead_job_title", "seen")], state=QueryNode.State.FIRED,
                     next_offset=100)
        _found(seen, fits=20 if qualified else 0, misses=0 if qualified else 20)
        _node(None, [("lead_job_title", "unseen")])

        assert select.next_node(_scores()) is not None

    def test_greedy_picks_the_best_estimate(self, db, monkeypatch):
        monkeypatch.setattr(select, "THOMPSON", False)
        bad = _node(None, [("lead_job_title", "bad")], state=QueryNode.State.FIRED,
                    next_offset=100)
        good = _node(None, [("lead_job_title", "good")], state=QueryNode.State.FIRED,
                     next_offset=100)
        _found(bad, fits=1, misses=9)
        _found(good, fits=8, misses=2)

        assert select.next_node(_scores()).pk == good.pk

    def test_thompson_still_favours_the_better_node_on_average(self, db):
        # A draw, not a shuffle: width tracks evidence, so a well-measured good node
        # wins most of the time without ever locking the frontier.
        bad = _node(None, [("lead_job_title", "bad")], state=QueryNode.State.FIRED,
                    next_offset=100)
        good = _node(None, [("lead_job_title", "good")], state=QueryNode.State.FIRED,
                     next_offset=100)
        _found(bad, fits=5, misses=45)
        _found(good, fits=40, misses=10)
        scores = _scores()

        rng = np.random.default_rng(0)
        wins = sum(select.next_node(scores, rng).pk == good.pk for _ in range(50))
        assert wins > 45


class TestExpansion:
    def test_children_are_the_node_plus_one_co_occurring_token(self, db):
        c = _campaign()
        _labelled(c, "founder cto ai", qualified=True)
        store = LabelStore.load()
        parent = _node(c, [("lead_job_title", "founder")])
        candidates = [("lead_job_title", t) for t in ("cto", "ai")]

        assert select.expand(parent, store, candidates) == 2
        children = QueryNode.objects.filter(parent=parent)
        assert {len(n.pairs) for n in children} == {2}

    def test_expansion_is_idempotent(self, db):
        c = _campaign()
        _labelled(c, "founder cto", qualified=True)
        store = LabelStore.load()
        parent = _node(c, [("lead_job_title", "founder")])
        candidates = [("lead_job_title", "cto")]

        select.expand(parent, store, candidates)
        assert select.expand(parent, store, candidates) == 0

    def test_a_field_holding_two_tokens_takes_no_third(self, db):
        """Same-field tokens are ANDed inside that field, so a third asks for a title
        carrying all three words — which nobody's does."""
        c = _campaign()
        _labelled(c, "founder cto ai", qualified=True)
        store = LabelStore.load()
        parent = _node(c, [("lead_job_title", "founder"), ("lead_job_title", "cto")])

        assert select.expand(parent, store, [("lead_job_title", "ai")]) == 0

    def test_a_capped_field_does_not_stop_another_axis(self, db):
        """The cap is per field: a node full on job title can still narrow on location."""
        c = _campaign()
        _labelled(c, "founder cto oman", qualified=True)
        store = LabelStore.load()
        parent = _node(c, [("lead_job_title", "founder"), ("lead_job_title", "cto")])

        assert select.expand(parent, store, [("lead_location", "Oman")]) == 1

    def test_a_closed_axis_never_holds_two_values(self, db):
        """Measured: `lead_seniority ["director founder"]` counts 0 where `["director"]`
        counts 5.0M, and the same for two places. A node that conjoined either would come
        back empty at offset 0 and prune its whole subtree over our own syntax."""
        c = _campaign()
        _labelled(c, "founder director cto oman spain", qualified=True)
        store = LabelStore.load()

        senior = _node(c, [("lead_seniority", "director")])
        assert select.expand(senior, store, [("lead_seniority", "founder")]) == 0

        place = _node(c, [("lead_location", "Oman")])
        assert select.expand(place, store, [("lead_location", "Spain")]) == 0

    def test_a_phrase_keyword_matches_on_its_words(self, db):
        """A location is one value carrying several words, and profiles are counted as a
        bag of words — so `California, United States` has to reach a Californian."""
        c = _campaign()
        _labelled(c, "founder california united states", qualified=True)
        store = LabelStore.load()
        parent = _node(c, [("lead_job_title", "founder")])

        assert store.counts([("lead_location", "California, United States")]) == (1, 0)
        assert select.expand(parent, store, [("lead_location", "California, United States")]) == 1
        assert select.expand(parent, store, [("lead_location", "Bavaria, Germany")]) == 0

    def test_a_dead_subset_prunes_the_child_before_it_is_created(self, db):
        # The anti-monotone half that survives the lattice being a DAG: a superset of an
        # empty conjunction is empty, whichever parent reaches it.
        c = _campaign()
        _labelled(c, "founder oman", qualified=True)
        store = LabelStore.load()
        _node(c, [("lead_location", "oman")], state=QueryNode.State.DEAD)
        parent = _node(c, [("lead_job_title", "founder")])

        assert select.expand(parent, store, [("lead_location", "oman")]) == 0

    def test_a_node_reached_twice_keeps_the_better_parent(self, db):
        c = _campaign()
        for _ in range(10):
            _labelled(c, "founder cto ai", qualified=True)
        for _ in range(10):
            _labelled(c, "cto agency", qualified=False)
        store = LabelStore.load()

        strong = _node(c, [("lead_job_title", "founder")])
        weak = _node(c, [("lead_job_title", "cto")])
        _found(strong, fits=9, misses=1)
        _found(weak, fits=1, misses=9)
        scores = _scores()
        select.expand(weak, store, [("lead_job_title", "founder")], scores)
        child = QueryNode.objects.get(
            token_key=token_key([("lead_job_title", "founder"), ("lead_job_title", "cto")]))
        assert child.parent_id == weak.pk

        select.expand(strong, store, [("lead_job_title", "cto")], scores)
        child.refresh_from_db()
        assert child.parent_id == strong.pk


class TestRetirement:
    def test_offset_zero_kills_the_node_and_its_subtree(self, db):
        c = _campaign()
        parent = _node(c, [("lead_job_title", "a")])
        child = _node(c, [("lead_job_title", "a"), ("lead_job_title", "b")], parent=parent)
        grandchild = _node(c, [("lead_job_title", "a"), ("lead_job_title", "c")], parent=child)

        assert select.retire(parent, at_offset=0) == "dead"
        for node in (parent, child, grandchild):
            node.refresh_from_db()
            assert node.state == QueryNode.State.DEAD

    def test_draining_below_the_cap_also_prunes_the_subtree(self, db):
        # Every match is already a Lead here, so a superset is drawn from an exhausted
        # population — there is nothing new below it.
        c = _campaign()
        parent = _node(c, [("lead_job_title", "a")], next_offset=400)
        child = _node(c, [("lead_job_title", "a"), ("lead_job_title", "b")], parent=parent)

        assert select.retire(parent, at_offset=400) == "drained"
        parent.refresh_from_db()
        child.refresh_from_db()
        assert parent.state == QueryNode.State.DRAINED
        assert child.state == QueryNode.State.DEAD

    def test_hitting_the_reach_cap_keeps_the_subtree_alive(self, db):
        # 10k is Elasticsearch's window, not the end of the population: adding a token
        # opens a fresh window over the part we could not reach.
        c = _campaign()
        parent = _node(c, [("lead_job_title", "a")], next_offset=REACH_CAP)
        child = _node(c, [("lead_job_title", "a"), ("lead_job_title", "b")], parent=parent)

        assert select.retire(parent, at_offset=REACH_CAP) == "capped"
        parent.refresh_from_db()
        child.refresh_from_db()
        assert parent.state == QueryNode.State.DRAINED
        assert child.state == QueryNode.State.FRONTIER

    def test_advance_moves_the_offset_and_keeps_the_node_fireable(self, db):
        c = _campaign()
        node = _node(c, [("lead_job_title", "a")])
        select.advance(node, leads_found=9027)
        node.refresh_from_db()
        assert node.next_offset == select.DISCOVERY_PAGE_SIZE
        assert node.state == QueryNode.State.FIRED
        assert node.leads_found == 9027


class TestCompanyAxis:
    """A company word is counted among the company's tags and always fires with a title
    (p1-e1-discovery-searches-the-buyers-company)."""

    @staticmethod
    def _store():
        # A cannabis extraction manager whose title never names the industry, and a
        # regulator whose title does — the two the old title-only seed confused.
        manager = _labelled(None, "head of extraction acme", qualified=True)
        manager.source_fields = {"contact_job_title": "head of extraction",
                                 "company_keywords": "cannabis, thc vape"}
        manager.save(update_fields=["source_fields"])
        _labelled(None, "cannabis compliance officer state of ohio", qualified=False)
        return LabelStore.load()

    def test_a_company_word_counts_the_tags_not_the_title(self, db):
        store = self._store()

        assert store.counts([("company_keywords", "cannabis")]) == (1, 0)
        assert store.counts([("lead_job_title", "cannabis")]) == (0, 1)
        assert store.counts([("company_keywords", "cannabis"),
                             ("lead_job_title", "extraction")]) == (1, 0)

    def test_a_title_node_grows_a_company_child(self, db):
        store = self._store()
        parent = _node(None, [("lead_job_title", "extraction")])

        assert select.expand(parent, store, [("company_keywords", "cannabis")]) == 1
        child = QueryNode.objects.get(parent=parent)
        assert set(child.pairs) == {("lead_job_title", "extraction"),
                                    ("company_keywords", "cannabis")}

    def test_a_node_with_no_title_grows_no_company_child(self, db):
        """It would page one employer's whole staff at a time."""
        _campaign(anchor_profiles=["head of extraction acme united states"])
        Lead.objects.filter(synthetic=True).update(
            source_fields={"company_keywords": "cannabis"})
        parent = _node(None, [("lead_location", "United States")])

        assert select.expand(parent, LabelStore.load(),
                             [("company_keywords", "cannabis")]) == 0

    def test_a_node_takes_one_company_word(self, db):
        """Two words in the field read as a phrase: "vape cannabis" counts 2."""
        _campaign(anchor_profiles=["head of extraction acme"])
        Lead.objects.filter(synthetic=True).update(
            source_fields={"company_keywords": "cannabis, vape"})
        parent = _node(None, [("lead_job_title", "extraction"),
                              ("company_keywords", "cannabis")])

        assert select.expand(parent, LabelStore.load(),
                             [("company_keywords", "vape")]) == 0

    def test_the_seed_opens_a_company_word_paired_with_each_title_word(self, db):
        keywords = [("company_keywords", "cannabis"), ("lead_job_title", "founder"),
                    ("lead_job_title", "purchasing"), ("lead_location", "United States")]
        select.seed_frontier(keywords, Seed(keywords, (2, 1000), "us"))

        opened = {frozenset(node.pairs) for node in QueryNode.objects.all()}
        assert frozenset({("company_keywords", "cannabis"),
                          ("lead_job_title", "purchasing")}) in opened
        assert frozenset({("company_keywords", "cannabis"),
                          ("lead_job_title", "founder")}) in opened
        assert frozenset({("company_keywords", "cannabis")}) not in opened
        assert len(opened) == 5

    def test_the_icp_place_rides_every_opening_node(self, db):
        """A US order's company + title nodes carried no country and surfaced Canadians."""
        keywords = [("company_keywords", "cannabis"), ("lead_job_title", "extraction"),
                    ("lead_location", "United States"), ("lead_location", "Canada")]
        select.seed_frontier(keywords, Seed(keywords, (2, 1000), "us", "United States"))

        opened = {frozenset(node.pairs) for node in QueryNode.objects.all()}
        us = ("lead_location", "United States")
        assert opened == {
            frozenset({("lead_job_title", "extraction"), us}),
            frozenset({("company_keywords", "cannabis"), ("lead_job_title", "extraction"), us}),
        }


class TestSeedFrontier:
    def test_opens_one_depth_one_node_per_keyword(self, db):
        # No root: the empty query matches everyone and its 10k window is the provider's
        # famous-company head, so the level comes from the label store instead.
        c = _campaign()
        keywords = [("lead_job_title", "founder"), ("lead_seniority", "founder")]
        seed = Seed(keywords, (10, 500), "us")
        assert select.seed_frontier(keywords, seed) == 2
        assert QueryNode.objects.filter(parent__isnull=True).count() == 2
        assert select.seed_frontier(keywords, seed) == 0

    def test_every_opened_node_carries_the_seed_s_band_and_country(self, db):
        # The band and the country are part of the query, not of the install: a node
        # opened by this seed keeps searching what this seed asked for even after a
        # later seed asks for something else.
        _campaign()
        seed = Seed([("lead_job_title", "founder")], (10, 500), "us")
        select.seed_frontier(seed.keywords, seed)

        node = QueryNode.objects.get()
        assert (node.headcount_min, node.headcount_max) == (10, 500)
        assert node.country_code == "us"
        assert node.to_filters()["company_headcount_min"] == 10

    def test_a_child_searches_its_parent_s_band(self, db):
        _campaign(anchor_profiles=["founder cto stealth"])
        parent = _node(None, [("lead_job_title", "founder")],
                       headcount_min=10, headcount_max=500, country_code="us")
        select.expand(parent, LabelStore.load(), [("lead_job_title", "cto")])

        child = QueryNode.objects.get(parent=parent)
        assert (child.headcount_min, child.headcount_max) == (10, 500)
        assert child.country_code == "us"
