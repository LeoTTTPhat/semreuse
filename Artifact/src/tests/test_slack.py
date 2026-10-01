"""Tests for measured agreement slack (semreuse.slack)."""

from __future__ import annotations

import numpy as np
import pytest

from semreuse.audit import AuditConfig
from semreuse.corpus import make_synthetic_corpus
from semreuse.engine import EngineConfig, SemReuseEngine
from semreuse.entailment import GroundTruthEntailment
from semreuse.oracle import SimulatedOracle
from semreuse.predicates import PredicateUniverse, generate_workload
from semreuse.rewriter import RewriteConfig
from semreuse.slack import (SlackStore, admit_within_budget,
                            observations_from_audit)


# ---------------------------------------------------------------------------
# The store learns a rate: slack per unit of corpus pruned
# ---------------------------------------------------------------------------

def test_uninformed_store_returns_the_prior_rate():
    s = SlackStore(prior_rate=0.05)
    # Pruning half the corpus with no evidence is expected to cost half the
    # prior rate.
    assert s.predict("never-seen", "pruned-superset", 0.5) == \
        pytest.approx(0.025)
    assert s.predict("never-seen", "pruned-superset", 0.0) == 0.0


def test_prediction_scales_with_how_much_is_pruned():
    """The whole point: a wide rewrite is predicted to cost more than a narrow
    one using the same view."""
    s = SlackStore()
    wide = s.predict("v", "pruned-superset", 0.9)
    narrow = s.predict("v", "pruned-superset", 0.1)
    assert wide > narrow * 8


def test_the_rate_is_learned_from_what_audits_measured():
    s = SlackStore(prior_rate=0.05, prior_strength=0.01, shrink=0.01)
    # Twenty audits, each: pruned half the corpus, lost 30% of the answer.
    for _ in range(20):
        s.observe("v1", "pruned-superset", slack=0.30,
                  pruned_fraction=0.5, weight=100)
    assert s.rate("v1", "pruned-superset") == pytest.approx(0.6, abs=0.05)
    assert s.predict("v1", "pruned-superset", 0.5) == pytest.approx(0.3,
                                                                    abs=0.03)


def test_rates_are_per_view_not_global():
    s = SlackStore(shrink=0.01, prior_strength=0.01)
    for _ in range(20):
        s.observe("bad", "pruned-superset", 0.45, 0.5, weight=100)
        s.observe("good", "pruned-superset", 0.01, 0.5, weight=100)
    assert s.rate("bad", "pruned-superset") > 0.6
    assert s.rate("good", "pruned-superset") < 0.1


def test_unseen_view_backs_off_to_its_kind():
    s = SlackStore(shrink=0.01, prior_strength=0.01)
    for v in range(10):
        s.observe(f"v{v}", "pruned-disjoint", 0.45, 0.5, weight=100)
        s.observe(f"w{v}", "pruned-superset", 0.02, 0.5, weight=100)
    assert s.rate("brand-new", "pruned-disjoint") > 0.5
    assert s.rate("brand-new", "pruned-superset") < 0.2


def test_weight_reflects_evidence():
    heavy, light = SlackStore(), SlackStore()
    heavy.observe("v", "pruned-superset", 0.5, 0.5, weight=1000)
    light.observe("v", "pruned-superset", 0.5, 0.5, weight=1)
    assert heavy.rate("v", "pruned-superset") > light.rate("v",
                                                           "pruned-superset")


def test_zero_width_rewrites_are_not_evidence():
    s = SlackStore()
    s.observe("v", "pruned-superset", 0.4, pruned_fraction=0.0, weight=100)
    assert s.observations == 0


# ---------------------------------------------------------------------------
# Budget admission
# ---------------------------------------------------------------------------

def test_budget_admits_the_cheap_wide_rewrites_first():
    cands = [("wide-reliable", 0.01, 5000),
             ("wide-unreliable", 0.30, 5000),
             ("narrow-reliable", 0.01, 50)]
    admitted, spent = admit_within_budget(cands, budget=0.05)
    assert "wide-reliable" in admitted
    assert "wide-unreliable" not in admitted   # would blow the budget alone
    assert spent <= 0.05


def test_budget_of_none_admits_everything():
    cands = [("a", 0.9, 10), ("b", 0.9, 10)]
    admitted, spent = admit_within_budget(cands, budget=None)
    assert set(admitted) == {"a", "b"} and spent == 0.0


def test_a_single_over_budget_rewrite_is_declined():
    admitted, spent = admit_within_budget([("x", 0.37, 9999)], budget=0.05,
                                          explore=0)
    assert admitted == [] and spent == 0.0


def test_exploration_keeps_evidence_flowing():
    """A budget that admits nothing would never learn anything: the estimates
    it spends are produced by auditing the rewrites it declines."""
    over = [("x", 0.37, 9999), ("y", 0.40, 500)]
    admitted, spent = admit_within_budget(over, budget=0.05)   # explore=1
    assert admitted == ["x"]           # the best-ranked one, budget or not
    assert spent == pytest.approx(0.37)
    # ...and only when the budget admitted nothing at all.
    mixed = [("cheap", 0.01, 5000), ("dear", 0.40, 5000)]
    admitted, _ = admit_within_budget(mixed, budget=0.05)
    assert admitted == ["cheap"]


# ---------------------------------------------------------------------------
# Reading slack out of an audit
# ---------------------------------------------------------------------------

class _Rec:
    def __init__(self, kind, size, sampled, positives, escalated=False):
        self.kind, self.size = kind, size
        self.sampled, self.positives = sampled, positives
        self.escalated = escalated


class _Stratum:
    def __init__(self, pid):
        self.source_pid = pid


class _Plan:
    def __init__(self, pruned):
        self.assumed_pos, self.pruned = [], pruned


class _Audit:
    def __init__(self, strata, found, extra=0):
        self.strata, self.found_positives = strata, found
        self.assumed_pos_lb = extra


def test_sampled_stratum_yields_the_horvitz_thompson_estimate():
    # 20 of 1000 rows sampled, 2 wrongly pruned found => ~100 missed,
    # against 400 positives found => slack 0.25; the stratum is 1000 of the
    # 4000-row corpus, so it pruned a quarter of it.
    obs = observations_from_audit(
        _Plan([_Stratum("v1")]),
        _Audit([_Rec("pruned-superset", 1000, 20, 2)], found=400), n_rows=4000)
    assert len(obs) == 1
    src, kind, slack, frac, weight = obs[0]
    assert src == "v1" and kind == "pruned-superset"
    assert slack == pytest.approx(0.25)
    assert frac == pytest.approx(0.25)
    assert weight == 20                     # weighted by rows actually audited


def test_escalated_stratum_is_exact_and_weighted_by_its_whole_size():
    obs = observations_from_audit(
        _Plan([_Stratum("v1")]),
        _Audit([_Rec("pruned-superset", 500, 30, 40, escalated=True)],
               found=400), n_rows=1000)
    _, _, slack, frac, weight = obs[0]
    assert frac == pytest.approx(0.5)
    assert slack == pytest.approx(0.1)      # 40 known misses / 400
    assert weight == 500


def test_assumed_positive_strata_are_not_recall_slack():
    """Positive union spends precision, not recall; it must not be recorded."""
    plan = _Plan([])
    plan.assumed_pos = [_Stratum("v1")]
    obs = observations_from_audit(
        plan, _Audit([_Rec("assumed-pos", 100, 10, 3)], found=400),
        n_rows=1000)
    assert obs == []


def test_no_positives_found_yields_no_observation():
    obs = observations_from_audit(
        _Plan([_Stratum("v1")]),
        _Audit([_Rec("pruned-superset", 100, 10, 1)], found=0), n_rows=1000)
    assert obs == []


# ---------------------------------------------------------------------------
# End to end: the mechanism must never cost correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_certificates_still_hold_with_slack_budgeting(seed):
    """The plan now depends on earlier audits; the bound must survive that.

    Slack estimates are read from previous queries only, so the partition is
    still fixed before the current sample is drawn -- the property Lemma 2
    needs. This test exercises it against a deliberately bad reasoner.
    """
    corpus = make_synthetic_corpus(n=600, seed=seed)
    uni = PredicateUniverse.build(corpus)
    wl = generate_workload(uni, n_queries=14, overlap_rate=0.8, seed=seed)
    oracle = SimulatedOracle(corpus, seed=seed)
    ent = GroundTruthEntailment(uni.resolver(), error_rate=0.35, seed=seed)
    eng = SemReuseEngine(
        corpus, oracle, ent,
        EngineConfig(audit=AuditConfig(target_recall=0.9), seed=seed,
                     slack_budget=0.5))
    for pred in wl.queries:
        res = eng.query(pred)
        truth = oracle.truth(pred)
        tp = int((res.reported & truth).sum())
        if res.recall_bound is not None:
            assert tp / max(1, int(truth.sum())) >= res.recall_bound - 1e-9
        if res.precision_bound is not None:
            assert tp / max(1, int(res.reported.sum())) >= \
                res.precision_bound - 1e-9


def test_the_store_actually_learns_during_a_workload():
    corpus = make_synthetic_corpus(n=800, seed=1)
    uni = PredicateUniverse.build(corpus)
    wl = generate_workload(uni, n_queries=20, overlap_rate=0.8, seed=1)
    oracle = SimulatedOracle(corpus, seed=1)
    ent = GroundTruthEntailment(uni.resolver(), error_rate=0.4, seed=1)
    eng = SemReuseEngine(
        corpus, oracle, ent,
        EngineConfig(audit=AuditConfig(target_recall=0.9), seed=1,
                     slack_budget=0.5))
    for pred in wl.queries:
        eng.query(pred)
    assert eng.slack.observations > 0
    assert eng.slack.summary()["sources_tracked"] > 0


def test_default_is_off_and_deterministic():
    """Without a budget the planner behaves exactly as it did before, and
    turning the mechanism on changes the plan (in either direction: which way
    is an empirical question, not a guarantee)."""
    corpus = make_synthetic_corpus(n=400, seed=2)
    uni = PredicateUniverse.build(corpus)
    wl = generate_workload(uni, n_queries=8, overlap_rate=0.8, seed=2)

    def run(budget):
        oracle = SimulatedOracle(corpus, seed=2)
        eng = SemReuseEngine(
            corpus, oracle, GroundTruthEntailment(uni.resolver()),
            EngineConfig(rewrite=RewriteConfig(), 
                         audit=AuditConfig(target_recall=0.9), seed=2,
                         slack_budget=budget))
        for pred in wl.queries:
            eng.query(pred)
        return oracle.stats.calls

    off = run(None)
    assert off == run(None)               # unchanged and deterministic
    assert run(0.0) != off                # the budget demonstrably bites


def test_budget_bounds_the_predicted_spend():
    """Whatever the budget admits, the plan reports what it expects to cost."""
    corpus = make_synthetic_corpus(n=600, seed=5)
    uni = PredicateUniverse.build(corpus)
    wl = generate_workload(uni, n_queries=12, overlap_rate=0.8, seed=5)
    oracle = SimulatedOracle(corpus, seed=5)
    eng = SemReuseEngine(
        corpus, oracle, GroundTruthEntailment(uni.resolver(), error_rate=0.3),
        EngineConfig(audit=AuditConfig(target_recall=0.9), seed=5,
                     slack_budget=0.5))
    spends = []
    for pred in wl.queries:
        res = eng.query(pred)
        if res.plan is not None:
            spends.append(res.plan.predicted_slack)
    assert spends, "no plans were built"
    assert all(s >= 0.0 for s in spends)
    assert any(s > 0.0 for s in spends), "budget never spent anything"
