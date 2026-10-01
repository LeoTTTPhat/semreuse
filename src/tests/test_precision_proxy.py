"""Tests for the precision certificate, the proxy cascade, and tier two."""

from __future__ import annotations

import numpy as np
import pytest

from semreuse.audit import AuditConfig, run_audit
from semreuse.corpus import make_synthetic_corpus
from semreuse.engine import EngineConfig, SemReuseEngine
from semreuse.entailment import GroundTruthEntailment, Judgment
from semreuse.oracle import SimulatedOracle
from semreuse.predicate_log import ExtensionalEntailment
from semreuse.predicates import (Predicate, PredicateUniverse, Relation,
                                 generate_workload)
from semreuse.proxy import (EmbeddingProxy, ProxyCascadeConfig,
                            ProxyCascadeEngine, choose_cutoff,
                            score_band_strata)
from semreuse.rewriter import RewritePlan, Stratum


# ---------------------------------------------------------------------------
# Precision certificate
# ---------------------------------------------------------------------------

def _plan_with_assumed(n, assumed_rows, candidates):
    return RewritePlan(
        n_rows=n, candidates=np.asarray(candidates, dtype=np.int64),
        assumed_pos=[Stratum(kind="assumed-pos", source_pid="v1",
                             source_text="q", confidence=0.9,
                             rows=np.asarray(assumed_rows, dtype=np.int64))],
        pruned=[], used_views=[])


def test_precision_bound_is_one_when_everything_verified():
    """No assumed-positive rows survive unverified => precision is exact."""
    corpus = make_synthetic_corpus(n=200, seed=0)
    uni = PredicateUniverse.build(corpus)
    pred = uni.by_name_prefix("leaf:baseball#")[0]
    oracle = SimulatedOracle(corpus, seed=0)
    plan = _plan_with_assumed(corpus.n, [], np.arange(corpus.n))
    res = run_audit(plan, pred, oracle, oracle.truth(pred),
                    AuditConfig(target_recall=0.9), np.random.default_rng(0))
    assert res.precision_lower_bound == 1.0
    assert res.unverified_reported == 0


def test_precision_bound_holds_on_a_dirty_stratum():
    """A stratum that is only half positive must not be certified as clean."""
    corpus = make_synthetic_corpus(n=400, seed=1)
    uni = PredicateUniverse.build(corpus)
    pred = uni.by_name_prefix("leaf:baseball#")[0]
    oracle = SimulatedOracle(corpus, seed=0)
    truth = oracle.truth(pred)
    pos = np.flatnonzero(truth)[:40]
    neg = np.flatnonzero(~truth)[:40]
    assumed = np.sort(np.concatenate([pos, neg]))     # 50% precision
    rest = np.setdiff1d(np.arange(corpus.n), assumed)
    plan = _plan_with_assumed(corpus.n, assumed, rest)
    res = run_audit(plan, pred, oracle, truth[rest],
                    AuditConfig(target_recall=0.9, precision_target=0.9),
                    np.random.default_rng(0))
    reported = plan.reported_mask(truth[rest])
    reported[res.corrections_pos] = True
    reported[res.corrections_neg] = False
    realized = (reported & truth).sum() / max(1, reported.sum())
    assert res.precision_lower_bound <= realized + 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_certificates_hold_end_to_end(seed):
    """Both published bounds must hold on every certified query."""
    corpus = make_synthetic_corpus(n=600, seed=seed)
    uni = PredicateUniverse.build(corpus)
    wl = generate_workload(uni, n_queries=12, overlap_rate=0.8, seed=seed)
    oracle = SimulatedOracle(corpus, seed=seed)
    ent = GroundTruthEntailment(uni.resolver(), error_rate=0.3, seed=seed)
    eng = SemReuseEngine(corpus, oracle, ent,
                         EngineConfig(audit=AuditConfig(target_recall=0.9),
                                      seed=seed))
    for pred in wl.queries:
        res = eng.query(pred)
        truth = oracle.truth(pred)
        tp = int((res.reported & truth).sum())
        rec = tp / max(1, int(truth.sum()))
        prec = tp / max(1, int(res.reported.sum()))
        if res.recall_bound is not None:
            assert rec >= res.recall_bound - 1e-9
        if res.precision_bound is not None:
            assert prec >= res.precision_bound - 1e-9


def test_precision_sample_size_is_independent_of_stratum_size():
    """Eq. (7): the precision-side sample must not grow with the stratum."""
    cfg = AuditConfig(precision_target=0.9)
    sizes = [10_000, 100_000, 1_000_000]
    ms = [cfg.adaptive_pos_sample_size(n, 0.01) for n in sizes]
    assert len(set(ms)) == 1, ms
    assert 20 <= ms[0] <= 200


# ---------------------------------------------------------------------------
# Proxy cascade and composition
# ---------------------------------------------------------------------------

def test_score_bands_partition_and_order():
    rows = np.arange(100)
    scores = np.linspace(1.0, 0.0, 100)
    strata = score_band_strata(rows, scores, 4, "pruned-proxy", "p")
    assert sum(s.size for s in strata) == 100
    assert len(np.unique(np.concatenate([s.rows for s in strata]))) == 100
    # Band 0 must hold the highest-scoring rows: escalation walks it first.
    assert scores[strata[0].rows].min() >= scores[strata[-1].rows].max()


def test_cutoff_declines_when_proxy_does_not_separate():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=500)                 # pure noise
    pilot = np.arange(200)
    answers = rng.random(200) < 0.3
    tau = choose_cutoff(scores, pilot, answers, target=0.9, rng=rng)
    assert tau is None


def test_cutoff_accepts_a_separating_proxy():
    rng = np.random.default_rng(0)
    truth = np.zeros(500, dtype=bool)
    truth[:100] = True
    scores = np.where(truth, 5.0, 0.0) + rng.normal(scale=0.1, size=500)
    pilot = np.arange(300)
    tau = choose_cutoff(scores, pilot, truth[pilot], target=0.9,
                        min_kept_recall=0.9, rng=rng)
    assert tau is not None and 0.5 < tau < 5.5


def test_proxy_cascade_certificates_hold():
    corpus = make_synthetic_corpus(n=500, seed=3)
    uni = PredicateUniverse.build(corpus)
    wl = generate_workload(uni, n_queries=6, overlap_rate=0.8, seed=3)
    oracle = SimulatedOracle(corpus, seed=3)
    from semreuse.baselines import HashingEmbedder
    proxy = EmbeddingProxy(corpus, HashingEmbedder(dim=64), mode="trained")
    eng = ProxyCascadeEngine(
        corpus, oracle, proxy,
        ProxyCascadeConfig(pilot_size=120, n_bands=4,
                           audit=AuditConfig(target_recall=0.9)), seed=3)
    for pred in wl.queries:
        res = eng.query(pred)
        truth = oracle.truth(pred)
        rec = int((res.reported & truth).sum()) / max(1, int(truth.sum()))
        if res.recall_bound is not None:
            assert rec >= res.recall_bound - 1e-9


def test_proxy_composition_never_breaks_the_certificate():
    corpus = make_synthetic_corpus(n=500, seed=4)
    uni = PredicateUniverse.build(corpus)
    wl = generate_workload(uni, n_queries=8, overlap_rate=0.8, seed=4)
    oracle = SimulatedOracle(corpus, seed=4)
    from semreuse.baselines import HashingEmbedder
    proxy = EmbeddingProxy(corpus, HashingEmbedder(dim=64))
    eng = SemReuseEngine(
        corpus, oracle, GroundTruthEntailment(uni.resolver(), error_rate=0.2),
        EngineConfig(audit=AuditConfig(target_recall=0.9), seed=4,
                     proxy_pilot=80), proxy=proxy)
    for pred in wl.queries:
        res = eng.query(pred)
        truth = oracle.truth(pred)
        rec = int((res.reported & truth).sum()) / max(1, int(truth.sum()))
        if res.recall_bound is not None:
            assert rec >= res.recall_bound - 1e-9


# ---------------------------------------------------------------------------
# Extensional relations
# ---------------------------------------------------------------------------

def test_extensional_relations_match_set_algebra_at_eps_zero():
    ext = {"a": np.array([1, 1, 0, 0], dtype=bool),
           "b": np.array([1, 1, 1, 0], dtype=bool),
           "c": np.array([0, 0, 0, 1], dtype=bool),
           "d": np.array([1, 1, 0, 0], dtype=bool),
           "e": np.array([0, 1, 1, 1], dtype=bool)}
    e = ExtensionalEntailment(lambda t: ext[t], eps=0.0)
    assert e.relation("a", "b") is Relation.FORWARD
    assert e.relation("b", "a") is Relation.BACKWARD
    assert e.relation("a", "c") is Relation.DISJOINT
    assert e.relation("a", "d") is Relation.EQUIV
    assert e.relation("b", "c") is Relation.DISJOINT      # {0,1,2} vs {3}
    assert e.relation("a", "e") is Relation.OVERLAP       # share row 1 only


def test_extensional_relations_tolerate_slack():
    """Real predicates nest only approximately; eps is what makes that usable."""
    P = np.zeros(1000, dtype=bool); P[:100] = True
    Q = np.zeros(1000, dtype=bool); Q[:200] = True
    P[999] = True                       # one row of p outside q (1% of P)
    ext = {"p": P, "q": Q}
    strict = ExtensionalEntailment(lambda t: ext[t], eps=0.0)
    slack = ExtensionalEntailment(lambda t: ext[t], eps=0.02)
    assert strict.relation("p", "q") is Relation.OVERLAP
    assert slack.relation("p", "q") is Relation.FORWARD


# ---------------------------------------------------------------------------
# Two-tier reasoner (arbiter stubbed: no network in unit tests)
# ---------------------------------------------------------------------------

class _StubArbiter:
    """Answers every pair DISJOINT; stands in for the LLM tier offline."""

    def __init__(self):
        from semreuse.arbiter import ArbiterStats
        from semreuse.entailment import RELATIONS
        self.stats = ArbiterStats()
        self.confidence = {r: 0.8 for r in RELATIONS}

    def relations(self, pairs):
        self.stats.calls += len(pairs)
        return [Relation.DISJOINT] * len(pairs)


class _StubTier1:
    """Tier one that is confident about one pair and unsure about the rest."""

    def __init__(self, confidences):
        from semreuse.entailment import EntailmentStats
        self.confidences = confidences
        self.stats = EntailmentStats()

    def judge_batch(self, p, qs):
        return [Judgment(Relation.FORWARD, c) for c in self.confidences]


def test_two_tier_escalates_only_unsure_pairs():
    from semreuse.arbiter import TwoTierEntailment

    tier1 = _StubTier1([0.95, 0.4, 0.6])
    arb = _StubArbiter()
    two = TwoTierEntailment(tier1, arb, escalate_below=0.7)
    qs = [Predicate(t, frozenset()) for t in ("a", "b", "c")]
    out = two.judge_batch(Predicate("p", frozenset()), qs)
    assert out[0].relation is Relation.FORWARD      # confident: untouched
    assert out[1].relation is Relation.DISJOINT     # escalated
    assert out[2].relation is Relation.DISJOINT     # escalated
    assert arb.stats.calls == 2
    assert two.stats.escalated == 2


def test_two_tier_respects_an_escalation_budget():
    from semreuse.arbiter import TwoTierEntailment

    tier1 = _StubTier1([0.1, 0.2, 0.3, 0.4])
    arb = _StubArbiter()
    two = TwoTierEntailment(tier1, arb, escalate_below=0.9,
                            max_escalations=2)
    qs = [Predicate(t, frozenset()) for t in "abcd"]
    two.judge_batch(Predicate("p", frozenset()), qs)
    assert arb.stats.calls == 2       # spent on the two least certain pairs
