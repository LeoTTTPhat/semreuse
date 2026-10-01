import numpy as np
import pytest

from semreuse.audit import AuditConfig
from semreuse.baselines import (ColdEngine, EmbeddingCacheEngine,
                                ExactCacheEngine, HashingEmbedder)
from semreuse.corpus import make_synthetic_corpus
from semreuse.engine import EngineConfig, SemReuseEngine
from semreuse.entailment import GroundTruthEntailment
from semreuse.metrics import run_workload, summarize
from semreuse.oracle import SimulatedOracle
from semreuse.predicates import PredicateUniverse, generate_workload


@pytest.fixture(scope="module")
def corpus2k():
    """Large enough that audit overhead is small relative to savings."""
    return make_synthetic_corpus(n=2000, seed=1)


@pytest.fixture(scope="module")
def universe2k(corpus2k):
    return PredicateUniverse.build(corpus2k)


def _audit_cfg():
    return AuditConfig(alpha=0.05, target_recall=0.8)


def _engine(corpus, oracle, universe, **cfg_kw):
    ent = GroundTruthEntailment(universe.resolver())
    cfg_kw.setdefault("audit", _audit_cfg())
    return SemReuseEngine(corpus, oracle, ent, EngineConfig(**cfg_kw))


def test_cold_engine_exact(corpus, oracle, universe):
    eng = ColdEngine(corpus, oracle)
    p = universe.by_name_prefix("group:sports#")[0]
    res = eng.query(p)
    assert (res.reported == oracle.truth(p)).all()
    assert res.oracle_calls == corpus.n


def test_exact_cache_hits(corpus, oracle, universe):
    eng = ExactCacheEngine(corpus, oracle)
    p = universe.by_name_prefix("group:sports#")[0]
    r1 = eng.query(p)
    r2 = eng.query(p)
    assert r1.oracle_calls == corpus.n and r2.oracle_calls == 0
    assert (r1.reported == r2.reported).all()


def test_embedding_cache_reuses_similar(corpus, oracle, universe):
    eng = EmbeddingCacheEngine(corpus, oracle, HashingEmbedder(), theta=0.99)
    p = universe.by_name_prefix("group:sports#")[0]
    r1 = eng.query(p)
    r2 = eng.query(p)  # identical text => sim = 1.0 >= theta
    assert r1.oracle_calls == corpus.n and r2.oracle_calls == 0


def test_semreuse_superset_saves_calls(corpus2k, universe2k):
    corpus, universe = corpus2k, universe2k
    oracle = SimulatedOracle(corpus, noise=0.0, seed=0)
    eng = _engine(corpus, oracle, universe)
    group = universe.by_name_prefix("group:sports#")[0]
    leaf = universe.by_name_prefix("leaf:baseball#")[0]
    r1 = eng.query(group)
    assert r1.reuse_kind == "cold" and r1.oracle_calls == corpus.n
    r2 = eng.query(leaf)
    assert r2.reuse_kind == "rewrite"
    assert r2.oracle_calls < corpus.n  # candidates restricted to sports rows
    # Perfect entailment + noiseless oracle => exact result.
    assert (r2.reported == oracle.truth(leaf)).all()
    assert r2.recall_bound is not None and r2.recall_bound <= 1.0


def test_semreuse_equivalence_paraphrase(corpus2k, universe2k):
    corpus, universe = corpus2k, universe2k
    oracle = SimulatedOracle(corpus, noise=0.0, seed=0)
    eng = _engine(corpus, oracle, universe)
    p1, p2 = universe.by_name_prefix("leaf:baseball#")[:2]
    eng.query(p1)
    r2 = eng.query(p2)  # paraphrase: EQUIV, fully reusable
    assert r2.candidate_calls == 0
    # audit-only cost: samples from the two strata, no candidates
    assert r2.oracle_calls <= 0.3 * corpus.n
    assert (r2.reported == oracle.truth(p2)).all()


def test_semreuse_bound_valid_under_entailment_errors(corpus, universe):
    """Wrong entailment judgments must not silently break the guarantee."""
    oracle = SimulatedOracle(corpus, noise=0.0, seed=0)
    ent = GroundTruthEntailment(universe.resolver(), error_rate=0.3, seed=3)
    eng = SemReuseEngine(corpus, oracle, ent, EngineConfig(
        audit=AuditConfig(alpha=0.05, target_recall=0.9,
                          budget_per_stratum=20)))
    wl = generate_workload(universe, n_queries=20, overlap_rate=0.9, seed=5)
    ms = run_workload(eng, wl, oracle)
    s = summarize(ms)
    # With alpha=0.05 and 20 queries, expected violations < 1; allow 2.
    assert s["bound_violations"] <= 2
    # Escalation should keep realized recall near the target.
    assert s["macro_recall"] >= 0.9


def test_semreuse_saves_vs_cold_on_workload(corpus2k, universe2k):
    corpus, universe = corpus2k, universe2k
    oracle_a = SimulatedOracle(corpus, seed=0)
    oracle_b = SimulatedOracle(corpus, seed=0)
    wl = generate_workload(universe, n_queries=25, overlap_rate=0.8, seed=2)
    cold = ColdEngine(corpus, oracle_a)
    sem = _engine(corpus, oracle_b, universe)
    m_cold = summarize(run_workload(cold, wl, oracle_a))
    m_sem = summarize(run_workload(sem, wl, oracle_b))
    assert m_sem["total_oracle_calls"] < 0.6 * m_cold["total_oracle_calls"]
    assert m_sem["macro_recall"] >= 0.97
    assert m_sem["macro_precision"] >= 0.97


def test_store_lineage(corpus, oracle, universe):
    eng = _engine(corpus, oracle, universe)
    group = universe.by_name_prefix("group:sports#")[0]
    leaf = universe.by_name_prefix("leaf:hockey#")[0]
    eng.query(group)
    eng.query(leaf)
    assert len(eng.store) == 2
    v = eng.store.views[1]
    assert v.derived_from  # leaf view records its source views
    assert eng.store.views[0].pid in v.derived_from
    assert v.frac_verified > 0
