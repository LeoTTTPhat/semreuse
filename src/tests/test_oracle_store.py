import numpy as np

from semreuse.oracle import SimulatedOracle
from semreuse.store import CachedView, PredicateStore


def test_oracle_exact_and_charged(corpus, universe):
    oracle = SimulatedOracle(corpus, noise=0.0)
    p = universe.by_name_prefix("leaf:baseball#")[0]
    rows = np.arange(corpus.n)
    res = oracle.evaluate(p, rows)
    assert (res == oracle.truth(p)).all()
    assert oracle.stats.calls == corpus.n


def test_oracle_noise_deterministic(corpus, universe):
    oracle = SimulatedOracle(corpus, noise=0.2, seed=7)
    p = universe.by_name_prefix("leaf:baseball#")[0]
    rows = np.arange(corpus.n)
    r1 = oracle.evaluate(p, rows)
    r2 = oracle.evaluate(p, rows[::-1])[::-1]
    assert (r1 == r2).all()  # same (predicate,row) -> same answer
    flip_rate = float((r1 != oracle.truth(p)).mean())
    assert 0.1 < flip_rate < 0.3


def test_oracle_noise_differs_across_predicates(corpus, universe):
    oracle = SimulatedOracle(corpus, noise=0.3, seed=7)
    p1, p2 = universe.by_name_prefix("leaf:baseball#")[:2]
    rows = np.arange(corpus.n)
    f1 = oracle.evaluate(p1, rows) != oracle.truth(p1)
    f2 = oracle.evaluate(p2, rows) != oracle.truth(p2)
    assert (f1 != f2).any()


def test_store_exact_match_normalization(corpus):
    store = PredicateStore(corpus.n)
    v = CachedView(pid="x", text="The document is about sports.",
                   reported=np.zeros(corpus.n, dtype=bool),
                   verified=np.ones(corpus.n, dtype=bool))
    store.add(v)
    assert store.find_exact("the document is about SPORTS") is v
    assert store.find_exact("the document is about hockey") is None
    assert store.memory_bytes() == 2 * corpus.n
