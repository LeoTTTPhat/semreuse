import numpy as np

from semreuse.predicates import (PredicateUniverse, Relation, generate_workload,
                                 labeled_pairs_for_calibration, normalize_text,
                                 true_relation)


def test_true_relation_algebra():
    a = frozenset({"baseball"})
    sports = frozenset({"baseball", "hockey"})
    cars = frozenset({"cars"})
    mixed = frozenset({"baseball", "cars"})
    assert true_relation(a, sports) is Relation.FORWARD
    assert true_relation(sports, a) is Relation.BACKWARD
    assert true_relation(a, a) is Relation.EQUIV
    assert true_relation(a, cars) is Relation.DISJOINT
    assert true_relation(sports, mixed) is Relation.OVERLAP


def test_universe_has_containment_structure(universe):
    leafs = universe.by_name_prefix("leaf:baseball#")
    groups = universe.by_name_prefix("group:sports#")
    assert leafs and groups
    assert true_relation(leafs[0].label_set, groups[0].label_set) is Relation.FORWARD
    # paraphrases of the same node are EQUIV with different surface text
    assert leafs[0].text != leafs[1].text
    assert true_relation(leafs[0].label_set, leafs[1].label_set) is Relation.EQUIV


def test_extension_matches_labels(corpus, universe):
    p = universe.by_name_prefix("group:sports#")[0]
    ext = corpus.extension(p.label_set)
    expected = np.array([l in ("baseball", "hockey") for l in corpus.leaf_labels])
    assert (ext == expected).all()


def test_workload_deterministic(universe):
    w1 = generate_workload(universe, n_queries=25, overlap_rate=0.7, seed=42)
    w2 = generate_workload(universe, n_queries=25, overlap_rate=0.7, seed=42)
    assert [p.text for p in w1.queries] == [p.text for p in w2.queries]
    w3 = generate_workload(universe, n_queries=25, overlap_rate=0.7, seed=43)
    assert [p.text for p in w3.queries] != [p.text for p in w1.queries]


def test_workload_overlap_monotone():
    # Use a wide taxonomy so low-overlap workloads can actually avoid
    # containment relations (with 3 groups everything ends up related).
    from semreuse.corpus import make_synthetic_corpus

    corpus = make_synthetic_corpus(n=100, seed=0, n_groups=12)
    uni = PredicateUniverse.build(corpus, include_unions=False)

    def n_related(wl):
        """Queries with a containment/equivalence relation to a predecessor."""
        rel = 0
        chain_rels = (Relation.EQUIV, Relation.FORWARD, Relation.BACKWARD)
        for i, p in enumerate(wl.queries):
            for q in wl.queries[:i]:
                if true_relation(p.label_set, q.label_set) in chain_rels:
                    rel += 1
                    break
        return rel

    lo = generate_workload(uni, n_queries=12, overlap_rate=0.05, seed=7)
    hi = generate_workload(uni, n_queries=12, overlap_rate=0.95, seed=7)
    assert n_related(hi) > n_related(lo)


def test_labeled_pairs_cover_all_relations(universe):
    pairs = labeled_pairs_for_calibration(universe, n_pairs=100, seed=0)
    rels = {r for _, _, r in pairs}
    assert rels == set(Relation)
    for p, q, r in pairs:
        assert true_relation(p.label_set, q.label_set) is r


def test_normalize_text():
    assert normalize_text("The document is about Sports.") == \
        normalize_text("the document is about sports")
