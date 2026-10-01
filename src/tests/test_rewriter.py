import numpy as np

from semreuse.entailment import Judgment
from semreuse.predicates import Relation
from semreuse.rewriter import RewriteConfig, build_plan
from semreuse.store import CachedView


def _view(pid, reported):
    reported = np.asarray(reported, dtype=bool)
    return CachedView(pid=pid, text=pid, reported=reported,
                      verified=np.ones_like(reported))


def test_superset_pruning():
    n = 10
    q = _view("q", [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    plan = build_plan(n, [(q, Judgment(Relation.FORWARD, 0.95))],
                      RewriteConfig())
    assert set(plan.candidates.tolist()) == {0, 1, 2}
    assert plan.n_pruned == 7
    assert plan.pruned[0].kind == "pruned-superset"


def test_positive_union():
    n = 10
    q = _view("q", [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    plan = build_plan(n, [(q, Judgment(Relation.BACKWARD, 0.95))],
                      RewriteConfig())
    assert plan.n_assumed_pos == 2
    assert set(plan.candidates.tolist()) == set(range(2, 10))
    # reported mask: assumed positives forced True
    res = plan.reported_mask(np.zeros(len(plan.candidates), dtype=bool))
    assert res[0] and res[1] and not res[2:].any()


def test_disjoint_elimination():
    n = 10
    q = _view("q", [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    plan = build_plan(n, [(q, Judgment(Relation.DISJOINT, 0.95))],
                      RewriteConfig())
    assert plan.n_pruned == 4
    assert plan.pruned[0].kind == "pruned-disjoint"
    assert set(plan.candidates.tolist()) == set(range(4, 10))


def test_equivalence_decomposes_to_both_directions():
    n = 8
    q = _view("q", [1, 1, 1, 0, 0, 0, 0, 0])
    plan = build_plan(n, [(q, Judgment(Relation.EQUIV, 0.95))],
                      RewriteConfig())
    # Everything is either assumed positive (q's positives) or pruned.
    assert len(plan.candidates) == 0
    assert plan.n_assumed_pos == 3
    assert plan.n_pruned == 5


def test_multi_view_combination():
    n = 12
    sup = _view("sup", [1] * 8 + [0] * 4)          # p => sup
    pos = _view("pos", [1, 1] + [0] * 10)          # pos => p
    dis = _view("dis", [0] * 6 + [1, 1] + [0] * 4)  # disjoint
    plan = build_plan(n, [
        (sup, Judgment(Relation.FORWARD, 0.9)),
        (pos, Judgment(Relation.BACKWARD, 0.95)),
        (dis, Judgment(Relation.DISJOINT, 0.92)),
    ], RewriteConfig())
    # candidates: inside sup (0..7), minus assumed pos (0,1), minus disjoint (6,7)
    assert set(plan.candidates.tolist()) == {2, 3, 4, 5}
    assert plan.n_assumed_pos == 2
    assert plan.n_pruned == 6  # rows 8..11 (superset) + rows 6,7 (disjoint)


def test_conflict_rows_go_back_to_candidates():
    n = 6
    # pos claims rows 0,1 positive; disjoint claims rows 1,2 negative.
    pos = _view("pos", [1, 1, 0, 0, 0, 0])
    dis = _view("dis", [0, 1, 1, 0, 0, 0])
    plan = build_plan(n, [
        (pos, Judgment(Relation.BACKWARD, 0.95)),
        (dis, Judgment(Relation.DISJOINT, 0.95)),
    ], RewriteConfig())
    # Row 1 is contested -> candidate. Row 0 assumed pos, row 2 pruned.
    assert 1 in plan.candidates.tolist()
    assert plan.n_assumed_pos == 1
    assert plan.n_pruned == 1


def test_thresholds_filter_low_confidence():
    n = 6
    q = _view("q", [1, 1, 1, 0, 0, 0])
    cfg = RewriteConfig(tau_forward=0.8)
    plan = build_plan(n, [(q, Judgment(Relation.FORWARD, 0.5))], cfg)
    assert len(plan.candidates) == n  # judgment ignored
    assert plan.used_views == []


def test_rule_ablation_flags():
    n = 6
    q = _view("q", [1, 1, 1, 0, 0, 0])
    cfg = RewriteConfig(enable_superset=False)
    plan = build_plan(n, [(q, Judgment(Relation.FORWARD, 0.95))], cfg)
    assert len(plan.candidates) == n


def test_plan_deterministic():
    n = 30
    rng = np.random.default_rng(0)
    views = [(_view(f"v{i}", rng.random(n) < 0.4),
              Judgment(Relation.FORWARD if i % 2 else Relation.DISJOINT, 0.9 + i / 100))
             for i in range(5)]
    p1 = build_plan(n, views, RewriteConfig())
    p2 = build_plan(n, views, RewriteConfig())
    assert (p1.candidates == p2.candidates).all()
    assert [s.rows.tolist() for s in p1.pruned] == \
        [s.rows.tolist() for s in p2.pruned]
