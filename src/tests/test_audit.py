import numpy as np
import pytest
from scipy.stats import hypergeom

from semreuse.audit import (AuditConfig, hypergeom_lower_bound,
                            hypergeom_upper_bound, run_audit)
from semreuse.rewriter import RewritePlan, Stratum


def test_hypergeom_bounds_exact_when_fully_sampled():
    assert hypergeom_upper_bound(50, 50, 7, 0.05) == 7
    assert hypergeom_lower_bound(50, 50, 7, 0.05) == 7


def test_hypergeom_bounds_bracket_truth():
    # U bounds the population total from above, L from below; both are
    # totals in [k, k + (N-m)] projected sensibly, with L <= U, and the
    # interval shrinks as the sample grows.
    N, alpha = 200, 0.05
    for m in (10, 50, 150):
        for k in (0, 3, m // 2):
            if k > m:
                continue
            U = hypergeom_upper_bound(N, m, k, alpha)
            L = hypergeom_lower_bound(N, m, k, alpha)
            assert 0 <= L <= U <= N
            assert U >= k              # observed positives are a floor
            assert L <= k + (N - m)    # cannot exceed k + unsampled
    assert hypergeom_upper_bound(N, 150, 3, alpha) < \
        hypergeom_upper_bound(N, 10, 3, alpha)


def test_hypergeom_lower_bound_coverage_simulated():
    rng = np.random.default_rng(1)
    N, T, m, alpha = 120, 40, 30, 0.10
    misses = 0
    trials = 400
    pop = np.zeros(N, dtype=bool)
    pop[:T] = True
    for _ in range(trials):
        sample = rng.choice(N, size=m, replace=False)
        k = int(pop[sample].sum())
        if hypergeom_lower_bound(N, m, k, alpha) > T:
            misses += 1
    assert misses / trials <= alpha + 0.03


def test_hypergeom_upper_bound_coverage_simulated():
    """Empirical coverage of the one-sided UCB >= 1 - alpha."""
    rng = np.random.default_rng(0)
    N, T, m, alpha = 120, 18, 30, 0.10
    misses = 0
    trials = 400
    pop = np.zeros(N, dtype=bool)
    pop[:T] = True
    for _ in range(trials):
        sample = rng.choice(N, size=m, replace=False)
        k = int(pop[sample].sum())
        if hypergeom_upper_bound(N, m, k, alpha) < T:
            misses += 1
    assert misses / trials <= alpha + 0.03  # slack for MC noise


def test_hypergeom_bound_definition_consistency():
    # U is the largest T with cdf(k) > alpha; check boundary property.
    N, m, k, alpha = 100, 20, 2, 0.05
    U = hypergeom_upper_bound(N, m, k, alpha)
    assert hypergeom.cdf(k, N, U, m) > alpha
    if U < N - (m - k):
        assert hypergeom.cdf(k, N, U + 1, m) <= alpha


def _mk_plan(n, candidates, pruned_rows, pos_rows):
    strata_p = [Stratum(kind="pruned-superset", source_pid="q1",
                        source_text="q1", confidence=0.9,
                        rows=np.asarray(pruned_rows))] if len(pruned_rows) else []
    strata_a = [Stratum(kind="assumed-pos", source_pid="q2", source_text="q2",
                        confidence=0.9,
                        rows=np.asarray(pos_rows))] if len(pos_rows) else []
    return RewritePlan(n_rows=n, candidates=np.asarray(candidates),
                       assumed_pos=strata_a, pruned=strata_p)


class _ArrayOracle:
    """Oracle over an explicit truth array, with call accounting."""

    class _Stats:
        def __init__(self):
            self.calls = 0

        def charge(self, tag, n):
            self.calls += n

    def __init__(self, truth):
        self.truth_arr = np.asarray(truth, dtype=bool)
        self.stats = self._Stats()

    def evaluate(self, predicate, rows, tag="eval"):
        rows = np.asarray(rows, dtype=np.int64)
        self.stats.charge(tag, len(rows))
        return self.truth_arr[rows]


def test_audit_corrections_and_bound(rng):
    n = 400
    truth = np.zeros(n, dtype=bool)
    truth[:80] = True  # positives are rows 0..79
    # Prune rows 60..199 (contains 20 true positives -> recall errors exist).
    pruned = np.arange(60, 200)
    candidates = np.concatenate([np.arange(0, 60), np.arange(200, 400)])
    plan = _mk_plan(n, candidates, pruned, [])
    oracle = _ArrayOracle(truth)
    cand_res = oracle.evaluate(None, candidates)
    cfg = AuditConfig(alpha=0.05, target_recall=None, budget_per_stratum=40, budget_fraction=None, mode="fixed",
                      escalate=False)
    res = run_audit(plan, None, oracle, cand_res, cfg, rng)
    # Bound is a valid lower bound on the true recall of the corrected output.
    reported = np.zeros(n, dtype=bool)
    reported[candidates] = cand_res
    reported[res.corrections_pos] = True
    tp = int((reported & truth).sum())
    fn = int((~reported & truth).sum())
    true_recall = tp / (tp + fn)
    assert res.recall_lower_bound <= true_recall + 1e-9
    assert res.audit_calls == 40
    # All corrections are genuine positives from the pruned stratum.
    assert truth[res.corrections_pos].all()


def test_audit_escalation_meets_target(rng):
    n = 300
    truth = np.zeros(n, dtype=bool)
    truth[:150] = True
    # Terrible pruning: half the positives pruned.
    pruned = np.arange(75, 225)
    candidates = np.concatenate([np.arange(0, 75), np.arange(225, 300)])
    plan = _mk_plan(n, candidates, pruned, [])
    oracle = _ArrayOracle(truth)
    cand_res = oracle.evaluate(None, candidates)
    cfg = AuditConfig(alpha=0.05, target_recall=0.98, budget_per_stratum=20, budget_fraction=None, mode="fixed",
                      escalate=True)
    res = run_audit(plan, None, oracle, cand_res, cfg, rng)
    assert res.recall_lower_bound >= 0.98
    # Escalation fully evaluated the stratum -> all pruned positives found.
    reported = np.zeros(n, dtype=bool)
    reported[candidates] = cand_res
    reported[res.corrections_pos] = True
    assert (reported == truth).all()
    # Cost accounting: audit + escalation covers the stratum exactly once.
    assert res.audit_calls + res.escalation_calls == len(pruned)


def test_audit_assumed_pos_lower_bound(rng):
    n = 200
    truth = np.zeros(n, dtype=bool)
    truth[:100] = True
    # Assume rows 0..99 positive (all correct), evaluate the rest.
    pos_rows = np.arange(0, 100)
    candidates = np.arange(100, 200)
    plan = _mk_plan(n, candidates, [], pos_rows)
    oracle = _ArrayOracle(truth)
    cand_res = oracle.evaluate(None, candidates)
    cfg = AuditConfig(alpha=0.05, target_recall=None, budget_per_stratum=30, budget_fraction=None, mode="fixed",
                      escalate=False)
    res = run_audit(plan, None, oracle, cand_res, cfg, rng)
    # No pruning -> no possible misses -> bound == 1.
    assert res.recall_lower_bound == 1.0
    assert len(res.corrections_neg) == 0


def test_small_stratum_fully_audited(rng):
    n = 50
    truth = np.zeros(n, dtype=bool)
    truth[:10] = True
    pruned = np.arange(8, 11)  # tiny stratum, below min_stratum_audit
    candidates = np.concatenate([np.arange(0, 8), np.arange(11, 50)])
    plan = _mk_plan(n, candidates, pruned, [])
    oracle = _ArrayOracle(truth)
    cand_res = oracle.evaluate(None, candidates)
    cfg = AuditConfig(alpha=0.05, target_recall=None, budget_per_stratum=25, budget_fraction=None, mode="fixed",
                      min_stratum_audit=5, escalate=False)
    res = run_audit(plan, None, oracle, cand_res, cfg, rng)
    # Fully audited stratum: exact, so bound reflects zero residual misses.
    assert res.recall_lower_bound == 1.0
    assert set(res.corrections_pos.tolist()) == {8, 9}


# ---------------------------------------------------------------------------
# Audit-design ablations (exp15): every variant must publish a valid bound.
# ---------------------------------------------------------------------------

def _multi_pruned_plan(n, candidates, pruned_groups, pos_rows=()):
    pruned = [Stratum(kind="pruned-superset", source_pid=f"q{i}",
                      source_text=f"q{i}", confidence=0.9,
                      rows=np.asarray(g, dtype=np.int64))
              for i, g in enumerate(pruned_groups)]
    pos = ([Stratum(kind="assumed-pos", source_pid="qa", source_text="qa",
                    confidence=0.9, rows=np.asarray(pos_rows, dtype=np.int64))]
           if len(pos_rows) else [])
    return RewritePlan(n_rows=n, candidates=np.asarray(candidates,
                                                       dtype=np.int64),
                       assumed_pos=pos, pruned=pruned)


def _realized(n, truth, candidates, cand_res, res, pos_rows=()):
    reported = np.zeros(n, dtype=bool)
    reported[candidates] = cand_res
    reported[np.asarray(pos_rows, dtype=np.int64)] = True
    reported[res.corrections_pos] = True
    reported[res.corrections_neg] = False
    tp = int((reported & truth).sum())
    rec = tp / max(1, int(truth.sum()))
    prec = tp / max(1, int(reported.sum()))
    return rec, prec


def test_separate_pooling_bound_is_valid_and_meets_target():
    n = 6000
    truth = np.zeros(n, dtype=bool)
    truth[:1000] = True                      # 1000 positives
    candidates = np.arange(0, 900)           # finds 900 of them
    groups = [np.arange(900, 2900), np.arange(2900, 4400),
              np.arange(4400, 6000)]         # 100 misses, all in group 0
    plan = _multi_pruned_plan(n, candidates, groups)
    oracle = _ArrayOracle(truth)
    viol = 0
    for seed in range(60):
        cand_res = oracle.evaluate(None, candidates)
        res = run_audit(plan, None, oracle, cand_res,
                        AuditConfig(target_recall=0.9,
                                    pruned_pooling="separate"),
                        np.random.default_rng(seed))
        rec, _ = _realized(n, truth, candidates, cand_res, res)
        assert res.recall_lower_bound >= 0.9
        viol += rec < res.recall_lower_bound
    assert viol <= 6                         # alpha = 0.05 over 60 runs


def test_separate_pooling_costs_more_on_equal_clean_strata():
    """J equal clean strata: one pooled sample vs. one sample per stratum."""
    n = 5000
    truth = np.zeros(n, dtype=bool)
    truth[:500] = True
    candidates = np.arange(0, 1000)          # all positives found
    groups = [np.arange(1000 + 1000 * i, 2000 + 1000 * i) for i in range(4)]
    plan = _multi_pruned_plan(n, candidates, groups)
    oracle = _ArrayOracle(truth)
    cand_res = oracle.evaluate(None, candidates)
    pooled = run_audit(plan, None, oracle, cand_res,
                       AuditConfig(target_recall=0.9),
                       np.random.default_rng(0))
    sep = run_audit(plan, None, oracle, cand_res,
                    AuditConfig(target_recall=0.9, pruned_pooling="separate"),
                    np.random.default_rng(0))
    assert pooled.recall_lower_bound >= 0.9 and sep.recall_lower_bound >= 0.9
    assert sep.audit_calls > 2 * pooled.audit_calls


def test_recall_trust_leaves_assumed_positives_unaudited():
    n = 2000
    truth = np.zeros(n, dtype=bool)
    truth[:300] = True
    pos_rows = np.concatenate([np.arange(0, 100), np.arange(1000, 1100)])
    candidates = np.setdiff1d(np.arange(0, 1000), pos_rows)
    plan = _multi_pruned_plan(n, candidates, [np.arange(1100, 2000)],
                              pos_rows)
    oracle = _ArrayOracle(truth)
    cand_res = oracle.evaluate(None, candidates)
    res = run_audit(plan, None, oracle, cand_res,
                    AuditConfig(target_recall=0.9, certify="recall-trust"),
                    np.random.default_rng(0))
    assert res.precision_lower_bound is None
    assert len(res.corrections_neg) == 0     # nothing on that side was read
    assert res.assumed_pos_lb == 0
    rec, prec = _realized(n, truth, candidates, cand_res, res, pos_rows)
    assert rec >= res.recall_lower_bound
    assert prec < 0.8                        # the unaudited half is wrong


def test_recall_only_never_demotes_but_joint_does():
    n = 1000
    truth = np.zeros(n, dtype=bool)
    truth[:300] = True
    pos_rows = np.concatenate([np.arange(0, 100), np.arange(500, 600)])
    candidates = np.setdiff1d(np.arange(n), pos_rows)
    plan = _multi_pruned_plan(n, candidates, [], pos_rows)
    oracle = _ArrayOracle(truth)
    cand_res = oracle.evaluate(None, candidates)
    joint = run_audit(plan, None, oracle, cand_res,
                      AuditConfig(target_recall=0.9),
                      np.random.default_rng(0))
    ronly = run_audit(plan, None, oracle, cand_res,
                      AuditConfig(target_recall=0.9, certify="recall"),
                      np.random.default_rng(0))
    assert joint.escalation_calls > 0 and ronly.escalation_calls == 0
    assert joint.audit_calls == ronly.audit_calls   # same samples
    assert joint.precision_lower_bound is not None
    assert ronly.precision_lower_bound is None
    _, p_joint = _realized(n, truth, candidates, cand_res, joint, pos_rows)
    _, p_ronly = _realized(n, truth, candidates, cand_res, ronly, pos_rows)
    assert p_joint == 1.0 and p_ronly < 0.8
    assert p_joint >= joint.precision_lower_bound
