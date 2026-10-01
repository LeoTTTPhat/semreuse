"""Experiment 16: certificate coverage and tightness curves.

Theorem 2 promises Pr[recall >= b_R and precision >= b_P] >= 1 - alpha for
every plan, however wrong the relation judgments were.  The workload runs
check this where realized recall is usually 1.0, which tests the bound only
loosely.  This experiment tests it where it is tight:

  primitives -- the exact non-coverage of the hypergeometric bounds, computed
                analytically over every true count T (no simulation);
  synthetic  -- controlled populations whose number of missed positives is
                placed relative to the certification boundary:
                lambda = misses / (G (1 - t) / t), so lambda = 1 means the
                unaudited plan has recall exactly t.  Each cell resamples the
                population and the audit R times and runs the shipped
                ``run_audit``; t in {0.8, 0.9, 0.95, 0.98} x alpha in
                {0.01, 0.05, 0.1}.  A precision sweep moves the assumed-positive
                stratum's precision across the demotion floor instead.
  replay     -- the real plans the engine built for a workload (the analyst
                log over a real LLM, or 20NG), re-audited with fresh randomness.

Reported: empirical violation rates against nominal alpha and against the
per-bound level the union bound assigns, and the gap between certified and
realized recall.

Usage:
    .venv/bin/python experiments/exp16_coverage.py --part primitives
    .venv/bin/python experiments/exp16_coverage.py --part synthetic --reps 20000
    .venv/bin/python experiments/exp16_coverage.py --part replay \
        --matrix data/llm_matrix/frozen_analyst_20ng_2000_llama3.1-8b.npz
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR  # noqa: E402

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402
from scipy.stats import beta as beta_dist, hypergeom  # noqa: E402

TARGETS = (0.8, 0.9, 0.95, 0.98)
ALPHAS = (0.01, 0.05, 0.1)
LAMBDAS = (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0)
PRECISIONS = (0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0)
DESIGNS = {"pooled": {}, "separate": {"pruned_pooling": "separate"}}

N_CAND, POS_CAND = 5000, 1000          # candidates: 1,000 positives found
SHAPES = {                             # (pruned stratum sizes, assumed-pos)
    "single": ((14500,), 0),
    "mixed": ((8000, 4000, 2000, 500), 500),
}


def _install_bound_cache() -> None:
    """Memoize the exact bounds: a cell asks for the same few (M, m, k, a)
    thousands of times.  Pure functions, so results are unchanged."""
    import semreuse.audit as au
    if not hasattr(au.hypergeom_upper_bound, "cache_info"):
        au.hypergeom_upper_bound = functools.lru_cache(maxsize=None)(
            au.hypergeom_upper_bound)
        au.hypergeom_lower_bound = functools.lru_cache(maxsize=None)(
            au.hypergeom_lower_bound)


class _ArrayOracle:
    class _Stats:
        def __init__(self):
            self.calls = 0

        def charge(self, tag, n):
            self.calls += n

    def __init__(self, truth):
        self.truth = truth
        self.stats = self._Stats()

    def evaluate(self, predicate, rows, tag="eval"):
        self.stats.charge(tag, len(rows))
        return self.truth[rows]


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = 1 - level
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------------------
# A. primitives: exact non-coverage, no simulation
# ---------------------------------------------------------------------------

def part_primitives() -> pd.DataFrame:
    from semreuse.audit import hypergeom_lower_bound, hypergeom_upper_bound
    rows = []
    for M, m in [(500, 30), (500, 100), (2000, 50), (2000, 400),
                 (14500, 400), (14500, 2000)]:
        for delta in ALPHAS:
            U = np.array([hypergeom_upper_bound(M, m, k, delta)
                          for k in range(m + 1)])
            L = np.array([hypergeom_lower_bound(M, m, k, delta)
                          for k in range(m + 1)])
            ks = np.arange(m + 1)
            worst_u = worst_l = 0.0
            for T in range(M + 1):
                pmf = hypergeom.pmf(ks, M, T, m)
                worst_u = max(worst_u, float(pmf[U < T].sum()))
                worst_l = max(worst_l, float(pmf[L > T].sum()))
            rows.append(dict(M=M, m=m, delta=delta,
                             max_noncoverage_upper=worst_u,
                             max_noncoverage_lower=worst_l))
            print(f"  M={M:6d} m={m:5d} delta={delta:.2f}: sup_T P[U<T]="
                  f"{worst_u:.4f}  sup_T P[L>T]={worst_l:.4f}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B. synthetic: controlled populations at the certification boundary
# ---------------------------------------------------------------------------

def build_population(shape: str, t: float, lam: float, pi: float,
                     rng: np.random.Generator):
    from semreuse.rewriter import RewritePlan, Stratum
    sizes, S = SHAPES[shape]
    n = N_CAND + S + sum(sizes)
    truth = np.zeros(n, dtype=bool)
    truth[:POS_CAND] = True
    pos_rows = np.arange(N_CAND, N_CAND + S)
    n_pos = int(round(pi * S))
    if S:
        truth[rng.choice(pos_rows, n_pos, replace=False)] = True
    # Misses placed relative to the allowance at the target: the plan's own
    # recall before any audit is t / (t + lam (1 - t)).
    misses = int(round(lam * (POS_CAND + n_pos) * (1 - t) / t))
    raw = np.array(sizes, dtype=float) * misses / sum(sizes)
    alloc = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - alloc))[: misses - alloc.sum()]:
        alloc[j] += 1
    start, strata = N_CAND + S, []
    for j, (size, tj) in enumerate(zip(sizes, alloc)):
        rows = np.arange(start, start + size)
        start += size
        if tj:
            truth[rng.choice(rows, int(tj), replace=False)] = True
        strata.append(Stratum(kind="pruned-superset", source_pid=f"q{j}",
                              source_text=f"q{j}", confidence=0.9, rows=rows))
    assumed = ([Stratum(kind="assumed-pos", source_pid="a", source_text="a",
                        confidence=0.9, rows=pos_rows)] if S else [])
    plan = RewritePlan(n_rows=n, candidates=np.arange(N_CAND),
                       assumed_pos=assumed, pruned=strata)
    return truth, plan


def _cell_seed(*key) -> int:
    return int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8],
                          "big")


def run_cell(spec: dict) -> dict:
    _install_bound_cache()
    from semreuse.audit import AuditConfig, run_audit
    sweep, shape, design = spec["sweep"], spec["shape"], spec["design"]
    t, alpha, lam, pi, reps = (spec["t"], spec["alpha"], spec["lam"],
                               spec["pi"], spec["reps"])
    cfg = AuditConfig(alpha=alpha, target_recall=t, **DESIGNS[design])
    seed = _cell_seed(sweep, shape, design, t, alpha, lam, pi)
    rng_pop = np.random.default_rng([seed, 1])
    rng_aud = np.random.default_rng([seed, 2])
    rv = pv = n_prec = certified_clean = 0
    b_r, r_real, gap, gap_clean, b_p, p_real, calls, r_pre = \
        [], [], [], [], [], [], [], []
    for _ in range(reps):
        truth, plan = build_population(shape, t, lam, pi, rng_pop)
        oracle = _ArrayOracle(truth)
        cand = truth[plan.candidates]
        res = run_audit(plan, None, oracle, cand, cfg, rng_aud)
        reported = plan.reported_mask(cand)
        reported[res.corrections_pos] = True
        reported[res.corrections_neg] = False
        n_true = int(truth.sum())
        tp = int((reported & truth).sum())
        rec = tp / n_true
        prec = tp / max(1, int(reported.sum()))
        pre = plan.reported_mask(cand)
        r_pre.append(int((pre & truth).sum()) / n_true)
        br = res.recall_lower_bound
        rv += rec < br - 1e-12
        b_r.append(br)
        r_real.append(rec)
        gap.append(rec - br)
        if res.escalation_calls == 0:
            certified_clean += 1
            gap_clean.append(rec - br)
        if res.precision_lower_bound is not None and plan.assumed_pos:
            n_prec += 1
            pv += prec < res.precision_lower_bound - 1e-12
            b_p.append(res.precision_lower_bound)
            p_real.append(prec)
        calls.append(res.audit_calls + res.escalation_calls)
    n_levels = len(SHAPES[shape][0]) + 1
    n_pools = 1 + int(SHAPES[shape][1] > 0)
    per_bound = (alpha / n_pools / n_levels if design == "pooled"
                 else alpha / n_pools / len(SHAPES[shape][0]))
    lo, hi = clopper_pearson(rv, reps)
    plo, phi = clopper_pearson(pv, max(1, n_prec))
    return dict(
        sweep=sweep, shape=shape, design=design, t=t, alpha=alpha, lam=lam,
        pi=pi, reps=reps, recall_pre_audit=float(np.mean(r_pre)),
        recall_violations=rv, recall_violation_rate=rv / reps,
        recall_viol_ci_lo=lo, recall_viol_ci_hi=hi,
        per_bound_level=per_bound,
        per_bound_assumed=(alpha / n_pools / max(1, int(SHAPES[shape][1] > 0))
                           if SHAPES[shape][1] else None),
        precision_certs=n_prec, precision_violations=pv,
        precision_violation_rate=pv / max(1, n_prec),
        precision_viol_ci_lo=plo, precision_viol_ci_hi=phi,
        mean_recall_bound=float(np.mean(b_r)),
        mean_recall_realized=float(np.mean(r_real)),
        mean_gap=float(np.mean(gap)),
        frac_certified_without_escalation=certified_clean / reps,
        mean_gap_certified_without_escalation=(float(np.mean(gap_clean))
                                               if gap_clean else None),
        mean_precision_bound=float(np.mean(b_p)) if b_p else None,
        mean_precision_realized=float(np.mean(p_real)) if p_real else None,
        mean_audit_plus_escalation_calls=float(np.mean(calls)))


def part_synthetic(reps: int, workers: int) -> pd.DataFrame:
    specs = []
    for t, alpha in itertools.product(TARGETS, ALPHAS):
        for lam in LAMBDAS:                       # recall sweep
            specs.append(dict(sweep="recall", shape="single",
                              design="pooled", t=t, alpha=alpha, lam=lam,
                              pi=1.0, reps=reps))
            for design in DESIGNS:
                specs.append(dict(sweep="recall", shape="mixed",
                                  design=design, t=t, alpha=alpha, lam=lam,
                                  pi=0.95, reps=reps))
        for pi in PRECISIONS:                     # precision sweep
            specs.append(dict(sweep="precision", shape="mixed",
                              design="pooled", t=t, alpha=alpha, lam=0.5,
                              pi=pi, reps=reps))
    print(f"[synthetic] {len(specs)} cells x {reps} replicates on "
          f"{workers} workers", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(run_cell, specs, chunksize=1)):
            out.append(row)
            if i % 20 == 0 or row["recall_violation_rate"] > row["alpha"]:
                print(f"  [{i+1}/{len(specs)}] {row['sweep']:9s} "
                      f"{row['shape']:6s} {row['design']:8s} t={row['t']} "
                      f"a={row['alpha']} lam={row['lam']} pi={row['pi']}: "
                      f"Rviol={row['recall_violation_rate']:.4f} "
                      f"Pviol={row['precision_violation_rate']:.4f} "
                      f"gap={row['mean_gap']:.4f}", flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# C. replay: real plans, fresh audit randomness
# ---------------------------------------------------------------------------

def capture_plans(args):
    """Run the shipped engine once and keep every rewritten query's plan."""
    from common import make_nli_entailment
    from semreuse.audit import AuditConfig
    from semreuse.engine import EngineConfig, SemReuseEngine

    nli = make_nli_entailment(seed=0, n_pairs=300)
    if args.matrix:
        from semreuse.corpus import Corpus, load_corpus
        from semreuse.llm_oracle import LLMOracle, LLMResponseMatrix
        from semreuse.predicate_log import analyst_log_predicates
        mat = LLMResponseMatrix.load(args.matrix)
        n = args.n_rows or mat.answers.shape[1]
        base = load_corpus("20ng", str(DATA_DIR), size=4000, seed=0)
        corpus = Corpus(name=f"20ng-pre{n}", docs=base.docs[:n],
                        leaf_labels=base.leaf_labels[:n],
                        taxonomy=dict(base.taxonomy))
        mat = LLMResponseMatrix(model=mat.model, corpus_name=mat.corpus_name,
                                texts=mat.texts, answers=mat.answers[:, :n],
                                meta=mat.meta)
        oracle = LLMOracle(corpus, matrix=mat)
        queries = [q for q in analyst_log_predicates()
                   if q.text in set(mat.texts)]
        label = f"{mat.model}-N{n}"
    else:
        from semreuse.corpus import load_corpus
        from semreuse.oracle import SimulatedOracle
        from semreuse.predicates import PredicateUniverse, generate_workload
        corpus = load_corpus("20ng", str(DATA_DIR), size=8000, seed=0)
        uni = PredicateUniverse.build(corpus)
        queries = generate_workload(uni, n_queries=120, overlap_rate=0.8,
                                    seed=0).queries
        oracle = SimulatedOracle(corpus, seed=0)
        label = f"20ng-sim-N{corpus.n}"
    eng = SemReuseEngine(corpus, oracle, nli, EngineConfig(
        audit=AuditConfig(alpha=0.05, target_recall=0.9,
                          budget_fraction=0.1), seed=0))
    plans = []
    for q in queries:
        res = eng.query(q)
        if res.reuse_kind != "rewrite" or res.plan is None:
            continue
        p = res.plan
        if p.n_pruned == 0 and p.n_assumed_pos == 0:
            continue
        truth = oracle.reference(q)
        plans.append((q.name, p, truth))
    return label, plans


def _replay_worker(job):
    _install_bound_cache()
    from semreuse.audit import AuditConfig, run_audit
    name, plan, truth, t, alpha, reps, seed = job
    cfg = AuditConfig(alpha=alpha, target_recall=t, budget_fraction=0.1)
    oracle = _ArrayOracle(truth)
    cand = truth[plan.candidates]
    rng = np.random.default_rng(seed)
    n_true = max(1, int(truth.sum()))
    rv = pv = npc = 0
    gaps, bounds, reals, calls = [], [], [], []
    for _ in range(reps):
        res = run_audit(plan, None, oracle, cand, cfg, rng)
        rep = plan.reported_mask(cand)
        rep[res.corrections_pos] = True
        rep[res.corrections_neg] = False
        tp = int((rep & truth).sum())
        rec = tp / n_true if truth.any() else 1.0
        prec = tp / max(1, int(rep.sum()))
        rv += rec < res.recall_lower_bound - 1e-12
        if res.precision_lower_bound is not None:
            npc += 1
            pv += prec < res.precision_lower_bound - 1e-12
        gaps.append(rec - res.recall_lower_bound)
        bounds.append(res.recall_lower_bound)
        reals.append(rec)
        calls.append(res.audit_calls + res.escalation_calls)
    return dict(query=name, t=t, alpha=alpha, reps=reps,
                recall_violations=rv, precision_certs=npc,
                precision_violations=pv, mean_recall_bound=np.mean(bounds),
                mean_recall_realized=np.mean(reals), mean_gap=np.mean(gaps),
                mean_audit_plus_escalation=np.mean(calls))


def part_replay(args) -> pd.DataFrame:
    label, plans = capture_plans(args)
    print(f"[replay] {label}: {len(plans)} rewritten plans", flush=True)
    jobs = [(name, plan, truth, t, alpha, args.reps,
             _cell_seed("replay", label, name, t, alpha))
            for (name, plan, truth) in plans
            for t, alpha in itertools.product(TARGETS, ALPHAS)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_replay_worker, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    df.insert(0, "workload", label)
    agg = (df.groupby(["t", "alpha"])
           .agg(audits=("reps", "sum"),
                recall_violations=("recall_violations", "sum"),
                precision_certs=("precision_certs", "sum"),
                precision_violations=("precision_violations", "sum"),
                mean_gap=("mean_gap", "mean"),
                mean_recall_bound=("mean_recall_bound", "mean"))
           .reset_index())
    for _, r in agg.iterrows():
        print(f"  t={r.t} alpha={r.alpha}: recall viol "
              f"{int(r.recall_violations)}/{int(r.audits)} "
              f"precision viol {int(r.precision_violations)}/"
              f"{int(r.precision_certs)} mean gap {r.mean_gap:.4f}",
              flush=True)
    return df


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["primitives", "synthetic", "replay"],
                    required=True)
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--workers", type=int,
                    default=max(1, min(16, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--matrix", default="",
                    help="replay: LLM matrix (empty = 20NG simulated)")
    ap.add_argument("--n-rows", type=int, default=0)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    tag = f"_{args.out_tag}" if args.out_tag else ""
    if args.part == "primitives":
        df = part_primitives()
    elif args.part == "synthetic":
        df = part_synthetic(args.reps, args.workers)
    else:
        df = part_replay(args)
    out = RESULTS_DIR / f"exp16_{args.part}{tag}.csv"
    df.to_csv(out, index=False)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
