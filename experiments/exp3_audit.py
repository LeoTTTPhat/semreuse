"""Experiment 3: audit budget vs guarantee tightness vs coverage (Fig. 6).

Stress-tests the audit layer with a *controlled* entailment error rate
(GroundTruthEntailment with injected wrong judgments) so we know exactly how
wrong the rewrites are, and sweeps:
  * audit budget fraction  x  entailment error rate  x  target recall

Reports per configuration (over many workload repetitions):
  * empirical bound coverage  (fraction of bounded queries whose realized
    recall >= reported bound; must be >= 1 - alpha),
  * realized recall, oracle calls, escalation frequency.

This isolates the paper's statistical claim from NLI model quality.

Usage:
    .venv/bin/python experiments/exp3_audit.py --scale small
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (base_parser, load_corpus_scaled,  # noqa: E402
                    scale_params, write_results)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--budgets", default="adaptive,0.02,0.05,0.1,0.2",
                    help="'adaptive' or audit sample fractions")
    ap.add_argument("--error-rates", default="0.0,0.1,0.3")
    ap.add_argument("--targets", default="0.8,0.9,0.95")
    ap.add_argument("--overlap", type=float, default=0.8)
    args = ap.parse_args()

    from semreuse.audit import AuditConfig
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.entailment import GroundTruthEntailment
    from semreuse.metrics import run_workload, summarize
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicates import PredicateUniverse, generate_workload

    corpus = load_corpus_scaled(args)
    _, n_queries = scale_params(args)
    universe = PredicateUniverse.build(corpus)
    print(f"[setup] corpus={corpus.name} n={corpus.n} "
          f"queries/workload={n_queries} repeats={args.repeats}")

    rows = []
    for budget_str in args.budgets.split(","):
        adaptive = budget_str == "adaptive"
        budget = None if adaptive else float(budget_str)
        for err in [float(x) for x in args.error_rates.split(",")]:
            for target in [float(x) for x in args.targets.split(",")]:
                viol = bounded = 0
                recalls, calls, escal = [], [], []
                for rep in range(args.repeats):
                    seed = args.seed * 1000 + rep
                    wl = generate_workload(universe, n_queries=n_queries,
                                           overlap_rate=args.overlap,
                                           seed=seed)
                    oracle = SimulatedOracle(corpus, seed=seed)
                    ent = GroundTruthEntailment(universe.resolver(),
                                                error_rate=err, seed=seed)
                    eng = SemReuseEngine(corpus, oracle, ent, EngineConfig(
                        audit=AuditConfig(
                            alpha=0.05, target_recall=target,
                            mode="adaptive" if adaptive else "fraction",
                            budget_fraction=budget),
                        seed=seed))
                    ms = run_workload(eng, wl, oracle)
                    s = summarize(ms)
                    viol += s["bound_violations"]
                    bounded += s["n_bounded"]
                    recalls.append(s["macro_recall"])
                    calls.append(s["total_oracle_calls"])
                    escal.append(sum(1 for m in ms
                                     if m.escalation_calls > 0))
                cold_calls = corpus.n * n_queries
                rows.append(dict(
                    budget=budget_str, entailment_error=err,
                    target_recall=target, alpha=0.05,
                    n_bounded=bounded, bound_violations=viol,
                    coverage=1 - viol / bounded if bounded else None,
                    mean_recall=float(np.mean(recalls)),
                    mean_calls=float(np.mean(calls)),
                    call_reduction=cold_calls / float(np.mean(calls)),
                    mean_escalated_queries=float(np.mean(escal)),
                    corpus=corpus.name, n_rows=corpus.n,
                    n_queries=n_queries, repeats=args.repeats,
                    seed=args.seed))
                r = rows[-1]
                print(f"budget={budget_str:>8s} err={err:.2f} "
                      f"target={target:.2f} -> coverage="
                      f"{r['coverage']:.3f} recall={r['mean_recall']:.3f} "
                      f"reduction={r['call_reduction']:.1f}x "
                      f"escalated={r['mean_escalated_queries']:.1f}")

    write_results(pd.DataFrame(rows), "exp3_audit", args)


if __name__ == "__main__":
    main()
