"""Experiment 7 (stage-2 addition): sensitivity of the NLI-driven system to
the rewrite confidence thresholds tau.

The exp5 ablation shows the NLI-vs-oracle-entailment gap is dominated by
lost disjoint elimination (and partly superset pruning).  Because the audit
layer guards recall regardless of judgment quality, lower taus can only
cost audit/escalation calls, not correctness -- so we sweep tau to see how
much of the gap aggressive trust recovers.

Usage:
    .venv/bin/python experiments/exp7_tau.py --scale full --corpus 20ng
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (base_parser, load_corpus_scaled,  # noqa: E402
                    make_nli_entailment, scale_params, write_results)

import pandas as pd  # noqa: E402

# (name, tau_equiv, tau_forward, tau_backward, tau_disjoint)
CONFIGS = [
    ("default", 0.8, 0.8, 0.9, 0.9),
    ("dis.7", 0.8, 0.8, 0.9, 0.7),
    ("dis.5", 0.8, 0.8, 0.9, 0.5),
    ("all.6", 0.6, 0.6, 0.6, 0.6),
    ("all.4", 0.4, 0.4, 0.4, 0.4),
]


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--target-recall", type=float, default=0.9)
    args = ap.parse_args()

    from semreuse.audit import AuditConfig
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.metrics import run_workload, summarize
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicates import PredicateUniverse, generate_workload
    from semreuse.rewriter import RewriteConfig

    corpus = load_corpus_scaled(args)
    _, n_queries = scale_params(args)
    universe = PredicateUniverse.build(corpus)
    wl = generate_workload(universe, n_queries=n_queries,
                           overlap_rate=args.overlap, seed=args.seed)
    nli = make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs)

    rows = []
    for name, te, tf, tb, td in CONFIGS:
        oracle = SimulatedOracle(corpus, seed=args.seed)
        eng = SemReuseEngine(corpus, oracle, nli, EngineConfig(
            rewrite=RewriteConfig(tau_equiv=te, tau_forward=tf,
                                  tau_backward=tb, tau_disjoint=td),
            audit=AuditConfig(alpha=0.05, target_recall=args.target_recall),
            seed=args.seed))
        ms = run_workload(eng, wl, oracle)
        s = summarize(ms)
        cold = corpus.n * n_queries
        rows.append(dict(
            config=name, tau_equiv=te, tau_forward=tf, tau_backward=tb,
            tau_disjoint=td,
            total_calls=s["total_oracle_calls"],
            call_reduction=cold / s["total_oracle_calls"],
            audit_calls=sum(m.audit_calls for m in ms),
            escalation_calls=sum(m.escalation_calls for m in ms),
            macro_precision=s["macro_precision"],
            macro_recall=s["macro_recall"],
            bound_violations=s["bound_violations"],
            n_bounded=s["n_bounded"],
            corpus=corpus.name, n_rows=corpus.n, n_queries=n_queries,
            overlap=args.overlap, target_recall=args.target_recall,
            seed=args.seed))
        r = rows[-1]
        print(f"[{name:8s}] reduction={r['call_reduction']:.2f}x "
              f"esc={r['escalation_calls']:7d} "
              f"P={r['macro_precision']:.3f} R={r['macro_recall']:.3f} "
              f"viol={r['bound_violations']}/{r['n_bounded']}")

    write_results(pd.DataFrame(rows), "exp7_tau", args)


if __name__ == "__main__":
    main()
