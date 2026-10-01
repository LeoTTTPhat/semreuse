"""Experiment 6: corpus-size scaling of the audit overhead (paper Fig. 8).

The audit sample needed for a per-query recall certificate is O(1) in the
corpus size (m ~ ln(1/alpha) / (delta * selectivity)), while reuse savings
grow with N -- so call reduction improves with corpus size.  This experiment
sweeps N with a fixed workload shape and reports the cost split
(candidates / audit / escalation) per method.

Usage:
    .venv/bin/python experiments/exp6_scaling.py --corpus 20ng
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA_DIR, base_parser, make_nli_entailment,  # noqa: E402
                    write_results)

import pandas as pd  # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--sizes", default="500,1000,2000,4000,8000")
    ap.add_argument("--n-queries", type=int, default=40)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--methods", default="semreuse,semreuse-gt")
    args = ap.parse_args()

    from semreuse.audit import AuditConfig
    from semreuse.corpus import load_corpus
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.entailment import GroundTruthEntailment
    from semreuse.metrics import run_workload, summarize
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicates import PredicateUniverse, generate_workload

    methods = args.methods.split(",")
    nli = make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs) \
        if "semreuse" in methods else None

    rows = []
    for size in [int(x) for x in args.sizes.split(",")]:
        corpus = load_corpus(args.corpus, str(DATA_DIR), size=size,
                             seed=args.seed)
        universe = PredicateUniverse.build(corpus)
        wl = generate_workload(universe, n_queries=args.n_queries,
                               overlap_rate=args.overlap, seed=args.seed)
        for method in methods:
            oracle = SimulatedOracle(corpus, seed=args.seed)
            ent = (nli if method == "semreuse"
                   else GroundTruthEntailment(universe.resolver()))
            eng = SemReuseEngine(corpus, oracle, ent, EngineConfig(
                audit=AuditConfig(alpha=0.05,
                                  target_recall=args.target_recall),
                seed=args.seed))
            ms = run_workload(eng, wl, oracle)
            s = summarize(ms)
            cold = corpus.n * args.n_queries
            rows.append(dict(
                n_rows=corpus.n, method=method,
                total_calls=s["total_oracle_calls"],
                candidate_calls=sum(m.candidate_calls for m in ms),
                audit_calls=sum(m.audit_calls for m in ms),
                escalation_calls=sum(m.escalation_calls for m in ms),
                call_reduction=cold / s["total_oracle_calls"]
                if s["total_oracle_calls"] else float("inf"),
                macro_recall=s["macro_recall"],
                macro_precision=s["macro_precision"],
                bound_violations=s["bound_violations"],
                n_bounded=s["n_bounded"],
                corpus=corpus.name, n_queries=args.n_queries,
                overlap=args.overlap, target_recall=args.target_recall,
                seed=args.seed))
            r = rows[-1]
            print(f"N={corpus.n:6d} {method:12s} "
                  f"reduction={r['call_reduction']:.2f}x "
                  f"audit={r['audit_calls']:7d} "
                  f"R={r['macro_recall']:.3f}")

    write_results(pd.DataFrame(rows), "exp6_scaling", args)


if __name__ == "__main__":
    main()
