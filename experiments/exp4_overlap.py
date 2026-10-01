"""Experiment 4: sensitivity to workload predicate overlap (paper Fig. 7).

The load-bearing empirical risk of the paper: how much overlap does a
workload need before SemReuse pays off?  Sweeps the workload generator's
overlap rate and reports call reduction + accuracy for SemReuse (NLI and
ground-truth entailment) vs the embedding cache.

Usage:
    .venv/bin/python experiments/exp4_overlap.py --scale small --corpus 20ng
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (base_parser, load_corpus_scaled,  # noqa: E402
                    make_nli_entailment, scale_params, write_results)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--overlaps", default="0.0,0.2,0.4,0.6,0.8,0.95")
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--methods", default="semreuse,semreuse-gt,embed@0.9")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from semreuse.audit import AuditConfig
    from semreuse.baselines import EmbeddingCacheEngine, SentenceEmbedder
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.entailment import GroundTruthEntailment
    from semreuse.metrics import run_workload, summarize
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicates import PredicateUniverse, generate_workload

    corpus = load_corpus_scaled(args)
    _, n_queries = scale_params(args)
    universe = PredicateUniverse.build(corpus)
    methods = args.methods.split(",")
    nli = make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs) \
        if "semreuse" in methods else None
    embedder = SentenceEmbedder() \
        if any(m.startswith("embed@") for m in methods) else None

    rows = []
    for overlap in [float(x) for x in args.overlaps.split(",")]:
        for method in methods:
            sums = []
            for rep in range(args.repeats):
                seed = args.seed * 1000 + rep
                wl = generate_workload(universe, n_queries=n_queries,
                                       overlap_rate=overlap, seed=seed)
                oracle = SimulatedOracle(corpus, seed=seed)
                audit_cfg = AuditConfig(alpha=0.05,
                                        target_recall=args.target_recall,
                                        budget_fraction=0.1)
                if method == "semreuse":
                    eng = SemReuseEngine(corpus, oracle, nli, EngineConfig(
                        audit=audit_cfg, seed=seed))
                elif method == "semreuse-gt":
                    ent = GroundTruthEntailment(universe.resolver())
                    eng = SemReuseEngine(corpus, oracle, ent, EngineConfig(
                        audit=audit_cfg, seed=seed))
                elif method.startswith("embed@"):
                    eng = EmbeddingCacheEngine(
                        corpus, oracle, embedder,
                        theta=float(method.split("@")[1]))
                else:
                    raise ValueError(method)
                sums.append(summarize(run_workload(eng, wl, oracle)))
            cold = corpus.n * n_queries
            mean_calls = float(np.mean([s["total_oracle_calls"]
                                        for s in sums]))
            rows.append(dict(
                overlap=overlap, method=method,
                mean_calls=mean_calls,
                call_reduction=cold / mean_calls if mean_calls else np.inf,
                macro_recall=float(np.mean([s["macro_recall"] for s in sums])),
                macro_precision=float(np.mean([s["macro_precision"]
                                               for s in sums])),
                bound_violations=sum(s["bound_violations"] for s in sums),
                n_bounded=sum(s["n_bounded"] for s in sums),
                corpus=corpus.name, n_rows=corpus.n, n_queries=n_queries,
                repeats=args.repeats, target_recall=args.target_recall,
                seed=args.seed))
            r = rows[-1]
            print(f"overlap={overlap:.2f} {method:12s} "
                  f"reduction={r['call_reduction']:.2f}x "
                  f"R={r['macro_recall']:.3f} P={r['macro_precision']:.3f}")

    write_results(pd.DataFrame(rows), "exp4_overlap", args)


if __name__ == "__main__":
    main()
