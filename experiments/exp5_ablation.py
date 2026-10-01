"""Experiment 5: rewrite-rule ablation (paper Table 3).

Contribution of each rewrite rule, cumulative and individual:
  superset-only, +positive-union, +disjoint (=full), and each rule alone.
Run with both NLI and ground-truth entailment to separate rule power from
reasoner quality.

Usage:
    .venv/bin/python experiments/exp5_ablation.py --scale small --corpus 20ng
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (base_parser, load_corpus_scaled,  # noqa: E402
                    make_nli_entailment, scale_params, write_results)

import pandas as pd  # noqa: E402

CONFIGS = {
    "superset-only":  dict(enable_superset=True, enable_positive_union=False,
                           enable_disjoint=False),
    "union-only":     dict(enable_superset=False, enable_positive_union=True,
                           enable_disjoint=False),
    "disjoint-only":  dict(enable_superset=False, enable_positive_union=False,
                           enable_disjoint=True),
    "superset+union": dict(enable_superset=True, enable_positive_union=True,
                           enable_disjoint=False),
    "full":           dict(enable_superset=True, enable_positive_union=True,
                           enable_disjoint=True),
    "none":           dict(enable_superset=False, enable_positive_union=False,
                           enable_disjoint=False),
}


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--entailments", default="nli,gt")
    args = ap.parse_args()

    from semreuse.audit import AuditConfig
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.entailment import GroundTruthEntailment
    from semreuse.metrics import run_workload, summarize
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicates import PredicateUniverse, generate_workload
    from semreuse.rewriter import RewriteConfig

    corpus = load_corpus_scaled(args)
    _, n_queries = scale_params(args)
    universe = PredicateUniverse.build(corpus)
    wl = generate_workload(universe, n_queries=n_queries,
                           overlap_rate=args.overlap, seed=args.seed)
    entailments = args.entailments.split(",")
    nli = make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs) if "nli" in entailments else None

    rows = []
    for ent_name in entailments:
        for cfg_name, flags in CONFIGS.items():
            oracle = SimulatedOracle(corpus, seed=args.seed)
            ent = (nli if ent_name == "nli"
                   else GroundTruthEntailment(universe.resolver()))
            eng = SemReuseEngine(corpus, oracle, ent, EngineConfig(
                rewrite=RewriteConfig(**flags),
                audit=AuditConfig(alpha=0.05,
                                  target_recall=args.target_recall,
                                  budget_fraction=0.1),
                seed=args.seed))
            s = summarize(run_workload(eng, wl, oracle))
            cold = corpus.n * n_queries
            rows.append(dict(
                entailment=ent_name, config=cfg_name,
                total_calls=s["total_oracle_calls"],
                call_reduction=cold / s["total_oracle_calls"]
                if s["total_oracle_calls"] else float("inf"),
                macro_recall=s["macro_recall"],
                macro_precision=s["macro_precision"],
                bound_violations=s["bound_violations"],
                n_bounded=s["n_bounded"], nli_scores=s["total_nli_scores"],
                corpus=corpus.name, n_rows=corpus.n, n_queries=n_queries,
                overlap=args.overlap, seed=args.seed))
            r = rows[-1]
            print(f"[{ent_name:3s}] {cfg_name:15s} "
                  f"reduction={r['call_reduction']:.2f}x "
                  f"R={r['macro_recall']:.3f} P={r['macro_precision']:.3f}")

    write_results(pd.DataFrame(rows), "exp5_ablation", args)


if __name__ == "__main__":
    main()
