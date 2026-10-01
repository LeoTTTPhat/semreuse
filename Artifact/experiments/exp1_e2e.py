"""Experiment 1: end-to-end workload comparison (paper Fig. 4 / Table 2).

Methods:
  cold           -- LOTUS-style, no reuse
  exact          -- exact-match predicate cache
  embed@T        -- GPTCache-style embedding-similarity cache, threshold T
  semreuse       -- NLI entailment + rewrite + audit (the system)
  semreuse-noaudit -- rewrite without the audit layer (ablation)
  semreuse-gt    -- perfect entailment oracle (upper bound)

Metrics per query: oracle calls (candidates/audit/escalation), NLI scorings,
precision/recall/F1 vs ground truth, recall bound + violation flag, latency.

Usage:
    .venv/bin/python experiments/exp1_e2e.py --scale small --corpus 20ng
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Timer, base_parser, load_corpus_scaled,  # noqa: E402
                    make_nli_entailment, scale_params, write_results)

import pandas as pd  # noqa: E402


def build_engine(name: str, corpus, oracle, universe, args, target_recall,
                 nli_cache=None):
    from semreuse.audit import AuditConfig
    from semreuse.baselines import (ColdEngine, EmbeddingCacheEngine,
                                    ExactCacheEngine, SentenceEmbedder)
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.entailment import GroundTruthEntailment

    audit_cfg = AuditConfig(alpha=0.05, target_recall=target_recall,
                            budget_fraction=0.1)
    if name == "cold":
        return ColdEngine(corpus, oracle)
    if name == "exact":
        return ExactCacheEngine(corpus, oracle)
    if name.startswith("embed@"):
        theta = float(name.split("@")[1])
        return EmbeddingCacheEngine(corpus, oracle, SentenceEmbedder(),
                                    theta=theta)
    if name == "semreuse":
        ent = nli_cache if nli_cache is not None else \
            make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs)
        return SemReuseEngine(corpus, oracle, ent, EngineConfig(
            audit=audit_cfg, seed=args.seed))
    if name == "semreuse-noaudit":
        ent = nli_cache if nli_cache is not None else \
            make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs)
        return SemReuseEngine(corpus, oracle, ent, EngineConfig(
            audit=audit_cfg, enable_audit=False, seed=args.seed))
    if name == "semreuse-gt":
        ent = GroundTruthEntailment(universe.resolver())
        return SemReuseEngine(corpus, oracle, ent, EngineConfig(
            audit=audit_cfg, seed=args.seed))
    raise ValueError(name)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--overlap", type=float, default=0.7)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--oracle-noise", type=float, default=0.0)
    ap.add_argument("--methods", default=(
        "cold,exact,embed@0.8,embed@0.9,semreuse,semreuse-noaudit,semreuse-gt"))
    args = ap.parse_args()

    from semreuse.metrics import metrics_to_rows, run_workload, summarize
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicates import PredicateUniverse, generate_workload

    corpus = load_corpus_scaled(args)
    _, n_queries = scale_params(args)
    universe = PredicateUniverse.build(corpus)
    workload = generate_workload(universe, n_queries=n_queries,
                                 overlap_rate=args.overlap, seed=args.seed)
    print(f"[setup] corpus={corpus.name} n={corpus.n} "
          f"queries={len(workload.queries)} overlap={args.overlap}")

    # Fit the NLI reasoner once; share across semreuse variants (same scores).
    methods = args.methods.split(",")
    nli = None
    if any(m.startswith("semreuse") and m != "semreuse-gt" for m in methods):
        nli = make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs)

    all_rows, summary_rows = [], []
    for method in methods:
        oracle = SimulatedOracle(corpus, noise=args.oracle_noise,
                                 seed=args.seed)
        if nli is not None and method.startswith("semreuse") \
                and method != "semreuse-gt":
            nli.stats.judgments = 0  # fresh judgment counters per method
        eng = build_engine(method, corpus, oracle, universe, args,
                           args.target_recall, nli_cache=nli)
        with Timer() as t:
            ms = run_workload(eng, workload, oracle)
        s = summarize(ms)
        s.update(method=method, corpus=corpus.name, n_rows=corpus.n,
                 overlap=args.overlap, target_recall=args.target_recall,
                 oracle_noise=args.oracle_noise, seed=args.seed,
                 wall_s=round(t.elapsed, 2))
        summary_rows.append(s)
        all_rows += metrics_to_rows(ms, method=method, corpus=corpus.name,
                                    n_rows=corpus.n, overlap=args.overlap,
                                    seed=args.seed)
        print(f"[{method:18s}] calls={s['total_oracle_calls']:8d} "
              f"P={s['macro_precision']:.3f} R={s['macro_recall']:.3f} "
              f"viol={s['bound_violations']}/{s['n_bounded']} "
              f"wall={s['wall_s']}s")

    write_results(pd.DataFrame(all_rows), "exp1_perquery", args)
    write_results(pd.DataFrame(summary_rows), "exp1_summary", args)


if __name__ == "__main__":
    main()
