"""Experiment 9: the proxy-cascade competitor, and its composition with reuse.

The question this answers is the one every reader of SUPG / NoScope / BARGAIN /
LOTUS asks first: *if a recall target below 1 is acceptable, why is cross-query
reuse needed at all?*  A cheap per-row proxy plus a recall-targeted cutoff
attacks the same budget with none of SemReuse's machinery.

Methods:
  cold                 -- no reuse, no proxy
  proxy                -- per-query proxy cascade, same certificate contract
  semreuse             -- entailment reuse, no proxy
  semreuse+proxy       -- both, certified by a single audit
  semreuse-gt(+proxy)  -- the same two with a perfect entailment oracle

Cost is reported in oracle calls, and separately in cheap-tier work, because
the two techniques have different cheap-tier scaling: entailment is
O(#views) per query, the proxy is O(N) per query plus a one-time O(N)
document-embedding pass.

Usage:
    .venv/bin/python experiments/exp9_proxy.py --scale full --corpus 20ng
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Timer, base_parser, load_corpus_scaled,  # noqa: E402
                    make_nli_entailment, scale_params, write_results)

import pandas as pd  # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--pilot", type=int, default=200)
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--min-kept-recall", type=float, default=0.95)
    ap.add_argument("--min-prune-fraction", type=float, default=0.2)
    ap.add_argument("--proxy-mode", default="trained",
                    choices=["trained", "cosine"])
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the proxy's pilot size and validation gate "
                         "and report its best configuration, so the baseline "
                         "is compared at its own optimum rather than ours")
    ap.add_argument("--methods", default=("cold,proxy,semreuse,semreuse+proxy,"
                                          "semreuse-gt,semreuse-gt+proxy"))
    args = ap.parse_args()

    from semreuse.audit import AuditConfig
    from semreuse.baselines import ColdEngine, SentenceEmbedder
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.entailment import GroundTruthEntailment
    from semreuse.metrics import metrics_to_rows, run_workload, summarize
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicates import PredicateUniverse, generate_workload
    from semreuse.proxy import (EmbeddingProxy, ProxyCascadeConfig,
                                ProxyCascadeEngine)

    corpus = load_corpus_scaled(args)
    _, n_queries = scale_params(args)
    universe = PredicateUniverse.build(corpus)
    workload = generate_workload(universe, n_queries=n_queries,
                                 overlap_rate=args.overlap, seed=args.seed)
    print(f"[setup] corpus={corpus.name} n={corpus.n} "
          f"queries={len(workload.queries)} overlap={args.overlap}")

    embedder = SentenceEmbedder()
    proxy = EmbeddingProxy(corpus, embedder, mode=args.proxy_mode)
    with Timer() as t_embed:
        proxy._docs()          # pay the one-time document pass up front
    print(f"[proxy] embedded {corpus.n} documents in {t_embed.elapsed:.1f}s")

    methods = args.methods.split(",")
    nli = (make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs)
           if any(m.startswith("semreuse") and "gt" not in m
                  for m in methods) else None)
    audit_cfg = AuditConfig(alpha=0.05, target_recall=args.target_recall,
                            budget_fraction=0.1)

    def build(name, oracle):
        use_proxy = name.endswith("+proxy")
        base = name[: -len("+proxy")] if use_proxy else name
        if base == "cold":
            return ColdEngine(corpus, oracle)
        if base == "proxy":
            return ProxyCascadeEngine(
                corpus, oracle, proxy,
                ProxyCascadeConfig(pilot_size=args.pilot, n_bands=args.bands,
                                   min_kept_recall=args.min_kept_recall,
                                   min_prune_fraction=args.min_prune_fraction,
                                   audit=audit_cfg), seed=args.seed)
        ent = (GroundTruthEntailment(universe.resolver())
               if base == "semreuse-gt" else nli)
        cfg = EngineConfig(audit=audit_cfg, seed=args.seed,
                           proxy_pilot=args.pilot, proxy_bands=args.bands,
                           proxy_min_kept_recall=args.min_kept_recall,
                           proxy_min_prune_fraction=args.min_prune_fraction)
        return SemReuseEngine(corpus, oracle, ent, cfg,
                              proxy=proxy if use_proxy else None)

    if args.sweep:
        methods = ["proxy"]
    rows, summary = [], []
    grid = ([(pi, k) for pi in (200, 400, 800) for k in (0.6, 0.7, 0.8, 0.9)]
            if args.sweep else [(args.pilot, args.min_kept_recall)])
    for method in methods:
      for pilot_i, keep_i in grid:
        args.pilot, args.min_kept_recall = pilot_i, keep_i
        oracle = SimulatedOracle(corpus, seed=args.seed)
        proxy.reset_training()   # each method/config trains its own proxies
        p0 = proxy.stats.row_scores
        eng = build(method, oracle)
        with Timer() as t:
            ms = run_workload(eng, workload, oracle)
        s = summarize(ms)
        s.update(method=method, corpus=corpus.name, n_rows=corpus.n,
                 overlap=args.overlap, target_recall=args.target_recall,
                 seed=args.seed, wall_s=round(t.elapsed, 2),
                 pilot=pilot_i, min_kept_recall=keep_i,
                 proxy_row_scores=proxy.stats.row_scores - p0,
                 proxy_doc_embeddings=(proxy.stats.doc_embeddings
                                       if "proxy" in method else 0))
        summary.append(s)
        rows += metrics_to_rows(ms, method=method, corpus=corpus.name,
                                n_rows=corpus.n, seed=args.seed)
        print(f"[{method:22s}] calls={s['total_oracle_calls']:8d} "
              f"P={s['macro_precision']:.3f} R={s['macro_recall']:.3f} "
              f"viol={s['bound_violations']}/{s['n_bounded']} "
              f"Pviol={s['precision_bound_violations']}/"
              f"{s['n_precision_bounded']} wall={s['wall_s']}s")

    tag = "exp9_sweep" if args.sweep else "exp9"
    write_results(pd.DataFrame(rows), f"{tag}_perquery", args)
    write_results(pd.DataFrame(summary), f"{tag}_summary", args)


if __name__ == "__main__":
    main()
