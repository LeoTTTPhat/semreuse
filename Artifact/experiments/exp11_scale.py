"""Experiment 11: server-scale corpora, and a multi-label topic hierarchy.

Two objections are answered here at once.

*Scale.*  Proposition 4 says the certificate's price is flat in N while reuse
savings grow with it, which is the paper's whole economic argument -- and the
original evaluation demonstrated it over N <= 7,951, three orders of magnitude
below the million-row corpora the introduction invokes.  This experiment sweeps
N up to the full 804,414-document RCV1-v2 collection and the full 120,000-row
AG News train split.

*Multi-label semantics.*  On a single-label corpus any two distinct topics are
automatically disjoint in extension, so disjoint elimination fires on almost
every pair (99.2% of queries on 20 Newsgroups) and genuine partial OVERLAP --
the relation from which no rewrite can extract anything -- barely exists.  RCV1
documents carry several hierarchically expanded topics, so containment is real
and deep (C1511 subset C151 subset C15 subset CCAT) while disjointness has to
be earned.  Ground-truth relations are therefore measured *extensionally* from
the label matrix, not assumed from label-set algebra.

Usage:
    .venv/bin/python experiments/exp11_scale.py --corpus rcv1 \
        --sizes 25000,100000,400000,804414 --n-queries 120
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, Timer, base_parser, write_results  # noqa: E402

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--sizes", default="25000,100000,400000,804414")
    ap.add_argument("--n-queries", type=int, default=120)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=0.0,
                    help="slack for extensional ground-truth relations")
    ap.add_argument("--methods",
                    default="cold,exact,semreuse,semreuse-gt")
    args = ap.parse_args()

    from semreuse.audit import AuditConfig
    from semreuse.baselines import ColdEngine, ExactCacheEngine
    from semreuse.corpus import load_corpus, load_rcv1
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.metrics import metrics_to_rows, run_workload, summarize
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicate_log import ExtensionalEntailment
    from semreuse.predicates import PredicateUniverse, generate_workload
    from semreuse.rcv1_predicates import (build_rcv1_universe,
                                          generate_drilldown_workload)

    from common import make_nli_entailment

    sizes = [int(x) for x in args.sizes.split(",")]
    methods = args.methods.split(",")
    nli = (make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs)
           if any(m == "semreuse" or m == "semreuse-noaudit" for m in methods)
           else None)

    base = (load_rcv1(str(DATA_DIR)) if args.corpus == "rcv1"
            else load_corpus(args.corpus, str(DATA_DIR)))
    print(f"[setup] base corpus={base.name} N={base.n}")

    rows, summary = [], []
    for size in sizes:
        corpus = base.subsample(size, seed=args.seed) if base.n > size else base
        if args.corpus == "rcv1":
            universe = build_rcv1_universe(corpus, seed=args.seed)
            workload = generate_drilldown_workload(
                universe, corpus, n_queries=args.n_queries,
                overlap_rate=args.overlap, seed=args.seed)
        else:
            universe = PredicateUniverse.build(corpus)
            workload = generate_workload(universe, n_queries=args.n_queries,
                                         overlap_rate=args.overlap,
                                         seed=args.seed)
        # Ground truth measured from extensions, never from label-set algebra:
        # with multi-label documents the two are different things.
        ext_cache: dict[str, np.ndarray] = {}
        text2labels = {p.text: p.label_set for p in universe.predicates}

        def ext_of(text: str) -> np.ndarray:
            if text not in ext_cache:
                ext_cache[text] = corpus.extension(text2labels[text])
            return ext_cache[text]

        sel = np.array([ext_of(q.text).mean() for q in workload.queries])
        print(f"[N={corpus.n}] queries={len(workload.queries)} "
              f"selectivity mean={sel.mean():.4f} median={np.median(sel):.4f}")

        for method in methods:
            oracle = SimulatedOracle(corpus, seed=args.seed)
            audit_cfg = AuditConfig(alpha=0.05,
                                    target_recall=args.target_recall,
                                    budget_fraction=0.1)
            if method == "cold":
                eng = ColdEngine(corpus, oracle)
            elif method == "exact":
                eng = ExactCacheEngine(corpus, oracle)
            elif method == "semreuse-gt":
                eng = SemReuseEngine(
                    corpus, oracle,
                    ExtensionalEntailment(ext_of, eps=args.eps),
                    EngineConfig(audit=audit_cfg, seed=args.seed))
            else:
                eng = SemReuseEngine(
                    corpus, oracle, nli,
                    EngineConfig(audit=audit_cfg, seed=args.seed,
                                 enable_audit=(method != "semreuse-noaudit")))
            with Timer() as t:
                ms = run_workload(eng, workload, oracle)
            s = summarize(ms)
            s.update(method=method, corpus=args.corpus, n_rows=corpus.n,
                     overlap=args.overlap, target_recall=args.target_recall,
                     seed=args.seed, wall_s=round(t.elapsed, 2),
                     mean_selectivity=float(sel.mean()),
                     audit_calls=sum(m.audit_calls for m in ms),
                     escalation_calls=sum(m.escalation_calls for m in ms),
                     candidate_calls=sum(m.candidate_calls for m in ms))
            summary.append(s)
            rows += metrics_to_rows(ms, method=method, corpus=args.corpus,
                                    n_rows=corpus.n, seed=args.seed)
            print(f"  [{method:14s}] calls={s['total_oracle_calls']:10d} "
                  f"P={s['macro_precision']:.3f} R={s['macro_recall']:.3f} "
                  f"Rviol={s['bound_violations']}/{s['n_bounded']} "
                  f"audit/q={s['audit_calls']/max(1,len(ms)):8.1f} "
                  f"wall={s['wall_s']}s", flush=True)
        write_results(pd.DataFrame(summary), "exp11_summary", args)
        write_results(pd.DataFrame(rows), "exp11_perquery", args)


if __name__ == "__main__":
    main()
