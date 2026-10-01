"""Experiment 15: direct ablation of the audit design.

Every other experiment evaluates the audit as part of the whole system.  This
one holds the reasoner, the rewrite rules and the workload fixed and swaps one
design decision of the audit at a time:

  pooling      -- one pooled sample and one exact bound for all pruned strata
                  (shipped) vs. a sample and a bound per stratum, with the miss
                  allowance split the way that minimizes the per-stratum design's
                  total sample ('separate'; 'separate-prop' splits by size);
  sizing       -- adaptive sample sizes (shipped) vs. a fixed sample per
                  stratum/pool and a fixed fraction of each;
  certificate  -- joint recall + precision (shipped) vs. recall-only: the
                  assumed-positive strata sampled for the recall numerator but
                  never demoted ('recall-only'), or not audited at all
                  ('recall-trust', all of alpha on the pruned pool).

Every variant publishes a valid certificate for what it certifies; they differ
in what they cost and in what they certify.  Settings: 20 Newsgroups (simulated
oracle; NLI and perfect reasoner), AG News at two corpus sizes (separating O(1)
from Theta(N) designs), and the real-LLM analyst log.

Usage:
    .venv/bin/python experiments/exp15_audit_design.py --setting 20ng
    .venv/bin/python experiments/exp15_audit_design.py --setting agnews \
        --sizes 8000,120000
    .venv/bin/python experiments/exp15_audit_design.py --setting real \
        --matrix data/llm_matrix/frozen_analyst_20ng_2000_llama3.1-8b.npz
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA_DIR, RESULTS_DIR, Timer,  # noqa: E402
                    make_nli_entailment)

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

# Overrides of the shipped AuditConfig, one design decision at a time.
DESIGNS = {
    "shipped": {},
    "separate": {"pruned_pooling": "separate"},
    "separate-prop": {"pruned_pooling": "separate",
                      "separate_allocation": "proportional"},
    "fixed-100": {"mode": "fixed", "budget_per_stratum": 100},
    "fixed-400": {"mode": "fixed", "budget_per_stratum": 400},
    "fixed-1600": {"mode": "fixed", "budget_per_stratum": 1600},
    "frac-0.02": {"mode": "fraction", "budget_fraction": 0.02,
                  "budget_per_stratum": 10 ** 9},
    "frac-0.05": {"mode": "fraction", "budget_fraction": 0.05,
                  "budget_per_stratum": 10 ** 9},
    "frac-0.10": {"mode": "fraction", "budget_fraction": 0.10,
                  "budget_per_stratum": 10 ** 9},
    "recall-only": {"certify": "recall"},
    "recall-trust": {"certify": "recall-trust"},
}


def audit_config(design: str, target: float):
    from semreuse.audit import AuditConfig
    kw = dict(alpha=0.05, target_recall=target, budget_fraction=0.1)
    kw.update(DESIGNS[design])
    return AuditConfig(**kw)


def load_setting(args, size):
    """(corpus, workload, oracle factory, perfect-reasoner factory, label)."""
    from semreuse.corpus import Corpus, load_corpus
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicate_log import ExtensionalEntailment
    from semreuse.predicates import (PredicateUniverse, Workload,
                                     generate_workload)

    if args.setting == "real":
        from semreuse.llm_oracle import LLMOracle, LLMResponseMatrix
        from semreuse.predicate_log import analyst_log_predicates
        matrix = LLMResponseMatrix.load(args.matrix)
        n_full = matrix.answers.shape[1]
        corpus = load_corpus("20ng", str(DATA_DIR), size=4000, seed=args.seed)
        n = size or n_full
        corpus = Corpus(name=f"{corpus.name}-pre{n}", docs=corpus.docs[:n],
                        leaf_labels=corpus.leaf_labels[:n],
                        taxonomy=dict(corpus.taxonomy))
        matrix = LLMResponseMatrix(
            model=matrix.model, corpus_name=matrix.corpus_name,
            texts=matrix.texts, answers=matrix.answers[:, :n],
            meta=matrix.meta)
        queries = [q for q in analyst_log_predicates()
                   if q.text in set(matrix.texts)]
        wl = Workload(queries=queries, universe=None,
                      params={"generator": "analyst-log"})
        idx = matrix.index

        def ext_of(t):
            return matrix.answers[idx[t]]
        return (corpus, wl, lambda: LLMOracle(corpus, matrix=matrix),
                lambda: ExtensionalEntailment(ext_of, eps=0.02),
                f"real-{matrix.model.split(':')[0]}-N{n}")

    base = load_corpus(args.corpus_name, str(DATA_DIR),
                       size=None if args.setting == "agnews" else 8000,
                       seed=args.seed)
    corpus = (base.subsample(size, seed=args.seed)
              if size and base.n > size else base)
    uni = PredicateUniverse.build(corpus)
    wl = generate_workload(uni, n_queries=args.n_queries,
                           overlap_rate=args.overlap, seed=args.seed)
    text2labels = {p.text: p.label_set for p in uni.predicates}
    cache: dict[str, np.ndarray] = {}

    def ext_of(t):
        if t not in cache:
            cache[t] = corpus.extension(text2labels[t])
        return cache[t]
    return (corpus, wl, lambda: SimulatedOracle(corpus, seed=args.seed),
            lambda: ExtensionalEntailment(ext_of, eps=0.0),
            f"{args.corpus_name}-N{corpus.n}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setting", choices=["20ng", "agnews", "real"],
                    default="20ng")
    ap.add_argument("--sizes", default="",
                    help="corpus sizes (agnews) or prefixes (real)")
    ap.add_argument("--matrix", default=str(
        DATA_DIR / "llm_matrix" / "frozen_analyst_20ng_2000_llama3.1-8b.npz"))
    ap.add_argument("--n-queries", type=int, default=120)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--reasoners", default="nli",
                    help="comma list of nli,gt")
    ap.add_argument("--designs", default=",".join(DESIGNS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-pairs", type=int, default=300)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    args.corpus_name = "20ng" if args.setting in ("20ng", "real") else "agnews"

    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.metrics import metrics_to_rows, run_workload, summarize

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else [0]
    reasoners = args.reasoners.split(",")
    designs = args.designs.split(",")
    nli = (make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs)
           if "nli" in reasoners else None)

    summary, perq = [], []
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_s = RESULTS_DIR / f"exp15_audit_design_{args.setting}{tag}.csv"
    out_q = RESULTS_DIR / f"exp15_perquery_{args.setting}{tag}.csv"
    for size in sizes:
        corpus, wl, mk_oracle, mk_gt, label = load_setting(args, size)
        cold = corpus.n * len(wl.queries)
        print(f"[setup] {label} queries={len(wl.queries)} cold={cold}",
              flush=True)
        for reasoner in reasoners:
            for design in designs:
                oracle = mk_oracle()
                ent = nli if reasoner == "nli" else mk_gt()
                eng = SemReuseEngine(corpus, oracle, ent, EngineConfig(
                    audit=audit_config(design, args.target_recall),
                    seed=args.seed))
                with Timer() as t:
                    ms = run_workload(eng, wl, oracle)
                s = summarize(ms)
                rew = [m for m in ms if m.reuse_kind == "rewrite"]
                calls = s["total_oracle_calls"]
                row = dict(
                    setting=args.setting, label=label, n_rows=corpus.n,
                    reasoner=reasoner, design=design,
                    target_recall=args.target_recall, alpha=0.05,
                    cold_calls=cold, calls=calls,
                    reduction=cold / max(1, calls),
                    candidate_calls=sum(m.candidate_calls for m in ms),
                    audit_calls=sum(m.audit_calls for m in ms),
                    escalation_calls=sum(m.escalation_calls for m in ms),
                    n_rewritten=len(rew),
                    audit_per_rewritten=(sum(m.audit_calls for m in rew)
                                         / max(1, len(rew))),
                    n_recall_certs=s["n_bounded"],
                    recall_violations=s["bound_violations_oracle"],
                    n_precision_certs=s["n_precision_bounded"],
                    precision_violations=s["precision_bound_violations"],
                    mean_recall_bound=s["mean_recall_bound"],
                    mean_precision_bound=s["mean_precision_bound"],
                    macro_precision=s["macro_precision"],
                    macro_recall=s["macro_recall_oracle"],
                    min_precision=float(min(m.precision for m in ms)),
                    wall_s=round(t.elapsed, 1), seed=args.seed)
                summary.append(row)
                perq += metrics_to_rows(ms, setting=args.setting,
                                        label=label, reasoner=reasoner,
                                        design=design, n_rows=corpus.n)
                print(f"  [{reasoner:3s} {design:14s}] calls={calls:9d} "
                      f"red={row['reduction']:6.2f}x "
                      f"audit={row['audit_calls']:7d} "
                      f"esc={row['escalation_calls']:8d} "
                      f"P={row['macro_precision']:.3f} "
                      f"R={row['macro_recall']:.3f} "
                      f"Rv={row['recall_violations']}/{row['n_recall_certs']} "
                      f"Pv={row['precision_violations']}/"
                      f"{row['n_precision_certs']} ({row['wall_s']}s)",
                      flush=True)
                pd.DataFrame(summary).to_csv(out_s, index=False)
    pd.DataFrame(perq).to_csv(out_q, index=False)
    print(f"[out] {out_s}\n[out] {out_q}")


if __name__ == "__main__":
    main()
