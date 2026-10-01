"""Experiment 8: end-to-end over a *real* LLM oracle and a real predicate log.

Everything else in this evaluation defines predicate semantics by corpus
labels.  This experiment removes both crutches at once:

  * the oracle is an instruction-tuned model answering a LOTUS-style
    ``sem_filter`` prompt per row (temperature 0, answers materialized once by
    ``build_llm_matrix.py`` and replayed under unit-cost accounting);
  * the workload is the hand-written analyst log of
    ``semreuse.predicate_log`` -- session-ordered filters about support
    requests, marketplace posts, tone, and rhetorical form, written without
    reference to the newsgroup labels.

Consequences worth stating before the numbers.  There is no label taxonomy to
appeal to, so ground truth for scoring *is* the model's own answers -- which is
exactly what the certificate of Theorem 2 promises fidelity to.  And true
containment between two natural predicates is never exact: the ground-truth
relation oracle therefore declares implication up to a slack ``eps``, and the
residual rows are real errors that the audit must catch.

Cost is reported in oracle calls, and translated into measured prompt tokens,
measured wall-clock seconds, and projected dollars at a published price for a
comparable hosted model.

Usage:
    .venv/bin/python experiments/exp8_realllm.py \
        --matrix data/llm_matrix/analyst_20ng_4000_llama3.1-8b-instruct-q4_K_M.npz
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA_DIR, RESULTS_DIR, Timer, base_parser,  # noqa: E402
                    make_nli_entailment, write_results)

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--n-rows", type=int, default=0,
                    help="use a prefix of the corpus (0 = all)")
    ap.add_argument("--sizes", default="",
                    help="comma-separated prefix sizes to sweep, e.g. "
                         "500,1000,2000. Because the response matrix already "
                         "holds every answer, a sweep over prefixes of the "
                         "(already shuffled) corpus costs no extra LLM calls "
                         "and shows the audit floor amortizing on the *real* "
                         "oracle rather than only on the simulated one.")
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.0,
                    help="if >0, set every rewrite threshold to this value "
                         "(the aggressive policy recommended in Section 7.8)")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="slack for ground-truth containment/disjointness")
    ap.add_argument("--methods", default=("cold,exact,embed@0.8,embed@0.9,"
                                          "semreuse,semreuse-noaudit,"
                                          "semreuse-gt"))
    args = ap.parse_args()

    from semreuse.llm_oracle import LLMResponseMatrix

    full_matrix = LLMResponseMatrix.load(args.matrix)
    sizes = ([int(x) for x in args.sizes.split(",")] if args.sizes
             else [args.n_rows or full_matrix.answers.shape[1]])
    rows_all, summary_all = [], []
    for size_i, size in enumerate(sizes):
        args.n_rows = size
        # The largest size is the headline, so it claims the
        # un-suffixed workload file the paper reads from.
        r, sm = run_one(args, full_matrix,
                        canonical=(size_i == len(sizes) - 1))
        rows_all += r
        summary_all += sm
    write_results(pd.DataFrame(rows_all), "exp8_perquery", args)
    write_results(pd.DataFrame(summary_all), "exp8_summary", args)


def run_one(args, full_matrix, canonical: bool):
    from semreuse.audit import AuditConfig
    from semreuse.baselines import (ColdEngine, EmbeddingCacheEngine,
                                    ExactCacheEngine, SentenceEmbedder)
    from semreuse.corpus import Corpus, load_corpus
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.llm_oracle import (REFERENCE_PRICE_IN_PER_MTOK,
                                     REFERENCE_PRICE_OUT_PER_MTOK,
                                     LLMOracle, LLMResponseMatrix)
    from semreuse.metrics import metrics_to_rows, run_workload, summarize
    from semreuse.predicate_log import (ExtensionalEntailment,
                                        analyst_log_predicates,
                                        designed_relation_slack,
                                        relation_distribution)
    from semreuse.predicates import Workload
    from semreuse.rewriter import RewriteConfig

    matrix = full_matrix
    n_full = matrix.answers.shape[1]
    base = load_corpus("20ng", str(DATA_DIR), size=n_full, seed=args.seed)
    n = args.n_rows or n_full
    if n < n_full:                       # prefix of an already-shuffled sample
        base = Corpus(name=f"{base.name}-pre{n}", docs=base.docs[:n],
                      leaf_labels=base.leaf_labels[:n],
                      taxonomy=dict(base.taxonomy))
        matrix = LLMResponseMatrix(
            model=matrix.model, corpus_name=matrix.corpus_name,
            texts=matrix.texts, answers=matrix.answers[:, :n],
            prompt_tokens=matrix.prompt_tokens,
            output_tokens=matrix.output_tokens, wall_s=matrix.wall_s,
            meta=matrix.meta)
    corpus = base
    queries = analyst_log_predicates()
    have = set(matrix.texts)
    queries = [q for q in queries if q.text in have]
    workload = Workload(queries=queries, universe=None,
                        params={"generator": "analyst-log"})
    idx = matrix.index
    ext_of = lambda t: matrix.answers[idx[t]]        # noqa: E731

    print(f"[setup] model={matrix.model} N={corpus.n} queries={len(queries)} "
          f"distinct={len(set(q.text for q in queries))}")

    # -- what the workload actually looks like under a real oracle ---------
    texts = sorted({q.text for q in queries})
    dist = relation_distribution(texts, ext_of, eps=args.eps)
    sel = np.array([ext_of(t).mean() for t in texts])
    slack = designed_relation_slack(ext_of)
    if slack:
        vals = np.array([r["slack"] for r in slack])
        print(f"[workload] designed implications: {len(slack)} pairs, slack "
              f"min={vals.min():.3f} median={np.median(vals):.3f} "
              f"max={vals.max():.3f}; exact (slack 0): "
              f"{int((vals == 0).sum())}")
        for r in sorted(slack, key=lambda r: -r["slack"])[:4]:
            print(f"           {r['a']}=>{r['b']:4s} {r['kind']:15s} "
                  f"slack={r['slack']:.3f} |A|={r['n_a']:4d} |B|={r['n_b']:4d}")
    # The pair-space mix is the wrong denominator on its own: reuse needs one
    # exploitable predecessor per query, not a mostly-nested pair space.  We
    # therefore also measure, for each query in issue order, whether any
    # *earlier* query stands in a relation a rewrite rule can use.
    ent = ExtensionalEntailment(ext_of, eps=args.eps)
    from semreuse.predicates import Relation
    usable = {Relation.EQUIV, Relation.FORWARD, Relation.BACKWARD,
              Relation.DISJOINT}
    have_pred = {"any": 0, "equiv": 0, "containment": 0, "disjoint": 0,
                 "exact_repeat": 0}
    seen: list[str] = []
    for q in queries:
        if q.text in seen:
            have_pred["exact_repeat"] += 1
        rels = [ent.relation(q.text, t) for t in seen if t != q.text]
        if any(r in usable for r in rels):
            have_pred["any"] += 1
        if any(r is Relation.EQUIV for r in rels):
            have_pred["equiv"] += 1
        if any(r in (Relation.FORWARD, Relation.BACKWARD) for r in rels):
            have_pred["containment"] += 1
        if any(r is Relation.DISJOINT for r in rels):
            have_pred["disjoint"] += 1
        seen.append(q.text)
    nq = max(1, len(queries))
    have_frac = {k: v / nq for k, v in have_pred.items()}
    print("[workload] relation mix over ordered pairs: "
          + ", ".join(f"{k}={v:.3f}" for k, v in dist.items()))
    print("[workload] queries with an exploitable predecessor: "
          + ", ".join(f"{k}={v:.3f}" for k, v in have_frac.items()))
    print(f"[workload] selectivity: mean={sel.mean():.3f} "
          f"median={np.median(sel):.3f} min={sel.min():.3f} max={sel.max():.3f}")

    # -- measured unit costs ------------------------------------------------
    # Per-call token counts and throughput are means over the calls the model
    # actually served during materialization (``live_calls``), not over matrix
    # cells: a resumed build serves fewer calls than the matrix has cells, and
    # dividing by cells would understate both.
    cells = max(1, int(matrix.meta.get("cells_answered")
                       or matrix.answers.size))
    live = max(1, int(matrix.meta.get("live_calls") or cells))
    ptok = matrix.prompt_tokens / cells      # exact mean over answered cells
    otok = matrix.output_tokens / cells
    # Measured serving throughput, or None when the matrix was assembled
    # entirely from cache and no live timing exists to report.
    rate = (live / matrix.wall_s) if matrix.wall_s > 0 else None
    per_call_dollars = (ptok / 1e6 * REFERENCE_PRICE_IN_PER_MTOK
                        + otok / 1e6 * REFERENCE_PRICE_OUT_PER_MTOK)
    print(f"[cost] measured {ptok:.1f} prompt + {otok:.1f} output tokens/call, "
          + (f"{rate:.1f} rows/s, " if rate else "throughput not timed in this "
             "assembly, ")
          + f"projected ${per_call_dollars*1000:.4f}/1k calls")

    audit_cfg = AuditConfig(alpha=0.05, target_recall=args.target_recall,
                            budget_fraction=0.1)
    rw = (RewriteConfig(tau_equiv=args.tau, tau_forward=args.tau,
                        tau_backward=args.tau, tau_disjoint=args.tau)
          if args.tau > 0 else RewriteConfig())
    methods = args.methods.split(",")
    nli = (make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs)
           if any(m.startswith("semreuse") and m != "semreuse-gt"
                  for m in methods) else None)

    def build(name, oracle):
        if name == "cold":
            return ColdEngine(corpus, oracle)
        if name == "exact":
            return ExactCacheEngine(corpus, oracle)
        if name.startswith("embed@"):
            return EmbeddingCacheEngine(corpus, oracle, SentenceEmbedder(),
                                        theta=float(name.split("@")[1]))
        if name == "semreuse-gt":
            ent = ExtensionalEntailment(ext_of, eps=args.eps)
            return SemReuseEngine(corpus, oracle, ent,
                                  EngineConfig(rewrite=rw, audit=audit_cfg,
                                               seed=args.seed))
        cfg = EngineConfig(rewrite=rw, audit=audit_cfg, seed=args.seed,
                           enable_audit=(name != "semreuse-noaudit"))
        return SemReuseEngine(corpus, oracle, nli, cfg)

    rows, summary = [], []
    for method in methods:
        oracle = LLMOracle(corpus, matrix=matrix)
        eng = build(method, oracle)
        with Timer() as t:
            ms = run_workload(eng, workload, oracle)
        s = summarize(ms)
        calls = s["total_oracle_calls"]
        s.update(method=method, corpus=corpus.name, n_rows=corpus.n,
                 oracle_model=matrix.model, seed=args.seed,
                 target_recall=args.target_recall, eps=args.eps,
                 tau=args.tau,
                 wall_s=round(t.elapsed, 2),
                 prompt_tokens=int(calls * ptok),
                 output_tokens=int(calls * otok),
                 projected_dollars=round(calls * per_call_dollars, 4),
                 oracle_seconds=(round(calls / rate, 1) if rate else None))
        summary.append(s)
        rows += metrics_to_rows(ms, method=method, corpus=corpus.name,
                                n_rows=corpus.n, seed=args.seed)
        print(f"[{method:18s}] calls={calls:8d} "
              f"P={s['macro_precision']:.3f} R={s['macro_recall']:.3f} "
              f"Rviol={s['bound_violations']}/{s['n_bounded']} "
              f"Pviol={s['precision_bound_violations']}/"
              f"{s['n_precision_bounded']} "
              f"${s['projected_dollars']:.2f} "
              + (f"{s['oracle_seconds']:.0f}s" if s['oracle_seconds'] else "--"))

    tag = f"_{args.out_tag}" if args.out_tag else ""
    meta_path = (RESULTS_DIR /
                 f"exp8_workload_{args.corpus}_{args.scale}{tag}.json")
    if not canonical:
        meta_path = meta_path.with_name(
            meta_path.stem + f"_n{corpus.n}" + meta_path.suffix)
    meta_path.write_text(json.dumps({
        "model": matrix.model, "n_rows": corpus.n, "queries": len(queries),
        "distinct": len(texts), "relation_mix": dist,
        "predecessor_fraction": have_frac,
        "designed_relations": slack,
        "selectivity_mean": float(sel.mean()),
        "selectivity_median": float(np.median(sel)),
        "selectivity_min": float(sel.min()),
        "selectivity_max": float(sel.max()),
        "prompt_tokens_per_call": ptok, "output_tokens_per_call": otok,
        "rows_per_s": rate, "dollars_per_call": per_call_dollars,
        "eps": args.eps}, indent=2))
    print(f"[out] {meta_path}")
    return rows, summary


if __name__ == "__main__":
    main()
