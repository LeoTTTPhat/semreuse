"""Experiment 14: does measured agreement slack pay for itself?

Section 7.2 diagnoses the real-oracle result: a declared implication that misses
by more than the recall allowance cannot be certified, so the audit escalates
and the escalation costs more than the pruning saved. ``semreuse.slack`` acts on
that diagnosis -- every audit measures what a rewrite cost, the estimates
accumulate per view, and the planner declines rewrites the budget cannot absorb.

This experiment asks whether the mechanism earns its place, on the setting that
motivated it (a real LLM oracle over the analyst log) and on the simulated
settings where the diagnosis does *not* apply, since a fix that helps where it
was designed to and hurts everywhere else is not a fix.

Usage:
  .venv/bin/python experiments/exp14_slack.py --oracle real \\
      --matrix data/llm_matrix/frozen_analyst_20ng_2000_llama3.1-8b.npz
  .venv/bin/python experiments/exp14_slack.py --oracle sim --corpus 20ng \\
      --scale full
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA_DIR, Timer, base_parser,  # noqa: E402
                    load_corpus_scaled, make_nli_entailment, scale_params,
                    write_results)

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--oracle", choices=["sim", "real"], default="sim")
    ap.add_argument("--matrix", default="")
    ap.add_argument("--n-rows", type=int, default=0)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--entailment-error", type=float, default=0.0,
                    help="corrupt a ground-truth reasoner at this rate "
                         "(simulated oracle only); 0 uses the NLI reasoner")
    ap.add_argument("--dump-slack", action="store_true",
                    help="write every slack observation with the features "
                         "available at plan time, to test what predicts it")
    ap.add_argument("--budgets", default="none,1.0,0.5,0.25,0.0",
                    help="slack budgets to compare; 'none' disables the "
                         "mechanism entirely")
    args = ap.parse_args()

    from semreuse.audit import AuditConfig
    from semreuse.baselines import ColdEngine
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.entailment import GroundTruthEntailment
    from semreuse.metrics import metrics_to_rows, run_workload, summarize
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicates import PredicateUniverse, Workload, generate_workload
    from semreuse.rewriter import RewriteConfig

    # -- the setting -------------------------------------------------------
    if args.oracle == "real":
        from semreuse.corpus import Corpus, load_corpus
        from semreuse.llm_oracle import LLMOracle, LLMResponseMatrix
        from semreuse.predicate_log import (ExtensionalEntailment,
                                            analyst_log_predicates)

        matrix = LLMResponseMatrix.load(args.matrix)
        n_full = matrix.answers.shape[1]
        n = args.n_rows or n_full
        base = load_corpus("20ng", str(DATA_DIR), size=n_full, seed=args.seed)
        if n < n_full:
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
        have = set(matrix.texts)
        queries = [q for q in analyst_log_predicates() if q.text in have]
        workload = Workload(queries=queries, universe=None,
                            params={"generator": "analyst-log"})
        idx = matrix.index
        ext_of = lambda t: matrix.answers[idx[t]]          # noqa: E731
        make_oracle = lambda: LLMOracle(corpus, matrix=matrix)   # noqa: E731
        reasoners = {
            "nli": lambda: make_nli_entailment(seed=args.seed,
                                               n_pairs=args.calib_pairs),
            "oracle-ent": lambda: ExtensionalEntailment(ext_of, eps=0.02),
        }
    else:
        corpus = load_corpus_scaled(args)
        _, n_queries = scale_params(args)
        universe = PredicateUniverse.build(corpus)
        workload = generate_workload(universe, n_queries=n_queries,
                                     overlap_rate=args.overlap, seed=args.seed)
        make_oracle = lambda: SimulatedOracle(corpus, seed=args.seed)  # noqa
        if args.entailment_error > 0:
            reasoners = {
                f"gt-err{args.entailment_error}":
                    lambda: GroundTruthEntailment(
                        universe.resolver(), error_rate=args.entailment_error,
                        seed=args.seed)}
        else:
            reasoners = {
                "nli": lambda: make_nli_entailment(seed=args.seed,
                                                   n_pairs=args.calib_pairs)}
    print(f"[setup] oracle={args.oracle} corpus={corpus.name} N={corpus.n} "
          f"queries={len(workload.queries)}")

    audit_cfg = AuditConfig(alpha=0.05, target_recall=args.target_recall,
                            budget_fraction=0.1)
    oracle = make_oracle()
    cold = summarize(run_workload(ColdEngine(corpus, oracle), workload,
                                  oracle))["total_oracle_calls"]
    print(f"[cold] {cold} calls")

    budgets = [None if b == "none" else float(b)
               for b in args.budgets.split(",")]
    rows, summary = [], []
    for rname, make_ent in reasoners.items():
        ent = make_ent()
        for budget in budgets:
            oracle = make_oracle()
            eng = SemReuseEngine(
                corpus, oracle, ent,
                EngineConfig(rewrite=RewriteConfig(),
                             audit=audit_cfg, seed=args.seed,
                             slack_budget=budget))
            feats = []
            if args.dump_slack:
                from semreuse.slack import observation_features
                orig = eng.query

                def traced(pred, _o=orig, _e=eng, _f=feats):
                    r = _o(pred)
                    if r.plan is not None and r.audit_result is not None:
                        _f.extend(observation_features(r.plan, r.audit_result,
                                                       corpus.n))
                    return r
                eng.query = traced
            with Timer() as t:
                ms = run_workload(eng, workload, oracle)
            if feats:
                tag = args.out_tag
                args.out_tag = f"{tag}_{rname}" if tag else rname
                write_results(pd.DataFrame(feats), "exp14_features", args)
                args.out_tag = tag
            s = summarize(ms)
            s.update(reasoner=rname, slack_budget=(-1.0 if budget is None
                                                   else budget),
                     budget_label=("off" if budget is None else f"{budget:g}"),
                     corpus=corpus.name, n_rows=corpus.n, oracle=args.oracle,
                     entailment_error=args.entailment_error, seed=args.seed,
                     cold_calls=cold,
                     reduction=round(cold / s["total_oracle_calls"], 4),
                     candidate_calls=sum(m.candidate_calls for m in ms),
                     audit_calls=sum(m.audit_calls for m in ms),
                     escalation_calls=sum(m.escalation_calls for m in ms),
                     slack_observations=eng.slack.observations,
                     slack_rate=eng.slack.summary()["global_rate"],
                     wall_s=round(t.elapsed, 2))
            summary.append(s)
            rows += metrics_to_rows(ms, reasoner=rname,
                                    budget_label=s["budget_label"],
                                    corpus=corpus.name, n_rows=corpus.n)
            print(f"[{rname:11s} budget={s['budget_label']:>4s}] "
                  f"calls={s['total_oracle_calls']:7d} "
                  f"{s['reduction']:5.2f}x  cand={s['candidate_calls']:7d} "
                  f"audit={s['audit_calls']:6d} esc={s['escalation_calls']:7d} "
                  f"P={s['macro_precision']:.3f} R={s['macro_recall']:.3f} "
                  f"viol={s['bound_violations']}/{s['n_bounded']} "
                  f"obs={s['slack_observations']} "
                  f"rate={s['slack_rate']}", flush=True)

    write_results(pd.DataFrame(summary), "exp14_summary", args)
    write_results(pd.DataFrame(rows), "exp14_perquery", args)


if __name__ == "__main__":
    main()
