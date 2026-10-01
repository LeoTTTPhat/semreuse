"""Experiment 10: tier two -- a small-LLM arbiter, evaluated and cost-accounted.

The cross-encoder tier's measured failure modes are DISJOINT (F1 0.72) and
OVERLAP (F1 0.05), and the rewrite ablation shows those are exactly where the
savings are lost.  This experiment builds the second tier the architecture has
always had an interface for, and answers three questions with numbers:

  Q-a  Does an instruction-tuned arbiter classify predicate relations better
       than the calibrated NLI head, and specifically on DISJOINT/OVERLAP?
  Q-b  What does escalating only the *unsure* pairs cost, in calls, tokens,
       and seconds, relative to the oracle calls it saves?
  Q-c  How much of the gap to the perfect-entailment upper bound does it close
       end to end?

Part 1 evaluates heads on balanced 20NG predicate pairs.  Part 2 runs the full
workload with tier one alone, tier one + tier two, and the ground-truth oracle.

Usage:
    .venv/bin/python experiments/exp10_arbiter.py --scale full --corpus 20ng \
        --arbiter-model qwen2.5:7b --parts pairs,e2e
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Timer, base_parser, load_corpus_scaled,  # noqa: E402
                    make_nli_entailment, scale_params, write_results)

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402


def score_head(name, judge_fn, pairs):
    from semreuse.entailment import RELATIONS

    t0 = time.perf_counter()
    preds = [judge_fn(p, q) for p, q, _ in pairs]
    dt = time.perf_counter() - t0
    y_true = [r for _, _, r in pairs]
    y_pred = [j.relation for j in preds]
    out = {"head": name, "n_pairs": len(pairs), "seconds": round(dt, 1),
           "accuracy": float(np.mean([t is p for t, p in zip(y_true, y_pred)]))}
    for rel in RELATIONS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t is rel and p is rel)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t is not rel and p is rel)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t is rel and p is not rel)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        out[f"f1_{rel.value}"] = (2 * prec * rec / (prec + rec)
                                  if prec + rec else 0.0)
    return out


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--arbiter-model", default="qwen2.5:7b")
    ap.add_argument("--eval-pairs", type=int, default=400)
    ap.add_argument("--escalate-below", type=float, default=0.7)
    ap.add_argument("--max-escalations", type=int, default=4,
                    help="cap on tier-two pairs per query (0 = unlimited); "
                         "spent on the views tier one is least sure about")
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--parts", default="pairs,e2e")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="if >0, set every rewrite threshold to this value")
    args = ap.parse_args()

    from semreuse.arbiter import LLMArbiter, TwoTierEntailment
    from semreuse.audit import AuditConfig
    from semreuse.baselines import ColdEngine
    from semreuse.corpus import make_synthetic_corpus
    from semreuse.engine import EngineConfig, SemReuseEngine
    from semreuse.entailment import GroundTruthEntailment
    from semreuse.metrics import metrics_to_rows, run_workload, summarize
    from semreuse.oracle import SimulatedOracle
    from semreuse.predicates import (PredicateUniverse, generate_workload,
                                     labeled_pairs_for_calibration)
    from semreuse.rewriter import RewriteConfig

    parts = args.parts.split(",")
    corpus = load_corpus_scaled(args)
    _, n_queries = scale_params(args)
    universe = PredicateUniverse.build(corpus)

    nli = make_nli_entailment(seed=args.seed, n_pairs=args.calib_pairs)
    arbiter = LLMArbiter(model=args.arbiter_model,
                         concurrency=args.concurrency, seed=args.seed)

    # Calibrate tier two exactly like tier one: on the synthetic dev taxonomy,
    # never on the evaluation corpus.
    dev_uni = PredicateUniverse.build(
        make_synthetic_corpus(n=50, seed=args.seed, n_groups=8))
    dev_pairs = labeled_pairs_for_calibration(dev_uni, n_pairs=200,
                                              seed=args.seed)
    cal = arbiter.calibrate(dev_pairs)
    print(f"[arbiter] {args.arbiter_model}: dev accuracy {cal['accuracy']:.3f}; "
          f"class confidences " +
          ", ".join(f"{k}={v:.2f}" for k, v in cal["confidence"].items()))

    rows = []
    if "pairs" in parts:
        pairs = labeled_pairs_for_calibration(universe,
                                              n_pairs=args.eval_pairs,
                                              seed=args.seed + 1)
        two = TwoTierEntailment(nli, arbiter,
                                escalate_below=args.escalate_below)
        rows.append(score_head("nli-calibrated", nli.judge, pairs))
        rows.append(score_head("arbiter-only", arbiter.judge, pairs))
        rows.append(score_head(
            "two-tier", lambda p, q: two.judge_batch(p, [q])[0], pairs))
        for r in rows:
            print(f"[pairs] {r['head']:16s} acc={r['accuracy']:.3f} "
                  + " ".join(f"{k[3:8]}={v:.2f}" for k, v in r.items()
                             if k.startswith("f1_"))
                  + f"  {r['seconds']}s")
        rows[-1]["escalated_pairs"] = two.stats.escalated
        rows[-1]["arbiter_calls"] = arbiter.stats.calls
        rows[-1]["arbiter_prompt_tokens"] = arbiter.stats.prompt_tokens
        write_results(pd.DataFrame(rows), "exp10_pairs", args)

    if "e2e" not in parts:
        return

    workload = generate_workload(universe, n_queries=n_queries,
                                 overlap_rate=args.overlap, seed=args.seed)
    audit_cfg = AuditConfig(alpha=0.05, target_recall=args.target_recall,
                            budget_fraction=0.1)
    rw = (RewriteConfig(tau_equiv=args.tau, tau_forward=args.tau,
                        tau_backward=args.tau, tau_disjoint=args.tau)
          if args.tau > 0 else RewriteConfig())
    summary = []
    for method in ["cold", "semreuse", "semreuse-2tier", "semreuse-gt"]:
        oracle = SimulatedOracle(corpus, seed=args.seed)
        a0 = (arbiter.stats.calls, arbiter.stats.prompt_tokens,
              arbiter.stats.wall_s)
        if method == "cold":
            eng = ColdEngine(corpus, oracle)
        else:
            if method == "semreuse":
                ent = nli
            elif method == "semreuse-2tier":
                ent = TwoTierEntailment(
                    nli, arbiter, escalate_below=args.escalate_below,
                    max_escalations=(args.max_escalations or None))
            else:
                ent = GroundTruthEntailment(universe.resolver())
            eng = SemReuseEngine(corpus, oracle, ent,
                                 EngineConfig(rewrite=rw, audit=audit_cfg,
                                              seed=args.seed))
        with Timer() as t:
            ms = run_workload(eng, workload, oracle)
        s = summarize(ms)
        s.update(method=method, corpus=corpus.name, n_rows=corpus.n,
                 overlap=args.overlap, target_recall=args.target_recall,
                 tau=args.tau, seed=args.seed, wall_s=round(t.elapsed, 2),
                 arbiter_model=args.arbiter_model,
                 arbiter_calls=arbiter.stats.calls - a0[0],
                 arbiter_prompt_tokens=arbiter.stats.prompt_tokens - a0[1],
                 arbiter_wall_s=round(arbiter.stats.wall_s - a0[2], 1),
                 max_escalations=args.max_escalations,
                 escalated_pairs=(ent.stats.escalated
                                  if method == "semreuse-2tier" else 0))
        summary.append(s)
        print(f"[e2e] {method:16s} calls={s['total_oracle_calls']:8d} "
              f"P={s['macro_precision']:.3f} R={s['macro_recall']:.3f} "
              f"viol={s['bound_violations']}/{s['n_bounded']} "
              f"arb={s['arbiter_calls']} ({s['arbiter_wall_s']}s) "
              f"wall={s['wall_s']}s", flush=True)
    write_results(pd.DataFrame(summary), "exp10_e2e", args)


if __name__ == "__main__":
    main()
