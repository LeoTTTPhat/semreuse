"""Experiment 2: entailment reasoner quality and calibration (paper Fig. 5).

Evaluates the NLI cross-encoder relation classifier on labeled predicate
pairs from the target corpus's predicate universe:
  * threshold head vs calibrated head (fit on synthetic dev pairs --
    dataset-independent, no leakage),
  * per-relation precision/recall/F1 + confusion matrix,
  * calibration: reliability bins + expected calibration error (ECE),
  * NLI scoring throughput (pairs/s) -- the cheap-tier cost argument.

Usage:
    .venv/bin/python experiments/exp2_entailment.py --scale small --corpus 20ng
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (base_parser, load_corpus_scaled,  # noqa: E402
                    make_nli_entailment, write_results)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def evaluate_head(ent, pairs, head_name):
    from semreuse.entailment import RELATIONS

    t0 = time.perf_counter()
    preds = []
    for p, q, _ in pairs:
        preds.append(ent.judge(p, q))
    dt = time.perf_counter() - t0

    rows = []
    y_true = [r for _, _, r in pairs]
    y_pred = [j.relation for j in preds]
    conf = [j.confidence for j in preds]
    for rel in RELATIONS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t is rel and p is rel)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t is not rel and p is rel)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t is rel and p is not rel)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(head=head_name, relation=rel.value, tp=tp, fp=fp,
                         fn=fn, precision=prec, recall=rec, f1=f1))
    acc = float(np.mean([t is p for t, p in zip(y_true, y_pred)]))

    # ECE over 10 confidence bins (correctness of the argmax relation).
    correct = np.array([t is p for t, p in zip(y_true, y_pred)], dtype=float)
    conf = np.asarray(conf)
    ece, bin_rows = 0.0, []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        mask = (conf >= lo) & (conf < hi if b < 9 else conf <= hi)
        if mask.sum() == 0:
            continue
        gap = abs(conf[mask].mean() - correct[mask].mean())
        ece += mask.mean() * gap
        bin_rows.append(dict(head=head_name, bin_lo=lo, bin_hi=hi,
                             n=int(mask.sum()),
                             mean_conf=float(conf[mask].mean()),
                             mean_acc=float(correct[mask].mean())))
    return rows, bin_rows, dict(head=head_name, accuracy=acc, ece=ece,
                                n_pairs=len(pairs),
                                judge_seconds=dt,
                                pairs_per_s=len(pairs) / dt if dt else 0)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--n-pairs", type=int, default=400)
    args = ap.parse_args()

    from semreuse.predicates import (PredicateUniverse,
                                     labeled_pairs_for_calibration)

    corpus = load_corpus_scaled(args)
    universe = PredicateUniverse.build(corpus)
    pairs = labeled_pairs_for_calibration(universe, n_pairs=args.n_pairs,
                                          seed=args.seed)
    print(f"[setup] {corpus.name}: {len(pairs)} labeled pairs")

    per_rel, bins, summaries = [], [], []

    # Threshold head (no training).
    ent = make_nli_entailment(calibrate=False, seed=args.seed)
    r, b, s = evaluate_head(ent, pairs, "threshold")
    per_rel += r; bins += b; summaries.append(s)
    print(f"[threshold ] acc={s['accuracy']:.3f} ece={s['ece']:.3f} "
          f"({s['pairs_per_s']:.0f} pairs/s)")

    # Calibrated head (fit on synthetic dev universe -- reuses same scores).
    ent2 = make_nli_entailment(calibrate=True, seed=args.seed, n_pairs=args.calib_pairs)
    ent2._score_cache = ent._score_cache  # share cached pair scores
    r, b, s = evaluate_head(ent2, pairs, "calibrated")
    per_rel += r; bins += b; summaries.append(s)
    print(f"[calibrated] acc={s['accuracy']:.3f} ece={s['ece']:.3f}")

    for row in per_rel:
        row.update(corpus=corpus.name, seed=args.seed)
    write_results(pd.DataFrame(per_rel), "exp2_per_relation", args)
    write_results(pd.DataFrame(bins), "exp2_reliability", args)
    write_results(pd.DataFrame(summaries), "exp2_summary", args)


if __name__ == "__main__":
    main()
