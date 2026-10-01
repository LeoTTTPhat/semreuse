"""Companion analysis for exp4: what the overlap knob actually generates.

For each nominal overlap rate (same seeds as exp4_overlap.py), measures the
*realized* workload composition: exact-text repeats, availability of an
EQUIV / containment / disjoint predecessor, and query selectivity.  This
contextualizes Fig. 7 -- in a small labeled universe, even 'fresh' chains
recycle broad predicates, so nominal overlap governs the workload mixture,
not total reusability.

Usage:
    .venv/bin/python experiments/exp4_workload_stats.py --scale full --corpus 20ng
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, load_corpus_scaled, scale_params, write_results  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--overlaps", default="0.0,0.2,0.4,0.6,0.8,0.95")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from semreuse.predicates import (PredicateUniverse, Relation,
                                     generate_workload, true_relation)

    corpus = load_corpus_scaled(args)
    _, n_queries = scale_params(args)
    universe = PredicateUniverse.build(corpus)

    rows = []
    for overlap in [float(x) for x in args.overlaps.split(",")]:
        for rep in range(args.repeats):
            seed = args.seed * 1000 + rep
            wl = generate_workload(universe, n_queries=n_queries,
                                   overlap_rate=overlap, seed=seed)
            seen_texts: set[str] = set()
            prior: list = []
            n_rep = n_equiv = n_contain = n_disj = n_any = 0
            sels = []
            for q in wl.queries:
                sels.append(corpus.extension(q.label_set).mean())
                if q.text in seen_texts:
                    n_rep += 1
                rels = {true_relation(q.label_set, p.label_set)
                        for p in prior if p.text != q.text}
                if Relation.EQUIV in rels:
                    n_equiv += 1
                if Relation.FORWARD in rels or Relation.BACKWARD in rels:
                    n_contain += 1
                if Relation.DISJOINT in rels:
                    n_disj += 1
                if rels - {Relation.OVERLAP}:
                    n_any += 1
                seen_texts.add(q.text)
                prior.append(q)
            nq = len(wl.queries)
            rows.append(dict(
                overlap=overlap, rep=rep, seed=seed, n_queries=nq,
                frac_exact_repeat=n_rep / nq,
                frac_equiv_prior=n_equiv / nq,
                frac_containment_prior=n_contain / nq,
                frac_disjoint_prior=n_disj / nq,
                frac_any_relation_prior=n_any / nq,
                mean_selectivity=float(np.mean(sels)),
                corpus=corpus.name, n_rows=corpus.n))
        r = rows[-1]
        print(f"overlap={overlap:.2f} repeat={r['frac_exact_repeat']:.2f} "
              f"equiv={r['frac_equiv_prior']:.2f} "
              f"contain={r['frac_containment_prior']:.2f} "
              f"disjoint={r['frac_disjoint_prior']:.2f} "
              f"sel={r['mean_selectivity']:.3f}")

    write_results(pd.DataFrame(rows), "exp4_workload_stats", args)


if __name__ == "__main__":
    main()
