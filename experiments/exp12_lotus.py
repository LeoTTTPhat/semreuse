"""Experiment 12: SemReuse under LOTUS, measured by LOTUS's own counters.

Everything else in this evaluation measures a cost model we implemented.  This
experiment measures somebody else's.  We install the released ``lotus-ai``
package, point it at a local Llama-3.1-8B endpoint, and run the same workload
twice:

  lotus          -- ``df.sem_filter(...)`` per query, as a LOTUS user writes it;
  lotus+semreuse -- the same calls routed through ``SemanticFilterService``,
                    whose oracle closure is *LOTUS's own* ``sem_filter`` over
                    the rows SemReuse could not eliminate.

The numbers reported are read out of ``lotus.models.LM.stats``: LLM calls,
prompt and completion tokens, as LOTUS counts them.  Whatever reduction appears
is therefore a reduction in LOTUS's bill, not in a reimplementation of it, and
the certificate is stated with respect to LOTUS's semantics, since the audit
calls go through the same accessor.

Run with the LOTUS virtualenv:
    .venv-lotus/bin/python experiments/exp12_lotus.py --n 400 --queries 14
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402


def lm_counters(lm) -> dict:
    u = lm.stats.physical_usage
    return {"prompt_tokens": int(u.prompt_tokens),
            "completion_tokens": int(u.completion_tokens),
            "total_tokens": int(u.total_tokens)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--queries", type=int, default=14)
    ap.add_argument("--model", default="ollama/llama3.1:8b-instruct-q4_K_M")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-doc-chars", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import lotus
    from lotus.models import LM

    from semreuse.corpus import load_corpus
    from semreuse.entailment import NLIEntailment
    from semreuse.integration import SemanticFilterService
    from semreuse.predicate_log import ANALYST_LOG_20NG

    corpus = load_corpus("20ng", str(ROOT / "data"), size=args.n,
                         seed=args.seed)
    docs = [d[: args.max_doc_chars] for d in corpus.docs]
    queries = [t for _, t in ANALYST_LOG_20NG][: args.queries]
    print(f"[setup] LOTUS {args.model} N={len(docs)} queries={len(queries)}")

    def make_lm():
        lm = LM(model=args.model, api_base=args.api_base,
                max_batch_size=args.batch)
        lotus.settings.configure(lm=lm)
        return lm

    def lotus_filter(text: str, rows: np.ndarray) -> np.ndarray:
        """One LOTUS sem_filter over exactly the requested rows."""
        df = pd.DataFrame({"text": [docs[int(r)] for r in rows]})
        kept = df.sem_filter(f"{{text}}: {text}")
        out = np.zeros(len(rows), dtype=bool)
        out[kept.index.to_numpy()] = True
        return out

    out = ROOT / "results" / f"exp12_lotus_{args.n}_{args.queries}.json"

    def dump(extra=None):
        """Write what we have after every arm.

        LOTUS drives the endpoint with a large concurrent batch, which we have
        seen wedge; losing a completed 30-minute arm to a stall in the next one
        is avoidable, so the file is written incrementally.
        """
        d = dict(results)
        if extra:
            d.update(extra)
        d["config"] = vars(args)
        out.write_text(json.dumps(d, indent=2))

    results = {}

    # -- arm 1: LOTUS as a user writes it ---------------------------------
    lm = make_lm()
    t0 = time.time()
    for i, q in enumerate(queries):
        lotus_filter(q, np.arange(len(docs)))
        print(f"  [lotus {i+1}/{len(queries)}] {time.time()-t0:6.0f}s",
              flush=True)
    results["lotus"] = {"seconds": round(time.time() - t0, 1),
                        "llm_calls": len(queries) * len(docs),
                        **lm_counters(lm)}
    print("[lotus]", json.dumps(results["lotus"]))
    dump()

    # -- arm 2: the same LOTUS calls, behind SemReuse ----------------------
    lm = make_lm()
    svc = SemanticFilterService(
        corpus, lotus_filter, entailment=NLIEntailment(),
        target_recall=args.target_recall, tau=args.tau, seed=args.seed,
        oracle_version=f"lotus:{args.model}")
    t0 = time.time()
    per_query = []
    for i, q in enumerate(queries):
        r = svc.filter(q)
        per_query.append({"query": q, "calls": r.oracle_calls,
                          "kind": r.reuse_kind,
                          "recall_bound": r.recall_bound,
                          "precision_bound": r.precision_bound,
                          "seconds": round(r.seconds_total, 1)})
        print(f"  [semreuse {i+1}/{len(queries)}] calls={r.oracle_calls:5d} "
              f"kind={r.reuse_kind:10s} b={r.recall_bound} "
              f"{time.time()-t0:6.0f}s", flush=True)
        dump({"per_query_partial": per_query,
              "semreuse_partial": {"queries_done": i + 1,
                                   "llm_calls": svc.stats.oracle_calls,
                                   "seconds": round(time.time() - t0, 1),
                                   **lm_counters(lm)}})
    results["lotus+semreuse"] = {"seconds": round(time.time() - t0, 1),
                                 "llm_calls": svc.stats.oracle_calls,
                                 **lm_counters(lm)}
    print("[lotus+semreuse]", json.dumps(results["lotus+semreuse"]))

    a, b = results["lotus"], results["lotus+semreuse"]
    results["reduction"] = {
        "llm_calls": round(a["llm_calls"] / max(1, b["llm_calls"]), 2),
        "total_tokens": round(a["total_tokens"] / max(1, b["total_tokens"]), 2),
        "wall_clock": round(a["seconds"] / max(1e-9, b["seconds"]), 2)}
    results["per_query"] = per_query
    dump()
    print(f"[out] {out}")
    print("[reduction]", json.dumps(results["reduction"]))


if __name__ == "__main__":
    main()
