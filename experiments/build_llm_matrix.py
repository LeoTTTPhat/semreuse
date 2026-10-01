"""Materialize a real LLM's semantic-filter answers over a corpus.

Runs an instruction-tuned model (via a local Ollama server) once per
(predicate, row) pair at temperature 0 and stores the dense boolean matrix,
together with token counts and wall-clock time.  Because decoding is
deterministic, replaying the matrix is observationally identical to live
calls, which is what makes a full-factorial comparison of six engines over a
real LLM oracle affordable (Section 8).

Answers are also cached in SQLite, so the job is resumable: re-running after
an interruption re-issues only the missing pairs.

Usage:
  python experiments/build_llm_matrix.py --workload analyst --n 1500 \
      --model llama3.1:8b-instruct-q4_K_M
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from common import DATA_DIR, RESULTS_DIR, ROOT  # noqa: F401

sys.path.insert(0, str(ROOT / "src"))

from semreuse.corpus import load_corpus                       # noqa: E402
from semreuse.llm_oracle import (LLMResponseMatrix,           # noqa: E402
                                 OllamaClient, PROMPT_VERSION)
from semreuse.predicate_log import (analyst_log_predicates,   # noqa: E402
                                    distinct_texts)
from semreuse.predicates import PredicateUniverse, generate_workload  # noqa: E402


def workload_texts(kind: str, corpus, n_queries: int, overlap: float,
                   seed: int) -> tuple[list[str], dict]:
    if kind == "analyst":
        preds = analyst_log_predicates()
        return distinct_texts(preds), {"n_queries": len(preds)}
    if kind == "taxonomy":
        uni = PredicateUniverse.build(corpus)
        wl = generate_workload(uni, n_queries=n_queries,
                               overlap_rate=overlap, seed=seed)
        return distinct_texts(wl.queries), {"n_queries": len(wl.queries),
                                            "overlap": overlap}
    raise ValueError(kind)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", choices=["analyst", "taxonomy"],
                    default="analyst")
    ap.add_argument("--corpus", default="20ng")
    ap.add_argument("--n", type=int, default=1500, help="corpus rows")
    ap.add_argument("--n-queries", type=int, default=40)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="llama3.1:8b-instruct-q4_K_M")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=240)
    ap.add_argument("--max-doc-chars", type=int, default=1200)
    ap.add_argument("--cache", default="",
                    help="answer-cache SQLite file (default: the shared "
                         "data/llm_cache/<corpus>_<n>_<seed>.sqlite)")
    ap.add_argument("--host", default="http://localhost:11434",
                    help="Ollama endpoint")
    ap.add_argument("--order", choices=["predicate", "doc"],
                    default="predicate",
                    help="issue order: all rows per predicate (original), or "
                         "all predicates per row, which keeps each "
                         "document's prompt prefix in the server's KV cache; "
                         "answers are the same either way")
    ap.add_argument("--row-start", type=int, default=0,
                    help="doc order only: first row this worker answers, so "
                         "several builders can split one matrix by rows")
    ap.add_argument("--no-assemble", action="store_true",
                    help="fill the answer cache and exit without writing the "
                         "matrix (for split builds; assemble afterwards)")
    ap.add_argument("--rows", type=int, default=0,
                    help="materialize only the first R rows (0 = all); the "
                         "corpus subsample is already shuffled, so a prefix "
                         "is a uniform sub-corpus and the answer cache stays "
                         "reusable if R is later raised")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus, str(DATA_DIR), size=args.n,
                         seed=args.seed)
    texts, meta = workload_texts(args.workload, corpus, args.n_queries,
                                 args.overlap, args.seed)
    print(f"[matrix] corpus={corpus.name} N={corpus.n} "
          f"workload={args.workload} distinct predicates={len(texts)} "
          f"model={args.model} prompt={PROMPT_VERSION}")

    cache = args.cache or str(DATA_DIR / "llm_cache" /
                              f"{args.corpus}_{args.n}_{args.seed}.sqlite")
    def _progress(done, total, rate):
        print(f"      .. {done}/{total} rows  ({rate:.1f} rows/s)", flush=True)

    client = OllamaClient(model=args.model, host=args.host, cache_path=cache,
                          concurrency=args.concurrency, chunk=args.chunk,
                          max_doc_chars=args.max_doc_chars, seed=args.seed,
                          progress=_progress)

    n_rows = args.rows or corpus.n
    rows = np.arange(n_rows)
    answers = np.zeros((len(texts), n_rows), dtype=bool)
    out_path = str(DATA_DIR / "llm_matrix" /
                   f"{args.workload}_{args.corpus}_{args.n}_"
                   f"{args.model.replace(':', '-').replace('/', '-')}.npz")
    print(f"[matrix] materializing rows [0,{n_rows}) -> {out_path}")
    t_start = time.time()
    if args.order == "doc":
        def _doc_progress(done, total, rate):
            el = time.time() - t_start
            eta = (total - done) * el / max(1, done) / 60
            print(f"      .. {done}/{total} pairs  ({rate:.1f}/s, "
                  f"ETA {eta:.0f} min)", flush=True)
        client.progress = _doc_progress
        client.answer_doc_major(corpus, texts, rows[args.row_start:])
    if args.no_assemble:
        print(json.dumps({"filled_rows": [args.row_start, n_rows],
                          "live_calls": client.stats.calls,
                          "wall_s": round(client.stats.wall_s, 1)}))
        return
    for i, text in enumerate(texts):
        t0 = time.time()
        answers[i] = client.answer(corpus, text, rows)
        dt = time.time() - t0
        done = i + 1
        rate = (client.stats.calls / max(1e-9, client.stats.wall_s)
                if client.stats.wall_s else 0.0)
        eta = (len(texts) - done) * (time.time() - t_start) / done / 60
        print(f"[{done}/{len(texts)}] pos={answers[i].mean():.3f} "
              f"{dt:6.1f}s  live_rate={rate:5.1f} rows/s  ETA={eta:5.1f} min"
              f"   | {text[:64]}", flush=True)
        # Checkpoint after every predicate: the SQLite cache makes the job
        # resumable, and the partial matrix is useful for smoke checks.
        cum = client.cumulative_cost()
        LLMResponseMatrix(
            model=args.model, corpus_name=f"{args.corpus}-{corpus.n}",
            texts=texts[:done], answers=answers[:done],
            prompt_tokens=cum["prompt_tokens"],
            output_tokens=cum["output_tokens"],
            wall_s=cum["wall_s"],
            meta={"workload": args.workload, "seed": args.seed,
                  "cache_hits": client.stats.cache_hits,
                  "live_calls": cum["calls"],
                  "cells_answered": cum["cells"],
                  "session_calls": client.stats.calls,
                  "retries": client.stats.retries,
                  "concurrency": args.concurrency,
                  "max_doc_chars": args.max_doc_chars, **meta},
        ).save(out_path)

    st = client.stats
    print(json.dumps({
        "out": out_path, "predicates": len(texts), "rows": n_rows,
        "pairs": len(texts) * n_rows, "live_calls": st.calls,
        "cache_hits": st.cache_hits, "prompt_tokens": st.prompt_tokens,
        "output_tokens": st.output_tokens, "wall_s": round(st.wall_s, 1),
        "rows_per_s": round(st.calls / max(1e-9, st.wall_s), 2),
        "projected_dollars": round(st.dollars, 4),
        "total_wall_min": round((time.time() - t_start) / 60, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
