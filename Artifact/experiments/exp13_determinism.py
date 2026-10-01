"""Experiment 13: is the oracle actually deterministic?

Two claims in this paper rest on it. Theorem 2 assumes a row evaluated twice
-- once as a candidate, again as an audit sample -- returns the same answer;
and the real-LLM study replays a materialized answer matrix on the grounds
that replay is observationally identical to live calls. Both are assumptions
about a piece of software we did not write, so we measure them rather than
assert them: re-issue a random sample of already-answered (predicate, row)
pairs and compare against the cache.

The sample is drawn from the cache with a fixed seed and re-issued through a
*fresh* client with the cache disabled, so nothing short-circuits.

Usage:
    .venv/bin/python experiments/exp13_determinism.py --n-pairs 200
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(DATA_DIR / "llm_cache" /
                                           "20ng_4000_0.sqlite"))
    ap.add_argument("--corpus", default="20ng")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--model", default="llama3.1:8b-instruct-q4_K_M")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--host", default="http://localhost:11434",
                    help="endpoint to re-issue through; pointing it at a "
                         "server configured differently from the one that "
                         "built the cache (e.g. one slot instead of eight) "
                         "also tests that answers do not depend on batching "
                         "or on the server's prompt-prefix cache")
    ap.add_argument("--rows", type=int, default=0,
                    help="sample only pairs with row < ROWS (0 = any)")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    from semreuse.corpus import load_corpus
    from semreuse.llm_oracle import OllamaClient

    corpus = load_corpus(args.corpus, str(DATA_DIR), size=args.n,
                         seed=args.seed)
    db = sqlite3.connect(args.cache)
    rows = db.execute(
        "SELECT claim, row, answer FROM ans WHERE model=? "
        + ("AND row < ? " if args.rows else "") + "ORDER BY claim, row",
        (args.model, args.rows) if args.rows else (args.model,)).fetchall()
    if not rows:
        raise SystemExit(f"no cached answers for {args.model} in {args.cache}")
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(rows), size=min(args.n_pairs, len(rows)),
                      replace=False)
    sample = [rows[int(i)] for i in pick]
    print(f"[setup] re-issuing {len(sample)} of {len(rows)} cached answers "
          f"through a fresh, cache-less client")

    # cache_path=None: nothing is memoized, so every call really goes out.
    client = OllamaClient(model=args.model, host=args.host, cache_path=None,
                          concurrency=args.concurrency, seed=args.seed)
    recs = []
    for claim, row, cached in sample:
        got = bool(client.answer(corpus, claim, np.array([int(row)]))[0])
        recs.append({"claim": claim[:60], "row": int(row),
                     "cached": bool(cached), "reissued": got,
                     "agree": got == bool(cached)})
    df = pd.DataFrame(recs)
    agree = float(df.agree.mean())
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out = RESULTS_DIR / f"exp13_determinism{tag}.csv"
    df.to_csv(out, index=False)
    summary = {"model": args.model, "host": args.host, "pairs": len(df),
               "agreement": round(agree, 4),
               "disagreements": int((~df.agree).sum()),
               "live_calls": client.stats.calls,
               "wall_s": round(client.stats.wall_s, 1),
               "retries": client.stats.retries}
    (RESULTS_DIR / f"exp13_determinism{tag}.json").write_text(
        json.dumps(summary, indent=2))
    print(f"[out] {out}")
    print("[determinism] " + json.dumps(summary))


if __name__ == "__main__":
    main()
