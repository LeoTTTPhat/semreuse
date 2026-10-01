"""Experiment 18: SemReuse under LOTUS at scale, with repeated, paired latency.

The first LOTUS run (exp12: 300 tuples, 6 filters) shows the seam works, but
at that size the audit floor consumes everything the rewrites free, so it
demonstrates compatibility rather than benefit.  This run puts the whole
analyst-written log (47 filters, in issue order) over the same tuples the
real-oracle study uses.

The GPU is shared with other long jobs whose load drifts over hours, so the
two arms are interleaved *per query* rather than run one after the other.
In repetition ``rep``, query ``i`` is answered by both arms back to back,

    (lotus, lotus+semreuse)   if (i + rep) is even,
    (lotus+semreuse, lotus)   otherwise,

so the two answers to a query see the same machine state and neither arm
systematically goes first.  Each arm has its own LOTUS ``LM``, switched in
with ``lotus.settings.configure(lm=...)`` before each of its calls, so LLM
tokens are read from LOTUS's own counters (``lotus.models.LM.stats``) per arm;
each repetition starts SemReuse from an empty store.  Latency is reported as
per-arm totals and as the distribution of paired per-query ratios.  The
LOTUS-alone answer to a query is the reference that query's SemReuse
certificate is checked against.  LOTUS runs at temperature 0, but behind a
batched serving endpoint it is not bitwise reproducible, so both arms' answer
masks are saved (``exp18_lotus_masks_{N}_{Q}.npz``, keys ``r{rep}_lotus_q{i}``
and ``r{rep}_semreuse_q{i}``) and the per-query CSV records each pair's
tuple-level agreement; LOTUS's own run-to-run agreement (cold SemReuse vs
LOTUS-alone, LOTUS-alone rep 0 vs rep 1) can then be separated from any
certificate violation.

Run with the LOTUS virtualenv (the proxy fans requests out over several Ollama
servers; see rr_proxy.py):
    .venv-lotus/bin/python experiments/exp18_lotus_scale.py --n 1000 \
        --repeats 2 --api-base http://127.0.0.1:11500 --batch 64
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
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402
import faulthandler     # noqa: E402
import signal           # noqa: E402

# `kill -USR1 <pid>` writes every thread's Python stack to the log; the memory
# guard does this before restarting a repetition whose log has gone silent.
faulthandler.register(signal.SIGUSR1, all_threads=True)

ARMS = ("lotus", "lotus+semreuse")


def lm_counters(lm) -> dict:
    u = lm.stats.physical_usage
    return {"prompt_tokens": int(u.prompt_tokens),
            "completion_tokens": int(u.completion_tokens),
            "total_tokens": int(u.total_tokens)}


def ratio_stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "median": round(float(q50), 4),
            "q25": round(float(q25), 4), "q75": round(float(q75), 4),
            "iqr": round(float(q75 - q25), 4)}


def paired_summary(rows: list[dict], n_queries: int) -> dict:
    """Paired wall-clock comparison over the query pairs completed so far."""
    d = pd.DataFrame(rows)
    if d.empty:
        return {}
    w = d.pivot_table(index=["rep", "query_index"], columns="arm",
                      values="seconds").dropna()
    if w.empty or not set(ARMS) <= set(w.columns):
        return {}
    order = (d[d.arm == "lotus"].set_index(["rep", "query_index"])["order"]
             .reindex(w.index))
    ratio = w["lotus+semreuse"] / w["lotus"]
    out = {"ratio": "lotus+semreuse seconds / lotus seconds",
           "per_query_ratio": ratio_stats(ratio),
           "per_query_ratio_by_order": {
               o: ratio_stats(ratio[order == o])
               for o in ("lotus-first", "semreuse-first")},
           "per_rep": []}
    for rep, g in w.groupby(level="rep"):
        s_l, s_s = float(g["lotus"].sum()), float(g["lotus+semreuse"].sum())
        out["per_rep"].append({
            "rep": int(rep), "pairs": int(len(g)),
            "complete": bool(len(g) == n_queries),
            "lotus_seconds": round(s_l, 1), "semreuse_seconds": round(s_s, 1),
            "summed_ratio": round(s_s / s_l, 4),
            "per_query_ratio": ratio_stats(g["lotus+semreuse"] / g["lotus"])})
    done = [r["summed_ratio"] for r in out["per_rep"] if r["complete"]]
    if done:
        out["summed_ratio_mean_over_complete_reps"] = round(
            float(np.mean(done)), 4)
    return out


def _cache_ollama_model_info() -> None:
    """Serve LiteLLM's Ollama model metadata from memory.

    Before every completion LiteLLM looks the model up with a POST to
    ``/api/show`` (``OllamaModelInfo.get_model_info``) through a module-level
    HTTP client whose timeout is 6,000 s; one lost response then stalls a
    whole LOTUS batch for up to 100 minutes, which is how two earlier runs
    hung.  The metadata cannot change during a run, so it is fetched once, with
    a 30-second limit and LiteLLM's own fallback, and reused.  LOTUS's prompts,
    batching and answer parsing are untouched, and both arms share it.
    """
    import threading
    from litellm.llms.ollama.common_utils import OllamaModelInfo
    original = OllamaModelInfo.get_model_info
    cache: dict = {}
    lock = threading.Lock()

    def fallback(model: str) -> dict:          # LiteLLM's on-error value
        return {"key": model, "litellm_provider": "ollama", "mode": "chat",
                "input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
                "max_tokens": None, "max_input_tokens": None,
                "max_output_tokens": None}

    def cached(self, model, api_base=None, api_key=None):
        key = (model, api_base)
        with lock:
            if key not in cache:
                box: dict = {}
                t = threading.Thread(daemon=True, target=lambda: box.update(
                    v=original(self, model, api_base=api_base,
                               api_key=api_key)))
                t.start()
                t.join(30)
                cache[key] = box["v"] if "v" in box else fallback(model)
                print(f"[litellm] model info for {model} "
                      f"{'fetched' if 'v' in box else 'timed out; fallback'}"
                      " once, reused for every call", flush=True)
            return cache[key]

    OllamaModelInfo.get_model_info = cached


def write_atomic(path: pathlib.Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=0, help="0 = whole log")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--model", default="ollama/llama3.1:8b-instruct-q4_K_M")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-doc-chars", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-rep", type=int, default=-1,
                    help="run just this repetition (0-based); repetitions are "
                         "independent (fresh store each), so they may run as "
                         "separate processes in parallel")
    ap.add_argument("--out-tag", default="",
                    help="suffix for the output files, e.g. _rep1")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run after its last completed "
                         "pair, from the outputs and exp18_state_*.pkl of the "
                         "same --n/--out-tag (SemReuse's store, answer memo "
                         "and RNG are restored, so the run continues exactly)")
    ap.add_argument("--attempts", type=int, default=3,
                    help="tries per LOTUS sem_filter call before giving up "
                         "(a retry is logged and counted per arm)")
    args = ap.parse_args()

    import lotus
    from lotus.models import LM
    _cache_ollama_model_info()

    from common import make_nli_entailment
    from semreuse.corpus import load_corpus
    from semreuse.integration import SemanticFilterService
    from semreuse.predicate_log import ANALYST_LOG_20NG

    # The same tuples as the real-oracle study: a prefix of the seed-0
    # 4,000-document subsample.
    base = load_corpus("20ng", str(ROOT / "data"), size=4000, seed=args.seed)
    from semreuse.corpus import Corpus
    corpus = Corpus(name=f"20ng-pre{args.n}", docs=base.docs[:args.n],
                    leaf_labels=base.leaf_labels[:args.n],
                    taxonomy=dict(base.taxonomy))
    docs = [d[: args.max_doc_chars] for d in corpus.docs]
    log = [t for _, t in ANALYST_LOG_20NG]
    queries = log[: args.queries] if args.queries else log
    nq = len(queries)
    nli = make_nli_entailment(seed=args.seed, n_pairs=300)
    print(f"[setup] LOTUS {args.model} @ {args.api_base} N={len(docs)} "
          f"queries={nq} repeats={args.repeats} (arms interleaved per query)",
          flush=True)

    def make_lm():
        return LM(model=args.model, api_base=args.api_base,
                  max_batch_size=args.batch)

    # Rows handed to LOTUS, seconds spent inside it, and retries.  The closure
    # is shared by both arms; which arm pays is whichever LM is configured.
    inside = {"rows": 0, "seconds": 0.0, "retries": 0}

    def lotus_filter(text: str, rows: np.ndarray) -> np.ndarray:
        df = pd.DataFrame({"text": [docs[int(r)] for r in rows]})
        t = time.perf_counter()
        for attempt in range(1, max(1, args.attempts) + 1):
            try:
                kept = df.sem_filter(f"{{text}}: {text}")
                break
            except Exception as e:           # transient server/HTTP failure
                if attempt >= args.attempts:
                    raise
                inside["retries"] += 1
                print(f"  [retry {attempt}/{args.attempts - 1}] "
                      f"{len(rows)} rows: {type(e).__name__}: {e}", flush=True)
                time.sleep(15)
        inside["seconds"] += time.perf_counter() - t
        inside["rows"] += len(rows)
        out = np.zeros(len(rows), dtype=bool)
        out[kept.index.to_numpy()] = True
        return out

    def run_arm(arm: str, lm, q: str, svc) -> dict:
        lotus.settings.configure(lm=lm)
        tok0 = lm_counters(lm)
        rows0, sec0, ret0 = inside["rows"], inside["seconds"], inside["retries"]
        started = time.time()
        t = time.perf_counter()
        if arm == "lotus":           # LOTUS as a user writes it
            res, mask = None, lotus_filter(q, np.arange(len(docs)))
        else:                        # the same LOTUS calls, behind SemReuse
            res = svc.filter(q)
            mask = res.mask
        dt = time.perf_counter() - t
        tok1 = lm_counters(lm)
        g = {"res": res, "mask": mask, "seconds": dt, "started_at": started,
             "llm_calls": inside["rows"] - rows0,
             "seconds_in_lotus": inside["seconds"] - sec0,
             "retries": inside["retries"] - ret0,
             **{k: tok1[k] - tok0[k] for k in tok1}}
        if res is not None and res.oracle_calls != g["llm_calls"]:
            print(f"  [warn] SemReuse counted {res.oracle_calls} calls, "
                  f"LOTUS received {g['llm_calls']} rows", flush=True)
        return g

    out_json = ROOT / "results" / f"exp18_lotus_{args.n}_{nq}{args.out_tag}.json"
    out_csv = ROOT / "results" / f"exp18_lotus_perquery_{args.n}_{nq}{args.out_tag}.csv"
    out_npz = ROOT / "results" / f"exp18_lotus_masks_{args.n}_{nq}{args.out_tag}.npz"
    summary: dict = {
        "config": vars(args),
        "design": ("arms interleaved per query: (lotus, lotus+semreuse) when "
                   "(query_index + rep) is even, reversed when odd; one LOTUS "
                   "LM per arm per repetition; fresh SemReuse store per "
                   "repetition; LOTUS-alone answers are the reference for "
                   "SemReuse's certificates; both arms' answer masks saved to "
                   f"{out_npz.name}"),
        "arms": []}
    rows: list[dict] = []
    # Answer masks, rewritten after every pair: r{rep}_lotus_q{i:02d} is the
    # LOTUS-alone answer, r{rep}_semreuse_q{i:02d} SemReuse's reported answer.
    masks: dict[str, np.ndarray] = {"queries": np.array(queries)}

    def save_masks(p: pathlib.Path) -> None:
        with open(p, "wb") as f:     # a file object: savez adds no suffix
            np.savez_compressed(f, **masks)

    def dump():
        summary["paired_wall_clock"] = paired_summary(rows, nq)
        write_atomic(out_json,
                     lambda p: p.write_text(json.dumps(summary, indent=2)))
        write_atomic(out_csv,
                     lambda p: pd.DataFrame(rows).to_csv(p, index=False))
        write_atomic(out_npz, save_masks)

    # Crash insurance: after every completed pair, SemReuse's mutable state
    # (view store, oracle answer memo, audit RNG, slack store, counters) is
    # pickled, so --resume continues the repetition exactly where it stopped.
    import pickle
    state_p = (ROOT / "results"
               / f"exp18_state_{args.n}_{nq}{args.out_tag}.pkl")
    resume = None
    if args.resume and state_p.exists():
        resume = pickle.loads(state_p.read_bytes())
        prev = json.loads(out_json.read_text())
        summary["arms"] = prev.get("arms", [])
        summary["resumed"] = prev.get("resumed", []) + [
            {"rep": resume["rep"], "after_pairs": resume["pairs_done"],
             "at": time.strftime("%Y-%m-%d %H:%M:%S")}]
        rows.extend(pd.read_csv(out_csv).to_dict("records"))
        with np.load(out_npz, allow_pickle=True) as z:
            masks.update({k: z[k] for k in z.files if k != "queries"})
        print(f"[resume] rep {resume['rep']} after {resume['pairs_done']}/"
              f"{nq} pairs ({len(rows)} rows, {len(masks) - 1} masks loaded)",
              flush=True)

    def save_state(rep: int, pairs_done: int, svc, tot, cert) -> None:
        eng = svc.engine
        st = {"rep": rep, "pairs_done": pairs_done, "store": eng.store,
              "rng": eng.rng.bit_generator.state,
              "query_index": eng._query_index, "slack": eng.slack,
              "memo": svc.oracle._memo, "oracle_stats": svc.oracle.stats,
              "svc_stats": svc.stats, "tot": tot, "cert": cert}
        write_atomic(state_p, lambda p: p.write_bytes(pickle.dumps(st)))

    t_start = time.time()
    reps = ([args.only_rep] if args.only_rep >= 0
            else list(range(args.repeats)))
    for rep in reps:
        lms = {arm: make_lm() for arm in ARMS}
        svc = SemanticFilterService(
            corpus, lotus_filter, entailment=nli,
            target_recall=args.target_recall, tau=args.tau, seed=args.seed,
            oracle_version=f"lotus:{args.model}")
        tot = {arm: {"seconds": 0.0, "llm_calls": 0, "seconds_in_lotus": 0.0,
                     "retries": 0} for arm in ARMS}
        cert = dict.fromkeys(("recall_certificates", "recall_violations",
                              "precision_certificates",
                              "precision_violations"), 0)
        start = 0
        if resume is not None and resume["rep"] > rep:
            continue                 # finished before the interruption
        if resume is not None and resume["rep"] == rep:
            eng = svc.engine
            eng.store, eng.slack = resume["store"], resume["slack"]
            eng.rng.bit_generator.state = resume["rng"]
            eng._query_index = resume["query_index"]
            svc.oracle._memo = resume["memo"]
            svc.oracle.stats = resume["oracle_stats"]
            svc.stats = resume["svc_stats"]
            tot, cert = resume["tot"], resume["cert"]
            start = resume["pairs_done"]
        for i, q in enumerate(queries):
            if i < start:
                continue
            order = ARMS if (i + rep) % 2 == 0 else ARMS[::-1]
            tag = "lotus-first" if order[0] == "lotus" else "semreuse-first"
            got: dict[str, dict] = {}
            for pos, arm in enumerate(order, 1):
                g = got[arm] = run_arm(arm, lms[arm], q, svc)
                g["position"] = pos
                for k in tot[arm]:
                    tot[arm][k] += g[k]
                extra = ""
                if g["res"] is not None:
                    r = g["res"]
                    extra = (f" kind={r.reuse_kind:9s} b_rec={r.recall_bound}"
                             f" b_prec={r.precision_bound}")
                print(f"  [rep {rep} q {i+1:2d}/{nq} #{pos} {arm:14s}] "
                      f"calls={g['llm_calls']:5d} {g['seconds']:7.1f}s "
                      f"({g['llm_calls'] / max(g['seconds'], 1e-9):6.2f} "
                      f"calls/s, {g['seconds_in_lotus']:7.1f}s in LOTUS) "
                      f"tok={g['total_tokens']}{extra}", flush=True)

            # SemReuse's certificate, checked against LOTUS's own cold answer.
            ref = got["lotus"]["mask"]
            r = got["lotus+semreuse"]["res"]
            tp = int((r.mask & ref).sum())
            rec = tp / max(1, int(ref.sum())) if ref.any() else 1.0
            prec = tp / max(1, int(r.mask.sum())) if r.mask.any() else 1.0
            rv = (r.recall_bound is not None
                  and rec < r.recall_bound - 1e-12)
            pv = (r.precision_bound is not None
                  and prec < r.precision_bound - 1e-12)
            cert["recall_certificates"] += r.recall_bound is not None
            cert["precision_certificates"] += r.precision_bound is not None
            cert["recall_violations"] += bool(rv)
            cert["precision_violations"] += bool(pv)

            # Tuple-level agreement of the two arms' answers (a cold SemReuse
            # query measures LOTUS's own run-to-run agreement).
            m_l = np.array(ref, dtype=bool)
            m_s = np.array(r.mask, dtype=bool)
            masks[f"r{rep}_lotus_q{i:02d}"] = m_l
            masks[f"r{rep}_semreuse_q{i:02d}"] = m_s
            agree = float((m_l == m_s).mean())
            only_l, only_s = int((m_l & ~m_s).sum()), int((m_s & ~m_l).sum())

            for arm in ARMS:
                g = got[arm]
                row = {"rep": rep, "arm": arm, "query_index": i, "query": q,
                       "order": tag, "position": g["position"],
                       "llm_calls": g["llm_calls"],
                       "positives": int((m_l if arm == "lotus" else m_s).sum()),
                       # pair-level: same value on both rows of the pair
                       "agreement": round(agree, 6),
                       "pos_only_lotus": only_l,
                       "pos_only_semreuse": only_s}
                if arm == "lotus+semreuse":
                    row.update({
                        "kind": r.reuse_kind,
                        "recall_bound": r.recall_bound,
                        "precision_bound": r.precision_bound,
                        "recall_vs_lotus": rec, "precision_vs_lotus": prec,
                        # the archive-wide coverage count reads these
                        "recall_oracle": rec,
                        "bound_violated_oracle": (None if r.recall_bound
                                                  is None else bool(rv)),
                        "precision_bound_violated": (
                            None if r.precision_bound is None
                            else bool(pv))})
                row.update({
                    "seconds": round(g["seconds"], 3),
                    "seconds_in_lotus": round(g["seconds_in_lotus"], 3),
                    "prompt_tokens": g["prompt_tokens"],
                    "completion_tokens": g["completion_tokens"],
                    "total_tokens": g["total_tokens"],
                    "retries": g["retries"],
                    "started_at": round(g["started_at"], 3)})
                rows.append(row)

            s_l, s_s = tot["lotus"]["seconds"], tot["lotus+semreuse"]["seconds"]
            print(f"  [rep {rep} q {i+1:2d}/{nq} pair {tag}] "
                  f"semreuse/lotus="
                  f"{got['lotus+semreuse']['seconds'] / got['lotus']['seconds']:.3f}"
                  f" R={rec:.3f} P={prec:.3f} viol(rec,prec)="
                  f"({int(rv)},{int(pv)}) agree={agree:.4f} "
                  f"only(lotus,semreuse)=({only_l},{only_s})"
                  f" | rep so far lotus {s_l:.0f}s "
                  f"semreuse {s_s:.0f}s ({s_s / s_l:.3f}) | elapsed "
                  f"{time.time() - t_start:.0f}s", flush=True)
            summary["progress"] = {"rep": rep, "pairs_done": i + 1,
                                   "queries": nq, "repeats": args.repeats,
                                   "elapsed_s": round(time.time() - t_start)}
            dump()
            save_state(rep, i + 1, svc, tot, cert)

        for arm in ARMS:
            entry = {"rep": rep, "arm": arm,
                     "seconds": round(tot[arm]["seconds"], 1),
                     "llm_calls": tot[arm]["llm_calls"],
                     "seconds_in_lotus": round(tot[arm]["seconds_in_lotus"], 1),
                     "retries": tot[arm]["retries"],
                     # summed per query, so a resumed repetition counts the
                     # tokens spent before the interruption too
                     **{k: int(sum(r[k] for r in rows
                                   if r["rep"] == rep and r["arm"] == arm))
                        for k in ("prompt_tokens", "completion_tokens",
                                  "total_tokens")}}
            if arm == "lotus+semreuse":
                entry.update(cert)
            summary["arms"].append(entry)
            print(f"[rep {rep} {arm}] {json.dumps(entry)}", flush=True)
        dump()

    arms = pd.DataFrame(summary["arms"])
    agg = arms.groupby("arm").agg(seconds_mean=("seconds", "mean"),
                                  seconds_std=("seconds", "std"),
                                  calls=("llm_calls", "mean"),
                                  tokens=("total_tokens", "mean"))
    a, b = agg.loc["lotus"], agg.loc["lotus+semreuse"]
    summary["reduction"] = {"llm_calls": round(a.calls / b.calls, 3),
                            "total_tokens": round(a.tokens / b.tokens, 3),
                            "wall_clock": round(a.seconds_mean
                                                / b.seconds_mean, 3)}
    summary.pop("progress", None)
    dump()
    print("[summary]", agg.round(1).to_dict())
    print("[reduction]", json.dumps(summary["reduction"]))
    print("[paired]", json.dumps(summary["paired_wall_clock"]))
    print(f"[out] {out_json}\n[out] {out_csv}\n[out] {out_npz}")


if __name__ == "__main__":
    main()
