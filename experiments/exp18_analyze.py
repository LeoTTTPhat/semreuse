"""Analyze exp18 (SemReuse under LOTUS at scale): economics, paired latency,
and certificates against a live, non-deterministic oracle.

LOTUS at temperature 0 behind a batched endpoint does not reproduce its own
answers exactly, so a certificate checked against one LOTUS-alone run mixes two
things: the certificate's own failures and the oracle disagreeing with itself.
Both arms' answer masks are saved per query (exp18_lotus_masks_*.npz), which
separates them:

  * run-to-run agreement -- LOTUS alone in repetition 0 vs. repetition 1,
    tuple by tuple: how deterministic the oracle actually is;
  * each certificate checked against the same repetition's LOTUS answers (as
    exp18 does), against the other repetition's, and against both: a
    violation that disappears against the other draw is the oracle's
    inconsistency, one that survives both is the certificate's.

Usage:
    .venv/bin/python experiments/exp18_analyze.py --n 1000 --queries 47
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def rp(s: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    tp = int((s & ref).sum())
    rec = tp / int(ref.sum()) if ref.any() else 1.0
    prec = tp / int(s.sum()) if s.any() else 1.0
    return rec, prec


def paired(pq: pd.DataFrame) -> dict:
    """Per-query SemReuse/LOTUS wall-clock ratios, overall and per repetition."""
    w = pq.pivot_table(index=["rep", "query_index"], columns="arm",
                       values="seconds").dropna()
    if w.empty:
        return {}
    r = w["lotus+semreuse"] / w["lotus"]

    def stats(x):
        return {"n": int(len(x)), "median": round(float(x.median()), 4),
                "q25": round(float(x.quantile(.25)), 4),
                "q75": round(float(x.quantile(.75)), 4)}
    out = {"per_query_ratio": stats(r), "per_rep": []}
    for rep, g in w.groupby(level=0):
        out["per_rep"].append({"rep": int(rep), "pairs": int(len(g)),
                               "summed_ratio": round(float(
                                   g["lotus+semreuse"].sum()
                                   / g["lotus"].sum()), 4),
                               "per_query_ratio": stats(
                                   g["lotus+semreuse"] / g["lotus"])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=47)
    ap.add_argument("--tags", nargs="*", default=["", "_rep1"],
                    help="output-file suffixes to merge (repetitions may run "
                         "as separate processes)")
    args = ap.parse_args()
    tag = f"{args.n}_{args.queries}"
    arms_all, frames, masks = [], [], {}
    for t in args.tags:
        js = RES / f"exp18_lotus_{tag}{t}.json"
        if not js.exists():
            continue
        arms_all += json.loads(js.read_text())["arms"]
        frames.append(pd.read_csv(RES / f"exp18_lotus_perquery_{tag}{t}.csv"))
        zz = np.load(RES / f"exp18_lotus_masks_{tag}{t}.npz", allow_pickle=True)
        masks.update({k: zz[k] for k in zz.files if k != "queries"})
    pq = pd.concat(frames, ignore_index=True)
    # A transient server error makes the runner re-issue a whole sem_filter
    # batch; LOTUS's counters and the clock then include both attempts.  Calls
    # (tuples handed to LOTUS) and answers are unaffected, but such a pair is
    # no clean token or latency measurement, so those use pairs without retries.
    worst = pq.groupby(["rep", "query_index"]).retries.transform("max")
    clean = pq[worst == 0]
    retried = (pq[worst > 0][["rep", "query_index"]].drop_duplicates()
               .astype(int).to_dict("records"))
    z = masks
    keys = set(z)
    summ = {"arms": arms_all, "reduction": None,
            "paired_wall_clock": paired(clean)}
    reps = sorted({int(k[1:k.index("_")]) for k in keys if k.startswith("r")})

    # -- the oracle against itself -----------------------------------------
    agree, flips = [], []
    for i in range(args.queries):
        k0, k1 = f"r0_lotus_q{i:02d}", f"r1_lotus_q{i:02d}"
        if k0 in keys and k1 in keys:
            a, b = z[k0].astype(bool), z[k1].astype(bool)
            agree.append(float((a == b).mean()))
            pos = int((a | b).sum())
            flips.append(int((a ^ b).sum()) / max(1, pos))

    # -- certificates against one draw, the other, and both ------------------
    sem = pq[pq.arm == "lotus+semreuse"]
    rows = []
    for _, r in sem.iterrows():
        rep, i = int(r.rep), int(r.query_index)
        ks, kl = f"r{rep}_semreuse_q{i:02d}", f"r{rep}_lotus_q{i:02d}"
        ko = f"r{1 - rep}_lotus_q{i:02d}"
        if ks not in keys or kl not in keys:
            continue
        s, same = z[ks].astype(bool), z[kl].astype(bool)
        other = z[ko].astype(bool) if ko in keys else None
        br = r.recall_bound if pd.notna(r.recall_bound) else None
        bp = r.precision_bound if pd.notna(r.precision_bound) else None
        rec_s, prec_s = rp(s, same)
        # Shortfall in tuples: how many more of LOTUS's positives SemReuse
        # would have to report (recall), or how many more of its reported
        # positives LOTUS would have to confirm (precision), to meet the bound.
        tp = int((s & same).sum())
        r_short = (max(0, int(np.ceil(br * int(same.sum()) - 1e-9)) - tp)
                   if br is not None else 0)
        p_short = (max(0, int(np.ceil(bp * int(s.sum()) - 1e-9)) - tp)
                   if bp is not None else 0)
        # LOTUS against itself on this query (both repetitions' LOTUS-alone
        # answers), in tuples.
        k0, k1 = f"r0_lotus_q{i:02d}", f"r1_lotus_q{i:02d}"
        l2l = (int((z[k0].astype(bool) ^ z[k1].astype(bool)).sum())
               if k0 in keys and k1 in keys else None)
        # A query SemReuse evaluated in full carries bounds of 1.0 that are
        # exact against the oracle's answers *in that arm*; any shortfall
        # against the LOTUS-alone arm is then the oracle disagreeing with
        # itself, not the certificate.
        full = int(r.llm_calls) >= args.n
        row = dict(rep=rep, query_index=i, evaluated_all=full,
                   recall_bound=br,
                   precision_bound=bp, recall_same=rec_s,
                   precision_same=prec_s,
                   recall_shortfall_tuples=r_short,
                   precision_shortfall_tuples=p_short,
                   lotus_vs_lotus_flips=l2l,
                   # A bound of exactly 1.0 is exact by construction: every
                   # reported answer (recall: every tuple; precision: every
                   # reported positive) came from an oracle call in
                   # SemReuse's own arm, so a shortfall against the other arm
                   # can only be the oracle answering differently there.
                   recall_bound_exact=br is not None and br >= 1 - 1e-12,
                   precision_bound_exact=bp is not None and bp >= 1 - 1e-12,
                   rviol_same=br is not None and rec_s < br - 1e-12,
                   pviol_same=bp is not None and prec_s < bp - 1e-12)
        if other is not None:
            rec_o, prec_o = rp(s, other)
            row.update(recall_other=rec_o, precision_other=prec_o,
                       rviol_other=br is not None and rec_o < br - 1e-12,
                       pviol_other=bp is not None and prec_o < bp - 1e-12)
            row["rviol_both"] = row["rviol_same"] and row["rviol_other"]
            row["pviol_both"] = row["pviol_same"] and row["pviol_other"]
        rows.append(row)
    cert = pd.DataFrame(rows)
    has_other = "rviol_other" in cert.columns
    nr = int(cert.recall_bound.notna().sum()) if len(cert) else 0
    npc = int(cert.precision_bound.notna().sum()) if len(cert) else 0

    arms = pd.DataFrame(summ["arms"])
    tot = pq.groupby("arm")[["llm_calls"]].sum()
    totc = clean.groupby("arm")[["total_tokens", "seconds"]].sum()
    if {"lotus", "lotus+semreuse"} <= set(tot.index):
        a, b = tot.loc["lotus"], tot.loc["lotus+semreuse"]
        ac, bc = totc.loc["lotus"], totc.loc["lotus+semreuse"]
        summ["reduction"] = {
            "llm_calls": round(float(a.llm_calls / b.llm_calls), 4),
            "total_tokens": round(float(ac.total_tokens / bc.total_tokens), 4),
            "wall_clock_summed": round(float(ac.seconds / bc.seconds), 4),
            "pairs": int(pq[pq.arm == "lotus+semreuse"].shape[0]),
            "clean_pairs": int(clean[clean.arm == "lotus+semreuse"].shape[0]),
            "retried_pairs": retried}
        # SemReuse's own work (planning, NLI, audit bookkeeping): the part of
        # its arm's wall-clock not spent inside LOTUS's sem_filter.
        sr = clean[clean.arm == "lotus+semreuse"]
        summ["semreuse_own_share"] = round(float(
            (sr.seconds - sr.seconds_in_lotus).sum() / sr.seconds.sum()), 4)
    out = {
        "repetitions_done": reps,
        "queries_per_rep": int(sem.groupby("rep").size().max()) if len(sem)
        else 0,
        "arms": arms.to_dict("records"),
        "reduction": summ.get("reduction"),
        "semreuse_own_share": summ.get("semreuse_own_share"),
        "paired_wall_clock": summ.get("paired_wall_clock"),
        "lotus_run_to_run": {
            "queries_compared": len(agree),
            "mean_tuple_agreement": float(np.mean(agree)) if agree else None,
            "min_tuple_agreement": float(np.min(agree)) if agree else None,
            "mean_positive_flip_rate": float(np.mean(flips)) if flips
            else None,
            "queries_identical": int(sum(a == 1.0 for a in agree))},
        "certificates": {
            "recall": nr, "precision": npc,
            "recall_viol_same": int(cert.rviol_same.sum()) if len(cert) else 0,
            "recall_viol_same_on_fully_evaluated": int(
                (cert.rviol_same & cert.evaluated_all).sum()) if len(cert)
            else 0,
            "precision_viol_same_on_fully_evaluated": int(
                (cert.pviol_same & cert.evaluated_all).sum()) if len(cert)
            else 0,
            "queries_fully_evaluated": int(cert.evaluated_all.sum())
            if len(cert) else 0,
            "precision_viol_same": int(cert.pviol_same.sum()) if len(cert)
            else 0,
            # shortfalls on bounds of exactly 1.0: the oracle disagreeing with
            # itself by construction; the rest could be the certificate's own
            "recall_viol_same_exact_bound": int(
                (cert.rviol_same & cert.recall_bound_exact).sum())
            if len(cert) else 0,
            "precision_viol_same_exact_bound": int(
                (cert.pviol_same & cert.precision_bound_exact).sum())
            if len(cert) else 0,
            "inexact_bound_shortfalls": (
                cert[(cert.rviol_same & ~cert.recall_bound_exact)
                     | (cert.pviol_same & ~cert.precision_bound_exact)][
                    ["rep", "query_index", "recall_bound", "recall_same",
                     "recall_shortfall_tuples", "precision_bound",
                     "precision_same", "precision_shortfall_tuples",
                     "lotus_vs_lotus_flips"]].to_dict("records")
                if len(cert) else []),
            "max_shortfall_tuples": int(max(
                cert.recall_shortfall_tuples.max(),
                cert.precision_shortfall_tuples.max())) if len(cert) else 0,
            **({"recall_viol_other": int(cert.rviol_other.fillna(False).sum()),
                "precision_viol_other": int(cert.pviol_other.fillna(False)
                                            .sum()),
                "recall_viol_both": int(cert.rviol_both.fillna(False).sum()),
                "precision_viol_both": int(cert.pviol_both.fillna(False)
                                           .sum())} if has_other else {})},
    }
    (RES / f"exp18_analysis_{tag}.json").write_text(json.dumps(out, indent=2,
                                                               default=float))
    cert.to_csv(RES / f"exp18_certificates_{tag}.csv", index=False)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
