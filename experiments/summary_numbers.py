"""Headline numbers, extracted from the archived result CSVs.

Writes ``results/summary_numbers.json`` and prints a human-readable digest,
including every recall/precision certificate in every per-query file checked
against its realized value (``coverage``).

Usage:
    .venv/bin/python experiments/summary_numbers.py
"""

from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def _summary(path):
    f = RES / path
    return pd.read_csv(f).set_index("method") if f.exists() else None


def e2e(path, label):
    d = _summary(path)
    if d is None:
        return None
    cold = d.loc["cold", "total_oracle_calls"]
    out = {"corpus": label, "n_rows": int(d.iloc[0]["n_rows"]),
           "n_queries": int(d.iloc[0]["n_queries"]), "methods": {}}
    for m, r in d.iterrows():
        out["methods"][m] = {
            "calls": int(r["total_oracle_calls"]),
            "reduction": round(cold / r["total_oracle_calls"], 2),
            "P": round(r["macro_precision"], 3),
            "R": round(r["macro_recall"], 3),
            "certificates": int(r["n_bounded"]),
            "recall_violations": int(r["bound_violations"]),
            "precision_certificates": int(r.get("n_precision_bounded", 0) or 0),
            "precision_violations": int(
                r.get("precision_bound_violations", 0) or 0),
            "nli_pair_scores": int(r["total_nli_scores"]),
        }
    # The number a reviewer asks for: what does entailment add over caching?
    if "exact" in d.index and "semreuse" in d.index:
        out["marginal_over_exact_cache"] = round(
            d.loc["exact", "total_oracle_calls"]
            / d.loc["semreuse", "total_oracle_calls"], 2)
    return out


def cost_split(path, label):
    f = RES / path
    if not f.exists():
        return None
    d = pd.read_csv(f)
    out = {}
    for m, g in d.groupby("method"):
        tot = g.oracle_calls.sum()
        if tot == 0:
            continue
        out[m] = {"total": int(tot),
                  "candidates": int(g.candidate_calls.sum()),
                  "audit": int(g.audit_calls.sum()),
                  "escalation": int(g.escalation_calls.sum()),
                  "kinds": g.reuse_kind.value_counts().to_dict()}
    return {"corpus": label, "split": out}


def scale(path, label):
    f = RES / path
    if not f.exists():
        return None
    d = pd.read_csv(f)
    cold = d[d.method == "cold"].set_index("n_rows")["total_oracle_calls"]
    rows = []
    for n in sorted(d.n_rows.unique()):
        sub = d[d.n_rows == n].set_index("method")
        row = {"n_rows": int(n),
               "mean_selectivity": round(float(sub.iloc[0]
                                               ["mean_selectivity"]), 4)}
        for m in sub.index:
            row[m] = {
                "reduction": round(cold[n] / sub.loc[m,
                                                     "total_oracle_calls"], 2),
                "audit_per_query": round(sub.loc[m, "audit_calls"]
                                         / sub.loc[m, "n_queries"], 1),
                "escalation_per_query": round(sub.loc[m, "escalation_calls"]
                                              / sub.loc[m, "n_queries"], 1),
                "violations": int(sub.loc[m, "bound_violations"]),
                "certificates": int(sub.loc[m, "n_bounded"])}
        rows.append(row)
    return {"corpus": label, "points": rows}


def coverage_everywhere():
    """Certificate coverage over every run in results/.

    Violations are counted against the *oracle's own* semantics, which is what
    Theorem 2 promises fidelity to (``bound_violated_oracle``).  Under a noisy
    oracle the label-space comparison is a different, and malformed, question
    -- cold evaluation itself does not achieve label-recall 1 -- so it is
    reported separately rather than folded in.

    The live-LOTUS runs (exp18) are excluded: there the oracle is queried
    afresh in each arm and is not bitwise deterministic, so "the oracle's own
    semantics" is not one fixed answer set and a shortfall against the other
    arm mixes certificate error with the oracle disagreeing with itself.
    exp18_analyze.py separates the two; ``lotus_live`` below reports it.
    """
    tot = viol = ptot = pviol = 0
    label_space = {"certificates": 0, "violations": 0}
    per_file = []
    for f in sorted(glob.glob(str(RES / "*perquery*.csv"))) + \
            sorted(glob.glob(str(RES / "exp3_audit*.csv"))):
        d = pd.read_csv(f)
        name = pathlib.Path(f).name
        if name.startswith("exp18_lotus"):
            continue
        if "recall_bound" in d.columns:
            b = d[d.recall_bound.notna()]
            col = ("bound_violated_oracle"
                   if "bound_violated_oracle" in b.columns
                   else "bound_violated")
            v = int(b[col].fillna(False).astype(bool).sum())
            tot += len(b)
            viol += v
            if "bound_violated" in b.columns:
                label_space["certificates"] += len(b)
                label_space["violations"] += int(
                    b["bound_violated"].fillna(False).astype(bool).sum())
            if v:
                per_file.append({"file": name, "certificates": len(b),
                                 "violations": v})
        elif "n_bounded" in d.columns:          # exp3 stores aggregates
            tot += int(d.n_bounded.sum())
            viol += int(d.bound_violations.sum())
        if "precision_bound" in d.columns:
            pb = d[d.precision_bound.notna()]
            ptot += len(pb)
            pviol += int(pb.precision_bound_violated.fillna(False)
                         .astype(bool).sum())
    return {"recall_certificates": tot, "recall_violations": viol,
            "coverage_recall": round(1 - viol / max(1, tot), 5),
            "precision_certificates": ptot, "precision_violations": pviol,
            "coverage_precision": round(1 - pviol / max(1, ptot), 5),
            "alpha": 0.05, "violating_files": per_file,
            "label_space_under_noise": label_space}


def entailment():
    f = RES / "exp2_summary_20ng_medium.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    per = pd.read_csv(RES / "exp2_per_relation_20ng_medium.csv")
    out = {}
    for _, r in d.iterrows():
        h = r["head"]
        out[h] = {"accuracy": round(r["accuracy"], 3), "ece": round(r["ece"], 3)}
        for _, q in per[per["head"] == h].iterrows():
            out[h][f"f1_{q['relation']}"] = round(q["f1"], 2)
    return out


def arbiter():
    fp = sorted(glob.glob(str(RES / "exp10_pairs_*.csv")))
    fe = sorted(glob.glob(str(RES / "exp10_e2e_*.csv")))
    out = {}
    if fp:
        d = pd.read_csv(fp[-1])
        out["pairs"] = d.to_dict(orient="records")
    if fe:
        e = pd.read_csv(fe[-1]).set_index("method")
        cold = e.loc["cold", "total_oracle_calls"]
        out["e2e"] = {m: {"calls": int(r["total_oracle_calls"]),
                          "reduction": round(cold / r["total_oracle_calls"], 2),
                          "P": round(r["macro_precision"], 3),
                          "R": round(r["macro_recall"], 3),
                          "violations": int(r["bound_violations"]),
                          "certificates": int(r["n_bounded"]),
                          "arbiter_calls": int(r.get("arbiter_calls", 0)),
                          "arbiter_prompt_tokens": int(
                              r.get("arbiter_prompt_tokens", 0)),
                          "arbiter_wall_s": float(r.get("arbiter_wall_s", 0))}
                      for m, r in e.iterrows()}
    return out or None


def real_llm():
    """The real-LLM headline numbers come from the canonical run, not from
    whatever file happens to sort last.

    Selecting by ``sorted(...)[-1]`` was a latent trap: adding any later-sorting
    exp8 artifact (an N-sweep, an ablation) silently re-pointed the
    headline numbers at it. Pin the canonical basename, and refuse to
    average over a multi-N sweep if one is ever passed here.
    """
    canon = RES / "exp8_summary_20ng_full.csv"
    fs = ([str(canon)] if canon.exists()
          else sorted(glob.glob(str(RES / "exp8_summary_*.csv"))))
    canon_w = RES / "exp8_workload_20ng_full.json"
    fw = ([str(canon_w)] if canon_w.exists()
          else sorted(glob.glob(str(RES / "exp8_workload_*.json"))))
    if not fs:
        return None
    d = pd.read_csv(fs[-1])
    if d["n_rows"].nunique() > 1:            # a sweep, not the headline run
        d = d[d.n_rows == d.n_rows.max()]
    d = d.set_index("method")
    cold = d.loc["cold", "total_oracle_calls"]
    out = {"oracle_model": str(d.iloc[0]["oracle_model"]),
           "n_rows": int(d.iloc[0]["n_rows"]),
           "n_queries": int(d.iloc[0]["n_queries"]), "methods": {}}
    for m, r in d.iterrows():
        out["methods"][m] = {
            "calls": int(r["total_oracle_calls"]),
            "reduction": round(cold / r["total_oracle_calls"], 2),
            "P": round(r["macro_precision"], 3),
            "R": round(r["macro_recall"], 3),
            "certificates": int(r["n_bounded"]),
            "recall_violations": int(r["bound_violations"]),
            "precision_violations": int(
                r.get("precision_bound_violations", 0) or 0),
            "dollars": round(float(r["projected_dollars"]), 3),
            "oracle_seconds": round(float(r["oracle_seconds"]), 1)}
    if fw:
        out["workload"] = json.loads(pathlib.Path(fw[-1]).read_text())
    return out


def lotus():
    fs = sorted(glob.glob(str(RES / "exp12_lotus_*.json")))
    if not fs:
        return None
    d = json.loads(pathlib.Path(fs[-1]).read_text())
    return {k: d[k] for k in ("lotus", "lotus+semreuse", "reduction", "config")
            if k in d}


def lotus_live():
    """exp18: the full log under LOTUS at N=1,000, analyzed by
    exp18_analyze.py (economics, paired latency, run-to-run agreement, and
    certificates checked against each LOTUS draw)."""
    fs = sorted(glob.glob(str(RES / "exp18_analysis_*.json")))
    if not fs:
        return None
    return json.loads(pathlib.Path(fs[-1]).read_text())


def ablation_and_tau():
    out = {}
    f = RES / "exp5_ablation_20ng_full.csv"
    if f.exists():
        d = pd.read_csv(f)
        out["ablation"] = d.pivot(index="config", columns="entailment",
                                  values="call_reduction").round(2).to_dict()
    for corpus in ("20ng", "agnews"):
        f = RES / f"exp7_tau_{corpus}_full.csv"
        if f.exists():
            d = pd.read_csv(f)
            out[f"tau_{corpus}"] = d[["config", "call_reduction", "n_bounded",
                                      "bound_violations"]].round(2).to_dict(
                                          orient="records")
    return out


def main() -> None:
    out = {
        "e2e_20ng": e2e("exp1_summary_20ng_full.csv", "20NG"),
        "e2e_agnews": e2e("exp1_summary_agnews_full.csv", "AG News"),
        "e2e_noise": e2e("exp1_summary_20ng_full_noise05.csv", "20NG +5% noise"),
        "cost_split_20ng": cost_split("exp1_perquery_20ng_full.csv", "20NG"),
        "scale_agnews": scale("exp11_summary_agnews_full.csv", "AG News"),
        "scale_rcv1": scale("exp11_summary_rcv1_full.csv", "RCV1"),
        "proxy": e2e("exp9_summary_20ng_full.csv", "20NG proxy study"),
        "entailment": entailment(),
        "arbiter": arbiter(),
        "real_llm": real_llm(),
        "lotus": lotus(),
        "lotus_live": lotus_live(),
        "rules": ablation_and_tau(),
        "coverage": coverage_everywhere(),
    }
    (RES / "summary_numbers.json").write_text(json.dumps(out, indent=2,
                                                       default=str))
    print(f"[out] {RES/'summary_numbers.json'}")
    for k in ("e2e_20ng", "e2e_agnews", "proxy"):
        if out[k]:
            print(f"\n== {k} (N={out[k]['n_rows']}, "
                  f"{out[k]['n_queries']} queries) ==")
            for m, v in out[k]["methods"].items():
                print(f"  {m:22s} {v['calls']:10,d} {v['reduction']:6.2f}x "
                      f"P={v['P']:.3f} R={v['R']:.3f} "
                      f"cert={v['certificates']:3d} viol={v['recall_violations']}"
                      f" Pviol={v['precision_violations']}")
            if "marginal_over_exact_cache" in out[k]:
                print(f"  -> entailment adds "
                      f"{out[k]['marginal_over_exact_cache']}x over exact cache")
    for k in ("scale_agnews", "scale_rcv1"):
        if out[k]:
            print(f"\n== {k} ==")
            for p in out[k]["points"]:
                sr = p.get("semreuse", {})
                gt = p.get("semreuse-gt", {})
                print(f"  N={p['n_rows']:>7,d} s={p['mean_selectivity']:.3f} "
                      f"semreuse {sr.get('reduction')}x "
                      f"(audit/q {sr.get('audit_per_query')}, "
                      f"esc/q {sr.get('escalation_per_query')})  "
                      f"gt {gt.get('reduction')}x "
                      f"(audit/q {gt.get('audit_per_query')})")
    print(f"\n== coverage ==\n  {json.dumps(out['coverage'])}")
    for k in ("real_llm", "arbiter", "lotus", "lotus_live"):
        if out[k]:
            print(f"\n== {k} ==\n  " + json.dumps(out[k])[:900])


if __name__ == "__main__":
    main()
