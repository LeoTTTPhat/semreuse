"""Experiment 17: the analyst-written log on several real LLM oracles.

The real-oracle result of Section 6.2 rests on one 8B model.  This script puts
the same fixed predicate log in front of every materialized oracle and
reports, per model:

  self-consistency -- how many of the log's 24 declared implications hold
                      exactly on the model's own extensions, by how much the
                      rest miss (median slack), how asymmetrically paraphrases
                      are honoured, and how many queries have a containment
                      predecessor at slack 0.02 -- all computed at a common
                      corpus size, because these statistics flatter small N;
  economics        -- reduction in oracle calls, and the escalation and audit
                      shares of cold cost (from the exp8 run on that matrix);
  validity         -- recall and precision certificate violations;
  determinism      -- re-issued pairs that reproduce the stored answer (exp13).

It also measures how often the models agree with *each other* tuple by tuple,
which is what makes a certificate relative to the host oracle, rather than to
some model-independent truth, the only well-posed promise.

Usage (after build_llm_matrix.py, exp8_realllm.py and exp13_determinism.py
have been run per model):
    .venv/bin/python experiments/exp17_multi_oracle.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR  # noqa: E402

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

M = DATA_DIR / "llm_matrix"
# (short name, display name, params in B, family, matrix, exp8 tag, exp13 tag)
MODELS = [
    ("llama8b", "Llama-3.1-8B", 8, "Llama",
     M / "frozen_analyst_20ng_2000_llama3.1-8b.npz", "", ""),
    ("qwen14b", "Qwen2.5-14B", 14, "Qwen",
     M / "analyst_20ng_4000_qwen2.5-14b-instruct-q4_K_M.npz", "qwen14b",
     "qwen14b"),
    ("mistral24b", "Mistral-Small-3.1-24B", 24, "Mistral",
     M / "analyst_20ng_4000_mistral-small3.1-24b.npz", "mistral24b",
     "mistral24b"),
    ("gemma27b", "Gemma-3-27B", 27, "Gemma",
     M / "analyst_20ng_4000_gemma3-27b.npz", "gemma27b_n500",
     "gemma27b"),
]


def self_consistency(matrix, n: int, eps: float = 0.02) -> dict:
    from semreuse.predicate_log import (ExtensionalEntailment,
                                        analyst_log_predicates,
                                        designed_relation_slack)
    from semreuse.predicates import Relation

    idx = matrix.index
    ans = matrix.answers[:, :n]

    def ext_of(t):
        return ans[idx[t]]
    rel = designed_relation_slack(ext_of)
    slack = np.array([r["slack"] for r in rel])
    para = [r for r in rel if r["kind"] == "paraphrase"]
    worse = np.array([max(r["slack"], r["slack_reverse"]) for r in para])
    better = np.array([min(r["slack"], r["slack_reverse"]) for r in para])
    ent = ExtensionalEntailment(ext_of, eps=eps)
    queries = [q for q in analyst_log_predicates() if q.text in idx]
    seen, cont, anyp = [], 0, 0
    usable = {Relation.EQUIV, Relation.FORWARD, Relation.BACKWARD,
              Relation.DISJOINT}
    for q in queries:
        rels = [ent.relation(q.text, t) for t in seen if t != q.text]
        cont += any(r in (Relation.FORWARD, Relation.BACKWARD) for r in rels)
        anyp += any(r in usable for r in rels)
        seen.append(q.text)
    return dict(
        n_rows=n, declared=len(rel), exact=int((slack == 0).sum()),
        median_slack=float(np.median(slack)),
        mean_slack=float(slack.mean()), max_slack=float(slack.max()),
        within_budget=int((slack <= 0.10).sum()),
        paraphrase_worse_median=float(np.median(worse)),
        paraphrase_better_median=float(np.median(better)),
        containment_predecessor=cont / len(queries),
        any_predecessor=anyp / len(queries),
        selectivity=float(ans.mean()))


def economics(e8_tag: str, n: int) -> dict:
    """End-to-end numbers for one oracle at one corpus size, from exp8."""
    tag = f"_{e8_tag}" if e8_tag else ""
    summ = RESULTS_DIR / f"exp8_summary_20ng_full{tag}.csv"
    if not summ.exists():
        return {}
    s = pd.read_csv(summ)
    if n not in set(s.n_rows):
        return {}
    s = s[s.n_rows == n].set_index("method")
    pq = pd.read_csv(RESULTS_DIR / f"exp8_perquery_20ng_full{tag}.csv")
    sr = pq[(pq.method == "semreuse") & (pq.n_rows == n)]
    cold = s.loc["cold", "total_oracle_calls"]
    r: dict = {"cold_calls": int(cold)}
    for meth, key in (("semreuse", "sr"), ("semreuse-noaudit", "noaudit"),
                      ("semreuse-gt", "gt"), ("exact", "exact"),
                      ("embed@0.8", "embed08")):
        if meth in s.index:
            r[f"{key}_calls"] = int(s.loc[meth, "total_oracle_calls"])
            r[f"{key}_reduction"] = float(
                cold / s.loc[meth, "total_oracle_calls"])
            r[f"{key}_precision"] = float(s.loc[meth, "macro_precision"])
            r[f"{key}_recall"] = float(s.loc[meth, "macro_recall"])
    r.update(
        sr_candidate_share=float(sr.candidate_calls.sum() / cold),
        sr_audit_share=float(sr.audit_calls.sum() / cold),
        sr_escalation_share=float(sr.escalation_calls.sum() / cold),
        sr_recall_certs=int(s.loc["semreuse", "n_bounded"]),
        sr_recall_viol=int(s.loc["semreuse", "bound_violations_oracle"]),
        sr_precision_certs=int(s.loc["semreuse", "n_precision_bounded"]),
        sr_precision_viol=int(s.loc["semreuse",
                                    "precision_bound_violations"]))
    if "semreuse-gt" in s.index:
        r.update(gt_recall_certs=int(s.loc["semreuse-gt", "n_bounded"]),
                 gt_recall_viol=int(s.loc["semreuse-gt",
                                          "bound_violations_oracle"]))
    return r


def main() -> None:
    from semreuse.llm_oracle import LLMResponseMatrix

    mats = {}
    for short, name, *_rest in MODELS:
        path = _rest[2]
        if Path(path).exists():
            m = LLMResponseMatrix.load(str(path))
            mats[short] = m
            print(f"[load] {name}: {m.answers.shape}")
        else:
            print(f"[skip] {name}: {path} not built")

    rows = []
    for short, name, params, family, path, e8, e13 in MODELS:
        if short not in mats:
            continue
        m = mats[short]
        width = m.answers.shape[1]
        for n in sorted({500, width} & set(range(1, width + 1))):
            r = dict(model=short, display=name, params_b=params,
                     family=family, **self_consistency(m, n))
            r.update(economics(e8, n))
            if n == width:
                det = RESULTS_DIR / (f"exp13_determinism_{e13}.json" if e13
                                     else "exp13_determinism.json")
                if det.exists():
                    d = json.loads(det.read_text())
                    r.update(determinism_pairs=int(d["pairs"]),
                             determinism_agree=int(round(d["agreement"]
                                                         * d["pairs"])))
                prompt = m.prompt_tokens / max(1, int(
                    m.meta.get("cells_answered") or m.answers.size))
                r.update(prompt_tokens_per_call=prompt,
                         matrix_cells=int(m.answers.size))
            rows.append(r)
            print(f"[{name:22s} N={n:5d}] exact {r['exact']}/{r['declared']}"
                  f"  median slack {r['median_slack']:.3f}  paraphrase "
                  f"worse/better {r['paraphrase_worse_median']:.3f}/"
                  f"{r['paraphrase_better_median']:.3f}  containment-pred "
                  f"{r['containment_predecessor']:.2f}"
                  + (f"  reduction {r['sr_reduction']:.2f}x esc "
                     f"{100*r['sr_escalation_share']:.1f}%"
                     if "sr_reduction" in r else ""), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp17_multi_oracle.csv", index=False)

    # Every declared implication on every oracle, at the prefix they share:
    # which ones each model honours, and by how much the others miss.
    from semreuse.predicate_log import designed_relation_slack
    common = min(m.answers.shape[1] for m in mats.values())
    imp = []
    for short, m in mats.items():
        for n in sorted({common, m.answers.shape[1]}):
            sub = m.answers[:, :n]
            for r in designed_relation_slack(
                    lambda t, sub=sub, m=m: sub[m.index[t]]):
                imp.append(dict(model=short, n_rows=n, **r))
    pd.DataFrame(imp).to_csv(RESULTS_DIR / "exp17_implications.csv",
                             index=False)

    # Tuple-level agreement between models, on the prefix they all share.
    agree = []
    for a, b in itertools.combinations(mats, 2):
        A, B = mats[a], mats[b]
        texts = [t for t in A.texts if t in B.index]
        x = np.stack([A.answers[A.index[t], :common] for t in texts])
        y = np.stack([B.answers[B.index[t], :common] for t in texts])
        po = float((x == y).mean())
        pa, pb = x.mean(), y.mean()
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (po - pe) / (1 - pe)
        both = (x & y).sum() / max(1, (x | y).sum())
        agree.append(dict(a=a, b=b, n_rows=common, predicates=len(texts),
                          agreement=po, kappa=float(kappa),
                          positive_jaccard=float(both)))
        print(f"[agree N={common}] {a:10s} vs {b:10s}: agreement {po:.3f} "
              f"kappa {kappa:.3f} positive Jaccard {both:.3f}")
    pd.DataFrame(agree).to_csv(RESULTS_DIR / "exp17_model_agreement.csv",
                               index=False)
    print(f"[out] {RESULTS_DIR/'exp17_multi_oracle.csv'}\n"
          f"[out] {RESULTS_DIR/'exp17_model_agreement.csv'}")


if __name__ == "__main__":
    main()
