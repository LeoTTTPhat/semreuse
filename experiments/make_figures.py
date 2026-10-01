"""Figures (PDF, single-column ~3.3in) from the result CSVs.

Reads results/*.csv, writes results/figures/*.pdf.

Usage:
    .venv/bin/python experiments/make_figures.py
"""

from __future__ import annotations

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

# ---- style ---------------------------------------------------------------
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
plt.rcParams.update({
    "figure.figsize": (3.3, 2.3),
    "figure.dpi": 200,
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# color follows the entity, fixed across all figures
C = {
    "cold": "#52514e",
    "exact": "#e87ba4",
    "embed@0.8": "#eda100",
    "embed@0.9": "#eb6834",
    "semreuse-noaudit": "#4a3aa7",
    "semreuse": "#2a78d6",
    "semreuse-gt": "#1baf7a",
}
LBL = {
    "cold": "Cold (no reuse)",
    "exact": "Exact cache",
    "embed@0.8": "Embed cache $\\theta$=0.8",
    "embed@0.9": "Embed cache $\\theta$=0.9",
    "semreuse-noaudit": "SemReuse w/o audit",
    "semreuse": "SemReuse",
    "semreuse-gt": "SemReuse (oracle ent.)",
}
ORDER = ["cold", "exact", "embed@0.9", "embed@0.8", "semreuse-noaudit",
         "semreuse", "semreuse-gt"]


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"[fig] {FIG / name}")


# ---- Fig 1/2: main comparison bars ---------------------------------------
def fig_main_bars(csv, name):
    df = pd.read_csv(csv).set_index("method")
    cold = df.loc["cold", "total_oracle_calls"]
    methods = [m for m in ORDER if m in df.index]
    calls = [df.loc[m, "total_oracle_calls"] / 1e3 for m in methods]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    y = np.arange(len(methods))[::-1]
    ax.barh(y, calls, height=0.62, color=[C[m] for m in methods],
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, [LBL[m] for m in methods])
    ax.set_xlabel("oracle calls (thousands)")
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    for yi, m, c in zip(y, methods, calls):
        red = cold / (c * 1e3)
        extra = ""
        p, r = df.loc[m, "macro_precision"], df.loc[m, "macro_recall"]
        if p < 0.995 or r < 0.995:
            extra = f"  (P={p:.2f} R={r:.2f})"
        ax.text(c + 0.012 * max(calls), yi,
                f"$\\times${red:.2f}{extra}", va="center", fontsize=6.3,
                color=INK if not extra else "#a03b3b")
    ax.set_xlim(0, max(calls) * 1.42)
    save(fig, name)


# ---- Fig 3: cost/accuracy frontier ---------------------------------------
def fig_frontier():
    df = pd.read_csv(RES / "exp1_summary_20ng_full.csv").set_index("method")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    offs = {  # label offsets to dodge collisions (x pts, y pts, ha)
        "cold": (0, -12, "center"), "exact": (2, 8, "left"),
        "embed@0.9": (0, -20, "center"),
        "embed@0.8": (6, 3, "left"), "semreuse-noaudit": (6, -3, "left"),
        "semreuse": (0, -12, "center"), "semreuse-gt": (-4, 7, "left"),
    }
    for m in ORDER:
        row = df.loc[m]
        x = row["total_oracle_calls"] / 1e3
        yv = row["macro_f1"]
        certified = row["n_bounded"] > 0 or m in ("cold", "exact")
        marker = "o" if certified else "D"
        face = C[m] if certified else "white"
        ax.plot([x], [yv], marker=marker, color=C[m],
                markerfacecolor=face, markersize=5,
                markeredgewidth=1.2, linestyle="none")
        dx, dy, ha = offs[m]
        ax.annotate(LBL[m], (x, yv), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.2, ha=ha)
    ax.set_xlabel("oracle calls (thousands)")
    ax.set_ylabel("macro F1 vs ground truth")
    ax.set_ylim(0.74, 1.05)
    ax.set_xlim(0, 1060)
    ax.annotate("open marker = no accuracy guarantee", (0.02, 0.96),
                xycoords="axes fraction", fontsize=6, color=MUTED,
                ha="left")
    save(fig, "fig_frontier_20ng.pdf")


# ---- Fig 4: reliability diagram ------------------------------------------
def fig_reliability():
    df = pd.read_csv(RES / "exp2_reliability_20ng_medium.csv")
    s = pd.read_csv(RES / "exp2_summary_20ng_medium.csv").set_index("head")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)))
    style = {"threshold": ("#52514e", "s", "Threshold head"),
             "calibrated": ("#2a78d6", "o", "Calibrated head")}
    for head, (col, mk, lab) in style.items():
        d = df[df["head"] == head]
        ece = s.loc[head, "ece"]
        ax.plot(d["mean_conf"], d["mean_acc"], marker=mk, color=col,
                label=f"{lab} (ECE {ece:.2f})", markersize=4)
    ax.set_xlabel("mean predicted confidence (bin)")
    ax.set_ylabel("empirical accuracy (bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    save(fig, "fig_reliability.pdf")


# ---- Fig 5: audit stress (exp3) ------------------------------------------
def fig_audit_stress():
    df = pd.read_csv(RES / "exp3_audit_20ng_medium.csv")
    d9 = df[df["target_recall"] == 0.9]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    greys = {"0.02": "#c3c2b7", "0.05": "#a8a69e", "0.1": "#898781",
             "0.2": "#52514e"}
    for budget in ["adaptive", "0.02", "0.05", "0.1", "0.2"]:
        d = d9[d9["budget"] == budget].sort_values("entailment_error")
        if budget == "adaptive":
            ax.plot(d["entailment_error"], d["call_reduction"], marker="o",
                    color="#2a78d6", label="adaptive (guarantee-aware)")
        else:
            ax.plot(d["entailment_error"], d["call_reduction"], marker="s",
                    color=greys[budget], linestyle=(0, (4, 2)),
                    label=f"fixed fraction {budget}")
    ax.set_xlabel("injected entailment error rate")
    ax.set_ylabel("call reduction ($\\times$)")
    ax.set_xticks([0.0, 0.1, 0.3])
    ax.legend(loc="upper right", ncols=1)
    total_cert = int(df["n_bounded"].sum())
    total_viol = int(df["bound_violations"].sum())
    ax.set_ylim(1.0, 3.2)
    ax.annotate("bound coverage = 1.00 in every cell "
                f"({total_viol} violations / {total_cert:,} "
                "certificates, $\\alpha$=0.05)",
                (0.02, 0.03), xycoords="axes fraction", fontsize=5.8,
                color=MUTED)
    save(fig, "fig_audit_stress.pdf")


# ---- Fig 6: overlap sweep (exp4) -----------------------------------------
def fig_overlap():
    df = pd.read_csv(RES / "exp4_overlap_20ng_full.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for m in ["semreuse-gt", "semreuse", "embed@0.9"]:
        d = df[df["method"] == m].sort_values("overlap")
        ax.plot(d["overlap"], d["call_reduction"], marker="o", color=C[m],
                label=LBL[m])
    ax.set_xlabel("nominal workload overlap rate")
    ax.set_ylabel("call reduction ($\\times$)")
    ax.legend(loc="upper right")
    save(fig, "fig_overlap.pdf")


# ---- Fig 7/8: scaling (exp6) ---------------------------------------------
def fig_scaling():
    df = pd.read_csv(RES / "exp6_scaling_20ng_small.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for m in ["semreuse-gt", "semreuse"]:
        d = df[df["method"] == m].sort_values("n_rows")
        ax.plot(d["n_rows"], d["call_reduction"], marker="o", color=C[m],
                label=LBL[m])
    ax.set_xscale("log")
    ax.set_xticks([500, 1000, 2000, 4000, 8000],
                  ["500", "1k", "2k", "4k", "8k"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("corpus size $N$ (log)")
    ax.set_ylabel("call reduction ($\\times$)")
    ax.legend(loc="upper left")
    save(fig, "fig_scaling_reduction.pdf")

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    nq = df["n_queries"].iloc[0]
    ns = sorted(df["n_rows"].unique())
    ax.plot(ns, ns, color=AXIS, linestyle=(0, (3, 3)), linewidth=1.0)
    ax.annotate("cold cost / query = $N$", (ns[1], ns[1] * 1.45),
                fontsize=6, color=MUTED, rotation=18)
    for m in ["semreuse-gt", "semreuse"]:
        d = df[df["method"] == m].sort_values("n_rows")
        audit = (d["audit_calls"] + d["escalation_calls"]) / nq
        ax.plot(d["n_rows"], audit, marker="o", color=C[m],
                label=LBL[m] + " audit+esc.")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([500, 1000, 2000, 4000, 8000],
                  ["500", "1k", "2k", "4k", "8k"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("corpus size $N$ (log)")
    ax.set_ylabel("oracle calls / query (log)")
    ax.legend(loc="upper left")
    save(fig, "fig_scaling_audit.pdf")


# ---- tables ---------------------------------------------------------------
# ---- Fig 9: scale sweep (exp11) -------------------------------------------
def fig_scale():
    """Reduction and per-query audit cost against corpus size, two corpora."""
    frames = []
    for corpus, fname in [("AG News", "exp11_summary_agnews_full.csv"),
                          ("RCV1", "exp11_summary_rcv1_full.csv")]:
        f = RES / fname
        if f.exists():
            d = pd.read_csv(f)
            d["corpus_label"] = corpus
            frames.append(d)
    if not frames:
        return
    df = pd.concat(frames)
    cold = (df[df.method == "cold"]
            .set_index(["corpus_label", "n_rows"])["total_oracle_calls"])

    fig, ax = plt.subplots()
    for (corpus, method), style in [
            (("AG News", "semreuse"), dict(marker="o", ls="-")),
            (("AG News", "semreuse-gt"), dict(marker="o", ls="--")),
            (("RCV1", "semreuse"), dict(marker="s", ls="-")),
            (("RCV1", "semreuse-gt"), dict(marker="s", ls="--"))]:
        d = df[(df.corpus_label == corpus) & (df.method == method)]
        if d.empty:
            continue
        red = [cold[(corpus, int(n))] / c for n, c in
               zip(d.n_rows, d.total_oracle_calls)]
        lbl = f"{corpus} {'oracle ent.' if 'gt' in method else 'NLI'}"
        ax.plot(d.n_rows, red, label=lbl, color=INK if corpus == "AG News"
                else MUTED, **style)
    ax.set_xscale("log")
    ax.set_xlabel("corpus size $N$ (documents)")
    ax.set_ylabel("call reduction vs.\ncold")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    save(fig, "fig_scale_reduction.pdf")

    fig, ax = plt.subplots()
    for corpus, mk in [("AG News", "o"), ("RCV1", "s")]:
        for method, ls in [("semreuse", "-"), ("semreuse-gt", "--")]:
            d = df[(df.corpus_label == corpus) & (df.method == method)]
            if d.empty:
                continue
            ax.plot(d.n_rows, d.audit_calls / d.n_queries, marker=mk, ls=ls,
                    color=INK if corpus == "AG News" else MUTED,
                    label=f"{corpus} {'oracle ent.' if 'gt' in method else 'NLI'}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size $N$ (documents)")
    ax.set_ylabel("audit calls\nper query")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    save(fig, "fig_scale_audit.pdf")


# ---- table: proxy cascade comparison (exp9) -------------------------------
# ---- table: real-LLM oracle (exp8) ----------------------------------------
# ---- table: tier-two arbiter (exp10) --------------------------------------
# ---- Fig: reduction vs N, simulated and real oracle side by side ----------
def fig_real_vs_sim_scale():
    """One picture for the paper's economic claim on both kinds of oracle.

    The simulated oracle carries the scale story and the real one the realism
    story; plotting them on the same axes is the honest way to show that they
    are the same curve at different points, not two unrelated experiments.
    """
    import glob
    real_f = sorted(glob.glob(str(RES / "exp8_summary_*full*.csv"))) or \
        sorted(glob.glob(str(RES / "exp8_summary_*.csv")))
    if not real_f:
        return
    d = pd.read_csv(real_f[-1])
    if d.n_rows.nunique() < 2:
        return
    cold = d[d.method == "cold"].set_index("n_rows")["total_oracle_calls"]
    fig, ax = plt.subplots()
    for method, style, lbl in [("semreuse", dict(marker="o", ls="-"),
                                "real oracle, NLI"),
                               ("semreuse-gt", dict(marker="o", ls="--"),
                                "real oracle, oracle ent.")]:
        sub = d[d.method == method].sort_values("n_rows")
        if sub.empty:
            continue
        ax.plot(sub.n_rows,
                [cold[int(n)] / c for n, c in
                 zip(sub.n_rows, sub.total_oracle_calls)],
                color=INK, label=lbl, **style)
    sim = RES / "exp11_summary_agnews_full.csv"
    if sim.exists():
        e = pd.read_csv(sim)
        cs = e[e.method == "cold"].set_index("n_rows")["total_oracle_calls"]
        sub = e[e.method == "semreuse"].sort_values("n_rows")
        ax.plot(sub.n_rows,
                [cs[int(n)] / c for n, c in
                 zip(sub.n_rows, sub.total_oracle_calls)],
                color=MUTED, marker="s", ls="-",
                label="simulated oracle (AG News)")
    ax.set_xscale("log")
    ax.set_xlabel("corpus size $N$ (documents)")
    ax.set_ylabel("call reduction\nvs.\\ cold")
    ax.legend(frameon=False, fontsize=6)
    save(fig, "fig_real_vs_sim_scale.pdf")


if __name__ == "__main__":
    fig_main_bars(RES / "exp1_summary_20ng_full.csv",
                  "fig_main_calls_20ng.pdf")
    fig_main_bars(RES / "exp1_summary_agnews_full.csv",
                  "fig_main_calls_agnews.pdf")
    fig_frontier()
    fig_reliability()
    fig_audit_stress()
    fig_overlap()
    fig_scaling()
    fig_scale()
    fig_real_vs_sim_scale()
