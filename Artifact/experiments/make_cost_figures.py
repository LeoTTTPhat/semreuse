"""Where the oracle calls go, and how that scales.

  fig_decomposition -- where the oracle calls go, as a share of cold, across
                       the four evaluation settings; the escalation segment is
                       the central diagnostic.
  fig_scaling       -- the certificate's price is flat in N while cold cost is
                       linear (a), and what that does to end-to-end reduction
                       on a reuse-friendly versus a multi-label corpus (b).

Reads results/*.csv, writes results/figures/*.pdf. Legends sit outside and
below the axes.

Usage:
    .venv/bin/python experiments/make_cost_figures.py
"""

from __future__ import annotations

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
CAND, AUDIT, ESC = "#4b6d8c", "#c9a227", "#a4453a"
plt.rcParams.update({
    "figure.dpi": 200, "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False,
})


def _row(df, method, n=None):
    d = df[df.method == method]
    if n is not None:
        d = d[d.n_rows == n]
    return d.iloc[0]


def decomposition() -> None:
    """Cold-normalized cost split. Short bar = big saving."""
    ag = pd.read_csv(RES / "exp11_summary_agnews_full.csv")
    rc = pd.read_csv(RES / "exp11_summary_rcv1_full.csv")
    ng = pd.read_csv(RES / "exp1_perquery_20ng_full.csv")
    rl = pd.read_csv(RES / "exp14_perquery_20ng_full_real.csv")

    ngs = ng[ng.method == "semreuse"]
    rls = rl[(rl.budget_label == "off") & (rl.reasoner == "nli")]
    rows = []
    for label, cold, cand, aud, esc in [
        ("AG News\n$N$=120k",
         _row(ag, "cold", 120000).total_oracle_calls,
         _row(ag, "semreuse", 120000).candidate_calls,
         _row(ag, "semreuse", 120000).audit_calls,
         _row(ag, "semreuse", 120000).escalation_calls),
        ("RCV1\n$N$=804k",
         _row(rc, "cold", 804414).total_oracle_calls,
         _row(rc, "semreuse", 804414).total_oracle_calls
         - _row(rc, "semreuse", 804414).audit_calls
         - _row(rc, "semreuse", 804414).escalation_calls,
         _row(rc, "semreuse", 804414).audit_calls,
         _row(rc, "semreuse", 804414).escalation_calls),
        ("20 Newsgroups\n$N$=7,951",
         _row(pd.read_csv(RES / "exp1_summary_20ng_full.csv"),
              "cold").total_oracle_calls,
         ngs.candidate_calls.sum(), ngs.audit_calls.sum(),
         ngs.escalation_calls.sum()),
        ("Real Llama-3.1-8B\n$N$=2,000",
         47 * 2000, rls.candidate_calls.sum(), rls.audit_calls.sum(),
         rls.escalation_calls.sum()),
    ]:
        rows.append((label, 100 * cand / cold, 100 * aud / cold,
                     100 * esc / cold, cold / (cand + aud + esc)))

    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    y = range(len(rows))
    c = [r[1] for r in rows]
    a = [r[2] for r in rows]
    e = [r[3] for r in rows]
    ax.barh(y, c, color=CAND, height=0.62, label="candidates")
    ax.barh(y, a, left=c, color=AUDIT, height=0.62, label="audit (certificate)")
    ax.barh(y, e, left=[x + z for x, z in zip(c, a)], color=ESC, height=0.62,
            label="escalation (repair)")
    for i, r in enumerate(rows):
        ax.text(r[1] + r[2] + r[3] + 2.5, i, f"{r[4]:.2f}$\\times$",
                va="center", fontsize=7)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("oracle calls, % of cold evaluation")
    ax.set_xlim(0, 110)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              columnspacing=1.0, handlelength=1.1, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "fig_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)
    for r in rows:
        print(f"  {r[0][:14]:16s} cand {r[1]:5.1f}%  audit {r[2]:5.2f}%  "
              f"esc {r[3]:5.1f}%  -> {r[4]:.2f}x")


def scaling() -> None:
    """(a) certificate price flat vs cold linear; (b) what that buys."""
    ag = pd.read_csv(RES / "exp11_summary_agnews_full.csv")
    rc = pd.read_csv(RES / "exp11_summary_rcv1_full.csv")
    NQ = 120
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.45))

    for df, name, col, mk in ((ag, "AG News", CAND, "o"),
                              (rc, "RCV1", ESC, "s")):
        s = df[df.method == "semreuse"].sort_values("n_rows")
        cd = df[df.method == "cold"].sort_values("n_rows")
        a1.plot(cd.n_rows, cd.total_oracle_calls / NQ, marker=mk, ms=3.0,
                lw=1.0, ls="--", color=col, alpha=0.55,
                label=f"{name}: cold")
        a1.plot(s.n_rows, s.audit_calls / NQ, marker=mk, ms=3.4, lw=1.5,
                color=col, label=f"{name}: audit")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylabel("oracle calls per query")
    a1.set_xlabel("corpus size $N$")
    a1.set_title("(a) cold cost is linear in $N$; the certificate is flat",
                 loc="left", pad=5)
    a1.annotate("$32\\times$ more tuples,\n$1.17\\times$ more audit",
                xy=(8e5, 534), xytext=(3.4e4, 1500), fontsize=6.5,
                color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=MUTED))

    for df, name, col, mk in ((ag, "AG News", CAND, "o"),
                              (rc, "RCV1", ESC, "s")):
        cd = df[df.method == "cold"].sort_values("n_rows")
        for meth, ls, tag in (("semreuse", "-", "NLI reasoner"),
                              ("semreuse-gt", ":", "oracle entailment")):
            s = df[df.method == meth].sort_values("n_rows")
            red = cd.total_oracle_calls.values / s.total_oracle_calls.values
            a2.plot(s.n_rows, red, marker=mk, ms=3.4, lw=1.5, ls=ls,
                    color=col, label=f"{name}, {tag}")
    a2.set_xscale("log")
    a2.set_yscale("log")
    from matplotlib.ticker import NullFormatter, NullLocator
    a2.yaxis.set_minor_locator(NullLocator())
    a2.yaxis.set_minor_formatter(NullFormatter())
    a2.set_yticks([2, 3, 5, 10, 20, 35])
    a2.set_yticklabels(["2", "3", "5", "10", "20", "35"])
    a2.set_ylabel("reduction vs. cold ($\\times$)")
    a2.set_xlabel("corpus size $N$")
    a2.set_title("(b) reduction grows only where entailment holds up",
                 loc="left", pad=5)

    for ax in (a1, a2):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
                  columnspacing=1.1, handlelength=1.7, handletextpad=0.4)
    fig.tight_layout(w_pad=2.4)
    fig.savefig(FIG / "fig_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    decomposition()
    scaling()
    print("wrote", FIG / "fig_decomposition.pdf", "and", FIG / "fig_scaling.pdf")
