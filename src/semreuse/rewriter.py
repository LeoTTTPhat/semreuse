"""Entailment-based rewrite planning.

Given a new predicate p, judgments against cached views, and per-relation
confidence thresholds, build a :class:`RewritePlan` that partitions the
corpus rows into:

  * ``candidates``      -- rows the oracle must evaluate,
  * assumed-positive strata -- rows assumed True (positive union: some cached
    q with q => p reported them positive),
  * pruned strata       -- rows assumed False, either because a superset view
    q (p => q) reported them negative, or because a disjoint view reported
    them positive.

Each assumed stratum records its provenance (source view, rule, confidence);
the audit layer samples *per stratum*, so a single low-quality judgment
cannot hide inside a big pool.

Rules (all optional, for ablations):
  R1 superset pruning     p => q  : candidates &= reported(q)
  R2 positive union       q => p  : rows in reported(q) assumed True
  R3 disjoint elimination p _|_ q : rows in reported(q) assumed False
  R4 equivalence          p == q  : R1 + R2 with the same view
  Multi-view combination: R1 intersects across all superset views; R2/R3
  union across views; conflicts (a row both assumed-True and pruned) are
  resolved by sending the row back to ``candidates`` (always safe).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semreuse.entailment import Judgment
from semreuse.predicates import Relation
from semreuse.store import CachedView


@dataclass
class RewriteConfig:
    tau_equiv: float = 0.8
    tau_forward: float = 0.8
    tau_backward: float = 0.9
    tau_disjoint: float = 0.9
    enable_superset: bool = True
    enable_positive_union: bool = True
    enable_disjoint: bool = True
    max_views: int = 8          # strongest-confidence views used per rule
    # Share of the recall allowance (1 - t) that pruning judgments may spend,
    # given what past audits measured their slack to be (semreuse.slack).
    # None disables the mechanism and restores the assume-nothing-is-lost
    # behaviour: every judgment clearing tau is used, whatever it has cost
    # before.
    slack_budget: float | None = None


@dataclass
class Stratum:
    """A set of rows sharing one assumed value and one provenance."""

    kind: str                   # 'pruned-superset' | 'pruned-disjoint' | 'assumed-pos'
    source_pid: str
    source_text: str
    confidence: float
    rows: np.ndarray            # int row ids

    @property
    def size(self) -> int:
        return len(self.rows)


@dataclass
class RewritePlan:
    n_rows: int
    candidates: np.ndarray                    # int row ids to evaluate
    assumed_pos: list[Stratum] = field(default_factory=list)
    pruned: list[Stratum] = field(default_factory=list)
    used_views: list[tuple[str, Relation, float]] = field(default_factory=list)
    predicted_slack: float = 0.0   # recall the admitted rewrites are expected
    # to cost, as a fraction of the predicate's positives (semreuse.slack)

    @property
    def n_pruned(self) -> int:
        return sum(s.size for s in self.pruned)

    @property
    def n_assumed_pos(self) -> int:
        return sum(s.size for s in self.assumed_pos)

    def reported_mask(self, candidate_results: np.ndarray) -> np.ndarray:
        """Assemble the reported extension: candidate oracle answers,
        assumed positives True, pruned rows False."""
        out = np.zeros(self.n_rows, dtype=bool)
        out[self.candidates] = candidate_results
        for s in self.assumed_pos:
            out[s.rows] = True
        return out


def build_plan(
    n_rows: int,
    judgments: list[tuple[CachedView, Judgment]],
    config: RewriteConfig,
    slack_predictor=None,
    recall_allowance: float | None = None,
) -> RewritePlan:
    """Construct a rewrite plan from per-view judgments.

    Deterministic: given the same store, judgments and slack estimates, the
    same plan.

    ``slack_predictor(source_pid, kind, pruned_fraction) -> float`` supplies
    each pruning judgment's expected recall cost from earlier audits, and ``recall_allowance`` is
    the ``1 - t`` the query has to spend. When both are given together with
    ``config.slack_budget``, pruning rewrites are admitted only while their
    predicted slack fits inside ``slack_budget * recall_allowance``, leaving
    the rest of the allowance for the audit's sampling uncertainty. The
    estimates come from *previous* queries, so the plan is still fixed before
    this query's sample is drawn and the certificate is untouched.
    """
    # Split EQUIV into its two one-directional components.
    forward: list[tuple[CachedView, float]] = []   # p => q
    backward: list[tuple[CachedView, float]] = []  # q => p
    disjoint: list[tuple[CachedView, float]] = []
    for view, j in judgments:
        if j.relation is Relation.EQUIV and j.confidence >= config.tau_equiv:
            forward.append((view, j.confidence))
            backward.append((view, j.confidence))
        elif j.relation is Relation.FORWARD and j.confidence >= config.tau_forward:
            forward.append((view, j.confidence))
        elif j.relation is Relation.BACKWARD and j.confidence >= config.tau_backward:
            backward.append((view, j.confidence))
        elif j.relation is Relation.DISJOINT and j.confidence >= config.tau_disjoint:
            disjoint.append((view, j.confidence))

    if not config.enable_superset:
        forward = []
    if not config.enable_positive_union:
        backward = []
    if not config.enable_disjoint:
        disjoint = []

    key = lambda vc: (-vc[1], vc[0].pid)  # confidence desc, pid tiebreak
    forward = sorted(forward, key=key)[: config.max_views]
    backward = sorted(backward, key=key)[: config.max_views]
    disjoint = sorted(disjoint, key=key)[: config.max_views]

    # Spend the recall allowance knowingly rather than discovering the bill
    # during the audit.
    slack_spent = 0.0
    if (config.slack_budget is not None and slack_predictor is not None
            and recall_allowance):
        from semreuse.slack import admit_within_budget

        cands = []
        for i, (v, c) in enumerate(forward):
            rows = int((~v.reported).sum())
            cands.append((("f", i),
                          slack_predictor(v.pid, "pruned-superset",
                                          rows / max(1, n_rows)), rows))
        for i, (v, c) in enumerate(disjoint):
            rows = int(v.reported.sum())
            cands.append((("d", i),
                          slack_predictor(v.pid, "pruned-disjoint",
                                          rows / max(1, n_rows)), rows))
        admitted, slack_spent = admit_within_budget(
            cands, config.slack_budget * recall_allowance)
        keep = set(admitted)
        forward = [vc for i, vc in enumerate(forward) if ("f", i) in keep]
        disjoint = [vc for i, vc in enumerate(disjoint) if ("d", i) in keep]

    used = ([(v.pid, Relation.FORWARD, c) for v, c in forward]
            + [(v.pid, Relation.BACKWARD, c) for v, c in backward]
            + [(v.pid, Relation.DISJOINT, c) for v, c in disjoint])

    # Assignment array: -1 unassigned (candidate), otherwise stratum index.
    assignment = np.full(n_rows, -1, dtype=np.int32)
    strata: list[Stratum] = []

    def claim(mask: np.ndarray, kind: str, view: CachedView,
              conf: float) -> None:
        rows = np.flatnonzero(mask & (assignment == -1))
        if len(rows) == 0:
            return
        assignment[rows] = len(strata)
        strata.append(Stratum(kind=kind, source_pid=view.pid,
                              source_text=view.text, confidence=conf,
                              rows=rows))

    # R2 positive union first (claims positives), highest confidence first.
    pos_mask = np.zeros(n_rows, dtype=bool)
    for view, conf in backward:
        claim(view.reported, "assumed-pos", view, conf)
        pos_mask |= view.reported

    # R1 superset pruning: rows outside reported(q) are assumed negative.
    # Conflict rule: a row already assumed positive stays positive *unless*
    # a superset view also excludes it -- then it's contested and goes back
    # to candidates (handled below by demoting contested rows).
    contested = np.zeros(n_rows, dtype=bool)
    for view, conf in forward:
        outside = ~view.reported
        contested |= outside & pos_mask
        claim(outside, "pruned-superset", view, conf)

    # R3 disjoint elimination: rows positive in a disjoint view assumed neg.
    for view, conf in disjoint:
        contested |= view.reported & pos_mask
        claim(view.reported, "pruned-disjoint", view, conf)

    # Demote contested rows to candidates (safe: they get oracle-evaluated).
    if contested.any():
        rows = np.flatnonzero(contested)
        assignment[rows] = -1
        for s in strata:
            s.rows = s.rows[~contested[s.rows]]

    strata = [s for s in strata if s.size > 0]
    candidates = np.flatnonzero(assignment == -1)
    # assignment indices may be stale after filtering; rebuild lists by kind.
    assumed_pos = [s for s in strata if s.kind == "assumed-pos"]
    pruned = [s for s in strata if s.kind.startswith("pruned")]

    return RewritePlan(n_rows=n_rows, candidates=candidates,
                       assumed_pos=assumed_pos, pruned=pruned,
                       used_views=used, predicted_slack=slack_spent)
