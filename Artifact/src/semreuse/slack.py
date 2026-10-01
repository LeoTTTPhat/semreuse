"""Measured agreement slack: what a rewrite actually costs in recall.

The real-oracle study says the binding constraint on reuse is not whether the
reasoner gets the relation right but *by how much it is wrong*. "The post
discusses space or astronomy" implies "the post discusses science" as a matter
of meaning, and on Llama-3.1-8B's own extensions it holds on 63% of the rows;
declaring the implication and pruning on it silently spends 37% of the recall
budget before the audit has drawn a single sample. At a target of 0.9 there is
only 10% to spend, so the audit cannot certify and escalates, and the escalation
costs more than the pruning saved.

The fix is to stop treating an implication as free. Every audit already measures
the damage a rewrite did: it samples the pruned pool uniformly and finds the
rows that should not have been pruned. Attributing those finds back to the view
and relation that pruned them yields an unbiased estimate of that judgment's
*slack* -- the fraction of the new predicate's positives it loses -- at zero
additional oracle cost. Accumulated over a workload, those estimates let the
rewriter predict what a rewrite will cost before committing to it, and decline
the ones the recall budget cannot absorb.

**Why this does not touch the certificate.** The store is written from the
audits of *earlier* queries and read when planning a *later* one, so the plan
is still a deterministic function of information fixed before the current
query's sample is drawn -- exactly the property Lemma 2 of Section 5.5 needs.
Using the current query's own sample to revise its own plan would break that,
and we never do it. As everywhere else in this design, a bad slack estimate
can cost oracle calls; it cannot cost correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Kinds of pruning a stratum can come from, as recorded on the plan. Only
# pruning spends *recall*; assumed-positive strata spend precision and are
# budgeted separately (Section 5.3).
PRUNING_KINDS = ("pruned-superset", "pruned-disjoint", "pruned-proxy")


@dataclass
class _Cell:
    """Sufficient statistics for a weighted regression through the origin."""

    wxx: float = 0.0        # sum w * x^2
    wxy: float = 0.0        # sum w * x * y
    weight: float = 0.0

    def add(self, pruned_fraction: float, slack: float,
            weight: float) -> None:
        self.wxx += weight * pruned_fraction * pruned_fraction
        self.wxy += weight * pruned_fraction * slack
        self.weight += weight


@dataclass
class SlackStore:
    """What a rewrite costs in recall, learned per view from past audits.

    The first version of this class predicted a view's slack directly. That is
    the wrong quantity to learn, and the data says so: across the real-oracle
    workload, five times more variance in slack sits *within* a source view
    than between views (0.156 against 0.031), and the reasoner's own confidence
    correlates with slack at only +0.18 -- with the wrong sign. What does
    predict it, at +0.58, is how much of the corpus the rewrite prunes, which
    is known exactly at planning time from the bitmaps and needs no learning
    at all.

    So the store learns a *rate*: slack per unit pruned, fitted by weighted
    least squares through the origin. On the real-oracle workload that
    one-parameter model reaches R^2 = 0.31 where predicting the mean reaches
    0.00, and it says something the identity model could not -- a rewrite that
    prunes three quarters of the corpus loses about half the answer, while one
    that prunes a tenth loses almost none.

    The consequence for planning is the useful part: since predicted slack
    grows with what a rewrite prunes, a fixed recall budget translates into a
    *maximum prunable fraction*. The planner stops pruning where the budget
    runs out rather than discovering the bill during the audit.

    Estimates for (source, kind) are shrunk toward the kind's rate, which is
    shrunk toward a global rate, which is shrunk toward ``prior_rate``. The
    prior is deliberately optimistic, so an uninformed engine behaves as it did
    before and tightens as evidence arrives; a pessimistic prior would decline
    every rewrite, and so never measure one, and so never learn (see
    ``admit_within_budget``).
    """

    prior_rate: float = 0.05
    prior_strength: float = 0.05
    shrink: float = 0.05
    _global: _Cell = field(default_factory=_Cell)
    _by_kind: dict[str, _Cell] = field(default_factory=dict)
    _by_source: dict[tuple[str, str], _Cell] = field(default_factory=dict)
    observations: int = 0

    # -- writing -----------------------------------------------------------

    def observe(self, source_pid: str, kind: str, slack: float,
                pruned_fraction: float, weight: float = 1.0) -> None:
        """Record one audited rewrite: it pruned this much and cost that much.

        ``weight`` reflects how much evidence the measurement carries -- the
        number of audited rows -- so a stratum sampled 400 times counts for
        more than one sampled 8 times.
        """
        slack = max(0.0, min(1.0, float(slack)))
        pruned_fraction = max(0.0, min(1.0, float(pruned_fraction)))
        weight = max(0.0, float(weight))
        if weight == 0.0 or pruned_fraction == 0.0:
            return
        self._global.add(pruned_fraction, slack, weight)
        self._by_kind.setdefault(kind, _Cell()).add(pruned_fraction, slack,
                                                    weight)
        self._by_source.setdefault((source_pid, kind), _Cell()).add(
            pruned_fraction, slack, weight)
        self.observations += 1

    # -- reading -----------------------------------------------------------

    def _rate(self, cell: _Cell | None, backoff: float, strength: float
              ) -> float:
        if cell is None:
            return backoff
        return (cell.wxy + strength * backoff) / (cell.wxx + strength)

    def rate(self, source_pid: str, kind: str) -> float:
        """Estimated slack per unit of corpus pruned."""
        glob = self._rate(self._global, self.prior_rate, self.prior_strength)
        by_kind = self._rate(self._by_kind.get(kind), glob, self.shrink)
        return max(0.0, self._rate(self._by_source.get((source_pid, kind)),
                                   by_kind, self.shrink))

    def predict(self, source_pid: str, kind: str,
                pruned_fraction: float) -> float:
        """Expected recall cost of pruning this much with this view."""
        return self.rate(source_pid, kind) * max(0.0, pruned_fraction)

    def summary(self) -> dict:
        return {
            "observations": self.observations,
            "global_rate": round(self._rate(self._global, self.prior_rate,
                                            self.prior_strength), 4),
            "by_kind_rate": {
                k: round(self._rate(c, self._rate(self._global,
                                                  self.prior_rate,
                                                  self.prior_strength),
                                    self.shrink), 4)
                for k, c in self._by_kind.items()},
            "sources_tracked": len(self._by_source),
        }


# ---------------------------------------------------------------------------
# Reading slack back out of a finished audit
# ---------------------------------------------------------------------------

def observations_from_audit(plan, audit_result, n_rows: int
                            ) -> list[tuple[str, str, float, float, float]]:
    """Turn one completed audit into (source, kind, slack, weight) records.

    The pruned pool is sampled uniformly, so a stratum that contributed
    ``sampled`` rows to the sample and ``positives`` wrongly-pruned rows among
    them has an unbiased estimated miss count of
    ``positives * size / sampled`` -- the Horvitz--Thompson estimate for
    without-replacement sampling from the pool. Dividing by the engine's own
    estimate of how many positives the predicate has turns that into a slack:
    the share of the answer this judgment would have cost.

    A stratum that was escalated is even better evidence: it was read in full,
    so its miss count is exact rather than estimated, and it is weighted by its
    whole size.
    """
    denom = float(audit_result.found_positives + audit_result.assumed_pos_lb)
    if denom <= 0:
        return []
    out: list[tuple[str, str, float, float]] = []
    strata = list(plan.assumed_pos) + list(plan.pruned)
    for rec, stratum in zip(audit_result.strata, strata):
        if rec.kind not in PRUNING_KINDS or rec.size == 0:
            continue
        if rec.escalated:
            # Read in full: the audit knows exactly what this judgment cost.
            missed = float(rec.positives)
            weight = float(rec.size)
        elif rec.sampled > 0:
            missed = rec.positives * rec.size / rec.sampled
            weight = float(rec.sampled)
        else:
            continue                      # no evidence about this stratum
        out.append((stratum.source_pid, rec.kind, missed / denom,
                    rec.size / max(1, n_rows), weight))
    return out


def observation_features(plan, audit_result, n_rows: int) -> list[dict]:
    """The same observations, with the features available at *plan* time.

    Used to answer an empirical question the design turns on: what, if
    anything, predicts how much a rewrite will cost before it is paid for?
    The source view's identity is one candidate; the reasoner's confidence and
    the stratum's relative size are the others, and unlike identity they are
    available for a (predicate, view) pair that has never been seen together.
    """
    denom = float(audit_result.found_positives + audit_result.assumed_pos_lb)
    if denom <= 0:
        return []
    rows = []
    strata = list(plan.assumed_pos) + list(plan.pruned)
    for rec, stratum in zip(audit_result.strata, strata):
        if rec.kind not in PRUNING_KINDS or rec.size == 0:
            continue
        if rec.escalated:
            missed, weight = float(rec.positives), float(rec.size)
        elif rec.sampled > 0:
            missed = rec.positives * rec.size / rec.sampled
            weight = float(rec.sampled)
        else:
            continue
        rows.append({"source_pid": stratum.source_pid, "kind": rec.kind,
                     "confidence": stratum.confidence,
                     "pruned_fraction": rec.size / max(1, n_rows),
                     "slack": missed / denom, "weight": weight,
                     "found_positives": audit_result.found_positives})
    return rows


# ---------------------------------------------------------------------------
# Spending the recall budget knowingly
# ---------------------------------------------------------------------------

def admit_within_budget(candidates, budget: float, floor: float = 0.002,
                        explore: int = 1):
    """Choose which pruning rewrites to use, given a slack budget.

    ``candidates`` is a list of ``(key, predicted_slack, rows_pruned)``, where
    the predicted slack already reflects how much each rewrite prunes. The
    budget is the share of the recall allowance that judgments may consume;
    whatever is left pays for the audit's own sampling uncertainty. Rewrites
    are admitted greedily by rows pruned per unit of predicted slack -- the
    standard knapsack heuristic -- so a wide, reliable superset view is taken
    before a narrow, unreliable one.

    ``explore`` is not optional garnish. The estimates this budget spends are
    themselves produced by audits of admitted rewrites, so a budget that
    admits nothing learns nothing and goes on admitting nothing forever: the
    mechanism starves itself, and a pessimistic prior guarantees it. Admitting
    the best-ranked candidate even when the budget says no keeps evidence
    flowing, and it is cheap to be wrong about -- a rewrite whose pruning is
    entirely undone by escalation costs candidates plus escalation, which
    together are the corpus, so the downside is bounded at roughly cold plus
    an audit rather than at anything catastrophic.

    Returns the admitted keys and the predicted slack they spend.
    """
    if budget is None:
        return [k for k, _, _ in candidates], 0.0
    ranked = sorted(candidates,
                    key=lambda c: (-(c[2] / max(c[1], floor)), c[0]))
    admitted, spent = [], 0.0
    for key, slack, _rows in ranked:
        if spent + slack <= budget:
            admitted.append(key)
            spent += slack
    if not admitted and explore > 0 and ranked:
        admitted = [k for k, _, _ in ranked[:explore]]
        spent = sum(sl for k, sl, _ in ranked[:explore])
    return admitted, spent
