"""Pooled, stratified audit sampling with exact finite-population recall bounds.

The rewrite plan assumes values for rows it does not evaluate.  The audit
layer converts those assumptions into a statistical guarantee:

  1. All pruned rows (assumed negative) form one *pool* with one bound:
     sampling at pool level rather than splitting the confidence level across
     per-view strata makes the recall certificate ~S times cheaper for S
     source views, while per-stratum attribution of the sample is kept for
     corrections and escalation.  Assumed-positive strata are instead sampled
     one by one, because the bound used there is per-stratum anyway and the
     size that certifies one is independent of how big it is.
  2. From each pool draw a uniform sample *without replacement* and evaluate
     it with the oracle.  Audited rows are corrected in the output (audit
     calls are never wasted).
  3. Exact hypergeometric confidence bounds (Clopper-Pearson-style CDF
     inversion) give
        U  = upper bound on true positives in the pruned pool,
        L  = lower bound on true positives in the assumed-positive pool,
     each at level alpha/2 (alpha if only one pool exists).
  4. With F = positives found (candidates + audit corrections), the reported
     recall lower bound is
        recall >= (F + (L - k_pos)) / (F + (L - k_pos) + (U - k_pruned))
     which holds with probability >= 1 - alpha.  It is valid regardless of
     how wrong the entailment judgments are: the audit is against the oracle
     itself, so guarantees do not compound across chained reuse.
  5. *Adaptive sizing*: the pruned-pool sample is sized so that a clean
     sample certifies the target -- m = ceil( ln(alpha_s) / ln(1 - A/N) )
     where A is the tolerable miss count implied by the target recall and
     the positives found so far.  (Binomial relaxation; conservative.)
  6. *Escalation*: if the bound still misses the target, the pruned stratum
     with the highest audited hit count is fully evaluated and removed from
     the pool; the bound is recomputed on the restricted sample (valid by
     conditioning: the sample restricted to the remaining subpopulation is
     uniform within it).  Escalation degrades gracefully toward cold
     evaluation and can always reach bound = 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import hypergeom


# ---------------------------------------------------------------------------
# Exact hypergeometric confidence bounds
# ---------------------------------------------------------------------------

def hypergeom_upper_bound(N: int, m: int, k: int, alpha: float) -> int:
    """Largest total-positive count T in a population of N consistent with
    observing k positives in a without-replacement sample of size m, at
    one-sided level alpha:  U = max{ T : P(X <= k | N, T, m) > alpha }.

    P(true T <= U) >= 1 - alpha.  If m == N the answer is exact (k).
    """
    if m >= N:
        return k
    lo, hi = k, N - (m - k)  # T cannot exceed k + (N - m)
    if hypergeom.cdf(k, N, hi, m) > alpha:
        return hi
    # cdf(k; N, T, m) is non-increasing in T: binary search the boundary.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom.cdf(k, N, mid, m) > alpha:
            lo = mid
        else:
            hi = mid - 1
    return lo


def hypergeom_lower_bound(N: int, m: int, k: int, alpha: float) -> int:
    """Smallest T consistent with the sample at one-sided level alpha:
    L = min{ T : P(X >= k | N, T, m) > alpha }.  P(true T >= L) >= 1-alpha."""
    if m >= N:
        return k
    lo, hi = k, N - (m - k)
    if hypergeom.sf(k - 1, N, lo, m) > alpha:
        return lo
    # sf(k-1; N, T, m) is non-decreasing in T: binary search the boundary.
    while lo < hi:
        mid = (lo + hi) // 2
        if hypergeom.sf(k - 1, N, mid, m) > alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# Configuration and result records
# ---------------------------------------------------------------------------

@dataclass
class AuditConfig:
    alpha: float = 0.05           # overall failure probability of the bound
    target_recall: float | None = 0.9  # escalate until bound >= target
    mode: str = "adaptive"        # 'adaptive' | 'fraction' | 'fixed'
    budget_per_stratum: int = 4000  # hard cap on a pool's audit sample
    budget_fraction: float | None = 0.1  # 'fraction' mode: sample this share
    allowance_safety: float = 0.8  # 'adaptive': spend-down of miss allowance
    min_sample: int = 20          # floor on a pool's sample size
    min_stratum_audit: int = 10   # pools smaller than this are fully read
    escalate: bool = True         # enforce target_recall via escalation
    escalation_order: str = "size-desc"  # 'size-desc' | 'conf-asc' | 'size-asc'
    # Any order fixed *before* sampling keeps the certificate valid (the chain
    # of active pools must be a priori; see Lemma 2 of the appendix), so the
    # choice is purely economic.  We default to 'size-desc' (most uncertain
    # mass first); walking the least-trusted strata first ('conf-asc') is the
    # intuitive alternative and measures within 0.4% of it on our workloads,
    # so the order is not a lever worth tuning.
    precision_target: float | None = 0.9  # demote assumed-pos strata whose
    # audited precision falls below this (rows get evaluated instead)
    # Audit-design ablations (exp15).  The defaults are the shipped design;
    # every alternative still publishes a valid certificate, so they differ in
    # price only.
    pruned_pooling: str = "pooled"   # 'pooled' | 'separate': one sample and
    # one bound for all pruned strata, or a sample and a bound per stratum
    separate_allocation: str = "sqrt"  # 'separate' only: how the miss
    # allowance is split across strata ('sqrt' minimizes the total sample;
    # 'proportional' splits it by stratum size)
    certify: str = "joint"           # 'joint' | 'recall' | 'recall-trust':
    # 'recall' samples assumed-positive strata for the recall numerator but
    # never demotes them and publishes no precision bound; 'recall-trust'
    # does not audit them at all and spends all of alpha on the pruned pool

    def sample_size(self, pool_size: int) -> int:
        """Sample size in 'fraction'/'fixed' modes (and for the assumed-
        positive pool in adaptive mode)."""
        if pool_size <= self.min_stratum_audit:
            return pool_size
        if self.mode != "fixed" and self.budget_fraction is not None:
            m = max(self.min_sample,
                    int(np.ceil(self.budget_fraction * pool_size)))
        else:
            m = self.budget_per_stratum
        return min(pool_size, m, self.budget_per_stratum)

    def adaptive_pos_sample_size(self, stratum_size: int,
                                 alpha_i: float) -> int:
        """Sample size for an assumed-positive stratum, independent of its size.

        A stratum is trusted only if its precision is at least
        ``precision_target`` = p.  If it were in fact worse, a clean sample of
        size m would occur with probability at most p^m (binomial relaxation
        of the hypergeometric, conservative), so m = ceil(ln alpha_i / ln p)
        suffices to either certify the stratum or expose it -- and that
        quantity does not depend on the stratum's size.

        This is the precision-side analogue of Eq. (5), and it is what makes
        the *whole* audit O(1) in corpus size: sampling a fixed 10% of every
        assumed-positive stratum, as a naive design does, costs Theta(N) and
        dominates the certificate's price on large corpora.
        """
        p = self.precision_target
        if stratum_size <= self.min_stratum_audit or p is None or p >= 1.0:
            return stratum_size
        m = int(np.ceil(np.log(max(alpha_i, 1e-12)) / np.log(p)))
        return min(stratum_size, max(self.min_sample, m),
                   self.budget_per_stratum)

    def adaptive_sample_size(self, pool_size: int, allowance: float,
                             alpha_s: float) -> int:
        """Minimal m such that a clean sample (k=0) certifies at most
        ``allowance`` missed positives in the pool: smallest m with
        (1 - A/N)^m <= alpha_s (binomial relaxation, conservative)."""
        N = pool_size
        if N <= self.min_stratum_audit:
            return N
        if allowance < 1.0:
            return N  # cannot certify anything: read the pool fully
        if allowance >= N:
            m = self.min_sample
        else:
            m = int(np.ceil(np.log(alpha_s) / np.log(1.0 - allowance / N)))
            m = max(m, self.min_sample)
        return min(N, m, self.budget_per_stratum)


@dataclass
class StratumAudit:
    """Per-stratum attribution of the pooled audit (diagnostics + escalation)."""

    kind: str
    source_pid: str
    size: int
    sampled: int                  # pool samples that fell in this stratum
    positives: int                # audited positives among those
    escalated: bool = False


@dataclass
class AuditResult:
    recall_lower_bound: float
    precision_lower_bound: float | None  # None: design certifies recall only
    audit_calls: int
    escalation_calls: int
    corrections_pos: np.ndarray   # rows flipped False->True (missed positives)
    corrections_neg: np.ndarray   # rows flipped True->False (false positives)
    strata: list[StratumAudit] = field(default_factory=list)
    found_positives: int = 0
    pruned_miss_ub: int = 0       # U - k on the (remaining) pruned pool
    assumed_pos_lb: int = 0       # max(0, L - k) on the assumed-positive pool
    unverified_reported: int = 0  # reported rows never oracle-checked


# ---------------------------------------------------------------------------
# Audit procedure
# ---------------------------------------------------------------------------

def _escalation_order(pruned, mode: str) -> list[int]:
    """A deterministic escalation order, fixed before any sample is drawn.

    Every ordering here is a function of the *plan* only -- stratum sizes,
    source confidences, and indices -- so the sequence of active pools is the
    a-priori nested chain the certificate's union bound is spent on.  Nothing
    about the audit's observations may enter, which is exactly why the obvious
    "escalate wherever the sample found the most misses" rule is unavailable.
    """
    idx = range(len(pruned))
    if mode == "size-asc":
        return sorted(idx, key=lambda i: (pruned[i].size, i))
    if mode == "conf-asc":
        return sorted(idx, key=lambda i: (pruned[i].confidence,
                                          -pruned[i].size, i))
    return sorted(idx, key=lambda i: (-pruned[i].size, i))


def _separate_sample_sizes(sizes: np.ndarray, allowance: float, eta: float,
                           config: AuditConfig) -> np.ndarray:
    """Per-stratum audit sizes when every pruned stratum has its own bound.

    The pooled design certifies the whole miss allowance ``A`` with one
    sample.  With one bound per stratum the allowance has to be split,
    ``A = sum_j A_j``, and a clean sample certifying ``A_j`` misses in a
    stratum of size ``s_j`` costs about ``s_j ln(1/eta) / A_j`` calls.
    Minimizing the total over the split gives ``A_j`` proportional to
    ``sqrt(s_j)`` -- the most favourable split this design admits, and the
    default, so the comparison with pooling is not won by a strawman.
    Strata whose share cannot be certified by sampling are read in full and
    release their share to the others.
    """
    J = len(sizes)
    m = np.zeros(J, dtype=np.int64)
    full = sizes <= config.min_stratum_audit
    while not full.all():
        w = (np.sqrt(sizes) if config.separate_allocation == "sqrt"
             else sizes.astype(float))
        w = np.where(full, 0.0, w)
        share = allowance * w / w.sum()
        released = False
        for j in np.flatnonzero(~full):
            mj = config.adaptive_sample_size(int(sizes[j]), float(share[j]),
                                             eta)
            if mj >= sizes[j]:
                full[j] = True
                released = True
            else:
                m[j] = mj
        if not released:
            break
    m[full] = sizes[full]
    return m


def run_audit(plan, predicate, oracle, candidate_results: np.ndarray,
              config: AuditConfig, rng: np.random.Generator) -> AuditResult:
    """Audit a rewrite plan and compute the recall lower bound.

    ``candidate_results``: oracle answers on plan.candidates (already paid).
    Returns corrections to fold into the reported extension.
    """
    # 'recall-trust' reports assumed positives unverified: nothing is sampled
    # there, the recall numerator counts verified positives only (still a
    # valid lower bound), and alpha goes entirely to the pruned pool.
    audit_pos = config.certify != "recall-trust"
    n_pools = (int(bool(plan.assumed_pos) and audit_pos)
               + int(bool(plan.pruned)))
    alpha_s = config.alpha / max(1, n_pools)

    audit_calls = 0
    escalation_calls = 0
    corrections_pos: list[np.ndarray] = []
    corrections_neg: list[np.ndarray] = []

    found = int(candidate_results.sum())  # positives found among candidates

    # ---- Phase 1: assumed-positive strata -----------------------------
    # Per-stratum sampling with per-stratum lower bounds L_i at level
    # alpha_s / S: a union bound over all strata makes every L_i valid
    # simultaneously, so the data-dependent choice of which strata to keep
    # (precision demotion below) cannot invalidate the certificate.
    pos_records = [StratumAudit(kind=s.kind, source_pid=s.source_pid,
                                size=s.size, sampled=0, positives=0)
                   for s in plan.assumed_pos]
    extra_pos_lb = 0
    # Reported rows that no oracle call ever touched: the unsampled part of
    # every *retained* assumed-positive stratum.  These are the only rows the
    # precision certificate has to reason about statistically.
    unverified_reported = 0
    sample_of: dict[int, np.ndarray] = {}
    if plan.assumed_pos and not audit_pos:
        unverified_reported = sum(s.size for s in plan.assumed_pos)
    if plan.assumed_pos and audit_pos:
        alpha_i = alpha_s / max(1, len(plan.assumed_pos))
        # Sample each assumed-positive stratum separately, at a size that does
        # not grow with the stratum: the bound we use is per-stratum anyway,
        # so pooling here would buy nothing and cost Theta(N) (see
        # ``adaptive_pos_sample_size``).
        for i, (rec, s) in enumerate(zip(pos_records, plan.assumed_pos)):
            m_i = (config.adaptive_pos_sample_size(s.size, alpha_i)
                   if config.mode == "adaptive"
                   else config.sample_size(s.size))
            full = m_i >= s.size
            sub = (s.rows if full
                   else s.rows[rng.choice(s.size, size=m_i, replace=False)])
            ans_i = oracle.evaluate(predicate, sub,
                                    tag="fallback" if full else "audit")
            # Reading a stratum in full is not certification, it is exhaustive
            # evaluation; charging it to the audit column would make the
            # certificate's price look like it grows with N when what actually
            # grew was the fallback of Proposition 3.
            if full:
                escalation_calls += len(sub)
            else:
                audit_calls += len(sub)
            neg = sub[~ans_i]
            if len(neg):  # wrongly assumed positive: correct them
                corrections_neg.append(neg)
            found += int(ans_i.sum())
            sample_of[i] = sub
            rec.sampled = int(len(sub))
            rec.positives = int(ans_i.sum())
            # Demotion is precision repair; a recall-only design never pays it.
            demote = (config.certify == "joint"
                      and config.precision_target is not None
                      and rec.sampled >= 5
                      and rec.positives
                      < config.precision_target * rec.sampled)
            if demote:
                # Assumed-positive rows of this stratum are evaluated
                # exactly instead of being trusted.
                rest = np.setdiff1d(s.rows, sample_of[i],
                                    assume_unique=True)
                if len(rest):
                    ans = oracle.evaluate(predicate, rest,
                                          tag="precision-escalate")
                    escalation_calls += len(rest)
                    found += int(ans.sum())
                    fneg = rest[~ans]
                    if len(fneg):
                        corrections_neg.append(fneg)
                rec.escalated = True
            else:
                L = hypergeom_lower_bound(rec.size, rec.sampled,
                                          rec.positives, alpha_i)
                extra_pos_lb += max(0, L - rec.positives)
                unverified_reported += rec.size - rec.sampled

    # ---- Phase 2: pruned pool -----------------------------------------
    pruned_records = [StratumAudit(kind=s.kind, source_pid=s.source_pid,
                                   size=s.size, sampled=0, positives=0)
                      for s in plan.pruned]
    # Mutable pool state (escalation removes strata and re-inverts).
    pool_N = 0
    samp_owner = np.empty(0, dtype=np.int64)
    samp_answers = np.empty(0, dtype=bool)
    active = np.ones(len(plan.pruned), dtype=bool)

    # Escalation removes strata from the pool in a *deterministic* order
    # (size desc, index asc) and re-inverts the bound on the restricted
    # sample.  Because the order is fixed a priori, the sequence of possible
    # active sets is a fixed nested chain; a union bound over its levels
    # keeps the certificate valid under the data-dependent stopping rule.
    esc_order = _escalation_order(plan.pruned, config.escalation_order)
    n_levels = len(plan.pruned) + 1
    alpha_pool = alpha_s / max(1, n_levels)
    separate = config.pruned_pooling == "separate" and bool(plan.pruned)

    if plan.pruned and not separate:
        pool = np.concatenate([s.rows for s in plan.pruned])
        owner = np.concatenate([np.full(s.size, i) for i, s in
                                enumerate(plan.pruned)])
        pool_N = len(pool)
        found_lb = found + extra_pos_lb
        if config.mode == "adaptive" and config.target_recall is not None:
            allowance = (found_lb * (1.0 - config.target_recall)
                         / max(config.target_recall, 1e-9)
                         * config.allowance_safety)
            m = config.adaptive_sample_size(pool_N, allowance, alpha_pool)
        else:
            m = config.sample_size(pool_N)
        full_pool = m >= pool_N
        idx = (np.arange(pool_N) if full_pool
               else rng.choice(pool_N, size=m, replace=False))
        sample, samp_owner = pool[idx], owner[idx]
        samp_answers = oracle.evaluate(
            predicate, sample, tag="fallback" if full_pool else "audit")
        # Same distinction as above: a full read is the Proposition 3 fallback
        # (the miss allowance was under one row, so nothing could be certified
        # by sampling), not the price of a certificate.
        if full_pool:
            escalation_calls += m
        else:
            audit_calls += m
        k = int(samp_answers.sum())
        if k:  # wrongly pruned positives found: correct them
            corrections_pos.append(sample[samp_answers])
        found += k
        for i, rec in enumerate(pruned_records):
            rec.sampled = int((samp_owner == i).sum())
            rec.positives = int(samp_answers[samp_owner == i].sum())
        if m == pool_N:  # fully read: exact
            active[:] = False
            for rec in pruned_records:
                rec.escalated = True
        # Rows sampled are known; keep per-row sample for escalation math.
        samp_rows = sample
    else:
        samp_rows = np.empty(0, dtype=np.int64)

    # The per-stratum alternative (ablation only): one exact bound per pruned
    # stratum at alpha_s / J.  The J bounds hold simultaneously, so strata may
    # be escalated in any order, including the data-dependent one used below
    # -- the one thing a separate design can do that a pooled one cannot.
    sep_sample_of: dict[int, np.ndarray] = {}
    sep_miss = np.zeros(len(plan.pruned), dtype=np.int64)
    if separate:
        eta = alpha_s / len(plan.pruned)
        sizes = np.array([s.size for s in plan.pruned], dtype=np.int64)
        if config.mode == "adaptive" and config.target_recall is not None:
            allowance = ((found + extra_pos_lb)
                         * (1.0 - config.target_recall)
                         / max(config.target_recall, 1e-9)
                         * config.allowance_safety)
            m_sep = _separate_sample_sizes(sizes, allowance, eta, config)
        else:
            m_sep = np.array([config.sample_size(int(z)) for z in sizes],
                             dtype=np.int64)
        for j, (rec, s) in enumerate(zip(pruned_records, plan.pruned)):
            full_j = bool(m_sep[j] >= s.size)
            sub = (s.rows if full_j else
                   s.rows[rng.choice(s.size, size=int(m_sep[j]),
                                     replace=False)])
            ans_j = oracle.evaluate(predicate, sub,
                                    tag="fallback" if full_j else "audit")
            if full_j:
                escalation_calls += len(sub)
            else:
                audit_calls += len(sub)
            k_j = int(ans_j.sum())
            if k_j:
                corrections_pos.append(sub[ans_j])
            found += k_j
            rec.sampled, rec.positives = int(len(sub)), k_j
            sep_sample_of[j] = sub
            if full_j:
                active[j] = False
                rec.escalated = True
            else:
                sep_miss[j] = hypergeom_upper_bound(s.size, len(sub), k_j,
                                                    eta) - k_j

    def pruned_miss_ub() -> int:
        """Exact UCB on missed positives among *active* (non-escalated)
        pruned strata, using the sample restricted to them (valid by
        conditioning on where the uniform sample fell)."""
        if not active.any():
            return 0
        if separate:
            return int(sep_miss[active].sum())
        N_act = sum(s.size for s, a in zip(plan.pruned, active) if a)
        in_act = np.isin(samp_owner, np.flatnonzero(active))
        m_act = int(in_act.sum())
        k_act = int(samp_answers[in_act].sum())
        U = hypergeom_upper_bound(N_act, m_act, k_act, alpha_pool)
        return U - k_act

    def bound() -> float:
        miss = pruned_miss_ub()
        denom = found + extra_pos_lb + miss
        return 1.0 if denom == 0 else (found + extra_pos_lb) / denom

    def precision_bound() -> float | None:
        """Exact lower bound on the precision of the reported extension.

        Every reported row is either oracle-verified positive (candidates,
        audit samples, corrections, escalated strata) -- ``found`` of them --
        or an unsampled row of a retained assumed-positive stratum.  The
        per-stratum lower bounds ``L_i``, already union-bounded at level
        ``alpha_s/S`` *before* any data was seen, lower-bound the positives
        among the latter by ``extra_pos_lb``.  Hence

            precision  >=  (found + extra_pos_lb) / (found + unverified)

        at the same level as the recall certificate and at *zero* extra
        oracle cost: it reuses the very samples the recall bound needed.
        """
        if config.certify != "joint":
            return None
        denom = found + unverified_reported
        return 1.0 if denom == 0 else min(
            1.0, (found + extra_pos_lb) / denom)

    # ---- Phase 3: escalation ------------------------------------------
    while (config.escalate and config.target_recall is not None
           and bound() < config.target_recall and active.any()):
        if separate:
            # Largest per-stratum miss bound first (valid: simultaneous bounds).
            i = max(np.flatnonzero(active),
                    key=lambda j: (sep_miss[j], plan.pruned[j].size, -j))
            already = sep_sample_of[i]
        else:
            # Deterministic escalation order (size desc): avoids the
            # selection effect of picking strata by observed audit hits.
            i = next(j for j in esc_order if active[j])
            already = samp_rows[samp_owner == i]
        s = plan.pruned[i]
        rest = np.setdiff1d(s.rows, already, assume_unique=True)
        if len(rest):
            answers = oracle.evaluate(predicate, rest, tag="escalate")
            escalation_calls += len(rest)
            pos = rest[answers]
            if len(pos):
                corrections_pos.append(pos)
            found += int(answers.sum())
        active[i] = False
        pruned_records[i].escalated = True

    cpos = (np.unique(np.concatenate(corrections_pos))
            if corrections_pos else np.empty(0, dtype=np.int64))
    cneg = (np.unique(np.concatenate(corrections_neg))
            if corrections_neg else np.empty(0, dtype=np.int64))
    return AuditResult(recall_lower_bound=bound(),
                       precision_lower_bound=precision_bound(),
                       audit_calls=audit_calls,
                       escalation_calls=escalation_calls,
                       corrections_pos=cpos, corrections_neg=cneg,
                       strata=pos_records + pruned_records,
                       found_positives=found,
                       pruned_miss_ub=pruned_miss_ub(),
                       assumed_pos_lb=extra_pos_lb,
                       unverified_reported=unverified_reported)
