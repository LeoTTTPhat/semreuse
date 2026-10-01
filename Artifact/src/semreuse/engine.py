"""The SemReuse engine: predicate store + entailment rewriter + audit.

Query path for a new predicate p over N rows:
  1. exact-match lookup (normalized text) -> free hit;
  2. judge p against every cached view with the entailment reasoner
     (cheap tier, counted separately);
  3. build a rewrite plan (superset pruning / positive union / disjoint
     elimination / multi-view combination);
  4. oracle-evaluate the candidate set;
  5. stratified audit -> corrections + exact recall lower bound;
     escalate if the bound misses the target;
  6. publish the reported extension into the store with lineage.

The engine never reads ground truth; metrics are computed outside.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semreuse.audit import AuditConfig, AuditResult, run_audit
from semreuse.predicates import Predicate
from semreuse.rewriter import RewriteConfig, RewritePlan, build_plan
from semreuse.store import CachedView, PredicateStore


@dataclass
class EngineConfig:
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    enable_audit: bool = True
    enable_exact_match: bool = True
    seed: int = 0
    # Optional composition with a per-query cheap proxy (Section 6.3): the
    # proxy prunes *within* the candidate set the rewrite already reduced,
    # and the same audit certifies both kinds of pruning at once.
    proxy_pilot: int = 200
    proxy_bands: int = 4
    proxy_safety: float = 0.5
    proxy_min_kept_recall: float = 0.95
    proxy_min_prune_fraction: float = 0.2
    # Learn each view's measured agreement slack from past audits and spend
    # the recall allowance on rewrites knowingly (semreuse.slack).  None keeps
    # the original behaviour: every judgment clearing tau is used.
    slack_budget: float | None = None


@dataclass
class QueryResult:
    predicate: Predicate
    reported: np.ndarray            # bool[N] final reported extension
    oracle_calls: int               # candidates + audit + escalation
    candidate_calls: int
    audit_calls: int
    escalation_calls: int
    nli_pair_scores: int
    recall_bound: float | None      # None => exact (cold or exact-match hit)
    reuse_kind: str                 # 'exact-hit' | 'rewrite' | 'cold'
    precision_bound: float | None = None  # published precision certificate
    plan: RewritePlan | None = None
    audit_result: AuditResult | None = None


class SemReuseEngine:
    def __init__(self, corpus, oracle, entailment,
                 config: EngineConfig | None = None, proxy=None):
        self.corpus = corpus
        self.oracle = oracle
        self.entailment = entailment
        self.proxy = proxy
        self.config = config or EngineConfig()
        self.store = PredicateStore(n_rows=corpus.n)
        self.rng = np.random.default_rng(self.config.seed)
        self._query_index = 0
        from semreuse.slack import SlackStore
        # Written from completed audits, read when planning later queries --
        # never the same query, which is what keeps Theorem 2 intact.
        self.slack = SlackStore()

    # ------------------------------------------------------------------

    def query(self, predicate: Predicate) -> QueryResult:
        cfg = self.config
        n = self.corpus.n
        self._query_index += 1
        calls0 = self.oracle.stats.calls
        nli0 = getattr(self.entailment, "stats", None)
        nli0 = nli0.pair_scores if nli0 else 0

        # 1. exact match
        if cfg.enable_exact_match:
            hit = self.store.find_exact(predicate.text)
            if hit is not None:
                return QueryResult(
                    predicate=predicate, reported=hit.reported.copy(),
                    oracle_calls=0, candidate_calls=0, audit_calls=0,
                    escalation_calls=0, nli_pair_scores=0,
                    recall_bound=hit.recall_bound,
                    precision_bound=hit.precision_bound,
                    reuse_kind="exact-hit")

        # 2. entailment judgments against cached views
        views = self.store.candidates_for_matching()
        judgments = list(zip(views, self.entailment.judge_batch(
            predicate, [self._as_predicate(v) for v in views]))) if views else []

        # 3. rewrite plan, spending the recall allowance on the rewrites whose
        #    measured slack it can actually absorb
        budget = (cfg.slack_budget if cfg.slack_budget is not None
                  else cfg.rewrite.slack_budget)
        if budget is not None:
            cfg.rewrite.slack_budget = budget
        allowance = (1.0 - cfg.audit.target_recall
                     if cfg.audit.target_recall else None)
        plan = build_plan(n, judgments, cfg.rewrite,
                          slack_predictor=(self.slack.predict
                                           if budget is not None else None),
                          recall_allowance=allowance)

        # 3b. optional proxy cutoff *inside* the reduced candidate set
        pilot, pilot_ans = None, None
        if self.proxy is not None and len(plan.candidates):
            plan, pilot, pilot_ans = self._apply_proxy(predicate, plan)

        # 4. evaluate candidates
        cand_results = (self.oracle.evaluate(predicate, plan.candidates,
                                             tag="candidates")
                        if len(plan.candidates) else np.zeros(0, dtype=bool))
        reported = plan.reported_mask(cand_results)
        verified = np.zeros(n, dtype=bool)
        verified[plan.candidates] = True
        if pilot is not None:
            reported[pilot] = pilot_ans
            verified[pilot] = True
            # Pilot answers are paid-for evidence: count them as found.
            cand_results = np.concatenate([cand_results, pilot_ans])

        # 5. audit
        audit_res: AuditResult | None = None
        recall_bound: float | None = None
        precision_bound: float | None = None
        if plan.n_pruned == 0 and plan.n_assumed_pos == 0:
            recall_bound = None  # pure cold evaluation: exact w.r.t. oracle
        elif cfg.enable_audit:
            audit_res = run_audit(plan, predicate, self.oracle, cand_results,
                                  cfg.audit, self.rng)
            reported[audit_res.corrections_pos] = True
            reported[audit_res.corrections_neg] = False
            for rec, rows in self._audited_rows(plan, audit_res):
                verified[rows] = True
            recall_bound = audit_res.recall_lower_bound
            precision_bound = audit_res.precision_lower_bound
            # Feed what this audit measured back into the slack store, for the
            # benefit of *later* queries only.
            from semreuse.slack import observations_from_audit
            for src, kind, sl, frac, w in observations_from_audit(
                    plan, audit_res, n):
                self.slack.observe(src, kind, sl, frac, w)

        # 6. publish
        view = CachedView(
            pid=predicate.pid, text=predicate.text, reported=reported.copy(),
            verified=verified,
            oracle_version=getattr(self.oracle, "version", "sim-v1"),
            query_index=self._query_index, recall_bound=recall_bound,
            precision_bound=precision_bound,
            derived_from=tuple(pid for pid, _, _ in plan.used_views))
        self.store.add(view)

        nli1 = getattr(self.entailment, "stats", None)
        nli1 = nli1.pair_scores if nli1 else 0
        total_calls = self.oracle.stats.calls - calls0
        reuse_kind = "rewrite" if plan.used_views else "cold"
        return QueryResult(
            predicate=predicate, reported=reported,
            oracle_calls=total_calls, candidate_calls=len(plan.candidates),
            audit_calls=audit_res.audit_calls if audit_res else 0,
            escalation_calls=audit_res.escalation_calls if audit_res else 0,
            nli_pair_scores=nli1 - nli0, recall_bound=recall_bound,
            precision_bound=precision_bound,
            reuse_kind=reuse_kind, plan=plan, audit_result=audit_res)

    # ------------------------------------------------------------------

    def _apply_proxy(self, predicate, plan):
        """Draw a pilot inside the candidate set, pick a recall-targeted
        proxy cutoff, and turn the rows it drops into pruned strata."""
        from semreuse.proxy import apply_proxy_cutoff, choose_cutoff

        cfg = self.config
        cands = plan.candidates
        m0 = min(len(cands), cfg.proxy_pilot)
        pilot = np.sort(self.rng.choice(cands, size=m0, replace=False))
        pilot_ans = self.oracle.evaluate(predicate, pilot, tag="pilot")
        scores = self.proxy.trained_scores(predicate, pilot, pilot_ans)
        tau = choose_cutoff(scores, pilot, pilot_ans,
                            cfg.audit.target_recall or 0.9, cfg.proxy_safety,
                            min_kept_recall=cfg.proxy_min_kept_recall,
                            min_prune_fraction=cfg.proxy_min_prune_fraction,
                            rng=self.rng)
        rest = np.setdiff1d(cands, pilot, assume_unique=True)
        plan = RewritePlan(n_rows=plan.n_rows, candidates=rest,
                           assumed_pos=plan.assumed_pos, pruned=plan.pruned,
                           used_views=plan.used_views)
        plan = apply_proxy_cutoff(plan, scores, tau, cfg.proxy_bands,
                                  predicate.pid)
        return plan, pilot, pilot_ans

    @staticmethod
    def _as_predicate(view: CachedView) -> Predicate:
        return Predicate(text=view.text, label_set=frozenset(), name=view.pid)

    @staticmethod
    def _audited_rows(plan: RewritePlan, audit_res: AuditResult):
        """Rows verified by audits (corrections are a subset)."""
        # Sampled rows are not stored per-stratum in the result; the
        # corrections plus escalated strata are what we can mark verified.
        # NOTE: audit order is assumed_pos first, then pruned (see run_audit).
        out = []
        strata = plan.assumed_pos + plan.pruned
        for rec, s in zip(audit_res.strata, strata):
            if rec.escalated or rec.sampled == rec.size:
                out.append((rec, s.rows))
        if len(audit_res.corrections_pos):
            out.append((None, audit_res.corrections_pos))
        if len(audit_res.corrections_neg):
            out.append((None, audit_res.corrections_neg))
        return out
