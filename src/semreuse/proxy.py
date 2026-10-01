"""Proxy-cascade evaluation: the standard single-query way to spend fewer
oracle calls, and its composition with entailment-based reuse.

A reader of SUPG, NoScope, BARGAIN, or LOTUS will ask the obvious question:
*if I am willing to accept a recall target below 1, why do I need cross-query
reuse at all?*  A cheap proxy model scores every row, the oracle is spent only
on the high-scoring ones, and a sample certifies what the low-scoring tail
cost.  That attacks the same budget with none of SemReuse's machinery, and it
is the baseline against which reuse has to justify itself.

This module implements it, and -- more importantly -- shows the two techniques
are orthogonal.  Both reduce the same object: the set of rows the oracle must
see.  Reuse shrinks it using *other queries'* answers; a proxy shrinks it using
*this query's* cheap scores.  Because SemReuse already expresses "rows I did
not evaluate, and why" as pruned strata, a proxy cutoff is just another kind of
stratum, and one audit certifies the combination.

Cost accounting.  Proxy scores are cheap-tier work, reported separately from
oracle calls, exactly like NLI pair scorings.  Unlike the entailment tier,
whose cost is O(#views) per query, the proxy tier costs O(N) *per query* plus a
one-time O(N) document-embedding pass -- a difference that matters at scale and
that the evaluation reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semreuse.audit import AuditConfig, run_audit
from semreuse.engine import QueryResult
from semreuse.predicates import Predicate
from semreuse.rewriter import RewritePlan, Stratum
from semreuse.store import CachedView, PredicateStore


@dataclass
class ProxyStats:
    doc_embeddings: int = 0     # one-time corpus pass
    row_scores: int = 0         # per-query O(N) scoring work


class EmbeddingProxy:
    """Cheap per-row relevance scores, zero-shot or trained per query.

    ``mode="cosine"`` scores rows by cosine(document, predicate): no labels
    needed, and the weakest honest version of the idea.  ``mode="trained"``
    is what SUPG and NoScope actually do -- fit a cheap classifier for *this*
    predicate on the pilot's oracle labels and score every row with it.  The
    trained variant is the baseline reuse has to beat, so it is the default.

    The document pass is done once for the corpus and amortized over the
    workload; scoring a predicate is one embedding plus a dense mat-vec (plus,
    in trained mode, a logistic-regression fit on a few hundred labelled rows).
    """

    def __init__(self, corpus, embedder, batch_size: int = 512,
                 mode: str = "trained"):
        self.corpus = corpus
        self.embedder = embedder
        self.batch_size = batch_size
        self.mode = mode
        self.stats = ProxyStats()
        self._doc_vecs: np.ndarray | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._trained: dict[str, np.ndarray] = {}

    def _docs(self) -> np.ndarray:
        if self._doc_vecs is None:
            vecs = []
            for i in range(0, self.corpus.n, self.batch_size):
                vecs.append(self.embedder.encode(
                    self.corpus.docs[i:i + self.batch_size]))
            self._doc_vecs = np.vstack(vecs).astype(np.float32)
            self.stats.doc_embeddings += self.corpus.n
        return self._doc_vecs

    def scores(self, predicate: Predicate) -> np.ndarray:
        """Zero-shot scores (also the fallback when training is impossible)."""
        if predicate.text not in self._cache:
            q = self.embedder.encode([predicate.text])[0].astype(np.float32)
            self._cache[predicate.text] = self._docs() @ q
            self.stats.row_scores += self.corpus.n
        return self._cache[predicate.text]

    def reset_training(self) -> None:
        """Forget per-query trained proxies.

        The trained scores are a function of the pilot, and the pilot depends
        on which engine drew it.  Sharing the cache across engines (or across
        configurations in a sweep) would silently let one method's pilot
        decide another method's cutoff, so every run that changes the engine
        or the configuration must call this first.
        """
        self._trained.clear()

    def trained_scores(self, predicate: Predicate, pilot_rows: np.ndarray,
                       pilot_answers: np.ndarray) -> np.ndarray:
        """Per-query proxy trained on the pilot's oracle labels.

        Falls back to zero-shot scores when the pilot is one-class (nothing to
        fit).  Training cost is cheap-tier: one logistic regression over a few
        hundred 384-dimensional vectors, milliseconds per query.
        """
        if self.mode != "trained":
            return self.scores(predicate)
        key = predicate.text
        if key in self._trained:
            return self._trained[key]
        y = np.asarray(pilot_answers, dtype=int)
        if y.min() == y.max():
            return self.scores(predicate)
        from sklearn.linear_model import LogisticRegression

        X = self._docs()
        clf = LogisticRegression(max_iter=1000, C=1.0,
                                 class_weight="balanced")
        clf.fit(X[pilot_rows], y)
        sc = clf.decision_function(X).astype(np.float32)
        self._trained[key] = sc
        self.stats.row_scores += self.corpus.n
        return sc


# ---------------------------------------------------------------------------
# Cutoff selection
# ---------------------------------------------------------------------------

def choose_cutoff(scores: np.ndarray, pilot_rows: np.ndarray,
                  pilot_answers: np.ndarray, target: float,
                  safety: float = 0.5, min_positives: int = 8,
                  min_kept_recall: float = 0.95,
                  min_prune_fraction: float = 0.2,
                  rng: np.random.Generator | None = None) -> float | None:
    """SUPG-style recall-targeted cutoff, chosen and *validated* on the pilot.

    The naive rule -- take the ``(1-target)*safety`` quantile of the sampled
    positives' scores -- is circular: it is fitted on the same positives it is
    then judged by, so it always looks good, and when the proxy does not in
    fact separate this predicate the cutoff prunes qualifying rows that the
    audit must later buy back with an expensive escalation.  Measured on 20
    Newsgroups, that failure mode costs more than the cutoff saves.

    We therefore split the pilot: the cutoff is fitted on one half and its
    kept-recall is estimated on the held-out half.  The cutoff is used only if
    it both keeps enough of the positives (``min_kept_recall``) and prunes
    enough rows to be worth the risk (``min_prune_fraction``).  Otherwise the
    proxy is declined for this query.  Correctness never depended on this
    decision -- the audit certifies whatever plan it is handed -- so the rule
    is a pure cost heuristic, which is exactly where a heuristic belongs.
    """
    pos_idx = np.flatnonzero(pilot_answers)
    if len(pos_idx) < min_positives:
        return None
    rng = rng or np.random.default_rng(0)
    perm = rng.permutation(len(pos_idx))
    half = len(perm) // 2
    fit = scores[pilot_rows[pos_idx[perm[:half]]]]
    val = scores[pilot_rows[pos_idx[perm[half:]]]]
    q = max(0.0, (1.0 - target) * safety)
    tau = float(np.quantile(fit, q))
    if len(val) and float((val >= tau).mean()) < min_kept_recall:
        return None                       # proxy does not separate: decline
    if float((scores >= tau).mean()) > 1.0 - min_prune_fraction:
        return None                       # too little pruned to be worth it
    return tau


def score_band_strata(rows: np.ndarray, scores: np.ndarray, n_bands: int,
                      kind: str, source: str) -> list[Stratum]:
    """Split pruned rows into score bands so escalation is graded.

    Escalating the *highest-scoring* pruned band first is what makes a proxy
    cascade recover cheaply from a bad cutoff; a single undifferentiated pool
    would force it straight to cold evaluation.
    """
    if len(rows) == 0:
        return []
    s = scores[rows]
    order = np.argsort(-s, kind="stable")
    parts = np.array_split(order, min(n_bands, max(1, len(rows))))
    out = []
    for b, part in enumerate(parts):
        if len(part) == 0:
            continue
        out.append(Stratum(kind=kind, source_pid=f"{source}#band{b}",
                           source_text=f"proxy score band {b}",
                           confidence=float(1.0 - b / max(1, len(parts))),
                           rows=np.sort(rows[part])))
    return out


# ---------------------------------------------------------------------------
# Standalone proxy-cascade engine (the baseline)
# ---------------------------------------------------------------------------

@dataclass
class ProxyCascadeConfig:
    pilot_size: int = 200
    n_bands: int = 4
    safety: float = 0.5
    min_kept_recall: float = 0.95
    min_prune_fraction: float = 0.2
    audit: AuditConfig = field(default_factory=AuditConfig)


class ProxyCascadeEngine:
    """Per-query proxy cascade with the same certificate contract.

    Pipeline: score all rows with the proxy; oracle-label a uniform pilot to
    locate a recall-targeted cutoff; oracle-evaluate everything above it;
    audit the tail with the *same* pooled hypergeometric machinery SemReuse
    uses, escalating band by band until the target bound is met.  Reported
    positives are all oracle-verified, so precision is exactly 1 -- this is a
    strong baseline, deliberately given every advantage except reuse.
    """

    def __init__(self, corpus, oracle, proxy: EmbeddingProxy,
                 config: ProxyCascadeConfig | None = None, seed: int = 0):
        self.corpus = corpus
        self.oracle = oracle
        self.proxy = proxy
        self.config = config or ProxyCascadeConfig()
        self.rng = np.random.default_rng(seed)
        self._qi = 0

    def query(self, predicate: Predicate) -> QueryResult:
        cfg = self.config
        n = self.corpus.n
        self._qi += 1
        calls0 = self.oracle.stats.calls
        # --- pilot -------------------------------------------------------
        m0 = min(n, cfg.pilot_size)
        pilot = np.sort(self.rng.choice(n, size=m0, replace=False))
        pilot_ans = self.oracle.evaluate(predicate, pilot, tag="pilot")
        scores = self.proxy.trained_scores(predicate, pilot, pilot_ans)
        tau = choose_cutoff(scores, pilot, pilot_ans,
                            cfg.audit.target_recall or 0.9, cfg.safety,
                            min_kept_recall=cfg.min_kept_recall,
                            min_prune_fraction=cfg.min_prune_fraction,
                            rng=self.rng)

        keep = (np.ones(n, dtype=bool) if tau is None else scores >= tau)
        keep[pilot] = False                      # already paid for
        cand = np.flatnonzero(keep)
        pruned_rows = np.flatnonzero(~keep)
        pruned_rows = pruned_rows[~np.isin(pruned_rows, pilot)]
        strata = score_band_strata(pruned_rows, scores, cfg.n_bands,
                                   "pruned-proxy", predicate.pid)
        plan = RewritePlan(n_rows=n, candidates=cand, assumed_pos=[],
                           pruned=strata, used_views=[])

        cand_ans = (self.oracle.evaluate(predicate, cand, tag="candidates")
                    if len(cand) else np.zeros(0, dtype=bool))
        reported = plan.reported_mask(cand_ans)
        reported[pilot] = pilot_ans              # fold in the pilot answers

        audit_res = None
        recall_bound = precision_bound = None
        if strata:
            # The pilot's own positives count as found evidence: pass them in
            # by extending the candidate answer vector (they were paid for).
            audit_res = run_audit(plan, predicate, self.oracle,
                                  np.concatenate([cand_ans, pilot_ans]),
                                  cfg.audit, self.rng)
            reported[audit_res.corrections_pos] = True
            reported[audit_res.corrections_neg] = False
            recall_bound = audit_res.recall_lower_bound
            precision_bound = audit_res.precision_lower_bound

        total = self.oracle.stats.calls - calls0
        return QueryResult(
            predicate=predicate, reported=reported, oracle_calls=total,
            candidate_calls=len(cand) + m0,
            audit_calls=audit_res.audit_calls if audit_res else 0,
            escalation_calls=audit_res.escalation_calls if audit_res else 0,
            nli_pair_scores=0, recall_bound=recall_bound,
            reuse_kind="proxy-cascade", precision_bound=precision_bound,
            plan=plan, audit_result=audit_res)


# ---------------------------------------------------------------------------
# Composition: a proxy cutoff inside a rewrite plan
# ---------------------------------------------------------------------------

def apply_proxy_cutoff(plan: RewritePlan, scores: np.ndarray,
                       tau: float | None, n_bands: int,
                       source: str) -> RewritePlan:
    """Shrink a rewrite plan's candidate set with a proxy cutoff.

    The removed rows become ordinary pruned strata, so the existing audit
    certifies the union of two very different assumptions -- "a cached view
    says these rows cannot qualify" and "a cheap model says these rows score
    too low" -- with one bound and one escalation chain.  Nothing about the
    certificate changes: it never cared *why* a row was pruned.
    """
    if tau is None or len(plan.candidates) == 0:
        return plan
    keep = scores[plan.candidates] >= tau
    dropped = plan.candidates[~keep]
    if len(dropped) == 0:
        return plan
    return RewritePlan(
        n_rows=plan.n_rows, candidates=plan.candidates[keep],
        assumed_pos=plan.assumed_pos,
        pruned=plan.pruned + score_band_strata(dropped, scores, n_bands,
                                               "pruned-proxy", source),
        used_views=plan.used_views)
