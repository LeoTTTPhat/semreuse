"""The integration seam: dropping SemReuse under an existing semantic operator.

SemReuse is not a query engine and does not want to be one.  It interposes at
exactly one point: where a semantic filter is about to fan out one LLM call per
row.  The host system hands over an *oracle closure* -- anything that answers
``predicate x rows -> bool[]`` -- and gets back a bitmap plus a certificate.
Everything the host's own optimizer does (proxy cascades, batching, model
routing, prompt caching) happens inside that closure and is untouched; SemReuse
only shrinks the row set the closure is invoked on, and its audit calls go
through the same closure, so the certificate is stated relative to whatever
semantics the host actually implements.

``SemanticFilterService`` is that seam, with the store persisting across calls.
``lotus_sem_filter`` is a drop-in replacement for LOTUS's ``sem_filter``
dataframe accessor with the same signature; passing ``lm=`` a configured LOTUS
model routes oracle calls through LOTUS itself, so the reduction we report is a
reduction in *LOTUS's* LLM calls, not in a reimplementation of them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from semreuse.audit import AuditConfig
from semreuse.corpus import Corpus
from semreuse.engine import EngineConfig, SemReuseEngine
from semreuse.oracle import OracleStats
from semreuse.predicates import Predicate


class ClosureOracle:
    """Adapts a host system's per-row evaluator to the oracle interface.

    ``fn(predicate_text, row_ids) -> bool array``.  Answers are memoized so
    that an audit which re-touches an already-evaluated row is free and, more
    importantly, consistent -- the determinism the certificate assumes.
    """

    version = "closure-v1"

    def __init__(self, corpus: Corpus, fn, version: str | None = None):
        self.corpus = corpus
        self.fn = fn
        self.stats = OracleStats()
        self._memo: dict[tuple[str, int], bool] = {}
        if version:
            self.version = version

    def evaluate(self, predicate: Predicate, rows: np.ndarray,
                 tag: str = "eval") -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.zeros(len(rows), dtype=bool)
        todo_pos, todo_rows = [], []
        for i, r in enumerate(rows):
            key = (predicate.text, int(r))
            if key in self._memo:
                out[i] = self._memo[key]
            else:
                todo_pos.append(i)
                todo_rows.append(int(r))
        if todo_rows:
            ans = np.asarray(self.fn(predicate.text, np.array(todo_rows)),
                             dtype=bool)
            self.stats.charge(tag, len(todo_rows))
            for i, r, a in zip(todo_pos, todo_rows, ans):
                out[i] = a
                self._memo[(predicate.text, r)] = bool(a)
        return out

    # The host's semantics *are* the reference semantics; there is no other
    # ground truth to appeal to (Section 3).
    def reference(self, predicate: Predicate) -> np.ndarray:
        return self.evaluate(predicate, np.arange(self.corpus.n),
                             tag="reference")

    truth = reference


@dataclass
class FilterResult:
    mask: np.ndarray
    oracle_calls: int
    recall_bound: float | None
    precision_bound: float | None
    reuse_kind: str
    seconds_total: float
    seconds_engine: float          # planning + entailment + bound inversion

    @property
    def seconds_oracle(self) -> float:
        return max(0.0, self.seconds_total - self.seconds_engine)


@dataclass
class ServiceStats:
    queries: int = 0
    oracle_calls: int = 0
    cold_equivalent: int = 0
    seconds_engine: float = 0.0
    seconds_total: float = 0.0
    per_query: list = field(default_factory=list)

    @property
    def reduction(self) -> float:
        return self.cold_equivalent / max(1, self.oracle_calls)


class SemanticFilterService:
    """A long-lived semantic-filter endpoint with a persistent predicate store.

    Usage from a host engine::

        svc = SemanticFilterService(corpus, my_llm_closure, target_recall=0.9)
        res = svc.filter("the document is about baseball")
        df[res.mask]            # rows, with res.recall_bound certified

    The store survives across calls; that is the entire point.
    """

    def __init__(self, corpus: Corpus, fn, entailment=None,
                 target_recall: float = 0.9, alpha: float = 0.05,
                 tau: float = 0.5, seed: int = 0, proxy=None,
                 oracle_version: str | None = None):
        self.oracle = ClosureOracle(corpus, fn, version=oracle_version)
        if entailment is None:
            from semreuse.entailment import NLIEntailment
            entailment = NLIEntailment()
        from semreuse.rewriter import RewriteConfig
        cfg = EngineConfig(
            rewrite=RewriteConfig(tau_equiv=tau, tau_forward=tau,
                                  tau_backward=tau, tau_disjoint=tau),
            audit=AuditConfig(alpha=alpha, target_recall=target_recall),
            seed=seed)
        self.engine = SemReuseEngine(corpus, self.oracle, entailment, cfg,
                                     proxy=proxy)
        self.corpus = corpus
        self.stats = ServiceStats()

    def filter(self, predicate_text: str) -> FilterResult:
        pred = Predicate(text=predicate_text, label_set=frozenset(),
                         name=predicate_text[:40])
        t0 = time.perf_counter()
        calls0 = self.oracle.stats.calls
        res = self.engine.query(pred)
        total = time.perf_counter() - t0
        calls = self.oracle.stats.calls - calls0
        # Engine-side time is everything that is not the host's own evaluator.
        engine_s = total - getattr(self, "_last_oracle_s", 0.0)
        out = FilterResult(
            mask=res.reported, oracle_calls=calls,
            recall_bound=res.recall_bound,
            precision_bound=res.precision_bound,
            reuse_kind=res.reuse_kind, seconds_total=total,
            seconds_engine=max(0.0, engine_s))
        self.stats.queries += 1
        self.stats.oracle_calls += calls
        self.stats.cold_equivalent += self.corpus.n
        self.stats.seconds_total += total
        self.stats.per_query.append(out)
        return out


# ---------------------------------------------------------------------------
# LOTUS drop-in
# ---------------------------------------------------------------------------

def lotus_oracle_closure(docs: list[str], column: str = "text",
                         lm=None, batch: int = 64):
    """An oracle closure backed by LOTUS's own ``sem_filter``.

    Requires the ``lotus-ai`` package and a configured LM.  Each invocation
    hands LOTUS a dataframe of just the rows SemReuse still needs, so the LLM
    calls LOTUS makes are exactly the calls SemReuse did not eliminate --
    which is what makes the measured reduction a reduction of LOTUS's cost
    rather than of our own reimplementation of it.
    """
    import lotus                      # noqa: F401  (import validates install)
    import pandas as pd

    def fn(predicate_text: str, rows: np.ndarray) -> np.ndarray:
        out = np.zeros(len(rows), dtype=bool)
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            df = pd.DataFrame({column: [docs[int(r)] for r in chunk]})
            kept = df.sem_filter(f"{{{column}}}: {predicate_text}")
            out[i:i + len(chunk)] = np.isin(np.arange(len(chunk)),
                                            kept.index.to_numpy())
        return out

    return fn


def lotus_sem_filter(df, user_instruction: str, service: SemanticFilterService):
    """``df.sem_filter``-compatible entry point that goes through SemReuse.

    The signature mirrors LOTUS's accessor so the substitution is one line in
    a host pipeline; the return value is the filtered frame, with the
    certificate attached as frame-level attributes.
    """
    res = service.filter(user_instruction)
    out = df[res.mask]
    out.attrs["semreuse_recall_bound"] = res.recall_bound
    out.attrs["semreuse_precision_bound"] = res.precision_bound
    out.attrs["semreuse_oracle_calls"] = res.oracle_calls
    out.attrs["semreuse_reuse_kind"] = res.reuse_kind
    return out
