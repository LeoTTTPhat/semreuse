"""Baseline engines: cold, exact-match cache, embedding-similarity cache.

All baselines share the SemReuse cost model (oracle calls are the currency)
and produce a reported extension per query so accuracy is comparable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from semreuse.engine import QueryResult
from semreuse.predicates import Predicate
from semreuse.store import CachedView, PredicateStore


class ColdEngine:
    """No reuse: every predicate is oracle-evaluated on every row (LOTUS-style
    single-query processing without cross-query state)."""

    def __init__(self, corpus, oracle):
        self.corpus = corpus
        self.oracle = oracle

    def query(self, predicate: Predicate) -> QueryResult:
        rows = np.arange(self.corpus.n)
        res = self.oracle.evaluate(predicate, rows, tag="cold")
        return QueryResult(predicate=predicate, reported=res,
                           oracle_calls=self.corpus.n,
                           candidate_calls=self.corpus.n, audit_calls=0,
                           escalation_calls=0, nli_pair_scores=0,
                           recall_bound=None, reuse_kind="cold")


class ExactCacheEngine:
    """Reuse only on exact (normalized) predicate-text match; else cold."""

    def __init__(self, corpus, oracle):
        self.corpus = corpus
        self.oracle = oracle
        self.store = PredicateStore(n_rows=corpus.n)
        self._qi = 0

    def query(self, predicate: Predicate) -> QueryResult:
        self._qi += 1
        hit = self.store.find_exact(predicate.text)
        if hit is not None:
            return QueryResult(predicate=predicate,
                               reported=hit.reported.copy(), oracle_calls=0,
                               candidate_calls=0, audit_calls=0,
                               escalation_calls=0, nli_pair_scores=0,
                               recall_bound=None, reuse_kind="exact-hit")
        rows = np.arange(self.corpus.n)
        res = self.oracle.evaluate(predicate, rows, tag="cold")
        self.store.add(CachedView(pid=predicate.pid, text=predicate.text,
                                  reported=res.copy(),
                                  verified=np.ones(self.corpus.n, dtype=bool),
                                  query_index=self._qi))
        return QueryResult(predicate=predicate, reported=res,
                           oracle_calls=self.corpus.n,
                           candidate_calls=self.corpus.n, audit_calls=0,
                           escalation_calls=0, nli_pair_scores=0,
                           recall_bound=None, reuse_kind="cold")


# ---------------------------------------------------------------------------
# Embedding-similarity cache (GPTCache-style)
# ---------------------------------------------------------------------------

class HashingEmbedder:
    """Deterministic char-3-gram hashing embedder (test/no-download fallback)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = re.sub(r"\s+", " ", t.lower())
            for j in range(len(t) - 2):
                g = t[j:j + 3]
                h = int.from_bytes(
                    hashlib.md5(g.encode()).digest()[:4], "big")
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceEmbedder:
    """sentence-transformers bi-encoder wrapper with a text cache."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in texts if t not in self._cache]
        if todo:
            vecs = self.model.encode(todo, normalize_embeddings=True,
                                     show_progress_bar=False)
            for t, v in zip(todo, vecs):
                self._cache[t] = np.asarray(v)
        return np.stack([self._cache[t] for t in texts])


@dataclass
class _EmbeddedView:
    view: CachedView
    vec: np.ndarray


class EmbeddingCacheEngine:
    """Semantic cache: if a cached predicate's embedding is within cosine
    threshold theta, return its cached result wholesale; else cold-evaluate.
    This is the GPTCache-style baseline: no logical containment, no partial
    reuse, no guarantee."""

    def __init__(self, corpus, oracle, embedder, theta: float = 0.9):
        self.corpus = corpus
        self.oracle = oracle
        self.embedder = embedder
        self.theta = theta
        self.entries: list[_EmbeddedView] = []
        self._qi = 0

    def query(self, predicate: Predicate) -> QueryResult:
        self._qi += 1
        vec = self.embedder.encode([predicate.text])[0]
        best, best_sim = None, -1.0
        for e in self.entries:
            sim = float(vec @ e.vec)
            if sim > best_sim:
                best, best_sim = e, sim
        if best is not None and best_sim >= self.theta:
            return QueryResult(predicate=predicate,
                               reported=best.view.reported.copy(),
                               oracle_calls=0, candidate_calls=0,
                               audit_calls=0, escalation_calls=0,
                               nli_pair_scores=0, recall_bound=None,
                               reuse_kind=f"embed-hit@{best_sim:.2f}")
        rows = np.arange(self.corpus.n)
        res = self.oracle.evaluate(predicate, rows, tag="cold")
        view = CachedView(pid=predicate.pid, text=predicate.text,
                          reported=res.copy(),
                          verified=np.ones(self.corpus.n, dtype=bool),
                          query_index=self._qi)
        self.entries.append(_EmbeddedView(view=view, vec=vec))
        return QueryResult(predicate=predicate, reported=res,
                           oracle_calls=self.corpus.n,
                           candidate_calls=self.corpus.n, audit_calls=0,
                           escalation_calls=0, nli_pair_scores=0,
                           recall_bound=None, reuse_kind="cold")
