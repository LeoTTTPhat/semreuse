"""LLM predicate oracles with cost accounting.

The "expensive LLM" that evaluates an NL predicate on a row is simulated:
ground truth comes from corpus labels, each call costs one unit, and the
system's objective is to minimize oracle calls at bounded recall loss.
A noise model makes the oracle imperfect in a *deterministic, repeatable*
way (the same (predicate, row) always gets the same answer), matching how a
temperature-0 LLM behaves.

`NLIOracle` optionally uses a local cross-encoder as a *real* noisy oracle,
scoring (document, predicate) pairs directly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from semreuse.corpus import Corpus
from semreuse.predicates import Predicate


@dataclass
class OracleStats:
    calls: int = 0
    per_query: dict[str, int] = field(default_factory=dict)

    def charge(self, tag: str, n: int) -> None:
        self.calls += n
        self.per_query[tag] = self.per_query.get(tag, 0) + n


class SimulatedOracle:
    """Ground-truth oracle over a labeled corpus, with optional noise.

    noise: probability that a given (predicate, row) answer is flipped.
    Flips are a deterministic function of (predicate text, row id, seed),
    so repeated evaluation is consistent -- crucially, audit samples see the
    same answer the original evaluation would have seen.
    """

    def __init__(self, corpus: Corpus, noise: float = 0.0, seed: int = 0):
        self.corpus = corpus
        self.noise = noise
        self.seed = seed
        self.stats = OracleStats()

    def _flip_mask(self, predicate: Predicate, rows: np.ndarray) -> np.ndarray:
        if self.noise <= 0:
            return np.zeros(len(rows), dtype=bool)
        h = hashlib.sha1(f"{predicate.text}|{self.seed}".encode()).digest()
        base = int.from_bytes(h[:8], "big")
        # Deterministic per-row uniform via splitmix-style hashing.
        x = (np.asarray(rows, dtype=np.uint64) + np.uint64(base)) * np.uint64(
            0x9E3779B97F4A7C15)
        x ^= x >> np.uint64(31)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        u = (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)
        return u < self.noise

    def truth(self, predicate: Predicate) -> np.ndarray:
        """Full ground-truth extension (label semantics; NOT charged).

        Used only for metrics, never by the engines.
        """
        return self.corpus.extension(predicate.label_set)

    def reference(self, predicate: Predicate) -> np.ndarray:
        """The oracle's *own* semantics on all rows (labels ^ noise flips),
        NOT charged.  This is the reference for the C2 recall certificate,
        which is conditional on the oracle being the reference semantics.
        Equal to ``truth`` when noise == 0."""
        ext = self.corpus.extension(predicate.label_set)
        rows = np.arange(self.corpus.n, dtype=np.int64)
        return ext ^ self._flip_mask(predicate, rows)

    def evaluate(self, predicate: Predicate, rows: np.ndarray,
                 tag: str = "eval") -> np.ndarray:
        """Answer the predicate on the given rows.  Charges len(rows) calls."""
        rows = np.asarray(rows, dtype=np.int64)
        self.stats.charge(tag, len(rows))
        answers = self.corpus.extension(predicate.label_set)[rows]
        flips = self._flip_mask(predicate, rows)
        return answers ^ flips


class NLIOracle:
    """A *real* local-model oracle: cross-encoder NLI on (document, predicate).

    Treats the document (truncated) as premise and the predicate as
    hypothesis; predicts True when P(entailment) > threshold.  Slow (real
    inference), noisy (real errors), useful for the realism ablation.
    """

    def __init__(self, corpus: Corpus,
                 model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
                 threshold: float = 0.5, max_chars: int = 800,
                 batch_size: int = 64, device: str | None = None):
        from sentence_transformers import CrossEncoder  # lazy import

        self.corpus = corpus
        self.model = CrossEncoder(model_name, device=device)
        self.threshold = threshold
        self.max_chars = max_chars
        self.batch_size = batch_size
        self.stats = OracleStats()
        self._cache: dict[tuple[str, int], bool] = {}

    def truth(self, predicate: Predicate) -> np.ndarray:
        return self.corpus.extension(predicate.label_set)

    def evaluate(self, predicate: Predicate, rows: np.ndarray,
                 tag: str = "eval") -> np.ndarray:
        import scipy.special

        rows = np.asarray(rows, dtype=np.int64)
        self.stats.charge(tag, len(rows))
        out = np.zeros(len(rows), dtype=bool)
        todo, todo_pos = [], []
        for i, r in enumerate(rows):
            key = (predicate.pid, int(r))
            if key in self._cache:
                out[i] = self._cache[key]
            else:
                todo.append((self.corpus.docs[r][: self.max_chars],
                             predicate.text))
                todo_pos.append(i)
        if todo:
            logits = self.model.predict(todo, batch_size=self.batch_size,
                                        convert_to_numpy=True,
                                        show_progress_bar=False)
            probs = scipy.special.softmax(logits, axis=1)
            # sentence-transformers NLI cross-encoders order labels
            # [contradiction, entailment, neutral].
            ent = probs[:, 1]
            for i, p_ent in zip(todo_pos, ent):
                val = bool(p_ent > self.threshold)
                out[i] = val
                self._cache[(predicate.pid, int(rows[i]))] = val
        return out
