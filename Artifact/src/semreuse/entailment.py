"""Entailment reasoners: classify the logical relation between NL predicates.

Given a new predicate p and a cached predicate q, the reasoner outputs a
:class:`Judgment` -- a relation in {EQUIV, FORWARD (p=>q), BACKWARD (q=>p),
DISJOINT, OVERLAP} plus a confidence in [0,1] that the rewriter thresholds
against and the audit layer treats as fallible.

Implementations:
  * :class:`GroundTruthEntailment` -- label-set algebra, optionally corrupted
    with a controlled error rate (for audit-mechanism experiments and tests).
  * :class:`NLIEntailment` -- a cross-encoder NLI model scored in *both
    directions*; a 6-dim feature vector (softmax probs of contradiction/
    entailment/neutral, forward and backward) is mapped to a relation either
    by interpretable thresholds or by a calibrated multinomial logistic head
    fit on a small set of labeled predicate pairs.

All NLI pair scorings are counted (`stats.pair_scores`) -- this is the cheap
tier whose cost we report alongside saved oracle calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from semreuse.predicates import Predicate, Relation, true_relation

RELATIONS = [Relation.EQUIV, Relation.FORWARD, Relation.BACKWARD,
             Relation.DISJOINT, Relation.OVERLAP]


@dataclass
class Judgment:
    relation: Relation
    confidence: float

    def __repr__(self) -> str:  # pragma: no cover
        return f"Judgment({self.relation.value}, {self.confidence:.2f})"


@dataclass
class EntailmentStats:
    pair_scores: int = 0     # cross-encoder forward passes (2 per pair)
    judgments: int = 0


class GroundTruthEntailment:
    """Perfect (or deliberately corrupted) entailment via label-set algebra.

    error_rate: probability a judgment is replaced by a *wrong* relation
    (deterministic per unordered pair, so repeated queries are consistent).
    Used to stress the audit layer with known entailment error rates.
    """

    def __init__(self, resolver: dict[str, frozenset[str]],
                 error_rate: float = 0.0, confidence: float = 0.99,
                 seed: int = 0):
        self.resolver = resolver
        self.error_rate = error_rate
        self.confidence = confidence
        self.seed = seed
        self.stats = EntailmentStats()

    def _corrupt(self, rel: Relation, key: str) -> Relation:
        h = hashlib.sha1(f"{key}|{self.seed}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / float(1 << 64)
        if u >= self.error_rate:
            return rel
        others = [r for r in RELATIONS if r is not rel]
        pick = int.from_bytes(h[8:12], "big") % len(others)
        return others[pick]

    def judge(self, p: Predicate, q: Predicate) -> Judgment:
        self.stats.judgments += 1
        a = self.resolver.get(p.text, p.label_set)
        b = self.resolver.get(q.text, q.label_set)
        rel = true_relation(a, b)
        if self.error_rate > 0:
            rel = self._corrupt(rel, f"{p.text}||{q.text}")
        return Judgment(rel, self.confidence)

    def judge_batch(self, p: Predicate,
                    qs: list[Predicate]) -> list[Judgment]:
        return [self.judge(p, q) for q in qs]


class NLIEntailment:
    """Two-direction cross-encoder NLI with a calibrated relation head.

    For a pair (p, q) we score:
        forward:  premise=p.text, hypothesis=q.text   (does p entail q?)
        backward: premise=q.text, hypothesis=p.text
    yielding softmax probabilities (c_f, e_f, n_f, c_b, e_b, n_b).

    Relation heads:
      * threshold head (no training):
            e_f>=t_e and e_b>=t_e            -> EQUIV
            e_f>=t_e                          -> FORWARD
            e_b>=t_e                          -> BACKWARD
            max(c_f, c_b)>=t_c                -> DISJOINT
            else                              -> OVERLAP
        confidence = the deciding probability.
      * calibrated head: multinomial logistic regression on the 6-dim feature
        vector, fit via :meth:`fit_calibration` on labeled pairs; confidence
        = predicted class probability (LR probabilities are near-calibrated;
        the experiment scripts additionally measure ECE).
    """

    LABELS = ["contradiction", "entailment", "neutral"]

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
                 device: str | None = None, batch_size: int = 64,
                 t_entail: float = 0.7, t_contra: float = 0.7):
        from sentence_transformers import CrossEncoder  # lazy import

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        self.t_entail = t_entail
        self.t_contra = t_contra
        self.head = None  # sklearn LogisticRegression once fit
        self.stats = EntailmentStats()
        self._score_cache: dict[tuple[str, str], np.ndarray] = {}

    # -- scoring ----------------------------------------------------------

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Softmax NLI probabilities for (premise, hypothesis) pairs."""
        import scipy.special

        todo = [pr for pr in pairs if pr not in self._score_cache]
        if todo:
            logits = self.model.predict(todo, batch_size=self.batch_size,
                                        convert_to_numpy=True,
                                        show_progress_bar=False)
            probs = scipy.special.softmax(np.atleast_2d(logits), axis=1)
            for pr, row in zip(todo, probs):
                self._score_cache[pr] = row
            self.stats.pair_scores += len(todo)
        return np.stack([self._score_cache[pr] for pr in pairs])

    def features(self, p_texts: list[str], q_texts: list[str]) -> np.ndarray:
        """6-dim feature vectors for pairs (p_i, q_i)."""
        fwd = [(a, b) for a, b in zip(p_texts, q_texts)]
        bwd = [(b, a) for a, b in zip(p_texts, q_texts)]
        f = self._score_pairs(fwd)
        b = self._score_pairs(bwd)
        return np.hstack([f, b])

    # -- relation heads ---------------------------------------------------

    def _threshold_head(self, feats: np.ndarray) -> list[Judgment]:
        out = []
        for c_f, e_f, _n_f, c_b, e_b, _n_b in feats:
            if e_f >= self.t_entail and e_b >= self.t_entail:
                out.append(Judgment(Relation.EQUIV, float(min(e_f, e_b))))
            elif e_f >= self.t_entail:
                out.append(Judgment(Relation.FORWARD, float(e_f)))
            elif e_b >= self.t_entail:
                out.append(Judgment(Relation.BACKWARD, float(e_b)))
            elif max(c_f, c_b) >= self.t_contra:
                out.append(Judgment(Relation.DISJOINT, float(max(c_f, c_b))))
            else:
                out.append(Judgment(Relation.OVERLAP,
                                    float(1 - max(e_f, e_b, c_f, c_b))))
        return out

    def fit_calibration(
        self,
        pairs: list[tuple[Predicate, Predicate, Relation]],
    ) -> float:
        """Fit the calibrated relation head on labeled pairs.

        Returns training accuracy.  Pairs come from a *development* predicate
        universe (e.g. a different dataset or held-out predicate split).
        """
        from sklearn.linear_model import LogisticRegression

        X = self.features([p.text for p, _, _ in pairs],
                          [q.text for _, q, _ in pairs])
        y = np.array([RELATIONS.index(r) for _, _, r in pairs])
        # lbfgs solver is multinomial by default in modern scikit-learn
        head = LogisticRegression(max_iter=2000, C=1.0)
        head.fit(X, y)
        self.head = head
        return float(head.score(X, y))

    def judge_batch(self, p: Predicate,
                    qs: list[Predicate]) -> list[Judgment]:
        if not qs:
            return []
        self.stats.judgments += len(qs)
        feats = self.features([p.text] * len(qs), [q.text for q in qs])
        if self.head is None:
            return self._threshold_head(feats)
        probs = self.head.predict_proba(feats)
        # Map head classes back to RELATIONS indices.
        out = []
        for row in probs:
            full = np.zeros(len(RELATIONS))
            for cls_idx, cls in enumerate(self.head.classes_):
                full[int(cls)] = row[cls_idx]
            k = int(np.argmax(full))
            out.append(Judgment(RELATIONS[k], float(full[k])))
        return out

    def judge(self, p: Predicate, q: Predicate) -> Judgment:
        return self.judge_batch(p, [q])[0]
