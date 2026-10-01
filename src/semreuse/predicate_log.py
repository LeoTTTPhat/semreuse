"""Hand-authored analyst predicate logs, and extensional relation oracles.

The generated workloads of :mod:`semreuse.predicates` derive every predicate
from the corpus label taxonomy, which makes the logical structure between
predicates an artifact of the generator: extensions are unions of label
classes, so containment is label-set containment and genuine partial overlap
is rare.  That is the single most load-bearing assumption in an evaluation of
cross-query reuse, and a reviewer is right to distrust it.

This module supplies the alternative.  ``ANALYST_LOG_20NG`` is a session-
ordered log of natural-language filters an analyst plausibly issues against a
Usenet archive -- support requests, marketplace posts, tone, rhetorical form,
topical drilldowns -- written *without reference to the newsgroup labels*.
Their semantics are therefore whatever a real LLM oracle says (there is no
label set to appeal to), and the true logical relation between two of them is
measured from the oracle's own answers by
:class:`ExtensionalEntailment`.

One consequence matters for the whole paper: with a real oracle, containment
between two natural predicates is essentially never *exact*.  "asks for help
with Macintosh hardware" implies "asks for help with computer hardware" as a
matter of meaning, yet the oracle's two extensions disagree on a handful of
rows.  We therefore define relations up to a slack ``eps`` -- and this is
precisely the regime the audit layer exists for: an approximate containment is
a wrong judgment on an eps-fraction of rows, which silent reuse would lose and
the audit repairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semreuse.entailment import Judgment
from semreuse.predicates import Predicate, Relation

# ---------------------------------------------------------------------------
# The analyst log
# ---------------------------------------------------------------------------
# Grouped into sessions, in issue order.  Each entry is (short name, text).
# Related predicates within a session create genuine containment, overlap, and
# paraphrase structure; predicates recur across sessions the way an analyst
# re-runs a filter days later.

ANALYST_LOG_20NG: list[tuple[str, str]] = [
    # Session A -- hardware support triage
    ("A1", "The post asks for technical help with computer hardware."),
    ("A2", "The post asks for technical help with a personal computer's hardware."),
    ("A3", "The post is a request for technical assistance concerning computer hardware."),
    ("A4", "The post asks for help with Macintosh hardware."),
    ("A5", "The post describes a hardware problem the author is experiencing."),
    # Session B -- the marketplace
    ("B1", "The post offers an item for sale."),
    ("B2", "The post offers a computer or a computer part for sale."),
    ("B3", "The post mentions a specific price in dollars."),
    ("B4", "The post is an advertisement."),
    ("B5", "The post is looking to buy something rather than to sell something."),
    # Session C -- sports desk
    ("C1", "The post discusses a professional sports team."),
    ("C2", "The post discusses a professional baseball team."),
    ("C3", "The post mentions a player's statistics."),
    ("C4", "The post is about a sporting event that has already taken place."),
    ("C5", "The post is about sports."),
    # Session D -- argument mining
    ("D1", "The post argues about politics."),
    ("D2", "The post argues about gun control policy."),
    ("D3", "The post expresses a religious belief."),
    ("D4", "The post argues about religion."),
    ("D5", "The post discusses the Middle East."),
    ("D6", "The post criticizes a government or a government policy."),
    # Session E -- cross-cutting form and tone (no topical alignment at all)
    ("E1", "The author of the post is frustrated or angry."),
    ("E2", "The post contains a question."),
    ("E3", "The post quotes another message."),
    ("E4", "The post uses technical jargon."),
    ("E5", "The post makes a factual claim that could be checked."),
    # Session F -- science drilldown
    ("F1", "The post discusses medicine or health."),
    ("F2", "The post discusses a specific disease or medical condition."),
    ("F3", "The post discusses space or astronomy."),
    ("F4", "The post discusses cryptography or encryption."),
    ("F5", "The post discusses computer security."),
    ("F6", "The post discusses science."),
    # Session G -- software refinement
    ("G1", "The post mentions an operating system."),
    ("G2", "The post mentions Microsoft Windows."),
    ("G3", "The post mentions the X Window System."),
    ("G4", "The post discusses software rather than hardware."),
    ("G5", "The post recommends a product or a service."),
    # Session H -- return visits: repeats, paraphrases, and new drilldowns
    ("H1", "The post asks for technical help with computer hardware."),      # exact repeat of A1
    ("H2", "The post is selling something."),                               # paraphrase of B1
    ("H3", "The post is about a motor vehicle."),
    ("H4", "The post is about a car or a motorcycle."),
    ("H5", "The post gives advice about maintaining a vehicle."),
    ("H6", "The tone of the post is angry or annoyed."),                    # paraphrase of E1
    ("H7", "The post is about a computer operating system."),               # paraphrase of G1
    ("H8", "The post is a for-sale advertisement."),                        # paraphrase of B1/B4
    ("H9", "The post discusses encryption algorithms or key management."),  # ~F4
    ("H10", "The post asks a question about computer hardware."),           # A1 and E2
]


# The relations the log was *written* to contain, by short name. These are
# claims about meaning, made before any oracle was run; the experiment measures
# how far the model's actual extensions fall short of them, which is the point
# (see ``designed_relation_slack``). ``kind`` is what a careful annotator would
# call the pair.
DESIGNED_RELATIONS: list[tuple[str, str, str]] = [
    ("A2", "A1", "specialisation"),   # personal-computer hardware => computer hardware
    ("A3", "A1", "paraphrase"),
    ("A4", "A1", "specialisation"),   # Macintosh hardware => computer hardware
    ("H1", "A1", "exact repeat"),
    ("H10", "A1", "specialisation"),
    ("B2", "B1", "specialisation"),   # computer for sale => item for sale
    ("B1", "B4", "specialisation"),   # for sale => advertisement
    ("H2", "B1", "paraphrase"),
    ("H8", "B1", "paraphrase"),
    ("C2", "C1", "specialisation"),   # baseball team => professional sports team
    ("C1", "C5", "specialisation"),
    ("D2", "D1", "specialisation"),   # gun control => politics
    ("D4", "D3", "specialisation"),
    ("F2", "F1", "specialisation"),   # specific disease => medicine
    ("F1", "F6", "specialisation"),
    ("F3", "F6", "specialisation"),
    ("F4", "F5", "specialisation"),   # cryptography => computer security
    ("H9", "F4", "paraphrase"),
    ("G2", "G1", "specialisation"),   # Windows => operating system
    ("G3", "G1", "specialisation"),
    ("H7", "G1", "paraphrase"),
    ("H4", "H3", "paraphrase"),
    ("H5", "H3", "specialisation"),
    ("H6", "E1", "paraphrase"),
]


def designed_relation_slack(extension_of) -> list[dict]:
    """Measure how far each designed implication misses being exact.

    For a declared ``a => b`` the slack is the fraction of a's positives that
    fall outside b's under the oracle's own answers. Zero would mean the model
    is perfectly consistent with the meaning we intended; it never is, and how
    much it misses by is what decides whether a recall budget can absorb the
    reuse (Section 7.2).
    """
    by_name = {name: text for name, text in ANALYST_LOG_20NG}
    out = []
    for a, b, kind in DESIGNED_RELATIONS:
        ta, tb = by_name.get(a), by_name.get(b)
        if ta is None or tb is None:
            continue
        try:
            A, B = extension_of(ta), extension_of(tb)
        except KeyError:
            continue                    # predicate not materialized yet
        na = int(A.sum())
        if na == 0:
            continue
        inter = int((A & B).sum())
        nb = int(B.sum())
        out.append({"a": a, "b": b, "kind": kind, "n_a": na, "n_b": nb,
                    # a's positives outside b: the slack of the declared
                    # implication.  For a paraphrase or an exact repeat the
                    # reverse direction is a claim too, so we report it as
                    # well -- a real oracle honours the two unequally.
                    "slack": (na - inter) / na,
                    "slack_reverse": (nb - inter) / nb if nb else 0.0,
                    "jaccard": inter / max(1, int((A | B).sum()))})
    return out


def analyst_log_predicates(names: list[str] | None = None) -> list[Predicate]:
    """The analyst log as :class:`Predicate` objects (empty label sets).

    ``label_set`` is deliberately empty: these predicates have no taxonomy
    semantics, so any engine that peeks at labels would score zero.  Ground
    truth comes from the LLM oracle's own answers.
    """
    keep = set(names) if names else None
    return [Predicate(text=text, label_set=frozenset(), name=f"log:{name}")
            for name, text in ANALYST_LOG_20NG
            if keep is None or name in keep]


def distinct_texts(preds: list[Predicate]) -> list[str]:
    seen: dict[str, None] = {}
    for p in preds:
        seen.setdefault(p.text, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Extensional (oracle-measured) relations
# ---------------------------------------------------------------------------

@dataclass
class ExtensionalEntailmentStats:
    judgments: int = 0
    pair_scores: int = 0


class ExtensionalEntailment:
    """Ground-truth relation oracle measured from an oracle's own extensions.

    ``eps`` is the slack with which containment and disjointness are declared:
    ``p => q`` iff at most an ``eps`` fraction of p's positives fall outside
    q's.  With ``eps = 0`` this is exact set containment (the right definition
    for label-defined predicates); with a real LLM oracle, ``eps`` around 2--5%
    is what a human annotator would call implication, and the residual rows are
    exactly the errors the audit has to catch.
    """

    def __init__(self, extension_of, eps: float = 0.02,
                 confidence: float = 1.0):
        self._ext = extension_of          # text -> bool[N]
        self.eps = eps
        self.confidence = confidence
        self.stats = ExtensionalEntailmentStats()

    def relation(self, p_text: str, q_text: str) -> Relation:
        P = self._ext(p_text)
        Q = self._ext(q_text)
        np_, nq = int(P.sum()), int(Q.sum())
        inter = int((P & Q).sum())
        if np_ == 0 and nq == 0:
            return Relation.EQUIV
        out_p = (np_ - inter) / np_ if np_ else 0.0     # fraction of P outside Q
        out_q = (nq - inter) / nq if nq else 0.0
        fwd = out_p <= self.eps                          # P subset~ Q
        bwd = out_q <= self.eps                          # Q subset~ P
        if fwd and bwd:
            return Relation.EQUIV
        if fwd:
            return Relation.FORWARD
        if bwd:
            return Relation.BACKWARD
        if np_ and nq and inter / min(np_, nq) <= self.eps:
            return Relation.DISJOINT
        return Relation.OVERLAP

    # Interface expected by the engine ------------------------------------

    def judge(self, p: Predicate, q: Predicate) -> Judgment:
        self.stats.judgments += 1
        return Judgment(self.relation(p.text, q.text), self.confidence)

    def judge_batch(self, p: Predicate,
                    qs: list[Predicate]) -> list[Judgment]:
        return [self.judge(p, q) for q in qs]


def relation_distribution(texts: list[str], extension_of,
                          eps: float = 0.02) -> dict[str, float]:
    """Share of each relation over all ordered pairs of ``texts``."""
    ent = ExtensionalEntailment(extension_of, eps=eps)
    counts = {r.value: 0 for r in Relation}
    total = 0
    for i, a in enumerate(texts):
        for j, b in enumerate(texts):
            if i == j:
                continue
            counts[ent.relation(a, b).value] += 1
            total += 1
    return {k: v / max(1, total) for k, v in counts.items()}


def selectivity_profile(texts: list[str], extension_of) -> np.ndarray:
    return np.array([extension_of(t).mean() for t in texts])
