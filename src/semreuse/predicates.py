"""Natural-language predicates with ground-truth extensions, and workloads.

A :class:`Predicate` is an NL filter ("The document is about sports.") whose
ground-truth semantics on a labeled corpus is a set of accepted *leaf* labels.
This makes the simulated oracle exact and lets us derive the true logical
relation between any two predicates (subsumption / equivalence / disjointness
/ overlap) from set algebra -- the reference against which the NLI entailment
reasoner is calibrated and evaluated.

Workload generation produces sequences of predicates with controlled overlap:
chains of related predicates (group -> leaves -> unions -> paraphrases) mixed
with unrelated ones.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field

import numpy as np

from semreuse.corpus import Corpus


class Relation(enum.Enum):
    """Logical relation of predicate p to predicate q (extensions P, Q)."""

    EQUIV = "equiv"        # P == Q
    FORWARD = "forward"    # P subset Q   (p implies q; q is a superset view)
    BACKWARD = "backward"  # Q subset P   (q implies p; q supplies positives)
    DISJOINT = "disjoint"  # P inter Q == empty
    OVERLAP = "overlap"    # none of the above


@dataclass(frozen=True)
class Predicate:
    """An NL predicate with ground-truth label-set semantics."""

    text: str                      # NL sentence form, e.g. "The document is about sports."
    label_set: frozenset[str]      # accepted leaf labels (ground truth)
    name: str = ""                 # short human-readable id

    @property
    def pid(self) -> str:
        return hashlib.sha1(self.text.encode()).hexdigest()[:12]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Predicate({self.name or self.text!r})"


def normalize_text(text: str) -> str:
    """Normalization used for exact-match caching."""
    return " ".join(text.lower().replace(".", " ").replace(",", " ").split())


def true_relation(a: frozenset[str], b: frozenset[str]) -> Relation:
    """Ground-truth relation of predicate with labels ``a`` to one with ``b``."""
    if a == b:
        return Relation.EQUIV
    if a <= b:
        return Relation.FORWARD
    if b <= a:
        return Relation.BACKWARD
    if not (a & b):
        return Relation.DISJOINT
    return Relation.OVERLAP


# ---------------------------------------------------------------------------
# Predicate universe construction
# ---------------------------------------------------------------------------

# A few surface-form templates so that exact-match caching is not trivially
# equivalent to semantic reuse.  {t} is the topic phrase.
TEMPLATES = [
    "The document is about {t}.",
    "This text discusses {t}.",
    "The passage is related to {t}.",
    "This article covers the topic of {t}.",
]

UNION_TEMPLATES = [
    "The document is about {a} or about {b}.",
    "This text discusses either {a} or {b}.",
]


@dataclass
class PredicateUniverse:
    """All predicates derivable from a corpus taxonomy, with ground truth."""

    corpus: Corpus
    predicates: list[Predicate] = field(default_factory=list)

    @classmethod
    def build(cls, corpus: Corpus, include_unions: bool = True) -> "PredicateUniverse":
        preds: list[Predicate] = []
        # Leaf and group predicates, all templates.
        for leaf in corpus.leaves:
            for ti, tpl in enumerate(TEMPLATES):
                preds.append(Predicate(tpl.format(t=leaf),
                                       frozenset([leaf]),
                                       name=f"leaf:{leaf}#t{ti}"))
        for group in corpus.groups:
            labels = frozenset(corpus.leaves_of(group))
            if len(labels) < 2:
                continue  # singleton group == its leaf; skip duplicates
            for ti, tpl in enumerate(TEMPLATES):
                preds.append(Predicate(tpl.format(t=group), labels,
                                       name=f"group:{group}#t{ti}"))
        # Union-of-two-leaves predicates from different groups (overlap-free
        # containment: leaf => union).
        if include_unions:
            leaves = corpus.leaves
            for i, a in enumerate(leaves):
                for b in leaves[i + 1:]:
                    if corpus.taxonomy[a] == corpus.taxonomy[b]:
                        continue
                    for ti, tpl in enumerate(UNION_TEMPLATES):
                        preds.append(Predicate(
                            tpl.format(a=a, b=b), frozenset([a, b]),
                            name=f"union:{a}|{b}#u{ti}"))
        return cls(corpus=corpus, predicates=preds)

    def by_name_prefix(self, prefix: str) -> list[Predicate]:
        return [p for p in self.predicates if p.name.startswith(prefix)]

    def resolver(self) -> dict[str, frozenset[str]]:
        """text -> label_set map (used by the ground-truth entailment oracle)."""
        return {p.text: p.label_set for p in self.predicates}


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

@dataclass
class Workload:
    """A sequence of predicates issued as queries, in order."""

    queries: list[Predicate]
    universe: PredicateUniverse
    params: dict = field(default_factory=dict)


def generate_workload(
    universe: PredicateUniverse,
    n_queries: int = 40,
    overlap_rate: float = 0.7,
    seed: int = 0,
) -> Workload:
    """Generate a workload with controlled predicate overlap.

    With probability ``overlap_rate`` the next query is *related* to an
    earlier query's topic chain (a sub-/super-predicate, a paraphrase, a
    union containing it, or a sibling leaf -- giving forward/backward/
    equivalence/disjointness opportunities); otherwise it starts a fresh
    chain on an unused group.

    Determinism: fixed seed => identical workload.
    """
    rng = np.random.default_rng(seed)
    corpus = universe.corpus
    groups = [g for g in corpus.groups if len(corpus.leaves_of(g)) >= 1]
    rng.shuffle(groups)
    queries: list[Predicate] = []
    active_groups: list[str] = []
    group_iter = iter(groups)

    def pick(preds: list[Predicate]) -> Predicate:
        return preds[rng.integers(len(preds))]

    def related_candidates(group: str) -> list[Predicate]:
        """A related predicate: group paraphrase, a leaf under the group, or
        (less often) a union containing one of its leaves.  Categories are
        weighted so the plentiful union predicates don't dominate chains."""
        group_preds = universe.by_name_prefix(f"group:{group}#")
        leaf_preds: list[Predicate] = []
        union_preds: list[Predicate] = []
        for leaf in corpus.leaves_of(group):
            leaf_preds += universe.by_name_prefix(f"leaf:{leaf}#")
            union_preds += [p for p in universe.predicates
                            if p.name.startswith("union:")
                            and leaf in p.label_set]
        r = rng.random()
        if r < 0.30 and group_preds:
            return group_preds
        if r < 0.80 and leaf_preds:
            return leaf_preds
        return union_preds or leaf_preds or group_preds

    while len(queries) < n_queries:
        start_fresh = (not active_groups) or rng.random() > overlap_rate
        if start_fresh:
            group = next(group_iter, None)
            if group is None:  # exhausted; recycle
                group = groups[rng.integers(len(groups))]
            active_groups.append(group)
            # A fresh chain usually starts with the broad (group) predicate.
            cands = (universe.by_name_prefix(f"group:{group}#")
                     or related_candidates(group))
        else:
            group = active_groups[rng.integers(len(active_groups))]
            cands = related_candidates(group)
        if not cands:
            continue
        q = pick(cands)
        queries.append(q)

    return Workload(queries=queries, universe=universe,
                    params=dict(n_queries=n_queries, overlap_rate=overlap_rate,
                                seed=seed))


def labeled_pairs_for_calibration(
    universe: PredicateUniverse,
    n_pairs: int = 400,
    seed: int = 0,
) -> list[tuple[Predicate, Predicate, Relation]]:
    """Sample predicate pairs with ground-truth relations, roughly balanced
    across relation classes, for fitting/evaluating the entailment reasoner."""
    rng = np.random.default_rng(seed)
    preds = universe.predicates
    buckets: dict[Relation, list[tuple[Predicate, Predicate, Relation]]] = {
        r: [] for r in Relation
    }
    # Enumerate a bounded random sample of ordered pairs.
    max_tries = n_pairs * 60
    for _ in range(max_tries):
        i, j = rng.integers(len(preds)), rng.integers(len(preds))
        if i == j:
            continue
        p, q = preds[i], preds[j]
        if p.text == q.text:
            continue
        rel = true_relation(p.label_set, q.label_set)
        buckets[rel].append((p, q, rel))
        if all(len(b) >= n_pairs // len(Relation) for b in buckets.values()):
            break
    per = n_pairs // len(Relation)
    out: list[tuple[Predicate, Predicate, Relation]] = []
    for r in Relation:
        b = buckets[r]
        rng.shuffle(b)
        out.extend(b[:per])
    rng.shuffle(out)
    return out
