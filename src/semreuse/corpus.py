"""Labeled text corpora with a two-level topic taxonomy.

A corpus is a list of documents, each carrying a *leaf* topic label. Leaves are
grouped into *groups* (e.g. ``baseball``/``hockey`` -> ``sports``), which gives
predicates genuine containment structure: "is about baseball" implies
"is about sports" both linguistically and in the ground truth.

Loaders cache raw data under ``data/`` and never re-download.
"""

from __future__ import annotations

import csv
import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Taxonomies
# ---------------------------------------------------------------------------

# 20 Newsgroups: newsgroup -> (leaf topic name, group topic name)
NEWSGROUP_TAXONOMY: dict[str, tuple[str, str]] = {
    "rec.sport.baseball": ("baseball", "sports"),
    "rec.sport.hockey": ("hockey", "sports"),
    "rec.autos": ("cars", "motor vehicles"),
    "rec.motorcycles": ("motorcycles", "motor vehicles"),
    "comp.graphics": ("computer graphics", "computers"),
    "comp.os.ms-windows.misc": ("the Microsoft Windows operating system", "computers"),
    "comp.sys.ibm.pc.hardware": ("PC hardware", "computers"),
    "comp.sys.mac.hardware": ("Macintosh hardware", "computers"),
    "comp.windows.x": ("the X Window System", "computers"),
    "sci.crypt": ("cryptography", "science"),
    "sci.electronics": ("electronics", "science"),
    "sci.med": ("medicine", "science"),
    "sci.space": ("space exploration", "science"),
    "talk.politics.guns": ("gun control", "politics"),
    "talk.politics.mideast": ("Middle East affairs", "politics"),
    "talk.politics.misc": ("political debate", "politics"),
    "soc.religion.christian": ("Christianity", "religion"),
    "talk.religion.misc": ("religious debate", "religion"),
    "alt.atheism": ("atheism", "religion"),
    "misc.forsale": ("items for sale", "classified ads"),
}

# AG News: class id -> (leaf name, group name).  AG News is flat, so we make
# each class its own group and rely on synthesized union predicates for
# containment structure.
AGNEWS_TAXONOMY: dict[int, tuple[str, str]] = {
    1: ("world news", "world news"),
    2: ("sports", "sports"),
    3: ("business and finance", "business and finance"),
    4: ("science and technology", "science and technology"),
}

AGNEWS_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
    "master/data/ag_news_csv/train.csv"
)


@dataclass
class Corpus:
    """A labeled corpus with a two-level taxonomy."""

    name: str
    docs: list[str]
    leaf_labels: list[str]  # leaf topic name per doc
    taxonomy: dict[str, str] = field(default_factory=dict)  # leaf -> group

    def __post_init__(self) -> None:
        assert len(self.docs) == len(self.leaf_labels)
        self._leaf_array = np.asarray(self.leaf_labels, dtype=object)

    def __len__(self) -> int:
        return len(self.docs)

    @property
    def n(self) -> int:
        return len(self.docs)

    @property
    def leaves(self) -> list[str]:
        return sorted(set(self.taxonomy))

    @property
    def groups(self) -> list[str]:
        return sorted(set(self.taxonomy.values()))

    def leaves_of(self, group: str) -> list[str]:
        return sorted(l for l, g in self.taxonomy.items() if g == group)

    def extension(self, label_set: frozenset[str]) -> np.ndarray:
        """Boolean mask of documents whose leaf label is in ``label_set``."""
        return np.isin(self._leaf_array, list(label_set))

    def subsample(self, n: int, seed: int = 0) -> "Corpus":
        """Stratified subsample of ~n documents (round-robin over leaves)."""
        rng = np.random.default_rng(seed)
        idx: list[int] = []
        per_leaf = max(1, n // max(1, len(self.leaves)))
        for leaf in self.leaves:
            leaf_idx = np.flatnonzero(self._leaf_array == leaf)
            take = min(per_leaf, len(leaf_idx))
            idx.extend(rng.choice(leaf_idx, size=take, replace=False).tolist())
        rng.shuffle(idx)
        idx = idx[:n]
        return Corpus(
            name=f"{self.name}-sub{len(idx)}",
            docs=[self.docs[i] for i in idx],
            leaf_labels=[self.leaf_labels[i] for i in idx],
            taxonomy=dict(self.taxonomy),
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_20newsgroups(data_dir: str, max_doc_chars: int = 2000) -> Corpus:
    """20 Newsgroups via scikit-learn (cached in data_dir/sklearn_20ng)."""
    from sklearn.datasets import fetch_20newsgroups

    raw = fetch_20newsgroups(
        data_home=os.path.join(data_dir, "sklearn_20ng"),
        subset="train",
        remove=("headers", "footers", "quotes"),
    )
    docs, labels = [], []
    for text, target in zip(raw.data, raw.target):
        group_name = raw.target_names[target]
        if group_name not in NEWSGROUP_TAXONOMY:
            continue
        text = " ".join(text.split())[:max_doc_chars]
        if len(text) < 50:  # drop empty/near-empty posts
            continue
        leaf, _ = NEWSGROUP_TAXONOMY[group_name]
        docs.append(text)
        labels.append(leaf)
    taxonomy = {leaf: grp for leaf, grp in NEWSGROUP_TAXONOMY.values()}
    return Corpus(name="20ng", docs=docs, leaf_labels=labels, taxonomy=taxonomy)


def load_agnews(data_dir: str, max_rows: int | None = None,
                max_doc_chars: int = 2000) -> Corpus:
    """AG News train split from a public CSV mirror (~29 MB, cached)."""
    path = os.path.join(data_dir, "ag_news_train.csv")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(AGNEWS_URL, tmp)
        os.replace(tmp, path)
    docs, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            cls, title, body = int(row[0]), row[1], row[2]
            leaf, _ = AGNEWS_TAXONOMY[cls]
            text = " ".join((title + ". " + body).split())[:max_doc_chars]
            docs.append(text)
            labels.append(leaf)
            if max_rows is not None and len(docs) >= max_rows:
                break
    taxonomy = {leaf: grp for leaf, grp in AGNEWS_TAXONOMY.values()}
    return Corpus(name="agnews", docs=docs, leaf_labels=labels, taxonomy=taxonomy)


def make_synthetic_corpus(n: int = 200, seed: int = 0,
                          n_groups: int | None = None,
                          leaves_per_group: int = 2) -> Corpus:
    """Tiny synthetic corpus for unit tests (no downloads).

    Default taxonomy: 3 groups x 2 leaves.  Pass ``n_groups`` for a larger
    generic taxonomy (groupK/topicK_J names).
    """
    if n_groups is None:
        taxonomy = {
            "baseball": "sports", "hockey": "sports",
            "cars": "vehicles", "motorcycles": "vehicles",
            "medicine": "science", "space": "science",
        }
    else:
        taxonomy = {f"topic{g}_{l}": f"group{g}"
                    for g in range(n_groups) for l in range(leaves_per_group)}
    rng = np.random.default_rng(seed)
    leaves = sorted(taxonomy)
    docs, labels = [], []
    for i in range(n):
        leaf = leaves[rng.integers(len(leaves))]
        docs.append(f"synthetic document {i} about {leaf}")
        labels.append(leaf)
    return Corpus(name="synthetic", docs=docs, leaf_labels=labels,
                  taxonomy=taxonomy)


def load_corpus(name: str, data_dir: str, size: int | None = None,
                seed: int = 0) -> Corpus:
    """Load a corpus by name, optionally subsampled to ``size`` documents."""
    if name == "rcv1":
        return load_rcv1(data_dir, size=size, seed=seed)
    if name == "20ng":
        c = load_20newsgroups(data_dir)
    elif name == "agnews":
        c = load_agnews(data_dir, max_rows=None if size is None else size * 20)
    elif name == "synthetic":
        c = make_synthetic_corpus(n=size or 200, seed=seed)
    else:
        raise ValueError(f"unknown corpus {name!r}")
    if size is not None and len(c) > size:
        c = c.subsample(size, seed=seed)
    return c


# ---------------------------------------------------------------------------
# Multi-label, hierarchical corpora
# ---------------------------------------------------------------------------

class MultiLabelCorpus(Corpus):
    """A corpus whose documents carry *sets* of labels from a deep hierarchy.

    Single-label corpora make the reuse problem easier than it is: two
    predicates over disjoint label sets are then automatically disjoint in
    extension, so disjoint elimination fires on essentially every pair (we
    measure 99.2% of queries on 20 Newsgroups).  When a document may carry
    several topics, disjointness has to be earned and genuine partial
    ``OVERLAP`` -- the relation no rewrite rule can exploit -- becomes the
    common case.  This class supports that setting, and with RCV1 it also
    supplies two orders of magnitude more rows.

    ``label_matrix`` is a scipy CSC boolean matrix (n_docs x n_labels);
    ``extension`` ORs the requested columns.  ``parents`` maps a label to its
    parent in the hierarchy (root labels map to ``None``), which lets the
    predicate universe build genuine multi-level containment chains.
    """

    def __init__(self, name: str, label_matrix, label_names: list[str],
                 parents: dict[str, str | None], docs: list[str] | None = None):
        import scipy.sparse as sp

        self.label_matrix = sp.csc_matrix(label_matrix, dtype=bool)
        self.label_names = list(label_names)
        self.parents = dict(parents)
        self._col = {name: i for i, name in enumerate(self.label_names)}
        n = self.label_matrix.shape[0]
        primary = self._primary_labels()
        # Two-level view (leaf -> group) so that all existing machinery keeps
        # working: "group" is the label's root ancestor.
        taxonomy = {lab: self.root_of(lab) for lab in self.label_names}
        super().__init__(name=name, docs=docs if docs is not None else [""] * n,
                         leaf_labels=primary, taxonomy=taxonomy)

    # -- hierarchy ---------------------------------------------------------

    def root_of(self, label: str) -> str:
        seen = set()
        cur = label
        while self.parents.get(cur) and cur not in seen:
            seen.add(cur)
            cur = self.parents[cur]
        return cur

    def children_of(self, label: str) -> list[str]:
        return sorted(l for l, p in self.parents.items() if p == label)

    def descendants_of(self, label: str) -> list[str]:
        out, stack = [], [label]
        while stack:
            cur = stack.pop()
            kids = self.children_of(cur)
            out.extend(kids)
            stack.extend(kids)
        return sorted(set(out))

    def _primary_labels(self) -> list[str]:
        """One representative label per document (for stratified subsampling)."""
        m = self.label_matrix.tocsr()
        names = self.label_names
        out = []
        indptr, indices = m.indptr, m.indices
        for i in range(m.shape[0]):
            lo, hi = indptr[i], indptr[i + 1]
            out.append(names[indices[lo]] if hi > lo else "<none>")
        return out

    # -- semantics ---------------------------------------------------------

    def extension(self, label_set: frozenset[str]) -> np.ndarray:
        cols = [self._col[l] for l in label_set if l in self._col]
        out = np.zeros(self.n, dtype=bool)
        for c in cols:
            sl = self.label_matrix.getcol(c)
            out[sl.indices] = True
        return out

    def subsample(self, n: int, seed: int = 0) -> "MultiLabelCorpus":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.n, size=min(n, self.n), replace=False))
        return MultiLabelCorpus(
            name=f"{self.name}-sub{len(idx)}",
            label_matrix=self.label_matrix.tocsr()[idx].tocsc(),
            label_names=self.label_names, parents=self.parents,
            docs=[self.docs[i] for i in idx] if any(self.docs) else None)


def _rcv1_parent(code: str, codes: set[str]) -> str | None:
    """Parent of an RCV1 topic code.

    RCV1-v2's topic hierarchy is prefix-structured under four roots
    (CCAT corporate, ECAT economics, GCAT government, MCAT markets):
    ``C1511 -> C151 -> C15 -> CCAT``.  Codes that are not prefix-nested
    (GPOL, GDIP, ...) hang directly off their letter's root.
    """
    if code.endswith("CAT"):
        return None
    for k in range(len(code) - 1, 0, -1):
        cand = code[:k]
        if cand in codes and cand != code and not cand.endswith("CAT"):
            return cand
    root = code[0] + "CAT"
    return root if root in codes else None


def load_rcv1(data_dir: str, size: int | None = None, seed: int = 0,
              min_docs: int = 200) -> MultiLabelCorpus:
    """RCV1-v2: 804,414 Reuters newswire stories, 103 hierarchical topics.

    Documents carry 3.2 topic labels on average, so predicate extensions
    genuinely overlap.  We use the label matrix only: the simulated oracle's
    semantics are label membership, and no baseline in this configuration
    needs the article text (which the public distribution ships as TF-IDF
    vectors, not as strings).
    """
    from sklearn.datasets import fetch_rcv1

    d = fetch_rcv1(data_home=os.path.join(data_dir, "sklearn_rcv1"),
                   download_if_missing=True)
    names = [str(x) for x in d.target_names]
    codes = set(names)
    parents = {c: _rcv1_parent(c, codes) for c in names}
    target = d.target.tocsc().astype(bool)
    # Drop labels too rare to certify anything at these sample sizes.
    keep = [i for i, c in enumerate(names)
            if target.getcol(i).nnz >= min_docs]
    names = [names[i] for i in keep]
    codes = set(names)
    parents = {c: _rcv1_parent(c, codes) for c in names}
    corpus = MultiLabelCorpus(name="rcv1", label_matrix=target[:, keep],
                              label_names=names, parents=parents)
    if size is not None and corpus.n > size:
        corpus = corpus.subsample(size, seed=seed)
    return corpus
