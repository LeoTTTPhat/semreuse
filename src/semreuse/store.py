"""The predicate store: cached natural-language views with lineage.

Every executed semantic filter is logged as a :class:`CachedView`:
  * the predicate text (and normalized form for exact-match lookup),
  * the *reported* extension over the corpus (a boolean bitmap of length N),
  * a *verified* bitmap marking rows whose value came from an actual oracle
    call (as opposed to being assumed by a rewrite),
  * lineage metadata: oracle/model version, query index, the audited recall
    lower bound the view was published with, and which cached views it was
    derived from.

Rewrites for later queries consume the *reported* extension; audits are
always performed against the oracle itself, so per-query guarantees do not
compound across chained reuse (see DESIGN.md, Sec. Guarantees).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from semreuse.predicates import normalize_text


@dataclass
class CachedView:
    pid: str
    text: str
    reported: np.ndarray          # bool[N] reported extension
    verified: np.ndarray          # bool[N] rows backed by oracle calls
    oracle_version: str = "sim-v1"
    query_index: int = -1
    recall_bound: float | None = None   # audited lower bound, if any
    precision_bound: float | None = None  # audited precision lower bound
    derived_from: tuple[str, ...] = ()  # pids of views used in the rewrite
    created_at: float = field(default_factory=time.time)

    @property
    def n_positive(self) -> int:
        return int(self.reported.sum())

    @property
    def frac_verified(self) -> float:
        return float(self.verified.mean()) if len(self.verified) else 0.0


class PredicateStore:
    """In-memory predicate store over a fixed corpus of N rows."""

    def __init__(self, n_rows: int):
        self.n_rows = n_rows
        self.views: list[CachedView] = []
        self._by_norm_text: dict[str, CachedView] = {}

    def __len__(self) -> int:
        return len(self.views)

    def add(self, view: CachedView) -> None:
        assert view.reported.shape == (self.n_rows,)
        assert view.verified.shape == (self.n_rows,)
        self.views.append(view)
        # Later views win exact-match lookups (freshest lineage).
        self._by_norm_text[normalize_text(view.text)] = view

    def find_exact(self, text: str) -> CachedView | None:
        return self._by_norm_text.get(normalize_text(text))

    def candidates_for_matching(self) -> list[CachedView]:
        """Views eligible as reuse sources (all, in insertion order)."""
        return list(self.views)

    def memory_bytes(self) -> int:
        """Approximate store footprint (bitmaps dominate)."""
        return sum(v.reported.nbytes + v.verified.nbytes for v in self.views)
