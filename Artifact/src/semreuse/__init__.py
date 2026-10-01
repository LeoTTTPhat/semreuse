"""SemReuse: answering semantic queries using cached natural-language views.

Core modules:
    corpus      -- labeled text corpora with a two-level topic taxonomy
    predicates  -- NL predicates with ground-truth extensions; workload generation
    oracle      -- simulated (and optional local-model) LLM predicate oracles with cost accounting
    store       -- the predicate store (cached NL views with lineage)
    entailment  -- entailment reasoners: ground-truth, noisy, and NLI cross-encoder (calibrated)
    rewriter    -- entailment-based rewrite planning (superset pruning, positive union,
                   disjoint elimination, multi-view combination)
    audit       -- stratified audit sampling with exact finite-population recall bounds
    engine      -- the SemReuse engine tying everything together
    baselines   -- cold / exact-match cache / embedding-similarity cache engines
    metrics     -- accuracy + cost accounting helpers
"""

__version__ = "0.1.0"

from semreuse.predicates import Predicate, Relation, true_relation  # noqa: F401
from semreuse.store import CachedView, PredicateStore  # noqa: F401
