"""Tier two: a small-LLM arbiter for predicate pairs the NLI head is unsure of.

The cross-encoder tier is fast and free but was trained on sentence-pair
inference, not on predicate containment; its measured failure modes are exactly
the relations that carry the savings -- DISJOINT and OVERLAP.  Tier two sends
only the pairs tier one is unsure about to an instruction-tuned model, which
sees the two predicates as what they are: two claims about a document.

The economics are what make a second tier defensible.  Tier one is O(#views)
per query; tier two is O(#views) *restricted to low-confidence pairs*, and each
call is one short prompt with no document in it.  A single semantic filter over
a corpus of N rows costs N oracle calls; arbitrating every pair in a
hundred-view store costs a hundred short calls, once, memoized for the rest of
the workload.  The tier's whole cost is therefore independent of N and, in our
runs, three to four orders of magnitude below the calls it saves.

Confidence.  Verbalized LLM confidences are not calibrated, so we do not ask
for one.  Instead the arbiter is calibrated the same way tier one is: its
predicted-class accuracy is measured once on the corpus-independent synthetic
dev taxonomy, and that per-class accuracy *is* the confidence the rewriter
thresholds on.  As everywhere else in this system, a miscalibrated confidence
can only cost oracle calls, never correctness.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from semreuse.entailment import RELATIONS, EntailmentStats, Judgment
from semreuse.predicates import Predicate, Relation

ARBITER_PROMPT_VERSION = "arbiter-v2-binary"

# A five-way multiple choice is the obvious prompt and the wrong one: small
# instruction-tuned models collapse onto a single option (Llama-3.1-8B answers
# "B is narrower" for every pair we tried, 2/8 correct).  Decomposing the
# relation into the three binary questions that define it recovers the
# accuracy, at two or three short calls per pair -- still nothing next to one
# oracle call per row.
IMPLIES_PROMPT = """A database applies two filters to documents.

Filter A: {a}
Filter B: {b}

Question: is every document that filter A selects also selected by filter B?
Think about whether A's meaning is more specific than B's.
Answer with exactly one word, yes or no.
Answer:"""

DISJOINT_PROMPT = """A database applies two filters to documents.

Filter A: {a}
Filter B: {b}

Question: is it impossible for a single document to be selected by both
filters at once, because their meanings exclude each other?
Answer with exactly one word, yes or no.
Answer:"""


@dataclass
class ArbiterStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    unparsed: int = 0


class LLMArbiter:
    """Relation classification for one predicate pair by a local LLM."""

    def __init__(self, model: str = "qwen2.5:7b",
                 host: str = "http://localhost:11434",
                 concurrency: int = 8, seed: int = 0, timeout: float = 300.0,
                 default_confidence: float = 0.75):
        self.model = model
        self.host = host.rstrip("/")
        self.concurrency = concurrency
        self.seed = seed
        self.timeout = timeout
        self.stats = ArbiterStats()
        self._cache: dict[tuple[str, str], Relation] = {}
        self._lock = threading.Lock()
        # class -> confidence, filled by ``calibrate``; until then a single
        # conservative prior for every class.
        self.confidence: dict[Relation, float] = {
            r: default_confidence for r in RELATIONS}

    # -- inference ---------------------------------------------------------

    def _ask(self, prompt: str) -> bool:
        body = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "keep_alive": "60m",
            "options": {"temperature": 0.0, "top_k": 1, "top_p": 1.0,
                        "seed": self.seed, "num_predict": 3},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        last: Exception | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read())
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
                time.sleep(1.0 * (attempt + 1))
        else:  # pragma: no cover
            raise RuntimeError(f"arbiter request failed: {last}")
        with self._lock:
            self.stats.calls += 1
            self.stats.prompt_tokens += int(d.get("prompt_eval_count") or 0)
            self.stats.output_tokens += int(d.get("eval_count") or 0)
        raw = (d.get("response") or "").strip().lower().lstrip("\"\'` \n")
        if raw.startswith("yes"):
            return True
        if raw.startswith("no"):
            return False
        with self._lock:
            self.stats.unparsed += 1
        return False        # unparsed => claim nothing, so no rewrite fires

    def _one(self, a: str, b: str) -> Relation:
        """Decide the relation from binary sub-questions.

        Two implication questions settle equivalence and both directions; a
        third question is asked only when neither implication holds, which is
        exactly when disjointness is still possible.
        """
        fwd = self._ask(IMPLIES_PROMPT.format(a=a, b=b))
        bwd = self._ask(IMPLIES_PROMPT.format(a=b, b=a))
        if fwd and bwd:
            return Relation.EQUIV
        if fwd:
            return Relation.FORWARD
        if bwd:
            return Relation.BACKWARD
        return (Relation.DISJOINT
                if self._ask(DISJOINT_PROMPT.format(a=a, b=b))
                else Relation.OVERLAP)

    def relation(self, a: str, b: str) -> Relation:
        key = (a, b)
        if key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]
        t0 = time.perf_counter()
        rel = self._one(a, b)
        with self._lock:
            self.stats.wall_s += time.perf_counter() - t0
        self._cache[key] = rel
        return rel

    def relations(self, pairs: list[tuple[str, str]]) -> list[Relation]:
        """Batch interface; concurrent, cache-aware, order-preserving."""
        todo = [(i, p) for i, p in enumerate(pairs) if p not in self._cache]
        if todo:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(self.concurrency) as ex:
                res = list(ex.map(lambda ip: self._one(*ip[1]), todo))
            with self._lock:
                self.stats.wall_s += time.perf_counter() - t0
            for (i, p), rel in zip(todo, res):
                self._cache[p] = rel
        self.stats.cache_hits += len(pairs) - len(todo)
        return [self._cache[p] for p in pairs]

    def judge(self, p: Predicate, q: Predicate) -> Judgment:
        rel = self.relation(p.text, q.text)
        return Judgment(rel, self.confidence[rel])

    def judge_batch(self, p: Predicate, qs: list[Predicate]) -> list[Judgment]:
        rels = self.relations([(p.text, q.text) for q in qs])
        return [Judgment(r, self.confidence[r]) for r in rels]

    # -- calibration -------------------------------------------------------

    def calibrate(self, pairs, floor: float = 0.3) -> dict:
        """Set each class's confidence to its measured precision on ``pairs``.

        ``pairs`` is a list of (p, q, true_relation) over the *synthetic* dev
        taxonomy, so nothing about an evaluation corpus leaks in.
        """
        preds = self.relations([(p.text, q.text) for p, q, _ in pairs])
        hit: dict[Relation, int] = {r: 0 for r in RELATIONS}
        tot: dict[Relation, int] = {r: 0 for r in RELATIONS}
        for (_, _, gold), pred in zip(pairs, preds):
            tot[pred] += 1
            hit[pred] += int(pred is gold)
        for r in RELATIONS:
            if tot[r]:
                self.confidence[r] = max(floor, hit[r] / tot[r])
        acc = sum(hit.values()) / max(1, len(pairs))
        return {"accuracy": acc,
                "confidence": {r.value: self.confidence[r] for r in RELATIONS},
                "support": {r.value: tot[r] for r in RELATIONS}}


# ---------------------------------------------------------------------------
# The two-tier reasoner
# ---------------------------------------------------------------------------

@dataclass
class TwoTierStats:
    pair_scores: int = 0        # tier-1 cross-encoder passes (mirrors NLI)
    judgments: int = 0
    escalated: int = 0          # pairs sent to tier two
    arbiter: ArbiterStats = field(default_factory=ArbiterStats)


class TwoTierEntailment:
    """Tier one (calibrated NLI) with tier-two arbitration of unsure pairs.

    A pair is escalated when tier one's confidence is below ``escalate_below``
    or when tier one predicts a relation in ``always_escalate`` -- by default
    OVERLAP, the class tier one almost never predicts correctly and the class
    on which no rewrite fires, so a wrong OVERLAP is a *missed saving* that
    costs nothing to re-examine.
    """

    def __init__(self, tier1, arbiter: LLMArbiter,
                 escalate_below: float = 0.7,
                 always_escalate: tuple[Relation, ...] = (Relation.OVERLAP,),
                 max_escalations: int | None = None):
        self.tier1 = tier1
        self.arbiter = arbiter
        self.escalate_below = escalate_below
        self.always_escalate = tuple(always_escalate)
        self.max_escalations = max_escalations
        self.stats = TwoTierStats(arbiter=arbiter.stats)

    def _needs_tier2(self, j: Judgment) -> bool:
        return (j.confidence < self.escalate_below
                or j.relation in self.always_escalate)

    def judge_batch(self, p: Predicate, qs: list[Predicate]) -> list[Judgment]:
        first = self.tier1.judge_batch(p, qs)
        self.stats.judgments += len(qs)
        self.stats.pair_scores = getattr(
            getattr(self.tier1, "stats", None), "pair_scores", 0)
        idx = [i for i, j in enumerate(first) if self._needs_tier2(j)]
        if self.max_escalations is not None:
            # Spend tier two where tier one is least sure.
            idx.sort(key=lambda i: first[i].confidence)
            idx = idx[: self.max_escalations]
        if not idx:
            return first
        self.stats.escalated += len(idx)
        rels = self.arbiter.relations([(p.text, qs[i].text) for i in idx])
        out = list(first)
        for i, rel in zip(idx, rels):
            out[i] = Judgment(rel, self.arbiter.confidence[rel])
        return out

    def judge(self, p: Predicate, q: Predicate) -> Judgment:
        return self.judge_batch(p, [q])[0]


def make_entailment_stats_shim(tt: TwoTierEntailment) -> EntailmentStats:
    """Engine-facing view of tier-one pair scorings (cost bookkeeping)."""
    st = EntailmentStats()
    st.pair_scores = tt.stats.pair_scores
    st.judgments = tt.stats.judgments
    return st
