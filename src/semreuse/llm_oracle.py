"""A *real* LLM oracle for semantic filters, served by a local Ollama endpoint.

The simulated oracle in :mod:`semreuse.oracle` defines predicate semantics by
corpus labels.  That buys exact ground truth but leaves the load-bearing
question open: does entailment-based reuse survive a *real* LLM whose per-row
answers are noisy, prompt-sensitive, and only loosely tied to any label
taxonomy?  ``LLMOracle`` answers semantic filters with an actual instruction
-tuned model (LOTUS-style ``sem_filter`` prompt, temperature 0), and accounts
for cost in three currencies: calls, tokens, and wall-clock seconds.

Two-phase design.  Real inference is slow (single-digit rows/s for an 8B model
on one machine), and a workload study needs to run *many* engines over the
*same* oracle.  Because decoding is deterministic (temperature 0, fixed seed),
the answer of the model on a (predicate, row) pair is a pure function; we
therefore materialize the full response matrix once
(:class:`LLMResponseMatrix`, built by ``experiments/build_llm_matrix.py``) and
replay it under unit-cost accounting.  Replay is observationally identical to
live calls -- every engine sees exactly the answers the model gave -- while
making a full-factorial comparison affordable.  ``LLMOracle`` supports both
modes: live (``client`` set) and replay (``matrix`` set).
"""

from __future__ import annotations

import http.client
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from semreuse.corpus import Corpus
from semreuse.oracle import OracleStats
from semreuse.predicates import Predicate

# Prompt template version -- part of every cache key, so a prompt edit never
# silently mixes semantics (the same discipline the predicate store applies to
# cached views).
PROMPT_VERSION = "semfilter-v1"

PROMPT_TEMPLATE = (
    "You are a careful data analyst labelling documents for a database query.\n"
    "\n"
    "Document:\n\"\"\"\n{doc}\n\"\"\"\n"
    "\n"
    "Claim: {claim}\n"
    "\n"
    "Does the claim hold for this document? Answer with exactly one word, "
    "yes or no.\nAnswer:"
)

# Published per-million-token prices of a representative commercial small model
# used only to translate measured token counts into a dollar figure; see the
# paper for the caveat that this is a projection, not a bill we paid.
REFERENCE_PRICE_IN_PER_MTOK = 0.15
REFERENCE_PRICE_OUT_PER_MTOK = 0.60


# ---------------------------------------------------------------------------
# Live client
# ---------------------------------------------------------------------------

@dataclass
class LLMCallStats:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    cache_hits: int = 0
    retries: int = 0

    def merge(self, other: "LLMCallStats") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.output_tokens += other.output_tokens
        self.wall_s += other.wall_s
        self.cache_hits += other.cache_hits

    @property
    def dollars(self) -> float:
        return (self.prompt_tokens / 1e6 * REFERENCE_PRICE_IN_PER_MTOK
                + self.output_tokens / 1e6 * REFERENCE_PRICE_OUT_PER_MTOK)


class OllamaClient:
    """Minimal, dependency-free client for a local Ollama server.

    Deterministic by construction: ``temperature=0``, ``top_k=1``, fixed seed,
    two output tokens.  Answers are memoized on disk (SQLite) keyed by
    (model, prompt version, claim, row) so that re-runs and audits observe the
    identical answer -- the determinism assumption of Section 3.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 cache_path: str | None = None, concurrency: int = 6,
                 max_doc_chars: int = 1200, timeout: float = 90.0,
                 seed: int = 0, chunk: int = 240,
                 keep_alive: str = "60m", progress=None):
        self.model = model
        self.chunk = chunk
        self.keep_alive = keep_alive
        self.progress = progress
        self.host = host.rstrip("/")
        self.concurrency = concurrency
        self.max_doc_chars = max_doc_chars
        self.timeout = timeout
        self.seed = seed
        self.stats = LLMCallStats()
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            self._db = sqlite3.connect(cache_path, check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS ans ("
                "model TEXT, pver TEXT, claim TEXT, row INTEGER, "
                "answer INTEGER, ptok INTEGER, otok INTEGER, raw TEXT, "
                "PRIMARY KEY (model, pver, claim, row))")
            # Cumulative cost of *live* inference, across restarts: a resumed
            # build serves few calls itself, so per-call token counts and
            # throughput have to be accumulated in the cache, not in memory.
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS runs ("
                "model TEXT, pver TEXT, calls INTEGER, wall_s REAL, "
                "ptok INTEGER, otok INTEGER)")
            self._db.commit()

    # -- cache -------------------------------------------------------------

    def _cached(self, claim: str, rows: list[int]) -> dict[int, bool]:
        if self._db is None:
            return {}
        out: dict[int, bool] = {}
        cur = self._db.cursor()
        for i in range(0, len(rows), 900):
            chunk = rows[i:i + 900]
            q = ("SELECT row, answer FROM ans WHERE model=? AND pver=? AND "
                 "claim=? AND row IN (%s)" % ",".join("?" * len(chunk)))
            for r, a in cur.execute(q, [self.model, PROMPT_VERSION, claim,
                                        *chunk]):
                out[int(r)] = bool(a)
        return out

    def _store(self, claim: str, recs: list[tuple[int, bool, int, int, str]]) -> None:
        if self._db is None or not recs:
            return
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO ans VALUES (?,?,?,?,?,?,?,?)",
                [(self.model, PROMPT_VERSION, claim, r, int(a), pt, ot, raw)
                 for r, a, pt, ot, raw in recs])
            self._db.commit()

    # -- inference ---------------------------------------------------------

    def _log_run(self, calls: int, wall_s: float, ptok: int,
                 otok: int) -> None:
        if self._db is None:
            return
        with self._lock:
            self._db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)",
                             (self.model, PROMPT_VERSION, calls, wall_s,
                              ptok, otok))
            self._db.commit()

    def cumulative_cost(self) -> dict:
        """Cost of every answer this cache holds for (model, prompt version).

        Token counts come from the answer table itself, so they are exact and
        complete even for cells produced by an earlier, interrupted build;
        wall-clock and call counts come from the per-chunk run log, which is
        what throughput should be measured from. Without this, a resumed build
        reports near-zero cost for work it inherited.
        """
        if self._db is None:
            return {"calls": self.stats.calls, "wall_s": self.stats.wall_s,
                    "prompt_tokens": self.stats.prompt_tokens,
                    "output_tokens": self.stats.output_tokens,
                    "cells": self.stats.calls}
        a = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(ptok),0), COALESCE(SUM(otok),0) "
            "FROM ans WHERE model=? AND pver=?",
            (self.model, PROMPT_VERSION)).fetchone()
        r = self._db.execute(
            "SELECT COALESCE(SUM(calls),0), COALESCE(SUM(wall_s),0) FROM runs "
            "WHERE model=? AND pver=?", (self.model, PROMPT_VERSION)).fetchone()
        return {"cells": int(a[0]), "prompt_tokens": int(a[1]),
                "output_tokens": int(a[2]),
                "calls": int(r[0]) or int(a[0]), "wall_s": float(r[1])}

    def _one(self, doc: str, claim: str) -> tuple[bool, int, int, str]:
        prompt = PROMPT_TEMPLATE.format(doc=doc[: self.max_doc_chars],
                                        claim=claim)
        body = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.0, "top_k": 1, "top_p": 1.0,
                        "seed": self.seed, "num_predict": 3},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    d = json.loads(resp.read())
                break
            except (urllib.error.URLError, TimeoutError, OSError,
                    http.client.HTTPException) as e:
                last_err = e
                with self._lock:
                    self.stats.retries += 1
                time.sleep(1.0 * (attempt + 1))
        else:  # pragma: no cover - only on a dead server
            raise RuntimeError(f"ollama request failed: {last_err}")
        raw = (d.get("response") or "").strip()
        ans = _parse_yes_no(raw)
        return ans, int(d.get("prompt_eval_count") or 0), \
            int(d.get("eval_count") or 0), raw[:16]

    def answer(self, corpus: Corpus, claim: str,
               rows: np.ndarray) -> np.ndarray:
        """Answer ``claim`` on ``rows``; cached rows cost nothing.

        Work proceeds in chunks that are committed to the answer cache as they
        complete, so a long materialization job is both observable and
        resumable: an interrupted run re-issues only the rows it never got to.
        """
        rows_l = [int(r) for r in rows]
        cached = self._cached(claim, rows_l)
        todo = [r for r in rows_l if r not in cached]
        self.stats.cache_hits += len(rows_l) - len(todo)
        for start in range(0, len(todo), self.chunk):
            batch = todo[start:start + self.chunk]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(self.concurrency) as ex:
                res = list(ex.map(
                    lambda r: self._one(corpus.docs[r], claim), batch))
            wall = time.perf_counter() - t0
            recs = [(r, a, pt, ot, raw)
                    for r, (a, pt, ot, raw) in zip(batch, res)]
            self._store(claim, recs)
            with self._lock:
                self.stats.calls += len(batch)
                self.stats.prompt_tokens += sum(pt for _, _, pt, _, _ in recs)
                self.stats.output_tokens += sum(ot for _, _, _, ot, _ in recs)
                self.stats.wall_s += wall
            cached.update({r: a for r, a, _, _, _ in recs})
            self._log_run(len(batch), wall,
                          sum(pt for _, _, pt, _, _ in recs),
                          sum(ot for _, _, _, ot, _ in recs))
            if self.progress is not None:
                self.progress(start + len(batch), len(todo),
                              len(batch) / max(1e-9, wall))
        return np.array([cached[r] for r in rows_l], dtype=bool)


    def answer_doc_major(self, corpus: Corpus, claims: list[str],
                         rows: np.ndarray, rows_per_commit: int = 32) -> None:
        """Fill the answer cache for every (claim, row) pair, one row at a time.

        The prompt puts the document before the claim, so the claims asked of
        one document share everything up to ``Claim:``.  Each worker takes a
        row and asks all of its missing claims back to back, which keeps that
        prefix in the server slot's KV cache and cuts prompt processing to the
        claim suffix.  Answers and their semantics are identical to
        :meth:`answer` (same prompt, same decoding options); only the issue
        order differs.  Resumable like :meth:`answer`: pairs already in the
        cache are never re-issued.
        """
        if self._db is None:
            raise ValueError("answer_doc_major needs an answer cache")
        done: set[tuple[str, int]] = set()
        for claim, row in self._db.execute(
                "SELECT claim, row FROM ans WHERE model=? AND pver=?",
                (self.model, PROMPT_VERSION)):
            done.add((claim, int(row)))
        todo = [(int(r), [c for c in claims if (c, int(r)) not in done])
                for r in rows]
        todo = [(r, cs) for r, cs in todo if cs]
        total = sum(len(cs) for _, cs in todo)
        self.stats.cache_hits += len(claims) * len(rows) - total

        def one_row(item):
            r, cs = item
            return [(c, r, *self._one(corpus.docs[r], c)) for c in cs]

        served = 0
        for start in range(0, len(todo), rows_per_commit):
            block = todo[start:start + rows_per_commit]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(self.concurrency) as ex:
                res = [x for lst in ex.map(one_row, block) for x in lst]
            wall = time.perf_counter() - t0
            with self._lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO ans VALUES (?,?,?,?,?,?,?,?)",
                    [(self.model, PROMPT_VERSION, c, r, int(a), pt, ot, raw)
                     for c, r, a, pt, ot, raw in res])
                self._db.commit()
                self.stats.calls += len(res)
                self.stats.prompt_tokens += sum(x[3] for x in res)
                self.stats.output_tokens += sum(x[4] for x in res)
                self.stats.wall_s += wall
            self._log_run(len(res), wall, sum(x[3] for x in res),
                          sum(x[4] for x in res))
            served += len(res)
            if self.progress is not None:
                self.progress(served, total, len(res) / max(1e-9, wall))

    def cached_matrix(self, claims: list[str], rows: np.ndarray) -> np.ndarray:
        """Dense bool[claims, rows] from the answer cache (must be complete)."""
        out = np.zeros((len(claims), len(rows)), dtype=bool)
        for i, c in enumerate(claims):
            got = self._cached(c, [int(r) for r in rows])
            missing = len(rows) - len(got)
            if missing:
                raise KeyError(f"{missing} answers missing for {c!r}")
            out[i] = [got[int(r)] for r in rows]
        return out


def _parse_yes_no(raw: str) -> bool:
    s = raw.strip().lower().lstrip("\"'` \n\t").rstrip(".,!\"'` \n\t")
    if s.startswith("yes") or s.startswith("true"):
        return True
    if s.startswith("no") or s.startswith("false"):
        return False
    # Degenerate completions are rare (<0.1% in our runs); treat as negative,
    # which is the conservative choice for a filter.
    return "yes" in s


# ---------------------------------------------------------------------------
# Materialized response matrix (replay mode)
# ---------------------------------------------------------------------------

@dataclass
class LLMResponseMatrix:
    """Dense (predicate x row) matrix of a real LLM's filter answers."""

    model: str
    corpus_name: str
    texts: list[str]                 # predicate texts, row order of ``answers``
    answers: np.ndarray              # bool[n_pred, n_rows]
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.texts)}

    def mean_prompt_tokens(self) -> float:
        n = max(1, self.answers.size)
        return self.prompt_tokens / n

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path, answers=self.answers,
            texts=np.array(self.texts, dtype=object),
            meta=np.array(json.dumps({
                "model": self.model, "corpus": self.corpus_name,
                "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens, "wall_s": self.wall_s,
                "prompt_version": PROMPT_VERSION, **self.meta}), dtype=object))

    @classmethod
    def load(cls, path: str) -> "LLMResponseMatrix":
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"].item()))
        return cls(model=meta["model"], corpus_name=meta["corpus"],
                   texts=[str(t) for t in z["texts"].tolist()],
                   answers=z["answers"],
                   prompt_tokens=int(meta.get("prompt_tokens", 0)),
                   output_tokens=int(meta.get("output_tokens", 0)),
                   wall_s=float(meta.get("wall_s", 0.0)), meta=meta)


class LLMOracle:
    """Oracle whose reference semantics are a real LLM's answers.

    In replay mode (``matrix`` given) the answers come from a materialized
    response matrix; calls are still charged, so cost accounting is exactly
    what a live run would report.  ``truth`` and ``reference`` both return the
    LLM's own answers: the certificate of Theorem 2 is a statement about
    fidelity to the oracle, and with a real LLM the oracle *is* the semantics.
    Label-space accuracy is reported separately by the experiment scripts.
    """

    version = "llm-v1"

    def __init__(self, corpus: Corpus,
                 matrix: LLMResponseMatrix | None = None,
                 client: OllamaClient | None = None):
        if matrix is None and client is None:
            raise ValueError("LLMOracle needs a matrix (replay) or a client")
        self.corpus = corpus
        self.matrix = matrix
        self.client = client
        self.stats = OracleStats()
        self._idx = matrix.index if matrix else {}
        if matrix is not None:
            self.version = f"llm-{matrix.model}-{PROMPT_VERSION}"
        elif client is not None:
            self.version = f"llm-{client.model}-{PROMPT_VERSION}"

    # -- reference semantics (never charged) -------------------------------

    def reference(self, predicate: Predicate) -> np.ndarray:
        if self.matrix is not None:
            i = self._idx.get(predicate.text)
            if i is None:
                raise KeyError(f"predicate not in response matrix: "
                               f"{predicate.text!r}")
            return self.matrix.answers[i]
        return self.client.answer(self.corpus, predicate.text,
                                  np.arange(self.corpus.n))

    def truth(self, predicate: Predicate) -> np.ndarray:
        """Ground truth for scoring *is* the oracle's own semantics here."""
        return self.reference(predicate)

    def labels(self, predicate: Predicate) -> np.ndarray:
        """The label-taxonomy extension, for label-space diagnostics only."""
        return self.corpus.extension(predicate.label_set)

    # -- charged evaluation ------------------------------------------------

    def evaluate(self, predicate: Predicate, rows: np.ndarray,
                 tag: str = "eval") -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        self.stats.charge(tag, len(rows))
        if self.matrix is not None:
            i = self._idx.get(predicate.text)
            if i is None:
                raise KeyError(f"predicate not in response matrix: "
                               f"{predicate.text!r}")
            return self.matrix.answers[i][rows]
        return self.client.answer(self.corpus, predicate.text, rows)
