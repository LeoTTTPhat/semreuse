"""Accuracy and cost accounting for workload runs."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class QueryMetrics:
    query_index: int
    predicate_name: str
    predicate_text: str
    reuse_kind: str
    oracle_calls: int
    candidate_calls: int
    audit_calls: int
    escalation_calls: int
    nli_pair_scores: int
    precision: float
    recall: float
    f1: float
    n_true_pos_rows: int
    recall_bound: float | None
    bound_violated: bool | None
    precision_bound: float | None
    precision_bound_violated: bool | None
    latency_s: float
    # Recall / bound check against the *oracle's own* semantics (labels ^
    # noise flips) -- the reference for the C2 certificate.  Identical to
    # the label-based columns when oracle noise is 0.
    recall_oracle: float | None = None
    bound_violated_oracle: bool | None = None


def score_result(reported: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    tp = int((reported & truth).sum())
    fp = int((reported & ~truth).sum())
    fn = int((~reported & truth).sum())
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def run_workload(engine, workload, oracle, verbose: bool = False) -> list[QueryMetrics]:
    """Execute a workload on an engine and score each query against the
    label-semantics ground truth (oracle.truth, never charged)."""
    out: list[QueryMetrics] = []
    for i, pred in enumerate(workload.queries):
        t0 = time.perf_counter()
        res = engine.query(pred)
        dt = time.perf_counter() - t0
        truth = oracle.truth(pred)
        prec, rec, f1 = score_result(res.reported, truth)
        violated = (None if res.recall_bound is None
                    else bool(rec < res.recall_bound - 1e-12))
        # C2 check: the certificate is w.r.t. the oracle's semantics.
        ref = (oracle.reference(pred) if hasattr(oracle, "reference")
               else truth)
        prec_o, rec_o, _ = score_result(res.reported, ref)
        violated_o = (None if res.recall_bound is None
                      else bool(rec_o < res.recall_bound - 1e-12))
        # Precision certificate: same reference, same convention.
        pbound = getattr(res, "precision_bound", None)
        pviol = (None if pbound is None
                 else bool(prec_o < pbound - 1e-12))
        out.append(QueryMetrics(
            query_index=i, predicate_name=pred.name,
            predicate_text=pred.text, reuse_kind=res.reuse_kind,
            oracle_calls=res.oracle_calls,
            candidate_calls=res.candidate_calls,
            audit_calls=res.audit_calls,
            escalation_calls=res.escalation_calls,
            nli_pair_scores=res.nli_pair_scores,
            precision=prec, recall=rec, f1=f1,
            n_true_pos_rows=int(truth.sum()),
            recall_bound=res.recall_bound, bound_violated=violated,
            precision_bound=pbound, precision_bound_violated=pviol,
            latency_s=dt,
            recall_oracle=rec_o, bound_violated_oracle=violated_o))
        if verbose:
            print(f"[{i:3d}] {pred.name:35s} kind={res.reuse_kind:12s} "
                  f"calls={res.oracle_calls:6d} P={prec:.3f} R={rec:.3f} "
                  f"bound={res.recall_bound}")
    return out


def summarize(metrics: list[QueryMetrics]) -> dict:
    calls = sum(m.oracle_calls for m in metrics)
    n = len(metrics)
    bounded = [m for m in metrics if m.recall_bound is not None]
    pbounded = [m for m in metrics if m.precision_bound is not None]
    return dict(
        n_queries=n,
        total_oracle_calls=calls,
        mean_calls_per_query=calls / n if n else 0.0,
        total_nli_scores=sum(m.nli_pair_scores for m in metrics),
        macro_precision=float(np.mean([m.precision for m in metrics])) if n else 1.0,
        macro_recall=float(np.mean([m.recall for m in metrics])) if n else 1.0,
        macro_f1=float(np.mean([m.f1 for m in metrics])) if n else 1.0,
        n_bounded=len(bounded),
        bound_violations=sum(1 for m in bounded if m.bound_violated),
        bound_violations_oracle=sum(
            1 for m in bounded if m.bound_violated_oracle),
        macro_recall_oracle=float(np.mean(
            [m.recall_oracle for m in metrics
             if m.recall_oracle is not None])) if n else 1.0,
        mean_recall_bound=float(np.mean([m.recall_bound for m in bounded]))
        if bounded else None,
        n_precision_bounded=len(pbounded),
        precision_bound_violations=sum(
            1 for m in pbounded if m.precision_bound_violated),
        mean_precision_bound=float(np.mean(
            [m.precision_bound for m in pbounded])) if pbounded else None,
        total_latency_s=sum(m.latency_s for m in metrics),
    )


def metrics_to_rows(metrics: list[QueryMetrics], **tags) -> list[dict]:
    """Flatten to CSV-ready dict rows, tagging every row with run params."""
    rows = []
    for m in metrics:
        d = dict(m.__dict__)
        d.update(tags)
        rows.append(d)
    return rows
