"""Shared harness for SemReuse experiments.

Every experiment script:
  * is parameterized by --scale {smoke,small,medium,full} (corpus size,
    query count) plus a fixed --seed,
  * writes tidy per-query CSVs and a summary CSV into results/,
  * uses only local models / the simulated oracle (no paid APIs),
  * caches datasets and models under data/ (HF_HOME=data/hf).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# scale -> (corpus_size, n_queries)
SCALES = {
    "smoke": (400, 12),
    "small": (1500, 30),
    "medium": (4000, 60),
    "full": (8000, 120),
}


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--scale", choices=SCALES, default="small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default="20ng",
                    choices=["20ng", "agnews", "synthetic", "rcv1"])
    ap.add_argument("--out-tag", default="", help="suffix for output files")
    ap.add_argument("--calib-pairs", type=int, default=300,
                    help="labeled synthetic pairs for NLI calibration")
    return ap


def scale_params(args) -> tuple[int, int]:
    return SCALES[args.scale]


def load_corpus_scaled(args):
    from semreuse.corpus import load_corpus

    size, _ = scale_params(args)
    return load_corpus(args.corpus, str(DATA_DIR), size=size, seed=args.seed)


def make_nli_entailment(universe=None, calibrate: bool = True, seed: int = 0,
                        n_pairs: int = 300):
    """NLI reasoner; calibrated on predicate pairs from a *synthetic*
    dev taxonomy (dataset-independent -> no leakage into test corpora)."""
    from semreuse.corpus import make_synthetic_corpus
    from semreuse.entailment import NLIEntailment
    from semreuse.predicates import (PredicateUniverse,
                                     labeled_pairs_for_calibration)

    ent = NLIEntailment(model_name=NLI_MODEL)
    if calibrate:
        # Scale the dev taxonomy with the requested pair count so the
        # class-balanced sampler can fill its buckets.
        n_groups = 8 if n_pairs <= 400 else 12
        dev_corpus = make_synthetic_corpus(n=50, seed=seed, n_groups=n_groups)
        dev_uni = PredicateUniverse.build(dev_corpus)
        pairs = labeled_pairs_for_calibration(dev_uni, n_pairs=n_pairs,
                                              seed=seed)
        acc = ent.fit_calibration(pairs)
        print(f"[calibration] fitted on {len(pairs)} synthetic pairs, "
              f"train acc {acc:.3f}")
    return ent


def write_results(df: pd.DataFrame, name: str, args) -> pathlib.Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    path = RESULTS_DIR / f"{name}_{args.corpus}_{args.scale}{tag}.csv"
    df.to_csv(path, index=False)
    print(f"[out] {path}  ({len(df)} rows)")
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
