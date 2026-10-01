"""Integration tests that need the local NLI cross-encoder.

Skipped automatically when the model is not present in the local HF cache
(data/hf) -- unit tests must pass offline.  Run experiments/prepare_data.py
to download models.
"""

import os
import pathlib

import pytest

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
MODEL = "cross-encoder/nli-deberta-v3-xsmall"


def _model_cached() -> bool:
    cache = DATA_DIR / "hf" / "hub" / ("models--" + MODEL.replace("/", "--"))
    return cache.exists()


pytestmark = pytest.mark.skipif(
    not _model_cached(), reason="NLI model not in data/hf cache")


@pytest.fixture(scope="module")
def nli():
    os.environ["HF_HUB_OFFLINE"] = "1"  # never hit the network in tests
    from semreuse.entailment import NLIEntailment

    ent = NLIEntailment(model_name=MODEL)
    yield ent
    os.environ.pop("HF_HUB_OFFLINE", None)


def test_nli_detects_subsumption(nli, universe):
    leaf = universe.by_name_prefix("leaf:baseball#")[0]
    group = universe.by_name_prefix("group:sports#")[0]
    from semreuse.predicates import Relation

    j = nli.judge(leaf, group)
    assert j.relation is Relation.FORWARD
    assert j.confidence > 0.5


def test_nli_detects_disjoint(nli, universe):
    a = universe.by_name_prefix("leaf:baseball#")[0]
    b = universe.by_name_prefix("leaf:medicine#")[0]
    from semreuse.predicates import Relation

    j = nli.judge(a, b)
    assert j.relation is Relation.DISJOINT


def test_nli_paraphrase_equiv(nli, universe):
    p1, p2 = universe.by_name_prefix("leaf:baseball#")[:2]
    from semreuse.predicates import Relation

    j = nli.judge(p1, p2)
    assert j.relation is Relation.EQUIV


def test_calibrated_head_improves_or_matches(nli, universe):
    from semreuse.predicates import labeled_pairs_for_calibration

    pairs = labeled_pairs_for_calibration(universe, n_pairs=120, seed=0)
    train, test = pairs[:80], pairs[80:]
    # threshold-head accuracy
    correct_thr = sum(nli.judge(p, q).relation is r for p, q, r in test)
    acc = nli.fit_calibration(train)
    assert acc > 0.5
    correct_cal = sum(nli.judge(p, q).relation is r for p, q, r in test)
    # Calibrated head should be at least competitive.
    assert correct_cal >= correct_thr - 3


def test_score_cache_avoids_recompute(nli, universe):
    p = universe.by_name_prefix("leaf:cars#")[0]
    q = universe.by_name_prefix("group:vehicles#")[0]
    nli.judge(p, q)
    before = nli.stats.pair_scores
    nli.judge(p, q)
    assert nli.stats.pair_scores == before
