import numpy as np
import pytest

from semreuse.corpus import make_synthetic_corpus
from semreuse.oracle import SimulatedOracle
from semreuse.predicates import PredicateUniverse


@pytest.fixture
def corpus():
    return make_synthetic_corpus(n=300, seed=1)


@pytest.fixture
def universe(corpus):
    return PredicateUniverse.build(corpus)


@pytest.fixture
def oracle(corpus):
    return SimulatedOracle(corpus, noise=0.0, seed=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
