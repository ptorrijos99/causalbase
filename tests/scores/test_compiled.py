"""The compiled BDeu engine behind the order searches, hill climbing, FLOP and exact search."""

import numpy as np
import pytest

from _samples import load_cell
from causalbase.scores import BDeuScore
from causalbase.scores._compiled import CompiledBDeu
from causalbase.scores._tetrad import TetradBDeuReference


def _tetrad(ds, **kwargs):
    """The engine with Tetrad 7.6.5's structure prior."""
    return CompiledBDeu(ds, prior="tetrad", **kwargs)


@pytest.mark.parametrize("net", ["alarm", "barley"])
def test_matches_fedcausal_bdeu(net):
    ds, _ = load_cell(net, 0)
    ref = TetradBDeuReference(ds)
    fast = _tetrad(ds)
    rng = np.random.default_rng(0)
    for _ in range(300):
        node = int(rng.integers(ds.n_vars))
        others = [v for v in range(ds.n_vars) if v != node]
        k = int(rng.integers(0, 6))
        parents = frozenset(int(p) for p in rng.choice(others, size=k, replace=False))
        assert fast.local_score(node, parents) == pytest.approx(
            ref.local_score(node, parents), rel=1e-10, abs=1e-8
        )


def test_sparse_path_with_many_parent_configurations():
    # barley has cardinalities up to 67, so 5 parents exceed the dense-table bound.
    ds, _ = load_cell("barley", 0)
    cards = ds.cardinalities
    node = int(np.argmin(cards))
    parents = frozenset(int(v) for v in np.argsort(cards)[::-1][:5] if v != node)
    q = int(np.prod([cards[p] for p in parents]))
    assert q * cards[node] > 4 * ds.n_rows
    ref = TetradBDeuReference(ds).local_score(node, parents)
    assert _tetrad(ds).local_score(node, parents) == pytest.approx(ref, rel=1e-10)


def test_no_structure_prior_matches():
    ds, _ = load_cell("alarm", 0)
    ref = TetradBDeuReference(ds, structure_prior=0.0)
    fast = _tetrad(ds, expected_parents=0.0)
    assert fast.local_score(3, frozenset({1, 7})) == pytest.approx(
        ref.local_score(3, frozenset({1, 7})), rel=1e-10
    )


@pytest.mark.parametrize("net", ["alarm", "barley"])
def test_batched_additions_equal_single_families_exactly(net):
    # Bitwise equality keeps runs reproducible whichever path scored a family.
    ds, _ = load_cell(net, 0)
    fast = _tetrad(ds)
    rng = np.random.default_rng(1)
    for _ in range(40):
        node = int(rng.integers(ds.n_vars))
        others = [v for v in range(ds.n_vars) if v != node]
        k = int(rng.integers(0, 4))
        parents = sorted(int(p) for p in rng.choice(others, size=k, replace=False))
        cands = sorted(v for v in others if v not in parents)
        batch = fast.local_scores_add(node, parents, cands)
        single = [fast.local_score(node, frozenset(parents) | {c}) for c in cands]
        assert list(batch) == single


def _random_batches(ds, seed, n, max_parents):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        node = int(rng.integers(ds.n_vars))
        others = [v for v in range(ds.n_vars) if v != node]
        k = int(rng.integers(0, max_parents + 1))
        parents = sorted(int(p) for p in rng.choice(others, size=k, replace=False))
        cands = [
            int(c)
            for c in rng.choice([v for v in others if v not in parents], size=8, replace=False)
        ]
        yield node, parents, cands


@pytest.mark.parametrize("net", ["alarm", "barley"])
def test_counting_paths_give_identical_scores(net):
    # popcount over bitsets, one dense pass over the rows, and rows grouped by
    # parent configuration must count the same cells, so the scores are equal bit for bit.
    ds, _ = load_cell(net, 0)
    popcount = _tetrad(ds, popcount_cells=10**9)
    dense = _tetrad(ds, popcount_cells=-1)
    grouped = _tetrad(ds, popcount_cells=-1, dense_cells=0)
    ref = TetradBDeuReference(ds)
    for node, parents, cands in _random_batches(ds, 2, 30, 4):
        a = popcount.local_scores_add(node, parents, cands)
        b = dense.local_scores_add(node, parents, cands)
        c = grouped.local_scores_add(node, parents, cands)
        assert list(a) == list(b) == list(c)
        expected = [ref.local_score(node, frozenset(parents) | {x}) for x in cands]
        assert list(a) == pytest.approx(expected, rel=1e-10, abs=1e-8)


def test_family_cache_budget_does_not_change_scores():
    ds, _ = load_cell("alarm", 0)
    roomy, starved = _tetrad(ds), _tetrad(ds, cache_bytes=0)
    for node, parents, cands in _random_batches(ds, 3, 30, 3):
        assert list(roomy.local_scores_add(node, parents, cands)) == list(
            starved.local_scores_add(node, parents, cands)
        )
        assert roomy.local_score(node, parents) == starved.local_score(node, parents)


@pytest.mark.parametrize("net", ["alarm", "barley"])
def test_n_jobs_give_identical_scores(net):
    ds, _ = load_cell(net, 0)
    serial, threaded = _tetrad(ds), _tetrad(ds, n_jobs=4)
    rng = np.random.default_rng(5)
    for _ in range(20):
        node = int(rng.integers(ds.n_vars))
        others = [v for v in range(ds.n_vars) if v != node]
        parents = sorted(
            int(p) for p in rng.choice(others, size=int(rng.integers(0, 4)), replace=False)
        )
        cands = [v for v in others if v not in parents]
        assert list(threaded.local_scores_add(node, parents, cands)) == list(
            serial.local_scores_add(node, parents, cands)
        )


def test_counts_every_family_it_scores():
    # Counting families separates an algorithm's work from how fast its kernel is.
    ds, _ = load_cell("alarm", 0)
    fast = _tetrad(ds)
    assert fast.n_families == 0
    fast.local_score(3, frozenset({1}))
    fast.local_scores_add(3, [1], [0, 2, 4, 5])
    assert fast.n_families == 5


@pytest.mark.parametrize("net", ["alarm", "barley"])
def test_kernels_give_identical_scores(net):
    # The cells kernel must return the floats of the kernel it replaced, whichever way it
    # splits a cell: through bitsets, by walking the rows, into one table of counters or
    # into four.
    ds, _ = load_cell(net, 0)
    legacy = _tetrad(ds, kernel="legacy")
    variants = [
        _tetrad(ds),
        _tetrad(ds, popcount_cells=-1),
        _tetrad(ds, popcount_cells=10**9),
        _tetrad(ds, popcount_cells=-1, lanes_rows=1),
        _tetrad(ds, popcount_cells=-1, lanes_rows=10**9),
    ]
    assert variants[0].kernel == "cells"
    for node, parents, cands in _random_batches(ds, 4, 30, 4):
        expected = list(legacy.local_scores_add(node, parents, cands))
        for fast in variants:
            assert list(fast.local_scores_add(node, parents, cands)) == expected
        assert variants[0].local_score(node, parents) == legacy.local_score(node, parents)
        # A family grouped already is scored through the batch path with a constant candidate.
        assert variants[0].local_score(node, parents) == legacy.local_score(node, parents)


def test_kernels_agree_on_few_rows_and_many_configurations():
    # Cells of one or two rows and configurations with fewer rows than the candidate has values.
    full, _ = load_cell("barley", 0)
    ds = full.subset(np.arange(300))
    legacy, cells = _tetrad(ds, kernel="legacy"), _tetrad(ds)
    for node, parents, cands in _random_batches(ds, 6, 40, 5):
        assert list(cells.local_scores_add(node, parents, cands)) == list(
            legacy.local_scores_add(node, parents, cands)
        )


def test_many_prior_values_do_not_change_the_scores():
    # The lgamma table is direct-mapped: with two rows nearly every prior evicts another one.
    ds, _ = load_cell("barley", 0)
    roomy, tight = _tetrad(ds), _tetrad(ds, lgamma_rows=2)
    for node, parents, cands in _random_batches(ds, 7, 30, 3):
        assert list(tight.local_scores_add(node, parents, cands)) == list(
            roomy.local_scores_add(node, parents, cands)
        )
        assert tight.local_score(node, parents) == roomy.local_score(node, parents)


def test_rejects_an_unknown_kernel():
    ds, _ = load_cell("alarm", 0)
    with pytest.raises(ValueError):
        _tetrad(ds, kernel="dense")


@pytest.mark.parametrize("net", ["alarm", "barley"])
def test_bernoulli_prior_matches_bdeu_score(net):
    # The order searches score with the engine; on the Bernoulli prior it must give
    # BDeuScore's values, which come from another kernel.
    ds, _ = load_cell(net, 0)
    for expected_parents in (1.0, None, 3.0):
        engine = CompiledBDeu(ds, expected_parents=expected_parents, prior="bernoulli")
        score = BDeuScore(ds, expected_parents=expected_parents)
        for node, parents, cands in _random_batches(ds, 8, 25, 4):
            assert engine.local_score(node, parents) == pytest.approx(
                score.local_score(node, parents), rel=1e-10
            )
            batch = engine.local_scores_add(node, parents, cands)
            assert list(batch) == pytest.approx(
                list(score.local_scores_with(node, frozenset(parents), cands)), rel=1e-10
            )
            assert list(batch) == [
                engine.local_score(node, frozenset(parents) | {c}) for c in cands
            ]


def test_the_two_priors_differ_only_in_the_prior():
    ds, _ = load_cell("alarm", 0)
    bernoulli = CompiledBDeu(ds, expected_parents=None, prior="bernoulli")
    tetrad = _tetrad(ds, expected_parents=0.0)
    for node, parents, cands in _random_batches(ds, 9, 20, 3):
        assert bernoulli.local_score(node, parents) == tetrad.local_score(node, parents)
        assert list(bernoulli.local_scores_add(node, parents, cands)) == list(
            tetrad.local_scores_add(node, parents, cands)
        )


def test_rejects_an_unknown_prior():
    ds, _ = load_cell("asia", 0)
    with pytest.raises(ValueError, match="prior"):
        CompiledBDeu(ds, prior="uniform")
