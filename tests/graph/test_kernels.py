"""The compiled graph kernels give exactly what the dense reference versions gave."""

from __future__ import annotations

import _reference_kernels as ref  # next to this file
import numpy as np

from causalbase.graph import _kernels as k


def _random_dag(n: int, density: float, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(n)
    dag = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                dag[order[i], order[j]] = 1
    return dag


def test_cpdag_and_extension_match_the_reference() -> None:
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 40))
        dag = _random_dag(n, float(rng.uniform(0.02, 0.4)), rng)
        cpdag = k.cpdag_of_dag(dag)
        assert np.array_equal(np.asarray(cpdag, dtype=np.uint8), ref.cpdag_of_dag(dag))
        pdag = np.asarray(cpdag, dtype=np.uint8)
        got, ok = k.consistent_extension(pdag)
        want, ok_ref = ref.consistent_extension(pdag)
        assert ok == ok_ref
        assert np.array_equal(got, want)


def test_extension_of_arbitrary_pdags_matches_the_reference() -> None:
    for seed in range(20):
        # Including PDAGs with no consistent extension.
        rng = np.random.default_rng(100 + seed)
        n = int(rng.integers(2, 25))
        adj = (rng.random((n, n)) < 0.15).astype(np.uint8)
        np.fill_diagonal(adj, 0)
        got, ok = k.consistent_extension(adj)
        want, ok_ref = ref.consistent_extension(adj)
        assert ok == ok_ref
        if ok:
            assert np.array_equal(got, want)


def _neighbour_lists(adj: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Padded neighbour lists of `adj`, each row in a random order."""
    n = adj.shape[0]
    skeleton = (adj | adj.T).astype(bool)
    deg = skeleton.sum(axis=1).astype(np.int32)
    nbr = np.zeros((n, max(1, int(deg.max(initial=0))) + 2), dtype=np.int32)
    for v in range(n):
        nbr[v, : deg[v]] = rng.permutation(np.flatnonzero(skeleton[v]))
    return nbr, deg


def _partly_oriented(dag: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A PDAG between `dag` and its CPDAG: some reversible edges keep the direction `dag` gives."""
    pdag = np.asarray(ref.cpdag_of_dag(dag), dtype=np.uint8)
    keep = rng.random(dag.shape) < 0.5
    pdag[(dag.T == 1) & keep.T] = 0
    return pdag


def test_complete_marks_gives_the_reference_cpdag() -> None:
    for seed in range(60):
        rng = np.random.default_rng(300 + seed)
        n = int(rng.integers(1, 40))
        dag = _random_dag(n, float(rng.uniform(0.02, 0.5)), rng)
        pdag = _partly_oriented(dag, rng)
        want = np.asarray(ref.cpdag_of_dag(dag), dtype=np.uint8)
        nbr, deg = _neighbour_lists(pdag, rng)
        got = pdag.copy()
        pairs, ok = k.complete_marks(got, nbr, deg)
        assert ok
        assert np.array_equal(got, want)
        assert np.array_equal(pairs, ref.changed_pairs(pdag, want))


def test_complete_marks_on_arbitrary_pdags_matches_the_reference() -> None:
    for seed in range(40):
        # Including PDAGs with no consistent extension, which are left untouched.
        rng = np.random.default_rng(400 + seed)
        n = int(rng.integers(2, 25))
        adj = (rng.random((n, n)) < 0.15).astype(np.uint8)
        np.fill_diagonal(adj, 0)
        dag, ok_ref = ref.consistent_extension(adj)
        nbr, deg = _neighbour_lists(adj, rng)
        got = adj.copy()
        pairs, ok = k.complete_marks(got, nbr, deg)
        assert ok == ok_ref
        if ok:
            want = np.asarray(ref.cpdag_of_dag(dag), dtype=np.uint8)
            assert np.array_equal(got, want)
            assert np.array_equal(pairs, ref.changed_pairs(adj, want))
        else:
            assert np.array_equal(got, adj)
            assert len(pairs) == 0


def test_reach_over_lists_matches_the_reference() -> None:
    for seed in range(30):
        rng = np.random.default_rng(500 + seed)
        n = int(rng.integers(2, 40))
        pdag = _partly_oriented(_random_dag(n, float(rng.uniform(0.05, 0.3)), rng), rng)
        nbr, deg = _neighbour_lists(pdag, rng)
        indptr = np.zeros(n + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(pdag.sum(axis=1))
        indices = np.nonzero(pdag)[1].astype(np.int32)
        for source in range(n):
            avoid = rng.choice(n, size=int(rng.integers(0, 4)), replace=False).astype(np.int32)
            got = k.semidirected_reach_lists(pdag, nbr, deg, source, avoid)
            assert np.array_equal(got, ref.semidirected_reach(indptr, indices, source, avoid))


def test_link_and_unlink_edit_the_lists() -> None:
    nbr = np.zeros((4, 3), dtype=np.int32)
    deg = np.zeros(4, dtype=np.int32)
    k.link(nbr, deg, 0, 1)
    k.link(nbr, deg, 0, 2)
    k.link(nbr, deg, 3, 0)
    assert sorted(nbr[0, : deg[0]].tolist()) == [1, 2, 3]
    k.unlink(nbr, deg, 1, 0)
    assert sorted(nbr[0, : deg[0]].tolist()) == [2, 3]
    assert deg.tolist() == [2, 0, 1, 1]
