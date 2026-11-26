"""PDAG: edits against a plain model, the cached arrays, and in-place completion."""

import numpy as np
import pytest

from causalbase.graph import DAG, PDAG
from causalbase.graph.cpdag import dag_to_cpdag


def _neighbours(pdag: PDAG) -> list[list[int]]:
    nbr, deg = pdag._neighbour_lists()
    return [sorted(nbr[v, : deg[v]].tolist()) for v in range(pdag.num_nodes)]


def test_random_edits_agree_with_a_plain_model_and_the_cached_arrays() -> None:
    """The live adjacency matrix and neighbour lists follow every edit, even past their width."""
    n = 12
    for seed in range(5):
        rng = np.random.default_rng(seed)
        pdag = PDAG(n)
        pdag._adjacency()
        pdag._neighbour_lists()
        directed: set[tuple[int, int]] = set()
        undirected: set[frozenset[int]] = set()
        for _ in range(300):
            u, v = (int(i) for i in rng.choice(n, size=2, replace=False))
            if v in pdag.adjacent(u):
                if frozenset((u, v)) in undirected and rng.random() < 0.5:
                    pdag.orient(u, v)
                    undirected.discard(frozenset((u, v)))
                    directed.add((u, v))
                else:
                    pdag.remove_edge(u, v)
                    undirected.discard(frozenset((u, v)))
                    directed -= {(u, v), (v, u)}
            elif rng.random() < 0.5:
                pdag.add_directed(u, v)
                directed.add((u, v))
            else:
                pdag.add_undirected(u, v)
                undirected.add(frozenset((u, v)))
            fresh = pdag.copy()
            assert fresh == pdag
            assert np.array_equal(pdag._adjacency(), fresh.adjacency())
            assert _neighbours(pdag) == [sorted(fresh.adjacent(w)) for w in range(n)]
        assert pdag.directed_edges() == sorted(directed)
        assert pdag.undirected_edges() == sorted(tuple(sorted(e)) for e in undirected)
        for w in range(n):
            assert pdag.parents(w) == {a for a, b in directed if b == w}
            neighbours = {x for e in undirected if w in e for x in e} - {w}
            assert pdag.undirected_neighbors(w) == neighbours


def test_malformed_edits_are_refused() -> None:
    pdag = PDAG(3)
    pdag.add_directed(0, 1)
    pdag.add_undirected(1, 2)
    pdag.add_directed(0, 1)  # adding an edge again changes nothing
    pdag.add_undirected(2, 1)
    assert (pdag.directed_edges(), pdag.undirected_edges()) == ([(0, 1)], [(1, 2)])
    for edit, args in [
        (pdag.add_directed, (1, 0)),  # the opposite direction
        (pdag.add_directed, (1, 2)),  # over an undirected edge
        (pdag.add_undirected, (0, 1)),  # over a directed edge
        (pdag.add_directed, (2, 2)),
    ]:
        with pytest.raises(ValueError):
            edit(*args)
    with pytest.raises(KeyError):
        pdag.orient(0, 1)
    with pytest.raises(KeyError):
        pdag.remove_edge(0, 2)


def test_to_dag_keeps_directed_edges_and_adds_no_v_structure() -> None:
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    assert PDAG.from_dag(dag).to_dag() == dag
    chain = PDAG(3)
    chain.add_undirected(0, 1)
    chain.add_undirected(1, 2)
    assert chain.to_dag().parents(1) != {0, 2}
    cycle = PDAG.from_dag(DAG.from_edges(3, [(0, 1), (1, 2)]))
    cycle.add_directed(2, 0)
    with pytest.raises(ValueError, match="no consistent"):
        cycle.to_dag()


def test_complete_in_place_gives_the_cpdag_and_the_pairs_it_changed() -> None:
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 25))
        order = rng.permutation(n)
        edges = [
            (int(order[i]), int(order[j]))
            for i in range(n)
            for j in range(i + 1, n)
            if rng.random() < 0.2
        ]
        dag = DAG.from_edges(n, edges)
        pdag = PDAG.from_dag(dag)
        before = pdag.adjacency()
        pairs = pdag._complete_in_place()
        want = dag_to_cpdag(dag)
        assert pdag == want
        assert np.array_equal(pdag._adjacency(), want.adjacency())
        changed = before != want.adjacency()
        assert sorted(map(tuple, pairs.tolist())) == sorted(
            {(min(u, v), max(u, v)) for u, v in zip(*np.nonzero(changed), strict=True)}
        )
        assert len(pdag._complete_in_place()) == 0


def test_complete_in_place_without_extension_raises_and_keeps_the_graph() -> None:
    pdag = PDAG(4)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        pdag.add_undirected(u, v)
    before = pdag.copy()
    with pytest.raises(ValueError, match="no consistent extension"):
        pdag._complete_in_place()
    assert pdag == before
