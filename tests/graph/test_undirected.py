"""Moralisation, triangulation, cliques and the treewidth bound."""

import itertools

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from causalbase.graph import DAG
from causalbase.graph.undirected import fill_in, maximal_cliques, mcs_order, moralize, treewidth


@st.composite
def _dags(draw: st.DrawFn) -> DAG:
    n = draw(st.integers(0, 9))
    order = draw(st.permutations(range(n)))
    edges = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n) if draw(st.booleans())]
    return DAG.from_edges(n, edges)


def test_moralize_marries_the_parents_of_each_node() -> None:
    # A diamond: 3's parents 1 and 2 are married; 0 and 3 stay apart.
    moral = moralize(DAG.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)]))
    expected = {(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)}
    assert {(u, v) for u, v in zip(*np.nonzero(np.triu(moral)), strict=True)} == expected
    assert np.array_equal(moral, moral.T)
    assert not moral.diagonal().any()


@given(_dags())
@settings(max_examples=60, deadline=None)
def test_fill_in_triangulates_and_the_cliques_cover_it(dag: DAG) -> None:
    moral = moralize(dag)
    chordal = fill_in(moral, mcs_order(moral))
    assert np.array_equal(chordal & moral, moral)
    assert np.array_equal(chordal, chordal.T)
    # Chordal iff, along an MCS order, each node's earlier neighbours form a clique.
    order = mcs_order(chordal)
    position = {v: i for i, v in enumerate(order)}
    for i, v in enumerate(order):
        earlier = [w for w in np.flatnonzero(chordal[v]) if position[w] < i]
        assert all(chordal[a, b] for a, b in itertools.combinations(earlier, 2))
    cliques = maximal_cliques(chordal, order)
    assert not any(a < b for a in cliques for b in cliques)
    for u, v in zip(*np.nonzero(chordal), strict=True):
        assert any({u, v} <= c for c in cliques)
    if dag.num_nodes:
        assert treewidth(dag) == max(len(c) for c in cliques) - 1


def test_treewidth_of_known_graphs() -> None:
    assert treewidth(DAG.from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])) == 2
    assert treewidth(DAG.from_edges(5, [(0, 1), (0, 2), (1, 3), (1, 4)])) == 1
    assert treewidth(DAG(0)) == treewidth(DAG(3)) == 0
