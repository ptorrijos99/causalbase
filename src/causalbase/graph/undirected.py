"""Moralization, triangulation, maximal cliques and treewidth.

Undirected graphs are symmetric boolean adjacency matrices with an empty
diagonal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from causalbase.graph.dag import DAG

__all__ = [
    "moralize",
    "mcs_order",
    "fill_in",
    "maximal_cliques",
    "treewidth",
]


def moralize(dag: DAG) -> npt.NDArray[np.bool_]:
    """Moralize a DAG into an undirected moral graph.

    The moral graph is the skeleton plus an edge between every two parents of
    a common child.

    Parameters
    ----------
    dag : DAG
        The directed acyclic graph to moralize.

    Returns
    -------
    np.ndarray
        A symmetric, hollow boolean adjacency matrix of shape
        ``(dag.num_nodes, dag.num_nodes)`` representing the moral graph.
    """
    directed = dag.adjacency()
    moral: npt.NDArray[np.bool_] = directed | directed.T
    for v in range(dag.num_nodes):
        parents = sorted(dag.parents(v))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                p, q = parents[i], parents[j]
                moral[p, q] = True
                moral[q, p] = True
    return moral


def mcs_order(adj: npt.NDArray[np.bool_]) -> list[int]:
    """Compute a maximum cardinality search (MCS) ordering.

    Repeatedly picks the unnumbered node adjacent to the most numbered ones,
    breaking ties by lowest index.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric boolean adjacency matrix of an undirected graph.

    Returns
    -------
    list[int]
        A permutation of ``range(adj.shape[0])``: the MCS visitation order.
    """
    n = adj.shape[0]
    numbered = [False] * n
    cardinality = [0] * n
    order: list[int] = []
    for _ in range(n):
        best = -1
        best_cardinality = -1
        for v in range(n):
            if not numbered[v] and cardinality[v] > best_cardinality:
                best_cardinality = cardinality[v]
                best = v
        order.append(best)
        numbered[best] = True
        for w in range(n):
            if adj[best, w] and not numbered[w]:
                cardinality[w] += 1
    return order


def fill_in(adj: npt.NDArray[np.bool_], order: list[int]) -> npt.NDArray[np.bool_]:
    """Triangulate a graph using the Tarjan and Yannakakis (1984) fill-in.

    Visits nodes in reverse `order` and connects every two neighbours of the
    current node that come earlier in `order`.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric boolean adjacency matrix of an undirected graph (typically
        a moral graph).
    order : list[int]
        A maximum cardinality ordering of `adj`, as returned by
        :func:`mcs_order`.

    Returns
    -------
    np.ndarray
        A new symmetric boolean adjacency matrix: a chordal (triangulated)
        supergraph of `adj`. The input `adj` is not modified.
    """
    n = len(order)
    filled = adj.copy()
    for i in range(n - 1, -1, -1):
        v = order[i]
        neighbors = [order[j] for j in range(i) if filled[v, order[j]]]
        for a in range(len(neighbors)):
            w = neighbors[a]
            for b in range(a + 1, len(neighbors)):
                x = neighbors[b]
                if not filled[w, x]:
                    filled[w, x] = True
                    filled[x, w] = True
    return filled


def maximal_cliques(adj: npt.NDArray[np.bool_], order: list[int]) -> list[frozenset[int]]:
    """Extract the maximal cliques of a chordal graph.

    Each node together with its neighbours earlier in `order` is a clique of a
    chordal graph; the ones contained in another are dropped.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric boolean adjacency matrix of a chordal (triangulated)
        undirected graph.
    order : list[int]
        A maximum cardinality ordering of `adj`, as returned by
        :func:`mcs_order`.

    Returns
    -------
    list[frozenset[int]]
        The maximal cliques of the graph. No returned clique is a subset of
        another.
    """
    n = len(order)
    candidates: list[frozenset[int]] = []
    for i in range(n - 1, -1, -1):
        v = order[i]
        clique = {v}
        for j in range(i):
            w = order[j]
            if adj[v, w]:
                clique.add(w)
        candidates.append(frozenset(clique))

    candidates.sort(key=len, reverse=True)

    kept: list[frozenset[int] | None] = list(candidates)
    for i in range(len(kept)):
        v1 = kept[i]
        if v1 is None:
            continue
        for j in range(i + 1, len(kept)):
            v2 = kept[j]
            if v2 is not None and v1 >= v2:
                kept[j] = None

    return [c for c in kept if c is not None]


def treewidth(dag: DAG) -> int:
    """Return an upper bound on the treewidth of `dag`'s moral graph.

    The moral graph is triangulated by maximum cardinality search and fill-in,
    and the bound is the size of its largest clique minus one. It is exact
    whenever the moral graph is already chordal.

    Parameters
    ----------
    dag : DAG
        The graph whose moral graph is measured.

    Returns
    -------
    int
        The bound; ``0`` for a graph with no edges.
    """
    moral = moralize(dag)
    triangulated = fill_in(moral, mcs_order(moral))
    cliques = maximal_cliques(triangulated, mcs_order(triangulated))
    return max((len(c) for c in cliques), default=1) - 1
