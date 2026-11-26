"""Directed acyclic graph over integer nodes."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np

from causalbase.exceptions import CycleError

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

__all__ = ["DAG", "CycleError"]


class DAG:
    """A directed acyclic graph over integer nodes ``0..num_nodes-1``.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph. Nodes are addressed by the integer
        indices ``0, 1, ..., num_nodes - 1``.

    Attributes
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_edges : int
        Current number of directed edges in the graph.

    """

    def __init__(self, num_nodes: int) -> None:
        if num_nodes < 0:
            raise ValueError(f"num_nodes must be non-negative, got {num_nodes}")
        self.num_nodes = num_nodes
        self.num_edges = 0
        self._parents: list[set[int]] = [set() for _ in range(num_nodes)]
        self._children: list[set[int]] = [set() for _ in range(num_nodes)]
        self._amat: npt.NDArray[np.bool_] | None = None

    def _check_node(self, v: int) -> None:
        # A negative index would otherwise wrap around to the last nodes.
        if not 0 <= v < self.num_nodes:
            raise IndexError(f"node {v} is not in 0..{self.num_nodes - 1}")

    def parents(self, v: int) -> frozenset[int]:
        """Return the parents of node `v`."""
        return frozenset(self._parents[v])

    def children(self, v: int) -> frozenset[int]:
        """Return the children of node `v`."""
        return frozenset(self._children[v])

    def adjacent(self, v: int) -> frozenset[int]:
        """Return the nodes adjacent to `v` (parents union children)."""
        return frozenset(self._parents[v] | self._children[v])

    def has_edge(self, u: int, v: int) -> bool:
        """Return whether the directed edge `u -> v` is present."""
        return v in self._children[u]

    def add_edge(self, u: int, v: int) -> None:
        """Add the directed edge `u -> v`.

        Parameters
        ----------
        u : int
            Source node index.
        v : int
            Target node index.

        Raises
        ------
        IndexError
            If `u` or `v` is not a node.
        CycleError
            If `u == v` (self-loop), or if a path `v -> ... -> u` already
            exists, which means adding `u -> v` would create a cycle. The
            graph is left unchanged when this is raised.

        Notes
        -----
        Adding an edge that already exists is a no-op: it does not raise
        and does not affect `num_edges`.
        """
        self._check_node(u)
        self._check_node(v)
        if self.has_edge(u, v):
            return
        if u == v:
            raise CycleError(f"self-loop rejected: {u} -> {v}")
        if self.has_path(v, u):
            raise CycleError(f"edge {u} -> {v} would create a cycle")
        self._parents[v].add(u)
        self._children[u].add(v)
        self.num_edges += 1
        self._amat = None

    def remove_edge(self, u: int, v: int) -> None:
        """Remove the directed edge `u -> v`.

        Parameters
        ----------
        u : int
            Source node index.
        v : int
            Target node index.

        Raises
        ------
        KeyError
            If `u -> v` is not an edge of the graph.
        """
        if not self.has_edge(u, v):
            raise KeyError(f"no such edge: {u} -> {v}")
        self._children[u].remove(v)
        self._parents[v].remove(u)
        self.num_edges -= 1
        self._amat = None

    def edges(self) -> list[tuple[int, int]]:
        """Return all edges of the graph, sorted."""
        return sorted((u, v) for u in range(self.num_nodes) for v in self._children[u])

    def adjacency(self) -> npt.NDArray[np.bool_]:
        """Return the boolean adjacency matrix of the graph."""
        if self._amat is None:
            amat = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
            for u in range(self.num_nodes):
                for v in self._children[u]:
                    amat[u, v] = True
            self._amat = amat
        return self._amat.copy()

    def topological_order(self) -> list[int]:
        """Return a deterministic topological order of the nodes.

        Uses Kahn's algorithm with a min-heap over ready nodes, so the
        result is deterministic (lowest-index-first among ties).

        Returns
        -------
        list[int]
            Node indices such that every edge `u -> v` has `u` appearing
            before `v`.
        """
        in_degree = [len(self._parents[v]) for v in range(self.num_nodes)]
        ready = [v for v in range(self.num_nodes) if in_degree[v] == 0]
        heapq.heapify(ready)
        order: list[int] = []
        while ready:
            u = heapq.heappop(ready)
            order.append(u)
            for v in sorted(self._children[u]):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    heapq.heappush(ready, v)
        return order

    def ancestors(self, v: int) -> frozenset[int]:
        """Return all ancestors of node `v`.

        Parameters
        ----------
        v : int
            Node index.

        Returns
        -------
        frozenset[int]
            All nodes `u != v` from which there is a directed path to `v`.
        """
        visited: set[int] = set()
        stack = list(self._parents[v])
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            stack.extend(self._parents[u])
        return frozenset(visited)

    def has_path(self, u: int, v: int) -> bool:
        """Return whether a directed path `u -> ... -> v` exists.

        Parameters
        ----------
        u : int
            Source node index.
        v : int
            Target node index.

        Returns
        -------
        bool
            ``True`` iff there is a nonempty directed path from `u` to `v`.
            Always ``False`` when `u == v`, since a self-loop is not a
            valid path in an acyclic graph.
        """
        if u == v:
            return False
        visited: set[int] = set()
        stack = list(self._children[u])
        while stack:
            w = stack.pop()
            if w == v:
                return True
            if w in visited:
                continue
            visited.add(w)
            stack.extend(self._children[w])
        return False

    def copy(self) -> DAG:
        """Return an independent deep copy of this graph.

        Returns
        -------
        DAG
            A new `DAG` with the same nodes and edges, sharing no mutable
            state with the original.
        """
        clone = DAG(self.num_nodes)
        clone._parents = [set(p) for p in self._parents]
        clone._children = [set(c) for c in self._children]
        clone.num_edges = self.num_edges
        return clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DAG):
            return NotImplemented
        return self.num_nodes == other.num_nodes and set(self.edges()) == set(other.edges())

    def __repr__(self) -> str:
        return f"DAG(num_nodes={self.num_nodes}, num_edges={self.num_edges})"

    @classmethod
    def from_edges(cls, num_nodes: int, edges: Iterable[tuple[int, int]]) -> DAG:
        """Build a `DAG` from a node count and an iterable of edges.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        edges : Iterable[tuple[int, int]]
            Edges `(u, v)` to add, in order.

        Returns
        -------
        DAG
            A new `DAG` with the given edges added.

        Raises
        ------
        IndexError
            If an edge names a node outside ``0..num_nodes-1``.
        CycleError
            If adding the edges in order would create a cycle or a
            self-loop.
        """
        dag = cls(num_nodes)
        for u, v in edges:
            dag._check_node(u)
            dag._check_node(v)
            if u == v:
                raise CycleError(f"self-loop rejected: {u} -> {v}")
            if v not in dag._children[u]:
                dag._children[u].add(v)
                dag._parents[v].add(u)
                dag.num_edges += 1
        if len(dag.topological_order()) < num_nodes:
            raise CycleError("the edges contain a directed cycle")
        return dag

    @classmethod
    def from_amat(cls, A: npt.ArrayLike) -> DAG:
        """Build a `DAG` from a square boolean (or 0/1) adjacency matrix.

        Parameters
        ----------
        A : np.ndarray
            Square matrix where a truthy `A[u, v]` denotes the edge
            `u -> v`.

        Returns
        -------
        DAG
            A new `DAG` with edges taken from the nonzero entries of `A`.

        Raises
        ------
        ValueError
            If `A` is not a square matrix.
        CycleError
            If `A` encodes a self-loop or a cycle.
        """
        matrix = np.asarray(A)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
        n = matrix.shape[0]
        edges = [(int(u), int(v)) for u, v in zip(*np.nonzero(matrix), strict=True)]
        return cls.from_edges(n, edges)
