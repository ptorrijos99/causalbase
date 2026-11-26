"""Partially directed acyclic graph: the working representation for CPDAGs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from causalbase.graph._kernels import (
    complete_marks,
    consistent_extension,
    link,
    unlink,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from causalbase.graph.dag import DAG

__all__ = ["PDAG"]


class PDAG:
    """A partially directed acyclic graph over integer nodes ``0..num_nodes-1``.

    A PDAG carries two kinds of edges: *directed* edges ``u -> v`` and
    *undirected* edges ``{u, v}``. At most one edge (of either kind, in either
    orientation) is stored between any pair of nodes.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph. Nodes are addressed by the integer
        indices ``0, 1, ..., num_nodes - 1``.

    Attributes
    ----------
    num_nodes : int
        Number of nodes in the graph.

    Notes
    -----
    Unlike :class:`DAG`, ``add_directed`` does not check for cycles. A directed
    cycle surfaces only when :meth:`to_dag` finds no consistent extension.
    """

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self._parents: list[set[int]] = [set() for _ in range(num_nodes)]
        self._children: list[set[int]] = [set() for _ in range(num_nodes)]
        self._undirected: list[set[int]] = [set() for _ in range(num_nodes)]
        # Built on first use, then edited by every mutation.
        self._amat: npt.NDArray[np.bool_] | None = None
        self._nbr: npt.NDArray[np.int32] | None = None
        self._deg: npt.NDArray[np.int32] | None = None

    def _link(self, u: int, v: int) -> None:
        if self._nbr is None or self._deg is None:
            return
        cap = self._nbr.shape[1]
        if self._deg[u] == cap or self._deg[v] == cap:
            self._nbr = np.concatenate([self._nbr, np.zeros_like(self._nbr)], axis=1)
        link(self._nbr, self._deg, u, v)

    def parents(self, v: int) -> frozenset[int]:
        """Return the parents of node `v`."""
        return frozenset(self._parents[v])

    def children(self, v: int) -> frozenset[int]:
        """Return the children of node `v`."""
        return frozenset(self._children[v])

    def undirected_neighbors(self, v: int) -> frozenset[int]:
        """Return the undirected neighbours of node `v`."""
        return frozenset(self._undirected[v])

    def adjacent(self, v: int) -> frozenset[int]:
        """Return the nodes adjacent to `v` (parents, children, and undirected)."""
        return frozenset(self._parents[v] | self._children[v] | self._undirected[v])

    # Allocation-free accessors for the search's inner loops, where copying into
    # a frozenset on every call dominates the runtime. The first three return
    # the graph's own storage: never mutate it, and never let it escape.

    def _children_set(self, v: int) -> set[int]:
        return self._children[v]

    def _parents_set(self, v: int) -> set[int]:
        return self._parents[v]

    def _undirected_set(self, v: int) -> set[int]:
        return self._undirected[v]

    def _adjacent_set(self, v: int) -> set[int]:
        # A fresh set, cheaper than `adjacent`, which also builds a frozenset.
        return self._parents[v] | self._children[v] | self._undirected[v]

    def has_directed(self, u: int, v: int) -> bool:
        """Return whether the directed edge `u -> v` is present."""
        return v in self._children[u]

    def has_undirected(self, u: int, v: int) -> bool:
        """Return whether the undirected edge `{u, v}` is present."""
        return v in self._undirected[u]

    def add_directed(self, u: int, v: int) -> None:
        """Add the directed edge `u -> v`.

        Adding an edge that already exists (same orientation) is a no-op. No
        cycle checking is performed; consistency is enforced only by
        :meth:`to_dag`.

        Parameters
        ----------
        u : int
            Source node index.
        v : int
            Target node index.

        Raises
        ------
        ValueError
            If `u == v` (self-loop), if the opposite directed edge `v -> u`
            already exists, or if the undirected edge `{u, v}` already
            exists -- at most one edge is allowed between any pair of nodes.
        """
        if u == v:
            raise ValueError(f"self-loop rejected: {u} -> {v}")
        if v in self._children[u]:
            return
        if u in self._children[v]:
            raise ValueError(f"opposite directed edge already exists: {v} -> {u}")
        if v in self._undirected[u]:
            raise ValueError(f"undirected edge already exists: {u} - {v}")
        self._children[u].add(v)
        self._parents[v].add(u)
        if self._amat is not None:
            self._amat[u, v] = True
        self._link(u, v)

    def add_undirected(self, u: int, v: int) -> None:
        """Add the undirected edge `{u, v}`.

        Adding an edge that already exists is a no-op.

        Parameters
        ----------
        u : int
            One endpoint index.
        v : int
            Other endpoint index.

        Raises
        ------
        ValueError
            If `u == v` (self-loop), or if a directed edge already exists
            between `u` and `v` (in either orientation) -- at most one edge
            is allowed between any pair of nodes.
        """
        if u == v:
            raise ValueError(f"self-loop rejected: {u} - {v}")
        if v in self._undirected[u]:
            return
        if v in self._children[u] or u in self._children[v]:
            raise ValueError(f"directed edge already exists between {u} and {v}")
        self._undirected[u].add(v)
        self._undirected[v].add(u)
        if self._amat is not None:
            self._amat[u, v] = self._amat[v, u] = True
        self._link(u, v)

    def orient(self, u: int, v: int) -> None:
        """Orient the undirected edge `{u, v}` as the directed edge `u -> v`.

        Parameters
        ----------
        u : int
            Node the oriented edge points *from*.
        v : int
            Node the oriented edge points *to*.

        Raises
        ------
        KeyError
            If `{u, v}` is not an undirected edge of the graph.
        """
        if v not in self._undirected[u]:
            raise KeyError(f"no such undirected edge: {u} - {v}")
        self._undirected[u].remove(v)
        self._undirected[v].remove(u)
        self._children[u].add(v)
        self._parents[v].add(u)
        if self._amat is not None:
            self._amat[v, u] = False

    def remove_edge(self, u: int, v: int) -> None:
        """Remove the edge between `u` and `v`, of either kind and orientation.

        Parameters
        ----------
        u : int
            One endpoint index.
        v : int
            Other endpoint index.

        Raises
        ------
        KeyError
            If there is no edge (directed either way, or undirected) between
            `u` and `v`.
        """
        if v in self._children[u]:
            self._children[u].remove(v)
            self._parents[v].remove(u)
        elif u in self._children[v]:
            self._children[v].remove(u)
            self._parents[u].remove(v)
        elif v in self._undirected[u]:
            self._undirected[u].remove(v)
            self._undirected[v].remove(u)
        else:
            raise KeyError(f"no such edge: {u} - {v}")
        if self._amat is not None:
            self._amat[u, v] = self._amat[v, u] = False
        if self._nbr is not None and self._deg is not None:
            unlink(self._nbr, self._deg, u, v)

    def _complete_in_place(self) -> npt.NDArray[np.int64]:
        """Make this PDAG the CPDAG of the equivalence class it extends to, in place.

        Returns the pairs ``(u, v)``, ``u < v``, whose marks changed, in
        increasing order. The work is proportional to the number of edges.

        Raises
        ------
        ValueError
            If this PDAG has no consistent extension; it is then left as it was.
        """
        amat = self._adjacency()
        nbr, deg = self._neighbour_lists()
        pairs: npt.NDArray[np.int64]
        pairs, ok = complete_marks(amat.view(np.uint8), nbr, deg)
        if not ok:
            raise ValueError("the graph has no consistent extension")
        for u, v in pairs.tolist():
            for a, b in ((u, v), (v, u)):
                self._undirected[a].discard(b)
                self._children[a].discard(b)
                self._parents[a].discard(b)
            if amat[u, v] and amat[v, u]:
                self._undirected[u].add(v)
                self._undirected[v].add(u)
            elif amat[u, v]:
                self._children[u].add(v)
                self._parents[v].add(u)
            else:
                self._children[v].add(u)
                self._parents[u].add(v)
        return pairs

    def directed_edges(self) -> list[tuple[int, int]]:
        """Return all directed edges, sorted."""
        return sorted((u, v) for u in range(self.num_nodes) for v in self._children[u])

    def undirected_edges(self) -> list[tuple[int, int]]:
        """Return all undirected edges as `(u, v)` pairs with `u < v`, sorted."""
        return sorted((u, v) for u in range(self.num_nodes) for v in self._undirected[u] if u < v)

    def adjacency(self) -> npt.NDArray[np.bool_]:
        """Return the boolean adjacency matrix of the graph.

        An undirected edge `{u, v}` sets both `A[u, v]` and `A[v, u]`; a
        directed edge `u -> v` sets only `A[u, v]`.

        Returns
        -------
        np.ndarray
            A copy of the cached boolean adjacency matrix.
        """
        return self._adjacency().copy()

    def _adjacency(self) -> npt.NDArray[np.bool_]:
        if self._amat is None:
            amat = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
            for u in range(self.num_nodes):
                for v in self._children[u]:
                    amat[u, v] = True
                for v in self._undirected[u]:
                    amat[u, v] = True
            self._amat = amat
        return self._amat

    def _neighbour_lists(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Return the live padded lists of adjacent nodes (read-only; do not mutate).

        ``nbr[v, :deg[v]]`` are the nodes adjacent to ``v`` by any edge, in no
        particular order. Later mutations of the graph show in `deg` and, until
        a row outgrows its width, in `nbr`: ask again after adding edges.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(nbr, deg)``, both ``int32``.
        """
        if self._nbr is None or self._deg is None:
            rows = [
                sorted(self._parents[v] | self._children[v] | self._undirected[v])
                for v in range(self.num_nodes)
            ]
            deg = np.array([len(row) for row in rows], dtype=np.int32)
            nbr = np.zeros((self.num_nodes, max(4, 2 * max(map(len, rows), default=0))), np.int32)
            for v, row in enumerate(rows):
                nbr[v, : len(row)] = row
            self._nbr, self._deg = nbr, deg
        return self._nbr, self._deg

    def copy(self) -> PDAG:
        """Return an independent deep copy of this graph.

        Returns
        -------
        PDAG
            A new `PDAG` with the same nodes and edges, sharing no mutable
            state with the original.
        """
        clone = PDAG(self.num_nodes)
        clone._parents = [set(p) for p in self._parents]
        clone._children = [set(c) for c in self._children]
        clone._undirected = [set(u) for u in self._undirected]
        # The derived arrays, when built, are copied rather than rebuilt in Python.
        if self._amat is not None:
            clone._amat = self._amat.copy()
        if self._nbr is not None and self._deg is not None:
            clone._nbr, clone._deg = self._nbr.copy(), self._deg.copy()
        return clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PDAG):
            return NotImplemented
        # The parent and undirected sets fix every edge, and compare without sorting.
        return (
            self.num_nodes == other.num_nodes
            and self._parents == other._parents
            and self._undirected == other._undirected
        )

    def __hash__(self) -> int:
        return hash((self.num_nodes, tuple(self.directed_edges()), tuple(self.undirected_edges())))

    def __repr__(self) -> str:
        n_dir = sum(len(c) for c in self._children)
        n_undir = sum(len(u) for u in self._undirected) // 2
        return f"PDAG(num_nodes={self.num_nodes}, directed={n_dir}, undirected={n_undir})"

    @classmethod
    def from_dag(cls, dag: DAG) -> PDAG:
        """Build a `PDAG` from a `DAG`, keeping every edge directed."""
        pdag = cls(dag.num_nodes)
        for u, v in dag.edges():
            pdag.add_directed(u, v)
        return pdag

    @classmethod
    def from_adjacency(cls, adj: npt.ArrayLike) -> PDAG:
        """Build a PDAG from a square matrix: ``A[u, v]`` alone is ``u -> v``, both is ``u - v``."""
        a = np.asarray(adj, dtype=bool)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"expected a square matrix, got shape {a.shape}")
        pdag = cls(a.shape[0])
        for u, v in zip(*np.nonzero(a & ~a.T), strict=True):
            pdag._children[u].add(int(v))
            pdag._parents[v].add(int(u))
        for u, v in zip(*np.nonzero(np.triu(a & a.T)), strict=True):
            pdag._undirected[u].add(int(v))
            pdag._undirected[v].add(int(u))
        return pdag

    def to_dag(self) -> DAG:
        """Return a DAG in the equivalence class this PDAG describes (Dor & Tarsi, 1992).

        Returns
        -------
        DAG
            Keeps every directed edge, orients every undirected one, and adds
            no v-structure.

        Raises
        ------
        ValueError
            If no such DAG exists.
        """
        from causalbase.graph.dag import DAG

        dag, ok = consistent_extension(self._adjacency().view(np.uint8))
        if not ok:
            raise ValueError("this PDAG has no consistent DAG extension")
        return DAG.from_amat(dag)
