"""The CPDAG of a DAG's Markov equivalence class, and Meek's orientation rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from causalbase.graph._kernels import cpdag_of_dag
from causalbase.graph.pdag import PDAG

if TYPE_CHECKING:
    from causalbase.graph.dag import DAG

__all__ = ["apply_meek_rules", "dag_to_cpdag"]


def dag_to_cpdag(dag: DAG) -> PDAG:
    """Return the CPDAG of `dag`: its compelled edges directed, the rest undirected.

    Uses Chickering's (1995) edge-labelling algorithm.
    """
    return PDAG.from_adjacency(cpdag_of_dag(dag.adjacency().view(np.uint8)))


def _common_adjacents(pdag: PDAG, x: int, y: int) -> set[int]:
    """Return nodes adjacent to both `x` and `y`."""
    return set(pdag.adjacent(x) & pdag.adjacent(y))


def _direct(pdag: PDAG, a: int, c: int, prevent_cycles: bool = False) -> bool:
    """Orient the undirected edge `a - c` as `a -> c`.

    Returns ``True`` iff `a - c` was undirected and has now been oriented.
    """
    if not pdag.has_undirected(a, c):
        return False
    if prevent_cycles and _directed_path(pdag, c, a):
        return False
    pdag.orient(a, c)
    return True


def _directed_path(pdag: PDAG, source: int, target: int) -> bool:
    """Return whether directed edges lead from `source` to `target`."""
    stack, seen = [source], {source}
    while stack:
        node = stack.pop()
        if node == target:
            return True
        for nxt in pdag._children_set(node):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _meek_r1(pdag: PDAG, b: int, c: int, pc: bool = False) -> bool:
    """Meek R1: if `a -> b`, `b - c`, and `a` not adjacent to `c`, then `b -> c`."""
    for a in sorted(pdag.parents(b)):
        if a in pdag.adjacent(c):
            continue
        if _direct(pdag, b, c, pc):
            return True
    return False


def _meek_r2(pdag: PDAG, a: int, c: int, pc: bool = False) -> bool:
    """Meek R2: if `a -> b -> c` and `a - c`, then `a -> c` (checked both ways)."""
    oriented = False
    for b in sorted(_common_adjacents(pdag, a, c)):
        if pdag.has_directed(a, b) and pdag.has_directed(b, c) and _direct(pdag, a, c, pc):
            oriented = True
        if pdag.has_directed(c, b) and pdag.has_directed(b, a) and _direct(pdag, c, a, pc):
            oriented = True
    return oriented


def _r3_helper(pdag: PDAG, a: int, d: int, b: int, c: int, pc: bool = False) -> bool:
    b4 = pdag.has_undirected(d, a)
    b5 = pdag.has_undirected(d, b)
    b6 = pdag.has_undirected(d, c)
    b7 = pdag.has_directed(b, a)
    b8 = pdag.has_directed(c, a)
    if b4 and b5 and b6 and b7 and b8:
        return _direct(pdag, d, a, pc)
    return False


def _meek_r3(pdag: PDAG, d: int, a: int, pc: bool = False) -> bool:
    """Meek R3: if `d - a`, `d - b`, `d - c`, `b -> a`, `c -> a`, `b` not adj `c`, then `d -> a`."""
    common = sorted(_common_adjacents(pdag, a, d))
    if len(common) < 2:
        return False
    oriented = False
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            b = common[i]
            c = common[j]
            if c not in pdag.adjacent(b) and _r3_helper(pdag, a, d, b, c, pc):
                oriented = True
    return oriented


def _meek_r4(pdag: PDAG, a: int, b: int, pc: bool = False) -> bool:
    """Meek R4: orient `a -> b` when a directed chain forces it.

    Given the undirected edge `a - b`, for each parent `c` of `b` and each
    common adjacent `d` of `a` and `c` (``d != b``) with `d` not adjacent to
    `b`, `d -> c` directed, and `a - d` undirected, orient `a -> b`.
    """
    oriented = False
    for c in sorted(pdag.parents(b)):
        adj = _common_adjacents(pdag, a, c)
        adj.discard(b)
        for d in sorted(adj):
            if d in pdag.adjacent(b):
                continue
            if not pdag.has_directed(d, c):
                continue
            if not pdag.has_undirected(a, d):
                continue
            if _direct(pdag, a, b, pc):
                oriented = True
    return oriented


def apply_meek_rules(pdag: PDAG, prevent_cycles: bool = False) -> None:
    """Close `pdag` under Meek's rules R1-R4 in place, to a fixpoint.

    Every scan runs in ascending index order, so the result never depends on
    set iteration order.

    Parameters
    ----------
    pdag : PDAG
        The graph to orient. Mutated in place; undirected edges compelled by the
        rules become directed.
    prevent_cycles : bool, default=False
        Leave an edge undirected when orienting it would close a directed
        cycle. The rules never do that on a PDAG with a consistent extension;
        on one assembled from votes, which need not have one, they can.
    """
    pc = prevent_cycles
    oriented = True
    while oriented:
        oriented = False
        for x, y in pdag.undirected_edges():
            # The edge may have been oriented earlier in this pass.
            if not pdag.has_undirected(x, y):
                continue
            if (
                _meek_r1(pdag, x, y, pc)
                or _meek_r1(pdag, y, x, pc)
                or _meek_r2(pdag, x, y, pc)
                or _meek_r2(pdag, y, x, pc)
                or _meek_r3(pdag, x, y, pc)
                or _meek_r3(pdag, y, x, pc)
                or _meek_r4(pdag, x, y, pc)
                or _meek_r4(pdag, y, x, pc)
            ):
                oriented = True
