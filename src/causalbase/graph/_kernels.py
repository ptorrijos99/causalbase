"""Compiled graph routines over adjacency matrices.

A matrix ``A`` encodes a PDAG: ``A[u, v] = 1`` and ``A[v, u] = 0`` is the edge
``u -> v``, and both set is the undirected edge ``u - v``.

The routines that take `nbr` and `deg` also read the skeleton as padded
neighbour lists: ``nbr[v, :deg[v]]`` are the nodes adjacent to `v`, in no
particular order. They cost time in the number of edges, not in ``n ** 2``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "complete_marks",
    "consistent_extension",
    "cpdag_of_dag",
    "link",
    "semidirected_reach_lists",
    "unlink",
]


@njit(cache=True)
def _neighbour_lists(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return CSR lists of each node's neighbours in `adj`, by any mark, in index order.

    Reads `adj` row by row and looks at ``adj[y, x]`` only for the few marks set,
    so the cost is one contiguous pass over the matrix plus the edges.
    """
    n = adj.shape[0]
    indptr = np.zeros(n + 1, dtype=np.int64)
    for x in range(n):
        row = adj[x]
        for y in range(n):
            if row[y] and (x < y or not adj[y, x]):  # each adjacent pair once
                indptr[x + 1] += 1
                indptr[y + 1] += 1
    for x in range(n):
        indptr[x + 1] += indptr[x]
    nbr = np.empty(indptr[n], dtype=np.int64)
    fill = indptr[:-1].copy()
    for x in range(n):
        row = adj[x]
        for y in range(n):
            if row[y] and (x < y or not adj[y, x]):
                nbr[fill[x]] = y
                fill[x] += 1
                nbr[fill[y]] = x
                fill[y] += 1
    for x in range(n):
        nbr[indptr[x] : indptr[x + 1]].sort()
    return indptr, nbr


@njit(cache=True)
def consistent_extension(adj: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a DAG extending the PDAG `adj` (Dor & Tarsi, 1992), and whether one exists.

    Repeatedly removes a sink `x`: a node with no outgoing directed edge whose
    undirected neighbours are each adjacent to every other neighbour of `x`,
    orienting its undirected edges into it. The sink removed is always the
    lowest-numbered one. `adj` is not modified.
    """
    n = adj.shape[0]
    dag = np.zeros((n, n), dtype=np.uint8)
    removed = np.zeros(n, dtype=np.bool_)
    indptr, nbr = _neighbour_lists(adj)
    # Outgoing directed edges of each node, kept up to date as sinks go.
    out = np.zeros(n, dtype=np.int64)
    for x in range(n):
        for a in range(indptr[x], indptr[x + 1]):
            y = nbr[a]
            if adj[x, y] and not adj[y, x]:
                out[x] += 1
    for _ in range(n):
        sink = -1
        for x in range(n):
            if removed[x] or out[x] > 0:
                continue
            ok = True
            for a in range(indptr[x], indptr[x + 1]):
                y = nbr[a]
                if removed[y] or not adj[x, y]:  # Gone, or a parent: no condition.
                    continue
                for b in range(indptr[x], indptr[x + 1]):
                    z = nbr[b]
                    if z != y and not removed[z] and not (adj[y, z] or adj[z, y]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sink = x
                break
        if sink < 0:
            return dag, False
        for a in range(indptr[sink], indptr[sink + 1]):
            y = nbr[a]
            if removed[y]:
                continue
            if adj[y, sink]:
                dag[y, sink] = 1
                if not adj[sink, y]:
                    out[y] -= 1
        removed[sink] = True
    return dag, True


@njit(cache=True)
def cpdag_of_dag(dag: np.ndarray) -> np.ndarray:
    """Return the CPDAG of `dag`, by Chickering's (1995) edge labelling."""
    n = dag.shape[0]
    # Parents of each node in index order, and children for the topological sort.
    pptr = np.zeros(n + 1, dtype=np.int64)
    cptr = np.zeros(n + 1, dtype=np.int64)
    for u in range(n):
        for v in range(n):
            if dag[u, v]:
                pptr[v + 1] += 1
                cptr[u + 1] += 1
    for v in range(n):
        pptr[v + 1] += pptr[v]
        cptr[v + 1] += cptr[v]
    pa = np.empty(pptr[n], dtype=np.int64)
    ch = np.empty(cptr[n], dtype=np.int64)
    pfill = pptr[:-1].copy()
    cfill = cptr[:-1].copy()
    for u in range(n):
        for v in range(n):
            if dag[u, v]:
                pa[pfill[v]] = u
                pfill[v] += 1
                ch[cfill[u]] = v
                cfill[u] += 1

    indegree = np.empty(n, dtype=np.int64)
    for v in range(n):
        indegree[v] = pptr[v + 1] - pptr[v]
    order = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for v in range(n):
        if indegree[v] == 0:
            order[tail] = v
            tail += 1
    while head < tail:
        u = order[head]
        head += 1
        for c in range(cptr[u], cptr[u + 1]):
            v = ch[c]
            indegree[v] -= 1
            if indegree[v] == 0:
                order[tail] = v
                tail += 1
    position = np.empty(n, dtype=np.int64)
    for i in range(n):
        position[order[i]] = i

    # 0 unlabelled, 1 compelled, 2 reversible
    label = np.zeros((n, n), dtype=np.int8)
    ranked = np.empty(n, dtype=np.int64)
    for y in order:
        lo, hi = pptr[y], pptr[y + 1]
        # Parents of y, latest in the order first: the order edges are labelled in.
        k = hi - lo
        for i in range(k):
            ranked[i] = pa[lo + i]
        keys = np.empty(k, dtype=np.int64)
        for i in range(k):
            keys[i] = -position[ranked[i]]
        idx = np.argsort(keys)
        for e in range(k):
            x = ranked[idx[e]]
            if label[x, y] != 0:
                continue
            settled = False
            for w_i in range(pptr[x], pptr[x + 1]):
                w = pa[w_i]
                if label[w, x] != 1:
                    continue
                if not dag[w, y]:
                    for z_i in range(lo, hi):
                        label[pa[z_i], y] = 1
                    settled = True
                    break
                label[w, y] = 1
            if settled:
                continue
            forced = False
            for z_i in range(lo, hi):
                z = pa[z_i]
                if z != x and not dag[z, x]:
                    forced = True
                    break
            value = 1 if forced else 2
            label[x, y] = value
            for z_i in range(lo, hi):
                z = pa[z_i]
                if label[z, y] == 0:
                    label[z, y] = value
    out = np.zeros((n, n), dtype=np.uint8)
    for v in range(n):
        for p in range(pptr[v], pptr[v + 1]):
            u = pa[p]
            out[u, v] = 1
            if label[u, v] == 2:
                out[v, u] = 1
    return out


@njit(cache=True)
def link(nbr: np.ndarray, deg: np.ndarray, u: int, v: int) -> None:
    """Add the adjacency of `u` and `v` to the neighbour lists, which must have room."""
    nbr[u, deg[u]] = v
    deg[u] += 1
    nbr[v, deg[v]] = u
    deg[v] += 1


@njit(cache=True)
def unlink(nbr: np.ndarray, deg: np.ndarray, u: int, v: int) -> None:
    """Remove the adjacency of `u` and `v` from the neighbour lists."""
    for a, b in ((u, v), (v, u)):
        last = deg[a] - 1
        for i in range(last + 1):
            if nbr[a, i] == b:
                nbr[a, i] = nbr[a, last]
                deg[a] = last
                break


@njit(cache=True)
def semidirected_reach_lists(
    adj: np.ndarray, nbr: np.ndarray, deg: np.ndarray, source: int, avoid: np.ndarray
) -> np.ndarray:
    """Return which nodes a semi-directed path from `source` reaches in the PDAG `adj`.

    Nodes listed in `avoid` are reached but not passed through.
    """
    n = adj.shape[0]
    blocked = np.zeros(n, dtype=np.bool_)
    for v in avoid:
        blocked[v] = True
    reached = np.zeros(n, dtype=np.bool_)
    stack = np.empty(n, dtype=np.int64)
    stack[0] = source
    top = 1
    reached[source] = True
    while top > 0:
        top -= 1
        v = stack[top]
        for k in range(deg[v]):
            w = nbr[v, k]
            if adj[v, w] and not reached[w]:
                reached[w] = True
                if not blocked[w]:
                    stack[top] = w
                    top += 1
    return reached


@njit(cache=True)
def complete_marks(adj: np.ndarray, nbr: np.ndarray, deg: np.ndarray) -> tuple[np.ndarray, bool]:
    """Turn the PDAG `adj` into the CPDAG of the class it extends to, in place.

    Returns the pairs ``(u, v)``, ``u < v``, whose marks changed, in increasing
    order, and whether `adj` has a consistent extension; without one `adj` is
    left as it was. Only marks change, so `nbr` and `deg` stay valid.

    A consistent extension is found by removing sinks (Dor & Tarsi, 1992) and
    its edges are labelled compelled or reversible (Chickering, 1995). An edge
    of the extension points from the endpoint removed later, so the extension
    is never stored: the removal order alone gives its parents and a
    topological order. The CPDAG does not depend on which extension is used.
    """
    n = adj.shape[0]
    cap = nbr.shape[1]

    # Outgoing directed edges of each node, kept up to date as sinks go.
    out = np.zeros(n, dtype=np.int64)
    for x in range(n):
        for a in range(deg[x]):
            y = nbr[x, a]
            if adj[x, y] and not adj[y, x]:
                out[x] += 1
    # A node is tested when it leaves the stack, and enters it again whenever a
    # neighbour is removed: nothing else changes whether it is a sink.
    removed = np.zeros(n, dtype=np.bool_)
    queued = np.ones(n, dtype=np.bool_)
    rank = np.empty(n, dtype=np.int64)  # position in the removal order
    by_rank = np.empty(n, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)
    for x in range(n):
        stack[x] = n - 1 - x
    top = n
    n_removed = 0
    while top > 0:
        top -= 1
        x = stack[top]
        queued[x] = False
        if out[x] > 0:
            continue
        ok = True
        for a in range(deg[x]):
            y = nbr[x, a]
            if removed[y] or not adj[x, y]:  # Gone, or a parent: no condition.
                continue
            for b in range(deg[x]):
                z = nbr[x, b]
                if z != y and not removed[z] and not (adj[y, z] or adj[z, y]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        removed[x] = True
        rank[x] = n_removed
        by_rank[n_removed] = x
        n_removed += 1
        for a in range(deg[x]):
            y = nbr[x, a]
            if removed[y]:
                continue
            if not adj[x, y]:
                out[y] -= 1
            if not queued[y]:
                queued[y] = True
                stack[top] = y
                top += 1
    if n_removed < n:
        return np.empty((0, 2), dtype=np.int64), False

    # label[y, a] is the label of the edge nbr[y, a] -> y of the extension:
    # 0 unlabelled, 1 compelled, 2 reversible.
    label = np.zeros((n, cap), dtype=np.int8)
    slots = np.empty(cap, dtype=np.int64)
    for i in range(n - 1, -1, -1):  # a topological order of the extension
        y = by_rank[i]
        # Parents of y, latest in the order first: the order edges are labelled in.
        k = 0
        for a in range(deg[y]):
            key = rank[nbr[y, a]]
            if key < i:
                continue
            j = k
            while j > 0 and rank[nbr[y, slots[j - 1]]] > key:
                slots[j] = slots[j - 1]
                j -= 1
            slots[j] = a
            k += 1
        for e in range(k):
            a = slots[e]
            if label[y, a] != 0:
                continue
            x = nbr[y, a]
            settled = False
            for b in range(deg[x]):
                w = nbr[x, b]
                if label[x, b] != 1 or rank[w] < rank[x]:
                    continue
                # w -> x is compelled; w is a parent of y exactly when adjacent to it.
                c = -1
                for j in range(deg[y]):
                    if nbr[y, j] == w:
                        c = j
                        break
                if c < 0:
                    for j in range(k):
                        label[y, slots[j]] = 1
                    settled = True
                    break
                label[y, c] = 1
            if settled:
                continue
            forced = False
            for j in range(k):
                z = nbr[y, slots[j]]
                if z != x and not ((adj[z, x] or adj[x, z]) and rank[z] > rank[x]):
                    forced = True
                    break
            value = 1 if forced else 2
            for j in range(k):
                if label[y, slots[j]] == 0:
                    label[y, slots[j]] = value

    count = 0
    for y in range(n):
        for a in range(deg[y]):
            x = nbr[y, a]
            if rank[x] > rank[y] and (not adj[x, y] or (adj[y, x] != 0) != (label[y, a] == 2)):
                count += 1
    codes = np.empty(count, dtype=np.int64)
    count = 0
    for y in range(n):
        for a in range(deg[y]):
            x = nbr[y, a]
            if rank[x] > rank[y] and (not adj[x, y] or (adj[y, x] != 0) != (label[y, a] == 2)):
                adj[x, y] = 1
                adj[y, x] = 1 if label[y, a] == 2 else 0
                codes[count] = min(x, y) * n + max(x, y)
                count += 1
    codes.sort()
    pairs = np.empty((count, 2), dtype=np.int64)
    for i in range(count):
        pairs[i, 0] = codes[i] // n
        pairs[i, 1] = codes[i] % n
    return pairs, True
