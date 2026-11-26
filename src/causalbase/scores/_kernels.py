"""Compiled counting kernels behind the discrete scores.

The likelihood of a family depends only on the multiset of its non-zero cell
counts ``n_jk``, the multiset of its configuration counts ``n_j``, and on ``q``
and ``r``. The kernels collect both multisets as histograms and sum over the
distinct count values in increasing order. A family therefore gets the same
float whichever way its rows were counted and whatever order its parents come
in, so a family scored as ``parents + [x]`` in a batch equals the same family
scored alone, bit for bit, and ``log Gamma`` is evaluated once per distinct
count rather than once per cell.

The rows are grouped once by the configuration of the shared parents. Each
extension `x` is then counted in one pass over the rows into a dense table
when ``G * |x| * r`` cells fit the buffer (``G`` realised configurations), or
group by group in a small ``|x| * r`` table otherwise, never by sorting rows.

A count's term depends only on the count and on the priors ``ess / q`` and
``ess / (q * r)``. Within a batch those are set by the cardinality of `x`, so
the extensions are scored by cardinality and each term is computed once per
cardinality and reused.
"""

from __future__ import annotations

import math

import numpy as np
from llvmlite import ir
from numba import njit, prange, types
from numba.extending import intrinsic

__all__ = ["extension_buffers", "family_extensions", "family_score", "value_bits"]

POPCOUNT_CELLS = 300
"""Count an extension with bitsets when ``G * r * (|x| - 1)`` is at most this."""


@intrinsic
def _popcount(typingctx, x):  # type: ignore[no-untyped-def]  # pragma: no cover - compiled
    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        fn = builder.module.declare_intrinsic("llvm.ctpop", [ir.IntType(64)])
        return builder.call(fn, [args[0]])

    return types.uint64(types.uint64), codegen


@njit(cache=True, nogil=True)
def _groups(
    values: np.ndarray, cards: np.ndarray, parents: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """Return each row's configuration of `parents` as a dense label, with ``q``.

    Also returns the number of labels; ``q`` is the true number of configurations.
    Labels are numbered by first occurrence, one variable at a time, so no code
    ever exceeds ``n_rows * card``, however many configurations there are.
    """
    n = values.shape[0]
    g = np.zeros(n, np.int64)
    n_groups = 1
    q = 1.0
    for p in parents:
        cp = cards[p]
        column = values[:, p]
        ids = np.full(n_groups * cp, -1, np.int64)
        k = 0
        for i in range(n):
            key = g[i] * cp + column[i]
            v = ids[key]
            if v < 0:
                v = k
                ids[key] = v
                k += 1
            g[i] = v
        n_groups = k
        q *= cp
    return g, n_groups, q


@njit(cache=True, nogil=True)
def _by_group(g: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Sort the rows by label (counting sort); return the permutation and each group's start."""
    start = np.zeros(n_groups + 1, np.int64)
    for i in range(g.shape[0]):
        start[g[i] + 1] += 1
    for k in range(n_groups):
        start[k + 1] += start[k]
    pos = start[:-1].copy()
    perm = np.empty(g.shape[0], np.int64)
    for i in range(g.shape[0]):
        perm[pos[g[i]]] = i
        pos[g[i]] += 1
    return perm, start


@njit(cache=True, nogil=True)
def value_bits(values: np.ndarray, cards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bitset of the rows taking each value of each variable, and each variable's first row.

    Row ``offsets[v] + a`` of the result marks the rows where variable `v` takes value `a`.
    """
    n, n_vars = values.shape
    words = (n + 63) // 64
    offsets = np.zeros(n_vars + 1, np.int64)
    for v in range(n_vars):
        offsets[v + 1] = offsets[v] + cards[v]
    bits = np.zeros((offsets[n_vars], words), np.uint64)
    for v in range(n_vars):
        for i in range(n):
            bits[offsets[v] + values[i, v], i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    return bits, offsets


@njit(cache=True, nogil=True)
def _cell_bits(
    g: np.ndarray, n_groups: int, child: np.ndarray, r: int, words: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bitset and count of the rows of every (configuration, child value) cell."""
    bits = np.zeros((n_groups * r, words), np.uint64)
    counts = np.zeros(n_groups * r, np.int64)
    for i in range(g.shape[0]):
        cell = g[i] * r + child[i]
        bits[cell, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
        counts[cell] += 1
    return bits, counts


@njit(cache=True, nogil=True)
def _count_bits(
    n_groups: int,
    r: int,
    cell_bits: np.ndarray,
    cell_counts: np.ndarray,
    x_bits: np.ndarray,
    x_first: int,
    cx: int,
    per_value: np.ndarray,
    h_cell: np.ndarray,
    seen_cell: np.ndarray,
    h_conf: np.ndarray,
    seen_conf: np.ndarray,
) -> tuple[int, int]:
    """:func:`_count` by intersecting each cell's rows with the rows of each value of `x`.

    The last value of `x` takes whatever rows of the cell the others left.
    """
    words = cell_bits.shape[1]
    n_cell = 0
    n_conf = 0
    for gi in range(n_groups):
        for a in range(cx):
            per_value[a] = 0
        for k in range(r):
            cell = gi * r + k
            rest = cell_counts[cell]
            if rest == 0:
                continue
            for a in range(cx):
                if a < cx - 1:
                    c = 0
                    for w in range(words):
                        c += _popcount(cell_bits[cell, w] & x_bits[x_first + a, w])
                    rest -= c
                else:
                    c = rest
                if c > 0:
                    per_value[a] += c
                    n_cell = _push(h_cell, seen_cell, n_cell, c)
        for a in range(cx):
            if per_value[a] > 0:
                n_conf = _push(h_conf, seen_conf, n_conf, per_value[a])
    return n_cell, n_conf


@njit(cache=True, nogil=True, inline="always")
def _push(hist: np.ndarray, seen: np.ndarray, n_seen: int, c: int) -> int:
    if hist[c] == 0:
        seen[n_seen] = c
        n_seen += 1
    hist[c] += 1
    return n_seen


@njit(cache=True, nogil=True, inline="always")
def _cell(c: int, a: float, lg_a: float, mle: bool) -> float:
    """One count's term: ``log Gamma(a + c) - log Gamma(a)``, or ``c log c`` for the MLE."""
    if mle:
        return c * math.log(c)
    return math.lgamma(a + c) - lg_a


@njit(cache=True, nogil=True)
def _sort(a: np.ndarray) -> None:
    """Sort `a` in place; by insertion when short, as the lists of distinct counts mostly are."""
    n = a.shape[0]
    if n > 64:
        a.sort()
        return
    for i in range(1, n):
        v = a[i]
        j = i
        while j > 0 and a[j - 1] > v:
            a[j] = a[j - 1]
            j -= 1
        a[j] = v


@njit(cache=True, nogil=True)
def _likelihood(
    h_cell: np.ndarray,
    seen_cell: np.ndarray,
    n_cell: int,
    h_conf: np.ndarray,
    seen_conf: np.ndarray,
    n_conf: int,
    q: float,
    r: int,
    ess: float,
    terms: np.ndarray,
    stamps: np.ndarray,
    tag: int,
) -> float:
    """Sum the BDeu (or maximised) likelihood over the two histograms, and clear them.

    ``terms[0, c]`` and ``terms[1, c]`` cache the term of a cell and of a
    configuration of count ``c``, valid where `stamps` holds `tag`. Calls that
    share `q`, `r` and `ess` may share a positive `tag`, and only those. Column
    0, which no count uses, holds ``log Gamma`` of the two priors.
    """
    mle = ess == 0.0
    a_j = ess / q
    a_jk = ess / (q * r)
    if stamps[0, 0] != tag:
        terms[0, 0] = 0.0 if mle else math.lgamma(a_jk)
        terms[1, 0] = 0.0 if mle else math.lgamma(a_j)
        stamps[0, 0] = tag
    lg_ajk = terms[0, 0]
    lg_aj = terms[1, 0]
    cells = seen_cell[:n_cell]
    confs = seen_conf[:n_conf]
    _sort(cells)
    _sort(confs)
    total = 0.0
    for c in cells:
        if stamps[0, c] != tag:
            terms[0, c] = _cell(c, a_jk, lg_ajk, mle)
            stamps[0, c] = tag
        total += h_cell[c] * terms[0, c]
        h_cell[c] = 0
    for c in confs:
        if stamps[1, c] != tag:
            terms[1, c] = _cell(c, a_j, lg_aj, mle)
            stamps[1, c] = tag
        total -= h_conf[c] * terms[1, c]
        h_conf[c] = 0
    return total


@njit(cache=True, nogil=True)
def _count(
    g: np.ndarray,
    n_groups: int,
    perm: np.ndarray,
    start: np.ndarray,
    child: np.ndarray,
    child_sorted: np.ndarray,
    r: int,
    x_values: np.ndarray,
    cx: int,
    table: np.ndarray,
    small: np.ndarray,
    touched: np.ndarray,
    per_value: np.ndarray,
    touched_x: np.ndarray,
    h_cell: np.ndarray,
    seen_cell: np.ndarray,
    h_conf: np.ndarray,
    seen_conf: np.ndarray,
) -> tuple[int, int]:
    """Histogram the cells of (configuration, `x`, child); return the distinct counts seen.

    Dense when the ``n_groups * cx * r`` cells fit `table`, else group by group
    over the rows sorted by `perm`, of which `child_sorted` is the child's
    column: the ``cx * r`` cells of a group go in `small` and its ``cx``
    configurations in `per_value`, and only those a row touched are read back.
    `table`, `small` and `per_value` must come in zeroed, and leave zeroed.
    """
    n_cell = 0
    n_conf = 0
    cells = n_groups * cx * r
    if cells <= table.shape[0]:
        for i in range(g.shape[0]):
            table[(g[i] * cx + x_values[i]) * r + child[i]] += 1
        for j in range(n_groups * cx):
            n_j = 0
            for k in range(j * r, j * r + r):
                c = table[k]
                if c > 0:
                    n_cell = _push(h_cell, seen_cell, n_cell, c)
                    n_j += c
                    table[k] = 0
            if n_j > 0:
                n_conf = _push(h_conf, seen_conf, n_conf, n_j)
        return n_cell, n_conf
    for gi in range(n_groups):
        n_touched = 0
        n_x = 0
        for t in range(start[gi], start[gi + 1]):
            a = x_values[perm[t]]
            idx = a * r + child_sorted[t]
            if small[idx] == 0:
                touched[n_touched] = idx
                n_touched += 1
            small[idx] += 1
            if per_value[a] == 0:
                touched_x[n_x] = a
                n_x += 1
            per_value[a] += 1
        for t in range(n_touched):
            n_cell = _push(h_cell, seen_cell, n_cell, small[touched[t]])
            small[touched[t]] = 0
        for t in range(n_x):
            n_conf = _push(h_conf, seen_conf, n_conf, per_value[touched_x[t]])
            per_value[touched_x[t]] = 0
    return n_cell, n_conf


@njit(cache=True, nogil=True)
def family_score(
    values: np.ndarray,
    cards: np.ndarray,
    node: int,
    parents: np.ndarray,
    ess: float,
    scratch: np.ndarray,
) -> float:
    """Return the log likelihood of `node` given `parents`: BDeu, or the MLE when `ess` is 0.

    Parameters
    ----------
    values : np.ndarray
        Integer-coded data, shape ``(n_rows, n_vars)``, column-major so that
        each column is contiguous.
    cards : np.ndarray
        Cardinality of each variable, ``int64``.
    node : int
        The child.
    parents : np.ndarray
        The parents, ``int64``, in any order.
    ess : float
        Equivalent sample size; 0 gives the maximised likelihood.
    scratch : np.ndarray
        ``int64`` work buffer; its length bounds the dense table.
    """
    n = values.shape[0]
    r = cards[node]
    child = values[:, node]
    g, n_groups, q = _groups(values, cards, parents)
    if n_groups * r <= scratch.shape[0]:
        perm = np.empty(0, np.int64)
        start = np.empty(0, np.int64)
        child_sorted = child[:0]
    else:
        perm, start = _by_group(g, n_groups)
        child_sorted = child[perm]
    zeros = np.zeros(n, np.int64)
    table = scratch[: min(scratch.shape[0], n_groups * r)]
    table[:] = 0
    small = np.zeros(r, np.int64)
    touched = np.empty(r, np.int64)
    per_value = np.zeros(1, np.int64)
    touched_x = np.empty(1, np.int64)
    h_cell = np.zeros(n + 1, np.int64)
    h_conf = np.zeros(n + 1, np.int64)
    seen_cell = np.empty(n + 1, np.int64)
    seen_conf = np.empty(n + 1, np.int64)
    n_cell, n_conf = _count(
        g,
        n_groups,
        perm,
        start,
        child,
        child_sorted,
        r,
        zeros,
        1,
        table,
        small,
        touched,
        per_value,
        touched_x,
        h_cell,
        seen_cell,
        h_conf,
        seen_conf,
    )
    terms = np.empty((2, n + 1))
    stamps = np.zeros((2, n + 1), np.int64)
    return _likelihood(
        h_cell, seen_cell, n_cell, h_conf, seen_conf, n_conf, q, r, ess, terms, stamps, 1
    )


def extension_buffers(
    n_rows: int, cards: np.ndarray, table_size: int, chunks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the work buffers :func:`family_extensions` keeps between calls, one row per chunk.

    A row of the first array holds, in this order: a counter of the tags
    handed to :func:`_likelihood`, the dense table (`table_size`), three
    arrays indexed by a value of `x` (the widest cardinality each), the two
    histograms and the two lists of distinct counts (``n_rows + 1`` each), and
    the stamps of the cached terms (``2 * (n_rows + 1)``). The second array
    holds the cached terms.
    """
    top = int(cards.max()) if len(cards) else 1
    ints = np.zeros((chunks, 1 + table_size + 3 * top + 6 * (n_rows + 1)), np.int64)
    return ints, np.zeros((chunks, 2, n_rows + 1))


@njit(cache=True, nogil=True, parallel=True)
def family_extensions(
    values: np.ndarray,
    cards: np.ndarray,
    node: int,
    parents: np.ndarray,
    extra: np.ndarray,
    ess: float,
    table_size: int,
    x_bits: np.ndarray,
    x_offsets: np.ndarray,
    ints: np.ndarray,
    floats: np.ndarray,
) -> np.ndarray:
    """Return :func:`family_score` of `node` given ``parents + [x]``, for each `x` in `extra`.

    The rows are grouped by the configurations of `parents` once and shared;
    the extensions are split into as many chunks as `ints` has rows, scored in
    parallel, each with its own row of the buffers `ints` and `floats` that
    :func:`extension_buffers` made for this `table_size`. The buffers are
    reused from call to call, which spares allocating them per call and
    thread. Every value equals the one :func:`family_score` gives. `x_bits`
    and `x_offsets` come from :func:`value_bits`; small tables are then
    counted with bitsets (pass an empty `x_bits` to always count rows).
    """
    n = values.shape[0]
    m = extra.shape[0]
    r = cards[node]
    child = values[:, node]
    g, n_groups, q_base = _groups(values, cards, parents)
    widest = 1
    for k in range(m):
        widest = max(widest, cards[extra[k]])
    if n_groups * widest * r > table_size:
        perm, start = _by_group(g, n_groups)
        child_sorted = child[perm]
    else:
        perm = np.empty(0, np.int64)
        start = np.empty(0, np.int64)
        child_sorted = child[:0]
    use_bits = x_bits.shape[0] > 0 and n_groups * r <= POPCOUNT_CELLS
    if use_bits:
        cell_bits, cell_counts = _cell_bits(g, n_groups, child, r, x_bits.shape[1])
    else:
        cell_bits = np.zeros((0, 1), np.uint64)
        cell_counts = np.empty(0, np.int64)
    chunks = max(1, min(ints.shape[0], m))
    top = (ints.shape[1] - 1 - table_size - 6 * (n + 1)) // 3
    grouped = perm.shape[0] > 0
    out = np.empty(m)
    for t in prange(chunks):
        lo = t * m // chunks
        hi = (t + 1) * m // chunks
        work = ints[t]
        at = 1
        table = work[at : at + table_size]
        at += table_size
        per_value = work[at : at + top]
        touched_x = work[at + top : at + 2 * top]
        bit_sums = work[at + 2 * top : at + 3 * top]
        at += 3 * top
        h_cell = work[at : at + n + 1]
        h_conf = work[at + n + 1 : at + 2 * (n + 1)]
        seen_cell = work[at + 2 * (n + 1) : at + 3 * (n + 1)]
        seen_conf = work[at + 3 * (n + 1) : at + 4 * (n + 1)]
        stamps = work[at + 4 * (n + 1) : at + 6 * (n + 1)].reshape(2, n + 1)
        terms = floats[t]
        # The cells of one group, needed only when some table is too wide to be dense.
        small = np.zeros(widest * r if grouped else 0, np.int64)
        touched = np.empty(widest * r if grouped else 0, np.int64)
        # By cardinality, so that each one's cached terms serve all its extensions.
        order = np.argsort(cards[extra[lo:hi]])
        last = 0
        for i in range(hi - lo):
            k = lo + order[i]
            x = extra[k]
            cx = cards[x]
            if cx != last:  # other priors: a tag never used with these buffers
                work[0] += 1
                last = cx
            if use_bits and n_groups * r * (cx - 1) <= POPCOUNT_CELLS:
                n_cell, n_conf = _count_bits(
                    n_groups,
                    r,
                    cell_bits,
                    cell_counts,
                    x_bits,
                    x_offsets[x],
                    cx,
                    bit_sums,
                    h_cell,
                    seen_cell,
                    h_conf,
                    seen_conf,
                )
            else:
                n_cell, n_conf = _count(
                    g,
                    n_groups,
                    perm,
                    start,
                    child,
                    child_sorted,
                    r,
                    values[:, x],
                    cx,
                    table,
                    small,
                    touched,
                    per_value,
                    touched_x,
                    h_cell,
                    seen_cell,
                    h_conf,
                    seen_conf,
                )
            out[k] = _likelihood(
                h_cell,
                seen_cell,
                n_cell,
                h_conf,
                seen_conf,
                n_conf,
                q_base * cx,
                r,
                ess,
                terms,
                stamps,
                work[0],
            )
    return out
