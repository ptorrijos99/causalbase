"""The compiled BDeu engine behind the order searches, hill climbing, FLOP and the exact searches.

:class:`CompiledBDeu` computes the BDeu log marginal likelihood of a family
plus the structure prior of a :class:`~causalbase.scores.BDeuScore`, either
the Bernoulli prior of :class:`~causalbase.scores.FamilyScore` or Tetrad
7.6.5's (the reference implementation of the latter is
:mod:`causalbase.scores._tetrad`). It is private: a ``BDeuScore`` builds one
on demand (``score._engine()``) and the searches score through it.

The likelihood of a family depends only on the multiset of non-zero cell
counts ``n_jk``, the multiset of configuration counts ``n_j``, and ``q`` and
``r``. The kernels collect those counts in two histograms and sum over count
values in increasing order, so the value does not depend on how the counts
were obtained: scoring ``parents + {c}`` in a batch gives the same float as
scoring that family alone. The rows of a parent set are grouped once (cached
per family), and each candidate ``c`` is then counted. The default kernel
(``kernel="cells"``) sorts the rows by (configuration, child value) cell and
splits every cell by the value of ``c``:

* with few cells, by intersecting the row bitset of each cell with the bitset
  of each value of ``c``;
* otherwise by walking the rows of the cell into ``cc`` counters, so that the
  work is one step per row plus one per non-empty cell of the family with
  ``c``, whatever the number of configurations.

The previous kernel (``kernel="legacy"``) is kept: bitsets for small tables,
one pass over the rows into a ``G * cc * r`` table that is then read whole, or
one ``cc * r`` table per configuration when that table would be too large.

``lgamma`` values are kept in a table per prior value, as they repeat across
families. Without numba it falls back to the uncompiled scores. It does not
cache scores: :class:`~causalbase.order.OrderScorer` caches families itself.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np

from causalbase.data import DiscreteDataset
from causalbase.graph import DAG
from causalbase.scores._tetrad import TetradBDeuReference
from causalbase.scores.bdeu import PRIORS, BDeuScore, Prior

__all__ = ["HAVE_NUMBA", "CompiledBDeu", "engine_of"]

try:
    from llvmlite import ir
    from numba import config, njit, prange, set_num_threads, types
    from numba.extending import intrinsic

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

if HAVE_NUMBA:

    @intrinsic
    def _popcount(typingctx, x):  # type: ignore[no-untyped-def]  # pragma: no cover - compiled
        def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
            fn = builder.module.declare_intrinsic("llvm.ctpop", [ir.IntType(64)])
            return builder.call(fn, [args[0]])

        return types.uint64(types.uint64), codegen

    @njit(cache=True, nogil=True)
    def _configs(
        cols: np.ndarray, cards: np.ndarray, parents: np.ndarray
    ) -> tuple[np.ndarray, int]:  # pragma: no cover - compiled
        """Return the configuration id of every row (by first occurrence) and their number."""
        n = cols.shape[1]
        g = np.zeros(n, dtype=np.int32)
        G = 1
        for p in parents:
            cp = cards[p]
            col = cols[p]
            ids = np.full(G * cp, -1, dtype=np.int32)
            new_g = 0
            for i in range(n):
                k = g[i] * cp + col[i]
                v = ids[k]
                if v < 0:
                    v = new_g
                    ids[k] = v
                    new_g += 1
                g[i] = v
            G = new_g
        return g, G

    @njit(cache=True, nogil=True)
    def _single(
        cols: np.ndarray,
        cards: np.ndarray,
        node: int,
        parents: np.ndarray,
        ess: float,
        prior: float,
        hist: np.ndarray,
        keys: np.ndarray,
        stamps: np.ndarray,
        table: np.ndarray,
    ) -> float:  # pragma: no cover - compiled
        """Score one family, counted without building the grouping that batches reuse."""
        n = cols.shape[1]
        r = cards[node]
        g, G = _configs(cols, cards, parents)
        child = cols[node]
        ids = np.full(G * r, -1, dtype=np.int32)
        cell = np.zeros(n, dtype=np.int64)
        nj = np.zeros(G, dtype=np.int64)
        n_cells = 0
        for i in range(n):
            k = g[i] * r + child[i]
            v = ids[k]
            if v < 0:
                v = n_cells
                ids[k] = v
                n_cells += 1
            cell[v] += 1
            nj[g[i]] += 1
        hk, hj, vk, vj = hist[0], hist[1], hist[2], hist[3]
        n_cell = 0
        n_conf = 0
        for x in range(n_cells):
            v = cell[x]
            if hk[v] == 0:
                vk[n_cell] = v
                n_cell += 1
            hk[v] += 1
        for x in range(G):
            v = nj[x]
            if hj[v] == 0:
                vj[n_conf] = v
                n_conf += 1
            hj[v] += 1
        qf = 1.0
        for p in parents:
            qf *= cards[p]
        s = _lik(hist, n_cell, n_conf, qf, r, ess, keys, stamps, table, _lg_tag(stamps)) + prior
        return s if math.isfinite(s) else np.nan

    @njit(cache=True, nogil=True)
    def _family(
        cols: np.ndarray,
        cards: np.ndarray,
        node: int,
        parents: np.ndarray,
        n_words: int,
        bit_cells: int,
    ) -> tuple[
        np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:  # pragma: no cover - compiled
        """Group the rows by parent configuration.

        Returns the configuration id of each row (numbered by first
        occurrence), the number ``G`` of realised configurations, the rows
        sorted by configuration with the group starts and their child values,
        the count of every (configuration, child) cell and, when ``G * r`` is
        at most ``bit_cells``, the rows of each cell as a bitset.
        """
        n = cols.shape[1]
        r = cards[node]
        g, G = _configs(cols, cards, parents)
        start = np.zeros(G + 1, dtype=np.int32)
        for i in range(n):
            start[g[i] + 1] += 1
        for k in range(G):
            start[k + 1] += start[k]
        pos = start[:-1].copy()
        perm = np.empty(n, dtype=np.int32)
        for i in range(n):
            perm[pos[g[i]]] = i
            pos[g[i]] += 1
        child = cols[node]
        ch = np.empty(n, dtype=np.int64)
        for t in range(n):
            ch[t] = child[perm[t]]
        njk = np.zeros(G * r, dtype=np.int64)
        for i in range(n):
            njk[g[i] * r + child[i]] += 1
        rows = G * r if G * r <= bit_cells else 0
        bits = np.zeros((rows, n_words), dtype=np.uint64)
        if rows:
            for i in range(n):
                bits[g[i] * r + child[i], i >> 6] |= np.uint64(1) << np.uint64(i & 63)
        return g, G, perm, start, ch, njk, bits

    @njit(cache=True, nogil=True)
    def _lg_row(
        keys: np.ndarray, n_keys: np.ndarray, a: float
    ) -> int:  # pragma: no cover - compiled
        """Return the row of a first-come lgamma table for the prior ``a``, -1 if it is full.

        Used by :mod:`causalbase.exact.dp`.
        """
        for t in range(n_keys[0]):
            if keys[t] == a:
                return t
        t = n_keys[0]
        if t == keys.shape[0]:
            return -1
        keys[t] = a
        n_keys[0] = t + 1
        return t

    @njit(cache=True, nogil=True)
    def _lg_tag(stamps: np.ndarray) -> int:  # pragma: no cover - compiled
        """Return a new tag for the rows of the lgamma table claimed by one batch.

        The tag is kept in the last entry of ``stamps``.
        """
        tag = stamps[stamps.shape[0] - 1] + 1
        stamps[stamps.shape[0] - 1] = tag
        return tag

    @njit(cache=True, nogil=True)
    def _lg_slot(
        keys: np.ndarray, stamps: np.ndarray, table: np.ndarray, a: float, tag: int
    ) -> int:  # pragma: no cover - compiled
        """Return the row of the lgamma table for the prior ``a``.

        The table is direct-mapped on the bits of ``a``: a row held by another
        prior is cleared and taken over, unless that prior claimed it with the
        same ``tag`` (the same batch), in which case -1 is returned and the
        caller computes lgamma directly.
        """
        m, e = math.frexp(a)
        h = np.int64(m * 4503599627370496.0)
        slot = (h ^ (h >> 21) ^ (h >> 42) ^ (e * 40503)) & (keys.shape[0] - 1)
        if keys[slot] == a:
            stamps[slot] = tag
            return slot
        if stamps[slot] == tag:
            return -1
        keys[slot] = a
        stamps[slot] = tag
        for c in range(table.shape[1]):
            table[slot, c] = np.nan
        return slot

    @njit(cache=True, nogil=True)
    def _lg(table: np.ndarray, row: int, a: float, c: int) -> float:  # pragma: no cover - compiled
        """Return ``lgamma(a + c)``, memoised in ``table[row, c]``."""
        if row < 0 or c >= table.shape[1]:
            return math.lgamma(a + c)
        v = table[row, c]
        if v != v:
            v = math.lgamma(a + c)
            table[row, c] = v
        return v

    @njit(cache=True, nogil=True)
    def _sort_prefix(a: np.ndarray, m: int) -> None:  # pragma: no cover - compiled
        if m > 64:
            a[:m].sort()
            return
        for i in range(1, m):
            x = a[i]
            j = i - 1
            while j >= 0 and a[j] > x:
                a[j + 1] = a[j]
                j -= 1
            a[j + 1] = x

    @njit(cache=True, nogil=True)
    def _lik_at(
        hist: np.ndarray,
        n_cell: int,
        n_conf: int,
        a_j: float,
        a_jk: float,
        row_j: int,
        row_jk: int,
        table: np.ndarray,
    ) -> float:  # pragma: no cover - compiled
        """Return the BDeu likelihood from the histograms of cell and configuration counts.

        The histograms are cleared.

        ``hist[0]``/``hist[2]``: cell-count histogram and its distinct values;
        ``hist[1]``/``hist[3]``: the same for configuration counts. ``a_j`` and
        ``a_jk`` are the configuration and cell priors, with their rows of the
        lgamma table.
        """
        _sort_prefix(hist[2], n_cell)  # increasing count values: the sum is order-free
        _sort_prefix(hist[3], n_conf)
        return _lik_sorted(hist, n_cell, n_conf, a_j, a_jk, row_j, row_jk, table)

    @njit(cache=True, nogil=True)
    def _lik_sorted(
        hist: np.ndarray,
        n_cell: int,
        n_conf: int,
        a_j: float,
        a_jk: float,
        row_j: int,
        row_jk: int,
        table: np.ndarray,
    ) -> float:  # pragma: no cover - compiled
        """:func:`_lik_at` when the distinct values of both histograms are already increasing."""
        lg_aj = _lg(table, row_j, a_j, 0)
        lg_ajk = _lg(table, row_jk, a_jk, 0)
        hk, hj, vk, vj = hist[0], hist[1], hist[2], hist[3]
        lik = 0.0
        for x in range(n_cell):
            c = vk[x]
            lik += hk[c] * (_lg(table, row_jk, a_jk, c) - lg_ajk)
            hk[c] = 0
        for x in range(n_conf):
            c = vj[x]
            lik += hj[c] * (lg_aj - _lg(table, row_j, a_j, c))
            hj[c] = 0
        return lik

    @njit(cache=True, nogil=True)
    def _lik(
        hist: np.ndarray,
        n_cell: int,
        n_conf: int,
        qf: float,
        r: int,
        ess: float,
        keys: np.ndarray,
        stamps: np.ndarray,
        table: np.ndarray,
        tag: int,
    ) -> float:  # pragma: no cover - compiled
        """:func:`_lik_at` for a family with ``qf`` configurations (a float, to avoid overflow)."""
        a_j = ess / qf
        a_jk = ess / (qf * r)
        row_j = _lg_slot(keys, stamps, table, a_j, tag)
        row_jk = _lg_slot(keys, stamps, table, a_jk, tag)
        return _lik_at(hist, n_cell, n_conf, a_j, a_jk, row_j, row_jk, table)

    @njit(cache=True, nogil=True)
    def _count(  # pragma: no cover - compiled
        c: int,
        cols: np.ndarray,
        cards: np.ndarray,
        value_bits: np.ndarray,
        offset: np.ndarray,
        node: int,
        g: np.ndarray,
        G: int,
        perm: np.ndarray,
        start: np.ndarray,
        ch: np.ndarray,
        njk: np.ndarray,
        bits: np.ndarray,
        popcount_cells: int,
        dense_cells: int,
        hist: np.ndarray,
        cnt: np.ndarray,
        aux: np.ndarray,
    ) -> tuple[int, int]:
        """Fill the histograms of the cell and configuration counts of the parents plus ``c``.

        Returns how many distinct values each histogram holds.
        """
        n = cols.shape[1]
        r = cards[node]
        child = cols[node]
        n_words = value_bits.shape[1]
        hk, hj, vk, vj = hist[0], hist[1], hist[2], hist[3]
        touched, touched_j, nja = aux[0], aux[1], aux[2]
        cc = cards[c]
        n_cell = 0
        n_conf = 0
        if G * r * (cc - 1) <= popcount_cells and (cc == 1 or bits.shape[0] > 0):
            # The last value of c takes the cell's remaining rows.
            for gi in range(G):
                for a in range(cc):
                    nja[a] = 0
                for k in range(r):
                    jk = gi * r + k
                    rest = njk[jk]
                    if rest == 0:
                        continue
                    for a in range(cc):
                        if a < cc - 1:
                            s = 0
                            row = offset[c] + a
                            for w in range(n_words):
                                s += _popcount(bits[jk, w] & value_bits[row, w])  # type: ignore[call-arg]
                            rest -= s
                        else:
                            s = rest
                        if s:
                            nja[a] += s
                            if hk[s] == 0:
                                vk[n_cell] = s
                                n_cell += 1
                            hk[s] += 1
                for a in range(cc):
                    s = nja[a]
                    if s:
                        if hj[s] == 0:
                            vj[n_conf] = s
                            n_conf += 1
                        hj[s] += 1
                        nja[a] = 0
        elif G * cc * r <= dense_cells:
            col = cols[c]
            gr = G * r
            for i in range(n):
                cnt[col[i] * gr + g[i] * r + child[i]] += 1
            for jj in range(G * cc):
                s = 0
                for k in range(jj * r, jj * r + r):
                    v = cnt[k]
                    if v:
                        s += v
                        if hk[v] == 0:
                            vk[n_cell] = v
                            n_cell += 1
                        hk[v] += 1
                        cnt[k] = 0
                if s:
                    if hj[s] == 0:
                        vj[n_conf] = s
                        n_conf += 1
                    hj[s] += 1
        else:
            col = cols[c]
            for gi in range(G):
                nt = 0
                ntj = 0
                for t in range(start[gi], start[gi + 1]):
                    a = col[perm[t]]
                    idx = a * r + ch[t]
                    if cnt[idx] == 0:
                        touched[nt] = idx
                        nt += 1
                    cnt[idx] += 1
                    if nja[a] == 0:
                        touched_j[ntj] = a
                        ntj += 1
                    nja[a] += 1
                for x in range(nt):
                    idx = touched[x]
                    v = cnt[idx]
                    if hk[v] == 0:
                        vk[n_cell] = v
                        n_cell += 1
                    hk[v] += 1
                    cnt[idx] = 0
                for x in range(ntj):
                    a = touched_j[x]
                    v = nja[a]
                    if hj[v] == 0:
                        vj[n_conf] = v
                        n_conf += 1
                    hj[v] += 1
                    nja[a] = 0
        return n_cell, n_conf

    @njit(cache=True, nogil=True)
    def _scores(  # pragma: no cover - compiled
        cols: np.ndarray,
        cards: np.ndarray,
        value_bits: np.ndarray,
        offset: np.ndarray,
        node: int,
        g: np.ndarray,
        G: int,
        perm: np.ndarray,
        start: np.ndarray,
        ch: np.ndarray,
        njk: np.ndarray,
        bits: np.ndarray,
        qf: float,
        cands: np.ndarray,
        ess: float,
        prior: float,
        popcount_cells: int,
        dense_cells: int,
        hist: np.ndarray,
        cnt: np.ndarray,
        aux: np.ndarray,
        keys: np.ndarray,
        stamps: np.ndarray,
        table: np.ndarray,
    ) -> np.ndarray:
        """Score ``node`` given the grouped parents plus each candidate, prior term included."""
        r = cards[node]
        tag = _lg_tag(stamps)
        out = np.empty(cands.shape[0])
        for j in range(cands.shape[0]):
            c = cands[j]
            n_cell, n_conf = _count(
                c,
                cols,
                cards,
                value_bits,
                offset,
                node,
                g,
                G,
                perm,
                start,
                ch,
                njk,
                bits,
                popcount_cells,
                dense_cells,
                hist,
                cnt,
                aux,
            )
            s = _lik(hist, n_cell, n_conf, qf * cards[c], r, ess, keys, stamps, table, tag) + prior
            out[j] = s if math.isfinite(s) else np.nan
        return out

    @njit(cache=True, nogil=True, parallel=True)
    def _scores_par(  # pragma: no cover - compiled
        cols: np.ndarray,
        cards: np.ndarray,
        value_bits: np.ndarray,
        offset: np.ndarray,
        node: int,
        g: np.ndarray,
        G: int,
        perm: np.ndarray,
        start: np.ndarray,
        ch: np.ndarray,
        njk: np.ndarray,
        bits: np.ndarray,
        qf: float,
        cands: np.ndarray,
        ess: float,
        prior: float,
        popcount_cells: int,
        dense_cells: int,
        hists: np.ndarray,
        cnts: np.ndarray,
        auxs: np.ndarray,
        keys: np.ndarray,
        stamps: np.ndarray,
        table: np.ndarray,
    ) -> np.ndarray:
        """:func:`_scores` with the candidates split into one chunk per buffer set.

        The lgamma rows are claimed before the parallel loop; table entries
        filled concurrently are written with the same value. Each candidate is
        computed exactly as in :func:`_scores`, so the results are identical.
        """
        r = cards[node]
        m = cands.shape[0]
        tag = _lg_tag(stamps)
        a_j = np.empty(m)
        a_jk = np.empty(m)
        row_j = np.empty(m, dtype=np.int64)
        row_jk = np.empty(m, dtype=np.int64)
        for j in range(m):
            q = qf * cards[cands[j]]
            a_j[j] = ess / q
            a_jk[j] = ess / (q * r)
            row_j[j] = _lg_slot(keys, stamps, table, a_j[j], tag)
            row_jk[j] = _lg_slot(keys, stamps, table, a_jk[j], tag)
        out = np.empty(m)
        chunks = hists.shape[0]
        for t in prange(chunks):
            for j in range(t * m // chunks, (t + 1) * m // chunks):
                n_cell, n_conf = _count(
                    cands[j],
                    cols,
                    cards,
                    value_bits,
                    offset,
                    node,
                    g,
                    G,
                    perm,
                    start,
                    ch,
                    njk,
                    bits,
                    popcount_cells,
                    dense_cells,
                    hists[t],
                    cnts[t],
                    auxs[t],
                )
                s = (
                    _lik_at(hists[t], n_cell, n_conf, a_j[j], a_jk[j], row_j[j], row_jk[j], table)
                    + prior
                )
                out[j] = s if math.isfinite(s) else np.nan
        return out

    @njit(cache=True, nogil=True)
    def _family_cells(
        cols: np.ndarray,
        cards: np.ndarray,
        node: int,
        parents: np.ndarray,
        n_words: int,
        bit_cells: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover - compiled
        """Sort the rows by (parent configuration, child value) cell.

        Returns the rows in that order (increasing row index inside a cell),
        the start of every realised cell, the configuration of each cell (the
        cells of a configuration are consecutive) and, when there are at most
        ``bit_cells`` cells, the rows of each one as a bitset.
        """
        n = cols.shape[1]
        r = cards[node]
        g, G = _configs(cols, cards, parents)
        child = cols[node]
        size = np.zeros(G * r, dtype=np.int32)
        for i in range(n):
            size[g[i] * r + child[i]] += 1
        K = 0
        for k in range(G * r):
            if size[k] > 0:
                K += 1
        cid = np.empty(G * r, dtype=np.int32)
        cstart = np.zeros(K + 1, dtype=np.int32)
        ccfg = np.empty(K, dtype=np.int32)
        x = 0
        for k in range(G * r):
            if size[k] > 0:
                cid[k] = x
                cstart[x + 1] = cstart[x] + size[k]
                ccfg[x] = k // r
                x += 1
        pos = cstart[:-1].copy()
        cperm = np.empty(n, dtype=np.int32)
        rows = K if bit_cells >= K else 0
        bits = np.zeros((rows, n_words), dtype=np.uint64)
        for i in range(n):
            x = cid[g[i] * r + child[i]]
            cperm[pos[x]] = i
            pos[x] += 1
            if rows:
                bits[x, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
        return cperm, cstart, ccfg, bits

    @njit(cache=True, nogil=True)
    def _count_cells(  # pragma: no cover - compiled
        c: int,
        cols: np.ndarray,
        cards: np.ndarray,
        value_bits: np.ndarray,
        offset: np.ndarray,
        cperm: np.ndarray,
        cstart: np.ndarray,
        ccfg: np.ndarray,
        bits: np.ndarray,
        popcount_cells: int,
        lanes_rows: int,
        hist: np.ndarray,
        work: np.ndarray,
    ) -> tuple[int, int]:
        """Fill the histograms of the cell and configuration counts of the family plus ``c``.

        Every (configuration, child) cell is split by the value of ``c``. With
        few cells (``cells * (cc - 1) <= popcount_cells``), by intersecting the
        row bitset of each cell with the bitset of each value of ``c``, the last
        value taking the rows that are left. Otherwise by walking the rows of
        the cell, which are consecutive, into a table of ``cc`` counters that
        is then emptied: by reading it whole or, when the cell has fewer rows
        than ``c`` has values, by reading those rows again. A cell of at least
        ``lanes_rows * cc`` rows is walked into four copies of the table,
        written in turn, because most of its consecutive rows share the value
        and would wait for each other's increment. The walk visits the column
        of ``c`` in an order no prefetcher follows, so the column is read once
        in order first. The counts of a configuration are accumulated on the
        way and emptied in the same manner when it ends.

        A count read from a table is always written to a list, which grows
        only if the count is not zero, so that no branch depends on the data;
        the histograms are filled from the lists. Returns how many distinct
        values each histogram holds.
        """
        lk, lj, small, nja = work[0], work[1], work[2], work[3]
        cc = cards[c]
        col = cols[c]
        K = ccfg.shape[0]
        p = 0  # cell counts so far
        q = 0  # configuration counts so far
        x = 0
        if bits.shape[0] == K and K * (cc - 1) <= popcount_cells:
            n_words = value_bits.shape[1]
            first = offset[c]
            while x < K:
                gi = ccfg[x]
                while x < K and ccfg[x] == gi:
                    rest = cstart[x + 1] - cstart[x]
                    for a in range(cc - 1):
                        s = 0
                        for w in range(n_words):
                            s += _popcount(bits[x, w] & value_bits[first + a, w])  # type: ignore[call-arg]
                        lk[p] = s
                        p += s != 0
                        nja[a] += s
                        rest -= s
                        if rest == 0:
                            break
                    lk[p] = rest
                    p += rest != 0
                    nja[cc - 1] += rest
                    x += 1
                for a in range(cc):
                    v = nja[a]
                    nja[a] = 0
                    lj[q] = v
                    q += v != 0
        else:
            warm = 0
            for i in range(0, col.shape[0], 64):
                warm += col[i]
            lk[0] = warm  # overwritten below; keeps the loop above
            lanes = lanes_rows * cc
            while x < K:
                gi = ccfg[x]
                g0 = cstart[x]
                while x < K and ccfg[x] == gi:
                    t0 = cstart[x]
                    t1 = cstart[x + 1]
                    x += 1
                    if t1 - t0 == 1:
                        lk[p] = 1
                        p += 1
                        nja[col[cperm[t0]]] += 1
                    elif t1 - t0 < cc:
                        for t in range(t0, t1):
                            small[col[cperm[t]]] += 1
                        for t in range(t0, t1):
                            a = col[cperm[t]]
                            v = small[a]
                            small[a] = 0
                            lk[p] = v
                            p += v != 0
                            nja[a] += v
                    elif t1 - t0 < lanes:
                        for t in range(t0, t1):
                            small[col[cperm[t]]] += 1
                        for a in range(cc):
                            v = small[a]
                            small[a] = 0
                            lk[p] = v
                            p += v != 0
                            nja[a] += v
                    else:
                        t4 = t0 + (((t1 - t0) >> 2) << 2)
                        for t in range(t0, t4, 4):
                            small[col[cperm[t]] << 2] += 1
                            small[(col[cperm[t + 1]] << 2) | 1] += 1
                            small[(col[cperm[t + 2]] << 2) | 2] += 1
                            small[(col[cperm[t + 3]] << 2) | 3] += 1
                        for t in range(t4, t1):
                            small[col[cperm[t]] << 2] += 1
                        for a in range(cc):
                            k = a << 2
                            v = small[k] + small[k + 1] + small[k + 2] + small[k + 3]
                            small[k] = 0
                            small[k + 1] = 0
                            small[k + 2] = 0
                            small[k + 3] = 0
                            lk[p] = v
                            p += v != 0
                            nja[a] += v
                t1 = cstart[x]
                if t1 - g0 < cc:
                    for t in range(g0, t1):
                        a = col[cperm[t]]
                        v = nja[a]
                        nja[a] = 0
                        lj[q] = v
                        q += v != 0
                else:
                    for a in range(cc):
                        v = nja[a]
                        nja[a] = 0
                        lj[q] = v
                        q += v != 0
        hk, hj, vk, vj = hist[0], hist[1], hist[2], hist[3]
        n_cell = 0
        n_conf = 0
        for y in range(p):
            v = lk[y]
            vk[n_cell] = v
            n_cell += hk[v] == 0
            hk[v] += 1
        for y in range(q):
            v = lj[y]
            vj[n_conf] = v
            n_conf += hj[v] == 0
            hj[v] += 1
        return n_cell, n_conf

    @njit(cache=True, nogil=True)
    def _order(a: np.ndarray, m: int, marks: np.ndarray) -> None:  # pragma: no cover - compiled
        """Sort the ``m`` distinct positive integers at the start of ``a`` through a bitmap.

        The bitmap ``marks`` is left cleared.
        """
        if m <= 24:
            _sort_prefix(a, m)
            return
        lo = marks.shape[0]
        hi = 0
        for x in range(m):
            v = a[x]
            w = v >> 6
            marks[w] |= np.uint64(1) << np.uint64(v & 63)
            if w < lo:
                lo = w
            if w > hi:
                hi = w
        k = 0
        for w in range(lo, hi + 1):
            word = marks[w]
            if word:
                marks[w] = 0
                base = w << 6
                while word:
                    low = word & (~word + np.uint64(1))
                    a[k] = base + np.int64(_popcount(low - np.uint64(1)))  # type: ignore[call-arg]
                    k += 1
                    word ^= low

    @njit(cache=True, nogil=True)
    def _scores_cells(  # pragma: no cover - compiled
        cols: np.ndarray,
        cards: np.ndarray,
        value_bits: np.ndarray,
        offset: np.ndarray,
        node: int,
        cperm: np.ndarray,
        cstart: np.ndarray,
        ccfg: np.ndarray,
        bits: np.ndarray,
        qf: float,
        cands: np.ndarray,
        ess: float,
        prior: float,
        popcount_cells: int,
        lanes_rows: int,
        hist: np.ndarray,
        work: np.ndarray,
        marks: np.ndarray,
        keys: np.ndarray,
        stamps: np.ndarray,
        table: np.ndarray,
    ) -> np.ndarray:
        """:func:`_scores` on a family grouped by :func:`_family_cells`."""
        r = cards[node]
        tag = _lg_tag(stamps)
        out = np.empty(cands.shape[0])
        for j in range(cands.shape[0]):
            c = cands[j]
            n_cell, n_conf = _count_cells(
                c,
                cols,
                cards,
                value_bits,
                offset,
                cperm,
                cstart,
                ccfg,
                bits,
                popcount_cells,
                lanes_rows,
                hist,
                work,
            )
            _order(hist[2], n_cell, marks)
            _order(hist[3], n_conf, marks)
            q = qf * cards[c]
            a_j = ess / q
            a_jk = ess / (q * r)
            row_j = _lg_slot(keys, stamps, table, a_j, tag)
            row_jk = _lg_slot(keys, stamps, table, a_jk, tag)
            s = _lik_sorted(hist, n_cell, n_conf, a_j, a_jk, row_j, row_jk, table) + prior
            out[j] = s if math.isfinite(s) else np.nan
        return out

    @njit(cache=True, nogil=True, parallel=True)
    def _scores_cells_par(  # pragma: no cover - compiled
        cols: np.ndarray,
        cards: np.ndarray,
        value_bits: np.ndarray,
        offset: np.ndarray,
        node: int,
        cperm: np.ndarray,
        cstart: np.ndarray,
        ccfg: np.ndarray,
        bits: np.ndarray,
        qf: float,
        cands: np.ndarray,
        ess: float,
        prior: float,
        popcount_cells: int,
        lanes_rows: int,
        hists: np.ndarray,
        works: np.ndarray,
        markss: np.ndarray,
        keys: np.ndarray,
        stamps: np.ndarray,
        table: np.ndarray,
    ) -> np.ndarray:
        """:func:`_scores_cells` with the candidates split into one chunk per buffer set.

        The candidates are split as in :func:`_scores_par`.
        """
        r = cards[node]
        m = cands.shape[0]
        tag = _lg_tag(stamps)
        a_j = np.empty(m)
        a_jk = np.empty(m)
        row_j = np.empty(m, dtype=np.int64)
        row_jk = np.empty(m, dtype=np.int64)
        for j in range(m):
            q = qf * cards[cands[j]]
            a_j[j] = ess / q
            a_jk[j] = ess / (q * r)
            row_j[j] = _lg_slot(keys, stamps, table, a_j[j], tag)
            row_jk[j] = _lg_slot(keys, stamps, table, a_jk[j], tag)
        out = np.empty(m)
        chunks = hists.shape[0]
        for t in prange(chunks):
            for j in range(t * m // chunks, (t + 1) * m // chunks):
                n_cell, n_conf = _count_cells(
                    cands[j],
                    cols,
                    cards,
                    value_bits,
                    offset,
                    cperm,
                    cstart,
                    ccfg,
                    bits,
                    popcount_cells,
                    lanes_rows,
                    hists[t],
                    works[t],
                )
                _order(hists[t, 2], n_cell, markss[t])
                _order(hists[t, 3], n_conf, markss[t])
                s = (
                    _lik_sorted(
                        hists[t], n_cell, n_conf, a_j[j], a_jk[j], row_j[j], row_jk[j], table
                    )
                    + prior
                )
                out[j] = s if math.isfinite(s) else np.nan
        return out


class CompiledBDeu:
    """BDeu with a structure prior, compiled: the engine of :class:`~causalbase.scores.BDeuScore`.

    The local score of a family is its BDeu log marginal likelihood plus the
    structure prior of ``k`` parents: ``k * log(p / (1 - p))`` with
    ``p = min(expected_parents / (n_vars - 1), 1/2)`` for ``prior="bernoulli"``
    (exactly as :class:`~causalbase.scores.FamilyScore` computes it), or
    Tetrad 7.6.5's ``k log(p) + (N - 1 - k) log(1 - p)`` with
    ``p = expected_parents / (N - 1)`` and ``N`` rows for ``prior="tetrad"``.
    The order searches, hill climbing, FLOP and the exact searches score with
    it; a ``BDeuScore`` builds one on demand.

    Parameters
    ----------
    data : DiscreteDataset
        Data to score; kept as ``data``, with its number of variables as ``n_vars``.
    ess : float, default=1.0
        Equivalent sample size of the BDeu prior.
    expected_parents : float or None, default=1.0
        The parameter of the structure prior; None (or 0 with ``"tetrad"``)
        leaves the prior out.
    n_jobs : int, default=1
        Threads for large batches of candidates, at most
        ``numba.config.NUMBA_NUM_THREADS``. Each candidate is scored as with
        one thread, so results do not depend on it.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior.
    kernel : {"cells", "legacy"}, default="cells"
        How a candidate of a batch is counted. ``"cells"`` sorts the rows of the
        grouped family by (configuration, child value) cell and splits every
        cell by the candidate (see :func:`_count_cells`). ``"legacy"`` chooses
        among bitsets, one dense table and one table per configuration (see
        ``dense_cells``, which only it reads). Both give the same floats.
    popcount_cells : int, default=300
        Count a candidate with bitsets when ``cells * (cc - 1)`` is at most this
        (``cc`` candidate values; the cells are the realised (configuration,
        child value) pairs for ``"cells"`` and all ``G * r`` of them for
        ``"legacy"``); -1 disables it.
    dense_cells : int or None, default=None
        Largest ``G * cc * r`` table counted in one pass over the rows
        (default ``4 * n_rows``); larger ones are counted group by group.
    cache_bytes : int, default=256 MiB
        Memory budget of the grouped parent sets kept between calls.
    lanes_rows : int, default=8
        ``"cells"``: a cell with at least ``lanes_rows * cc`` rows is walked
        into four copies of the table of counters, written in turn.
    lgamma_rows : int, default=2048
        Rows of the table of lgamma values, one per prior value in use
        (rounded up to a power of two; 8 kB each).

    Attributes
    ----------
    n_families : int
        Families scored so far, single or batched: the work of a search,
        separate from the speed of the kernel.
    """

    def __init__(
        self,
        data: DiscreteDataset,
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        n_jobs: int = 1,
        *,
        prior: Prior = "bernoulli",
        kernel: str = "cells",
        popcount_cells: int = 300,
        dense_cells: int | None = None,
        cache_bytes: int = 256 * 2**20,
        lanes_rows: int = 8,
        lgamma_rows: int = 2048,
    ) -> None:
        if kernel not in ("cells", "legacy"):
            raise ValueError(f"kernel must be 'cells' or 'legacy', not {kernel!r}")
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}, got {prior!r}")
        self.kernel = kernel
        self._lanes_rows = int(lanes_rows)
        self.data = data
        self.n_vars = data.n_vars
        self.ess = float(ess)
        self.prior: Prior = prior
        self.expected_parents = None if expected_parents is None else float(expected_parents)
        if prior == "bernoulli" and self.expected_parents is not None:
            # As FamilyScore computes it, so that both add the same prior.
            p = min(self.expected_parents / max(data.n_vars - 1, 1), 0.5)
            self._log_odds = math.log(p) - math.log1p(-p)
        else:
            self._log_odds = 0.0
        self._n_rows = data.n_rows
        self.n_families = 0  # families scored so far, single or batched
        if not HAVE_NUMBA:  # pragma: no cover
            self._fallback: BDeuScore | TetradBDeuReference = (
                TetradBDeuReference(data, ess, self.expected_parents or 0.0)
                if prior == "tetrad"
                else BDeuScore(data, ess=ess, expected_parents=self.expected_parents)
            )
            return
        n = data.n_rows
        values = np.asarray(data.values, dtype=np.int64).T
        cards = np.asarray(data.cardinalities, dtype=np.int64)
        # A constant last column of cardinality 1: "parents + const" is "parents".
        cards = np.append(cards, 1)
        values = np.vstack([values, np.zeros((1, n), dtype=np.int64)])
        dtype = np.uint8 if cards.max() <= 256 else np.int32
        self._cols = np.ascontiguousarray(values.astype(dtype))
        self._cards = cards
        self._const = np.array([len(cards) - 1], dtype=np.int64)
        n_words = (n + 63) // 64
        self._offset = np.zeros(len(cards) + 1, dtype=np.int64)
        self._offset[1:] = np.cumsum(cards)
        self._value_bits = np.zeros((int(self._offset[-1]), n_words), dtype=np.uint64)
        for v in range(len(cards)):
            for a in range(int(cards[v])):
                idx = np.flatnonzero(self._cols[v] == a)
                np.bitwise_or.at(
                    self._value_bits[self._offset[v] + a],
                    idx >> 6,
                    np.uint64(1) << (idx & 63).astype(np.uint64),
                )
        self._popcount_cells = int(popcount_cells)
        self._bit_cells = min(
            max(self._popcount_cells, 0), 4096
        )  # bitsets cost G * r * n / 8 bytes
        mc = int(cards.max())
        dense = 4 * n if dense_cells is None else int(dense_cells)
        self._dense_cells = dense
        # numba refuses more threads than it started with (the number of cores by default).
        self._threads = max(1, min(int(n_jobs), config.NUMBA_NUM_THREADS))  # type: ignore[attr-defined]
        self._hists = np.zeros((self._threads, 4, n + 1), dtype=np.int64)
        self._cnts = np.zeros((self._threads, max(dense, mc * mc)), dtype=np.int32)
        self._auxs = np.zeros((self._threads, 3, max(n, mc)), dtype=np.int64)
        self._hist, self._cnt, self._aux = self._hists[0], self._cnts[0], self._auxs[0]
        # "cells": per thread, the lists of cell and configuration counts and the two small
        # tables.
        self._works = np.zeros((self._threads, 4, max(n + 2, 4 * mc)), dtype=np.int32)
        self._marks = np.zeros(
            (self._threads, (n >> 6) + 2), dtype=np.uint64
        )  # bitmap that orders the counts
        rows = (
            1 << max(0, int(lgamma_rows) - 1).bit_length()
        )  # a power of two: rows are found by masking
        self._lg_keys = np.zeros(rows)
        self._lg_stamps = np.zeros(rows + 1, dtype=np.int64)  # the last entry counts the batches
        self._lg_table = np.full((rows, min(n + 1, 1024)), np.nan)
        self._cache_bytes = int(cache_bytes)
        self._families: dict[tuple[int, tuple[int, ...]], tuple[Any, ...]] = {}
        self._used = 0

    def _prior_term(self, num_parents: int) -> float:
        if self.prior == "bernoulli":
            return num_parents * self._log_odds
        e = self.expected_parents
        if not e:
            return 0.0
        vm = self._n_rows - 1
        return num_parents * math.log(e / vm) + (vm - num_parents) * math.log(1.0 - (e / vm))

    def _grouped(self, node: int, par: tuple[int, ...]) -> tuple[Any, ...]:
        key = (node, par)
        fam = self._families.get(key)
        if fam is None:
            arr = np.asarray(par, dtype=np.int64)
            qf = 1.0
            for p in par:
                qf *= float(self._cards[p])
            if self.kernel == "cells":
                fam = (
                    *_family_cells(
                        self._cols,
                        self._cards,
                        node,
                        arr,
                        self._value_bits.shape[1],
                        self._bit_cells,
                    ),
                    qf,
                )
            else:
                fam = (
                    *_family(
                        self._cols,
                        self._cards,
                        node,
                        arr,
                        self._value_bits.shape[1],
                        self._bit_cells,
                    ),
                    qf,
                )
            size = sum(x.nbytes for x in fam if isinstance(x, np.ndarray))
            if self._used + size > self._cache_bytes:
                self._families.clear()
                self._used = 0
            if size <= self._cache_bytes:
                self._families[key] = fam
                self._used += size
        return fam

    def _run(self, node: int, par: tuple[int, ...], cands: np.ndarray, prior: float) -> np.ndarray:
        if self.kernel == "cells":
            cperm, cstart, ccfg, bits, qf = self._grouped(node, par)
            if self._threads > 1 and cands.shape[0] >= 8 * self._threads:
                set_num_threads(self._threads)
                return _scores_cells_par(
                    self._cols,
                    self._cards,
                    self._value_bits,
                    self._offset,
                    node,
                    cperm,
                    cstart,
                    ccfg,
                    bits,
                    qf,
                    cands,
                    self.ess,
                    prior,
                    self._popcount_cells,
                    self._lanes_rows,
                    self._hists,
                    self._works,
                    self._marks,
                    self._lg_keys,
                    self._lg_stamps,
                    self._lg_table,
                )
            return _scores_cells(
                self._cols,
                self._cards,
                self._value_bits,
                self._offset,
                node,
                cperm,
                cstart,
                ccfg,
                bits,
                qf,
                cands,
                self.ess,
                prior,
                self._popcount_cells,
                self._lanes_rows,
                self._hist,
                self._works[0],
                self._marks[0],
                self._lg_keys,
                self._lg_stamps,
                self._lg_table,
            )
        g, G, perm, start, ch, njk, bits, qf = self._grouped(node, par)
        if self._threads > 1 and cands.shape[0] >= 8 * self._threads:
            set_num_threads(self._threads)
            return _scores_par(
                self._cols,
                self._cards,
                self._value_bits,
                self._offset,
                node,
                g,
                G,
                perm,
                start,
                ch,
                njk,
                bits,
                qf,
                cands,
                self.ess,
                prior,
                self._popcount_cells,
                self._dense_cells,
                self._hists,
                self._cnts,
                self._auxs,
                self._lg_keys,
                self._lg_stamps,
                self._lg_table,
            )
        return _scores(
            self._cols,
            self._cards,
            self._value_bits,
            self._offset,
            node,
            g,
            G,
            perm,
            start,
            ch,
            njk,
            bits,
            qf,
            cands,
            self.ess,
            prior,
            self._popcount_cells,
            self._dense_cells,
            self._hist,
            self._cnt,
            self._aux,
            self._lg_keys,
            self._lg_stamps,
            self._lg_table,
        )

    def score_dag(self, dag: DAG) -> float:
        """Return the score of `dag`: the sum of its families' local scores."""
        return float(sum(self.local_score(v, dag.parents(v)) for v in range(dag.num_nodes)))

    def local_score(self, node: int, parents: Collection[int]) -> float:
        """Return the local score of ``node`` given ``parents``."""
        self.n_families += 1
        if not HAVE_NUMBA:  # pragma: no cover
            return self._fallback.local_score(node, parents)
        par = sorted(int(p) for p in parents)
        if (node, tuple(par)) in self._families:  # grouped already: the batch path is cheaper
            return float(
                self._run(int(node), tuple(par), self._const, self._prior_term(len(par)))[0]
            )
        return _single(
            self._cols,
            self._cards,
            int(node),
            np.asarray(par, dtype=np.int64),
            self.ess,
            self._prior_term(len(par)),
            self._hist,
            self._lg_keys,
            self._lg_stamps,
            self._lg_table,
        )

    def local_scores_add(
        self, node: int, parents: Sequence[int], cands: Sequence[int] | np.ndarray
    ) -> np.ndarray:
        """Return the local scores of ``node`` given ``parents`` plus each one of ``cands``.

        Equal, bit for bit, to calling :meth:`local_score` once per candidate.
        """
        self.n_families += len(cands)
        if not HAVE_NUMBA:  # pragma: no cover
            return np.array([self.local_score(node, frozenset(parents) | {int(c)}) for c in cands])
        par = tuple(sorted(int(p) for p in parents))
        cand = np.asarray(cands, dtype=np.int64)
        return self._run(int(node), par, cand, self._prior_term(len(par) + 1))


def engine_of(score: Any) -> Any:
    """Return what the order searches, hill climbing, FLOP and the exact searches score with.

    That is ``score._engine()`` for a :class:`~causalbase.scores.BDeuScore`.
    Any other object with ``local_score(node, parents)`` and
    ``local_scores_add(node, parents, candidates)`` is returned as it is: every
    other :class:`~causalbase.scores.FamilyScore`, an engine with other kernel
    settings, or a score of one's own.

    Raises
    ------
    TypeError
        If `score` is neither.
    """
    if isinstance(score, BDeuScore):
        return score._engine()
    if hasattr(score, "local_score") and hasattr(score, "local_scores_add"):
        return score
    raise TypeError(
        "the search takes a score with local_score and local_scores_add, such as a "
        f"FamilyScore, got {type(score).__name__}"
    )
