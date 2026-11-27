"""The compiled parent-set walks of the exact searches for the Gaussian scores.

A Gaussian score reads its family from the Schur complement of the parents in
one matrix (see :mod:`causalbase.scores._gaussian_kernels`). Walking the parent
sets depth first, a set and its extension by ``j`` differ by one elimination
step, ``M' = M - M[:, j] M[j, :] / M[j, j]``, so each set costs a rank-one
update of the rows still reachable from it (the later candidates and the
child) instead of a factorisation, and ``log |M_PP|`` is the running sum of
the logarithms of the pivots. A candidate whose pivot falls below
``PIVOT`` times its diagonal entry is a linear combination of the set and is
skipped, as the single-family kernels skip it.

The matrices given here are ``M`` restricted to the other variables, in
increasing order, followed by the child, which is therefore the last row.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from causalbase.scores._gaussian_kernels import PIVOT

__all__ = ["gaussian_all_scores", "gaussian_limited_scores"]


@njit(cache=True, nogil=True)
def _eliminate(
    src: np.ndarray, dst: np.ndarray, j: int, diag: float, lo: int, hi: int, node: int
) -> float:
    """Write into `dst` the rows ``lo..hi-1`` and `node` of `src` with variable `j` eliminated.

    Returns the logarithm of the pivot, or 0 when `j` is dependent (then the
    rows are copied unchanged).
    """
    piv = src[j, j]
    if piv > PIVOT * diag:
        for i in range(lo, hi):
            f = src[i, j] / piv
            for t in range(lo, hi):
                dst[i, t] = src[i, t] - f * src[j, t]
            dst[i, node] = src[i, node] - f * src[j, node]
            dst[node, i] = dst[i, node]
        dst[node, node] = src[node, node] - src[node, j] * src[j, node] / piv
        return math.log(piv)
    for i in range(lo, hi):
        for t in range(lo, hi):
            dst[i, t] = src[i, t]
        dst[i, node] = src[i, node]
        dst[node, i] = src[node, i]
    dst[node, node] = src[node, node]
    return 0.0


@njit(cache=True, nogil=True)
def _value(
    mat: np.ndarray,
    node: int,
    log_det: float,
    k: int,
    a: np.ndarray,
    b: np.ndarray,
    c: float,
    floor: float,
) -> float:
    s = a[k] + b[k] * math.log(max(mat[node, node], floor)) + c * log_det
    return s if math.isfinite(s) else -np.inf


@njit(cache=True, nogil=True, parallel=True)
def gaussian_all_scores(
    M: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: float,
    floor: float,
    high: int,
    bounds: np.ndarray,
    tol: float,
    out: np.ndarray,
) -> np.ndarray:
    """Score every parent set (a bitmask over the first ``m`` rows of `M`) into `out`.

    As the BDeu walk of :mod:`causalbase.exact.dp`: the top `high` bits split
    the sets into chunks walked in parallel, and the supersets of a set of
    ``k`` variables are skipped when ``bounds[k]`` is below the best score
    along its chain of subsets by more than `tol`. Returns the sets scored per
    chunk.
    """
    m = M.shape[0] - 1
    node = m
    lo = m - high
    n_chunks = 1 << high
    done = np.zeros(n_chunks, dtype=np.int64)
    for ch in prange(n_chunks):
        mats = np.empty((lo + 2, m + 1, m + 1))
        dets = np.zeros(lo + 2)
        cur = M.copy()
        log_det = 0.0
        best = _value(cur, node, 0.0, 0, a, b, c, floor)
        s = best
        size = 0
        closed = bounds[0] < best - tol
        base = ch << lo
        for bit in range(lo, m):
            if (base >> bit) & 1:
                nxt_mat = np.empty_like(cur)
                log_det += _eliminate(cur, nxt_mat, bit, M[bit, bit], 0, m, node)
                cur = nxt_mat
                size += 1
                s = _value(cur, node, log_det, size, a, b, c, floor)
                best = max(best, s)
                if bounds[size] < best - tol:
                    closed = True
        out[base] = s
        cnt = 1
        if closed:
            done[ch] = cnt
            continue
        mats[0] = cur
        dets[0] = log_det
        masks = np.zeros(lo + 1, dtype=np.int64)
        nxt = np.zeros(lo + 1, dtype=np.int64)
        bests = np.zeros(lo + 1)
        masks[0] = base
        bests[0] = best
        depth = 0
        while depth >= 0:
            j = nxt[depth]
            if j >= lo:
                depth -= 1
                continue
            nxt[depth] = j + 1
            d = dets[depth] + _eliminate(mats[depth], mats[depth + 1], j, M[j, j], j + 1, lo, node)
            S = masks[depth] | (1 << j)
            k = size + depth + 1
            s = _value(mats[depth + 1], node, d, k, a, b, c, floor)
            out[S] = s
            cnt += 1
            bd = max(bests[depth], s)
            if bounds[k] < bd - tol:
                continue  # no superset of S in this subtree can be useful
            depth += 1
            dets[depth] = d
            masks[depth] = S
            nxt[depth] = j + 1
            bests[depth] = bd
        done[ch] = cnt
    return done


@njit(cache=True, nogil=True, parallel=True)
def gaussian_limited_scores(
    M: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: float,
    floor: float,
    kmax: int,
    starts: np.ndarray,
    out_masks: np.ndarray,
    out_scores: np.ndarray,
) -> None:
    """Score every non-empty parent set of at most `kmax` variables.

    As the BDeu kernel of :mod:`causalbase.exact.ilp`: chunk ``i`` holds the
    sets whose smallest variable is ``i``, written from ``starts[i]`` in
    depth-first order.
    """
    m = M.shape[0] - 1
    node = m
    for i in prange(m):
        mats = np.empty((kmax + 1, m + 1, m + 1))
        dets = np.zeros(kmax + 1)
        masks = np.zeros(kmax + 1, dtype=np.int64)
        nxt = np.zeros(kmax + 1, dtype=np.int64)
        pos = starts[i]
        dets[1] = _eliminate(M, mats[1], i, M[i, i], i + 1, m, node)
        S = np.int64(1) << i
        out_masks[pos] = S
        out_scores[pos] = _value(mats[1], node, dets[1], 1, a, b, c, floor)
        pos += 1
        depth = 1
        masks[1] = S
        nxt[1] = i + 1
        while depth >= 1:
            j = nxt[depth]
            if j >= m or depth >= kmax:
                depth -= 1
                continue
            nxt[depth] = j + 1
            d = dets[depth] + _eliminate(mats[depth], mats[depth + 1], j, M[j, j], j + 1, m, node)
            S = masks[depth] | (np.int64(1) << j)
            out_masks[pos] = S
            out_scores[pos] = _value(mats[depth + 1], node, d, depth + 1, a, b, c, floor)
            pos += 1
            depth += 1
            dets[depth] = d
            masks[depth] = S
            nxt[depth] = j + 1
