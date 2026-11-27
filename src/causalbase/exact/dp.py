"""Globally optimal DAG by dynamic programming over subsets.

The score is any :class:`~causalbase.scores.FamilyScore`, and
:func:`exact_search` is the entry point to both exact searches. A
:class:`~causalbase.scores.BDeuScore`, with either prior, is scored with the
compiled kernels below; any other score family by family, in batches of
extensions, as described at the end.

The search follows Silander and Myllymaki (2006). For every variable ``v`` the
local score of every parent set ``S`` of the other variables is computed (bit
for bit the value the engine's ``local_score`` returns), and a parent set is
kept as a candidate only if it scores strictly better than all of its subsets
(the subset dominance of de Campos and Ji 2011). ``best(v, U)``, the best
candidate inside ``U``, is a max-over-subsets transform of those scores. The
optimal network over a variable set ``W`` is then

    F(W) = max over v in W of F(W - v) + best(v, W - v),

the sink ``v`` being the last variable of an optimal order of ``W``.

Enumerating the ``2^(p-1)`` parent sets of a variable is the expensive part,
and it skips every superset of a set ``S`` that cannot be useful. The bound
is sound for BDeu: in the sequential (Polya urn) form of the marginal
likelihood every factor is at most 1, and the first row of each non-empty
(configuration, child value) cell contributes at most ``1/r``. A strict
superset of ``S`` has at least the ``K`` non-empty cells of ``S`` and at least
one more parent, so its score is at most ``-K log r + prior(|S| + 1)``; when
that is no better than a subset of ``S`` already scored, the supersets reached
from ``S`` are left out (scored as ``-inf``). Skipped sets are dominated by a
subset with a finite score, so neither the candidates nor ``best(v, U)``
change.

Memory is ``2^(p-1)`` float64 per variable while it is scored, ``p 2^(p-1)``
candidate indices for the dynamic programming and ``2^p`` float64 for ``F``:
about 6 GB at ``p = 27``.

The other scores go through the same depth-first walk, written in Python: the
extensions of a set by every later variable are scored in one batch, and the
supersets of a set are skipped when the score's ``_superset_bound`` proves
them useless. The discrete BIC bounds them with the argument of de Campos and
Ji (the log likelihood is at most 0, so the penalty of one more parent alone
bounds the score); the Gaussian BIC with the likelihood of the family of
every other variable, which no family of the node can exceed. BGe has no such
bound here (its marginal likelihood is not bounded by a penalty), so every
parent set is scored. The walk costs a batch call per parent set, which keeps
these scores to about 20 variables.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from numbers import Integral
from typing import Any, NamedTuple

import numpy as np
from numba import njit, prange

from causalbase._parallel import numba_threads
from causalbase._validation import check_time_limit
from causalbase.data import DiscreteDataset
from causalbase.exact._gaussian import gaussian_all_scores
from causalbase.graph import DAG
from causalbase.scores._compiled import CompiledBDeu, _lg_row, _lik_at, _sort_prefix, engine_of
from causalbase.scores._family import FamilyScore
from causalbase.scores.bdeu import BICScore
from causalbase.scores.gaussian import _GaussianScore
from causalbase.search._result import SearchResult

__all__ = ["exact_search"]


@njit(cache=True, nogil=True)
def _refine(
    g: np.ndarray,
    col: np.ndarray,
    card: int,
    g_out: np.ndarray,
    ids: np.ndarray,
    made: np.ndarray,
    child: np.ndarray,
    r: int,
    cell: np.ndarray,
) -> int:
    """Refine the grouping ``g`` by ``col`` (ids by first occurrence).

    The cells of the refined grouping are counted into ``cell``.
    """
    n = g.shape[0]
    new_g = 0
    for i in range(n):
        k = g[i] * card + col[i]
        v = ids[k]
        if v < 0:
            v = new_g
            ids[k] = v
            made[v] = k
            new_g += 1
        g_out[i] = v
        cell[v * r + child[i]] += 1
    for v in range(new_g):
        ids[made[v]] = -1
    return new_g


@njit(cache=True, nogil=True)
def _score(
    G: int,
    r: int,
    qf: float,
    ess: float,
    prior: float,
    cell: np.ndarray,
    hist: np.ndarray,
    keys: np.ndarray,
    n_keys: np.ndarray,
    table: np.ndarray,
    kind: int,
    per: float,
) -> tuple[float, int]:
    """Return the score from the counts of ``G`` configurations, and the non-empty cells.

    The counts are cleared. ``kind`` 0 is BDeu, summed exactly as the compiled
    engine's ``_lik_at``: cell counts then configuration counts, each over its
    distinct values in increasing order, with the same memoised ``lgamma``.
    ``kind`` 1 is the BIC, ``sum n_jk log n_jk - sum n_j log n_j`` over the
    same histograms less ``per`` per free parameter.
    """
    hk, hj, vk, vj = hist[0], hist[1], hist[2], hist[3]
    n_cell = 0
    n_conf = 0
    K = 0
    for x in range(G):
        t = 0
        for y in range(x * r, x * r + r):
            c = cell[y]
            if c:
                t += c
                K += 1
                if hk[c] == 0:
                    vk[n_cell] = c
                    n_cell += 1
                hk[c] += 1
                cell[y] = 0
        if hj[t] == 0:
            vj[n_conf] = t
            n_conf += 1
        hj[t] += 1
    if kind == 1:
        _sort_prefix(vk, n_cell)
        _sort_prefix(vj, n_conf)
        lik = 0.0
        for x in range(n_cell):
            c = vk[x]
            lik += hk[c] * c * math.log(c)
            hk[c] = 0
        for x in range(n_conf):
            t = vj[x]
            lik -= hj[t] * t * math.log(t)
            hj[t] = 0
        s = lik - per * qf * (r - 1) + prior
        return (s if math.isfinite(s) else np.nan), K
    a_j = ess / qf
    a_jk = ess / (qf * r)
    lik = _lik_at(
        hist,
        n_cell,
        n_conf,
        a_j,
        a_jk,
        _lg_row(keys, n_keys, a_j),
        _lg_row(keys, n_keys, a_jk),
        table,
    )
    s = lik + prior
    return (s if math.isfinite(s) else np.nan), K


@njit(cache=True, nogil=True)
def _qf(mask: int | np.integer[Any], cards: np.ndarray, m: int) -> float:
    """Return the number of configurations of ``mask`` as a float.

    It is multiplied in variable order, like the compiled engine.
    """
    q = 1.0
    for b in range(m):
        if (mask >> b) & 1:
            q *= cards[b]
    return q


@njit(cache=True, nogil=True)
def _bound(
    kind: int, K: int, log_r: float, qf: float, cmin: float, per: float, r: int, prior: float
) -> float:
    """Return a bound on the score of every strict superset of a set.

    The set has ``K`` non-empty cells and ``qf`` configurations, and a strict
    superset at most the structure prior ``prior``. BDeu: every factor of the
    Polya urn form is at most 1, and the first row of each of the ``K`` cells
    contributes at most ``1 / r``. BIC: the log likelihood is at most 0, and one
    more parent multiplies the configurations by at least ``cmin``.
    """
    if kind == 1:
        return -per * (r - 1) * qf * cmin + prior
    return -K * log_r + prior


@njit(cache=True, nogil=True, parallel=True)
def _all_scores(
    cols: np.ndarray,
    cards: np.ndarray,
    child: np.ndarray,
    r: int,
    ess: float,
    priors: np.ndarray,
    high: int,
    tol: float,
    out: np.ndarray,
    kind: int,
    per: float,
) -> np.ndarray:
    """Score every parent set (a bitmask over the rows of ``cols``) into ``out``.

    The top ``high`` bits split the sets into independent chunks; each chunk
    is a depth-first walk over the remaining bits in which a set's grouping
    refines its parent's. ``kind`` and ``per`` choose the score, as in
    :func:`_score`. Returns how many sets were scored per chunk.
    """
    m, n = cols.shape
    lo = m - high
    n_chunks = 1 << high
    mc = 1
    cmin = 1.0 * cards[0] if m else 1.0
    for b in range(m):
        mc = max(mc, cards[b])
        cmin = min(cmin, 1.0 * cards[b])
    log_r = math.log(r)
    done = np.zeros(n_chunks, dtype=np.int64)
    for c in prange(n_chunks):
        gs = np.zeros((lo + 1, n), dtype=np.int16)
        Gs = np.zeros(lo + 1, dtype=np.int64)
        tmp = np.zeros((2, n), dtype=np.int16)
        ids = np.full(n * mc, -1, dtype=np.int16)
        made = np.zeros(n, dtype=np.int32)
        cell = np.zeros(n * r, dtype=np.int16)
        hist = np.zeros((4, n + 1), dtype=np.int64)
        keys = np.zeros(256)
        n_keys = np.zeros(1, dtype=np.int64)
        table = np.full((256, min(n + 1, 512)), np.nan)
        # Base set of the chunk, built along a chain of its subsets that only serves the bound.
        for i in range(n):
            cell[child[i]] += 1
        s, K = _score(1, r, 1.0, ess, priors[0], cell, hist, keys, n_keys, table, kind, per)
        best = s
        G = 1
        cur = 0
        size = 0
        closed = _bound(kind, K, log_r, 1.0, cmin, per, r, priors[1]) < best - tol
        base = c << lo
        for b in range(lo, m):
            if (base >> b) & 1:
                G = _refine(tmp[cur], cols[b], cards[b], tmp[1 - cur], ids, made, child, r, cell)
                cur = 1 - cur
                size += 1
                qb = _qf(base & ((1 << (b + 1)) - 1), cards, m)
                s, K = _score(
                    G, r, qb, ess, priors[size], cell, hist, keys, n_keys, table, kind, per
                )
                best = max(best, s)
                if _bound(kind, K, log_r, qb, cmin, per, r, priors[size + 1]) < best - tol:
                    closed = True
        out[base] = s
        cnt = 1
        if closed:
            done[c] = cnt
            continue
        gs[0, :] = tmp[cur]
        Gs[0] = G
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
            Gn = _refine(gs[depth], cols[j], cards[j], gs[depth + 1], ids, made, child, r, cell)
            S = masks[depth] | (1 << j)
            k = size + depth + 1
            qs = _qf(S, cards, m)
            s, K = _score(Gn, r, qs, ess, priors[k], cell, hist, keys, n_keys, table, kind, per)
            out[S] = s
            cnt += 1
            bd = max(bests[depth], s)
            if _bound(kind, K, log_r, qs, cmin, per, r, priors[k + 1]) < bd - tol:
                continue  # no superset of S in this subtree can be useful
            depth += 1
            Gs[depth] = Gn
            masks[depth] = S
            nxt[depth] = j + 1
            bests[depth] = bd
        done[c] = cnt
    return done


@njit(cache=True, parallel=True)
def _subset_max(f: np.ndarray, m: int) -> None:
    """In place: ``f[S]`` becomes the maximum of ``f`` over the subsets of ``S``."""
    size = f.shape[0]
    for b in range(m):
        bit = 1 << b
        for t in prange(size >> 1):
            S = ((t >> b) << (b + 1)) | (t & (bit - 1)) | bit
            x = f[S ^ bit]
            if x > f[S]:
                f[S] = x


@njit(cache=True, parallel=True)
def _candidate_flags(scores: np.ndarray, f: np.ndarray, m: int) -> np.ndarray:
    """Flag the sets scoring strictly better than every strict subset (``f``: subset maxima)."""
    size = scores.shape[0]
    keep = np.zeros(size, dtype=np.bool_)
    for S in prange(size):
        s = scores[S]
        if not s > -np.inf:
            continue
        ok = True
        for b in range(m):
            if (S >> b) & 1 and f[S ^ (1 << b)] >= s:
                ok = False
                break
        keep[S] = ok
    return keep


@njit(cache=True, parallel=True)
def _best_index(cand_masks: np.ndarray, cand_scores: np.ndarray, m: int, idx: np.ndarray) -> None:
    """Fill ``idx[U]``: the index of the best candidate that is a subset of ``U``.

    The candidates are sorted by score.
    """
    size = idx.shape[0]
    f = np.full(size, -np.inf)
    for t in range(cand_masks.shape[0]):
        f[cand_masks[t]] = cand_scores[t]
        idx[cand_masks[t]] = t
    for b in range(m):
        bit = 1 << b
        for t in prange(size >> 1):
            S = ((t >> b) << (b + 1)) | (t & (bit - 1)) | bit
            x = f[S ^ bit]
            if x > f[S] or (x == f[S] and idx[S ^ bit] < idx[S]):
                f[S] = x
                idx[S] = idx[S ^ bit]


def _priors(fast: CompiledBDeu, m: int) -> np.ndarray:
    return np.array([fast._prior_term(k) for k in range(m + 2)])


class _Counted(NamedTuple):
    """What the counting kernels need to score a discrete score: BDeu (kind 0) or BIC (kind 1)."""

    data: DiscreteDataset
    kind: int
    ess: float
    per: float
    priors: np.ndarray


def counted(score: Any, m: int) -> _Counted | None:
    """Return how the counting kernels score `score`, or None if they cannot.

    They take BDeu (through its engine) and the discrete BIC, on fewer than
    32768 rows (counts are int16) and at most 255 levels per variable.
    """
    if isinstance(score, CompiledBDeu):
        found = _Counted(score.data, 0, score.ess, 0.0, _priors(score, m))
    elif isinstance(score, BICScore):
        priors = np.arange(m + 2) * score._log_odds
        found = _Counted(score.data, 1, 0.0, score._per_parameter, priors)
    else:
        return None
    data = found.data
    if data.n_rows >= 2**15 or max(data.cardinalities, default=1) > 255:
        return None
    return found


def gaussian_parts(
    score: _GaussianScore, v: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return the matrix of `score` over the other variables then `v`, coefficients and floor."""
    M, a, b, c = score._exact_parts()
    idx = [u for u in range(score.n_vars) if u != v] + [v]
    return np.ascontiguousarray(M[np.ix_(idx, idx)]), a, b, c, score._floor(v)


def walk_parent_sets(
    score: Any, v: int, max_size: int, prune: bool, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Score the parent sets of ``v`` of at most ``max_size`` variables, depth first.

    The scores are any score's (see :func:`~causalbase.scores._compiled.engine_of`),
    a :class:`~causalbase.scores.FamilyScore`'s without its cache. Returns the
    sets scored, as bitmasks over the other variables in increasing order, and
    their scores, with ``-inf`` for a score that is not finite. With ``prune``
    the supersets of a set are left out when the score's ``_superset_bound``
    for it is below the best score along its chain of subsets, by more than
    ``tol``.
    """
    p = int(score.n_vars)
    others = np.array([u for u in range(p) if u != v], dtype=np.int64)
    m = len(others)
    size = min(max_size, m)
    family = isinstance(score, FamilyScore)
    bound = getattr(score, "_superset_bound", None) if prune else None
    empty = np.zeros(0, dtype=np.int64)
    first = float(score.local_score(v, ()))
    first = first if math.isfinite(first) else -math.inf
    masks, values = [0], [first]
    stack: list[tuple[int, np.ndarray, float, int]] = [(0, empty, first, 0)]
    while stack:
        mask, parents, best, start = stack.pop()
        if parents.size >= size or start >= m:
            continue
        if bound is not None and bound(v, parents) < best - tol:
            continue  # no superset of this set can be useful
        bits = np.arange(start, m)
        if family:
            vals = score._extensions_uncached(v, parents, others[bits])
        else:
            vals = score.local_scores_add(v, parents.tolist(), others[bits])
        vals = np.asarray(vals, dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, -np.inf)
        for b, s in zip(bits.tolist(), vals.tolist(), strict=True):
            S = mask | (1 << b)
            masks.append(S)
            values.append(s)
            stack.append((S, np.append(parents, others[b]), max(best, s), b + 1))
    return np.array(masks, dtype=np.int64), np.array(values)


def parent_set_scores(
    score: Any,
    v: int,
    high: int = 8,
    tol: float = 1e-6,
    prune: bool = True,
) -> tuple[np.ndarray, int]:
    """Local score of ``v`` given every subset of the other variables.

    Bit ``b`` of the index stands for the ``b``-th other variable in
    increasing order. Sets that the superset bound proves useless are
    ``-inf`` (all sets are scored when ``prune`` is false). Returns the scores
    and the number of sets actually scored. BDeu (a
    :class:`~causalbase.scores.BDeuScore` or its compiled engine) and the
    discrete BIC are scored by the counting kernels, the Gaussian scores by
    the kernels of :mod:`causalbase.exact._gaussian`, and any other score by
    :func:`walk_parent_sets`. A score uses as many threads as its ``n_jobs``.
    """
    score = engine_of(score)
    m = int(score.n_vars) - 1
    out = np.full(1 << m, -np.inf)
    with numba_threads(_threads_of(score)):
        found = counted(score, m)
        if found is not None:
            data = found.data
            others = np.array([u for u in range(m + 1) if u != v], dtype=np.int64)
            values = np.asarray(data.values)
            done = _all_scores(
                np.ascontiguousarray(values[:, others].T.astype(np.uint8)),
                np.asarray(data.cardinalities, dtype=np.int64)[others].copy(),
                np.ascontiguousarray(values[:, v].astype(np.uint8)),
                int(data.cardinalities[v]),
                found.ess,
                found.priors,
                min(high, m),
                tol if prune else np.inf,
                out,
                found.kind,
                found.per,
            )
            return out, int(done.sum())
        if isinstance(score, _GaussianScore):
            M, a, b, c, floor = gaussian_parts(score, v)
            others = np.array([u for u in range(m + 1) if u != v], dtype=np.int64)
            bounds = np.array(
                [score._superset_bound(v, others[:k]) if prune else np.inf for k in range(m + 1)]
            )
            done = gaussian_all_scores(M, a, b, c, floor, min(high, m), bounds, tol, out)
            return out, int(done.sum())
    masks, values = walk_parent_sets(score, v, m, prune, tol)
    out[masks] = values
    return out, len(masks)


def _threads_of(score: Any) -> int:
    """Return the threads `score` runs its kernels on: its ``n_jobs``, 1 for a score without."""
    return int(getattr(score, "_threads", 1))


def prune_candidates(scores: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Candidate parent sets (masks, scores), sorted by decreasing score."""
    f = scores.copy()
    _subset_max(f, m)
    keep = _candidate_flags(scores, f, m)
    masks = np.flatnonzero(keep).astype(np.int64)
    sc = scores[masks]
    order = np.lexsort((masks, -sc))
    return masks[order], sc[order]


@njit(cache=True)
def _order_dp(
    idx_tables: np.ndarray, cand_scores: np.ndarray, cand_offsets: np.ndarray, p: int
) -> tuple[float, np.ndarray]:
    """``F(W)`` for every variable set ``W`` and the sink achieving it."""
    size = 1 << p
    F = np.empty(size)
    sink = np.zeros(size, dtype=np.int8)
    F[0] = 0.0
    for W in range(1, size):
        best = -np.inf
        arg = -1
        for v in range(p):
            if (W >> v) & 1:
                U = W ^ (1 << v)
                low = U & ((1 << v) - 1)
                Uc = ((U >> (v + 1)) << v) | low
                s = F[U] + cand_scores[cand_offsets[v] + idx_tables[v, Uc]]
                if s > best:
                    best = s
                    arg = v
        F[W] = best
        sink[W] = arg
    return F[size - 1], sink


def _dp_search(score: Any, prune: bool, verbose: bool) -> SearchResult:
    """Find the optimal DAG for the local score of ``score`` by dynamic programming."""
    p = int(score.n_vars)
    m = p - 1
    t0 = time.perf_counter()
    cand_masks: list[np.ndarray]
    cand_scores: list[np.ndarray]
    n_scored: list[int]
    cand_masks, cand_scores, n_scored = [], [], []
    for v in range(p):
        tv = time.perf_counter()
        scores, done = parent_set_scores(score, v, prune=prune)
        masks, sc = prune_candidates(scores, m)
        del scores
        cand_masks.append(masks)
        cand_scores.append(sc)
        n_scored.append(done)
        if verbose:
            print(
                f"  var {v}: scored {done} of {1 << m} sets, {len(masks)} candidates, "
                f"{time.perf_counter() - tv:.1f}s",
                flush=True,
            )
    n_max = max(len(c) for c in cand_masks)
    dtype = np.uint16 if n_max < 2**16 else np.uint32
    idx_tables = np.zeros((p, 1 << m), dtype=dtype)
    for v in range(p):
        _best_index(cand_masks[v], cand_scores[v], m, idx_tables[v])
    t1 = time.perf_counter()
    offsets = np.zeros(p + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(c) for c in cand_masks])
    best, sink = _order_dp(idx_tables, np.concatenate(cand_scores), offsets, p)
    t2 = time.perf_counter()
    parents: list[Any] = [None] * p

    W = (1 << p) - 1
    while W:
        v = int(sink[W])
        U = W ^ (1 << v)
        Uc = ((U >> (v + 1)) << v) | (U & ((1 << v) - 1))
        mask = int(cand_masks[v][idx_tables[v, Uc]])
        parents[v] = sorted(b if b < v else b + 1 for b in range(m) if (mask >> b) & 1)
        W = U
    del idx_tables, sink
    dag = DAG.from_edges(p, [(u, v) for v in range(p) for u in parents[v]])
    sizes = [max(bin(int(x)).count("1") for x in c) for c in cand_masks]
    info = {
        "parents": parents,
        "n_candidates": [len(c) for c in cand_masks],
        "n_scored": n_scored,
        "time_scores": t1 - t0,
        "time_dp": t2 - t1,
        "max_candidate_size": sizes,
    }
    return SearchResult(dag, float(best), 0, [(t2 - t0, 0, float(best))], info)


def exact_search(
    score: FamilyScore,
    initial_order: Sequence[int] | None = None,
    *,
    max_parents: int | None = None,
    time_limit: float | None = None,
    prune: bool = True,
    verbose: bool = False,
) -> SearchResult:
    """Find the network of highest score, or the best one under a parent limit.

    Without a limit, dynamic programming over variable subsets (Silander and
    Myllymaki, 2006) finds the global optimum; memory grows as ``p 2^p``, so
    about 27 variables fit in 32 GB. With `max_parents`, an integer program
    solved with SCIP (which needs ``pip install "causalbase[exact]"``) finds
    the optimum among the DAGs whose parent sets have at most `max_parents`
    variables; see :mod:`causalbase.exact.ilp`.

    Parameters
    ----------
    score : FamilyScore
        The score. BDeu, the discrete BIC, the Gaussian BIC and BGe are
        scored by compiled kernels on as many threads as the score's
        ``n_jobs`` (the counting kernels of BDeu and the BIC on fewer than
        32768 rows and at most 255 levels per variable; larger discrete data
        goes through the score's own batches, more slowly). Any other score
        is scored family by family in Python.
    initial_order : sequence of int, optional
        Integer program only: a variable order whose best consistent network
        seeds SCIP with a first solution.
    max_parents : int or None, default=None
        Most parents of a node. None runs the dynamic programming.
    time_limit : float or None, default=None
        Integer program only: SCIP's time limit in seconds. The result is then
        the best network found, and ``info["status"]`` says whether it is
        proved optimal.
    prune : bool, default=True
        Dynamic programming only: skip the parent sets that the superset bound
        of the score proves useless (the result is the same). BGe has no bound,
        so it scores every set either way.
    verbose : bool, default=False
        Print progress per variable, and SCIP's log.

    Returns
    -------
    SearchResult
        ``n_iter`` is 0 and ``trace`` holds a single entry. ``info`` holds
        ``"parents"`` (sorted parents per node), ``"n_candidates"`` and
        ``"n_scored"`` (parent sets kept and scored, per node),
        ``"max_candidate_size"`` (largest candidate per node),
        ``"time_scores"`` and ``"time_dp"`` (seconds scoring the parent sets
        and searching, by dynamic programming or the integer program), and,
        with `max_parents`, SCIP's ``"status"`` ("optimal" when proved),
        ``"gap"``, ``"dual_bound"`` and ``"n_clusters"`` (cluster constraints
        added).

    Raises
    ------
    ValueError
        If `max_parents` is out of range, or, with `max_parents`, there are
        more than 64 variables.

    Examples
    --------
    >>> from causalbase.datasets import load_asia
    >>> from causalbase.exact import exact_search
    >>> from causalbase.scores import BDeuScore
    >>> result = exact_search(BDeuScore(load_asia().data))
    >>> result.dag.num_nodes
    8
    """
    if max_parents is not None and (
        isinstance(max_parents, bool) or not isinstance(max_parents, Integral) or max_parents < 0
    ):
        raise ValueError(f"max_parents must be a non-negative integer or None, got {max_parents!r}")
    check_time_limit(time_limit)
    engine = engine_of(score)
    if max_parents is None:
        return _dp_search(engine, prune, verbose)
    from causalbase.exact.ilp import _ilp_search

    return _ilp_search(engine, int(max_parents), initial_order, time_limit, verbose)
