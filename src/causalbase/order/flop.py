"""FLOP (Wienöbst, Henckel and Weichwald, ICLR 2026) ported to every score here (a baseline).

Follows ``flopsearch`` (v0.3.0): for each node, a non-greedy grow-shrink (accept
any parent that improves, candidates in shuffled order, then remove any that
improves) warm-started from the previous parent set, with its two early breaks
(an added candidate that does not improve leaves the parents as they were; a
removed node that was not a parent changes nothing); reinsertion of every
variable at its best position, visiting the variables in their current order;
and iterated local search from the best order with ``round(ln p)`` random swaps,
each restart rebuilding the parent sets from scratch and accepted only when
strictly better. The local score is any :class:`~causalbase.scores.FamilyScore` (a
:class:`~causalbase.scores.BDeuScore` is scored with its compiled engine), with a per-node
cache, so only the evaluator differs from
:class:`~causalbase.order.ils.IteratedLocalSearch`.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from causalbase._validation import check_count, check_time_limit, resolve_seed
from causalbase.graph import DAG
from causalbase.order.polished import start_order
from causalbase.scores._compiled import engine_of
from causalbase.search import SearchResult

if TYPE_CHECKING:
    from causalbase.scores import FamilyScore

__all__ = ["flop"]

EPS = 1e-9


def _bits(mask: int) -> list[int]:
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


class _Evaluator:
    def __init__(self, fast: Any, n: int, seed: int) -> None:
        self.fast = fast
        self.rng = random.Random(seed)
        self.cache: list[dict[int, float]] = [{} for _ in range(n)]

    def ls(self, v: int, mask: int) -> float:
        c = self.cache[v]
        s = c.get(mask)
        if s is None:
            s = self.fast.local_score(v, _bits(mask))
            c[mask] = s
        return s

    def add(self, v: int, mask: int, w: int) -> float:
        c = self.cache[v]
        m = mask | (1 << w)
        s = c.get(m)
        if s is None:
            s = float(self.fast.local_scores_add(v, _bits(mask), [w])[0])
            c[m] = s
        return s

    def grow(self, v: int, P: int, cur: float, Z: int) -> tuple[int, float]:
        while True:
            changed = False
            cand = _bits(Z & ~P)
            self.rng.shuffle(cand)
            for w in cand:
                s = self.add(v, P, w)
                if s > cur:
                    P |= 1 << w
                    cur = s
                    changed = True
            if not changed:
                return P, cur

    def shrink(self, v: int, P: int, cur: float) -> tuple[int, float]:
        while True:
            changed = False
            par = _bits(P)
            self.rng.shuffle(par)
            for w in par:
                s = self.ls(v, P & ~(1 << w))
                if s > cur:
                    P &= ~(1 << w)
                    cur = s
                    changed = True
            if not changed:
                return P, cur

    def fit(self, v: int, Z: int) -> tuple[int, float]:
        return self.shrink(v, *self.grow(v, 0, self.ls(v, 0), Z))

    def minus(self, v: int, P: int, cur: float, Z: int, r: int) -> tuple[int, float]:
        if not (P >> r) & 1:
            return P, cur
        P &= ~(1 << r)
        return self.shrink(v, *self.grow(v, P, self.ls(v, P), Z))

    def plus(self, v: int, P: int, cur: float, Z: int, r: int) -> tuple[int, float]:
        s = self.add(v, P, r)
        if not s > cur:
            return P, cur
        return self.shrink(v, *self.grow(v, P | (1 << r), s, Z))


def _reinsert(ev: _Evaluator, perm: list[int], par: list[int], sc: list[float], v: int) -> float:
    n = len(perm)
    i = perm.index(v)
    pre = [0] * (n + 1)
    for j, u in enumerate(perm):
        pre[j + 1] = pre[j] | (1 << u)
    best, best_pos = EPS, i
    v_at: dict[int, tuple[int, float]] = {}
    z_at: dict[int, tuple[int, float]] = {}
    Pv, sv, diff = par[v], sc[v], 0.0
    for pos in range(i - 1, -1, -1):
        z = perm[pos]
        Pv, ns = ev.minus(v, Pv, sv, pre[pos], z)
        diff += ns - sv
        sv = ns
        zP, zs = ev.plus(z, par[z], sc[z], pre[pos] | (1 << v), v)
        diff += zs - sc[z]
        z_at[pos] = (zP, zs)
        if diff > best:
            best, best_pos = diff, pos
            v_at[pos] = (Pv, sv)
    Pv, sv, diff = par[v], sc[v], 0.0
    for pos in range(i + 1, n):
        z = perm[pos]
        Pv, ns = ev.plus(v, Pv, sv, pre[pos + 1] & ~(1 << v), z)
        diff += ns - sv
        sv = ns
        zP, zs = ev.minus(z, par[z], sc[z], pre[pos] & ~(1 << v), v)
        diff += zs - sc[z]
        z_at[pos] = (zP, zs)
        if diff > best:
            best, best_pos = diff, pos
            v_at[pos] = (Pv, sv)
    if best_pos == i:
        return 0.0
    par[v], sc[v] = v_at[best_pos]
    span = range(best_pos, i) if best_pos < i else range(i + 1, best_pos + 1)
    for pos in span:
        par[perm[pos]], sc[perm[pos]] = z_at[pos]
    perm.pop(i)
    perm.insert(best_pos, v)
    return best


def _local_search(
    ev: _Evaluator, perm: list[int], deadline: float | None
) -> tuple[list[int], list[int], list[float]]:
    n = len(perm)
    par, sc = [0] * n, [0.0] * n
    pre = 0
    for v in perm:
        par[v], sc[v] = ev.fit(v, pre)
        pre |= 1 << v
    while True:
        last = sum(sc)
        for v in list(perm):
            _reinsert(ev, perm, par, sc, v)
            if deadline is not None and time.perf_counter() > deadline:
                return perm, par, sc
        if sum(sc) - last <= EPS:
            return perm, par, sc


def flop(
    score: FamilyScore,
    initial_order: Sequence[int] | None = None,
    *,
    max_iter: int | None = None,
    time_limit: float | None = None,
    random_state: int | None = None,
) -> SearchResult:
    """Run FLOP: its local search over orders, then its iterated local search.

    One iteration is one kick of the iterated local search: ``round(ln p)``
    random swaps of the best order, a local search from there with the parent
    sets rebuilt from scratch, and a move there if it is strictly better.

    Parameters
    ----------
    score : FamilyScore
        The score: BDeu with either prior (scored with its compiled engine),
        BIC, Gaussian BIC or BGe, or a score of one's own.
    initial_order : sequence of int, optional
        Order the first local search starts from. By default a topological
        order of the network GES learns on the same score, learned before the
        time limit starts.
    max_iter : int or None, default=None
        Kicks after the first local search; 0 runs the first local search only.
    time_limit : float or None, default=None
        Seconds of search, checked inside the local searches too. At least one
        of `max_iter` and `time_limit` must be set.
    random_state : int or None, default=None
        Seeds the candidate orders of the grow-shrink and the kicks; None draws
        a fresh seed.

    Returns
    -------
    SearchResult
        ``n_iter`` counts the kicks. ``info`` holds ``"order"`` (the best order),
        ``"parents"`` (the parent set of each node, as frozensets, consistent
        with that order) and ``"restarts"`` (local searches run, ``n_iter + 1``).

    Raises
    ------
    ValueError
        If neither limit is set, or an option is out of range.

    Examples
    --------
    >>> from causalbase.datasets import load_asia
    >>> from causalbase.order import flop
    >>> from causalbase.scores import BDeuScore
    >>> flop(BDeuScore(load_asia().data), max_iter=2, random_state=0).n_iter
    2
    """
    check_count("max_iter", max_iter, allow_none=True)
    check_time_limit(time_limit)
    if max_iter is None and time_limit is None:
        raise ValueError("set max_iter, time_limit or both, or the search never ends")
    seed = resolve_seed(random_state)
    start = start_order(score, initial_order)
    # FLOP counts local searches: the first one, then one per kick.
    max_restarts = None if max_iter is None else max_iter + 1
    n = len(start)
    ev = _Evaluator(engine_of(score), n, seed)
    rng = random.Random(seed + 1)
    k = max(1, round(math.log(n)))
    t0 = time.perf_counter()
    deadline = None if time_limit is None else t0 + time_limit
    best_perm, best_par, best_sc = _local_search(ev, list(start), deadline)
    best = sum(best_sc)
    trace = [(time.perf_counter() - t0, 0, best)]
    restarts = 1
    while (max_restarts is None or restarts < max_restarts) and (
        deadline is None or time.perf_counter() < deadline
    ):
        perm = list(best_perm)
        for _ in range(k):
            i, j = rng.randrange(n), rng.randrange(n)
            perm[i], perm[j] = perm[j], perm[i]
        perm, par, sc = _local_search(ev, perm, deadline)
        restarts += 1
        if sum(sc) > best + EPS:
            best_perm, best_par, best_sc, best = perm, par, sc, sum(sc)
            trace.append((time.perf_counter() - t0, restarts - 1, best))
    parents = [frozenset(_bits(m)) for m in best_par]
    dag = DAG.from_edges(n, [(u, v) for v in range(n) for u in parents[v]])
    info = {"order": list(best_perm), "parents": parents, "restarts": restarts}
    return SearchResult(dag, sum(best_sc), restarts - 1, trace, info)
