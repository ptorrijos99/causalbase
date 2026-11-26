"""Optimal DAG under a parent-set size limit, by integer programming.

It is meant for networks too large for :mod:`causalbase.exact.dp`. The local
score of every parent set of at most ``max_size`` variables is computed with
the kernels of :mod:`causalbase.exact.dp` (bit for bit the compiled engine), the
sets that do not beat all their subsets are dropped, and the remaining
candidates enter an integer program in the style of GOBNILP (Cussens 2011;
Bartlett and Cussens 2017), solved with SCIP through PySCIPOpt: one binary per
candidate parent set, exactly one chosen per variable, and acyclicity through
cluster constraints (every set ``C`` of variables has a member with no parent
in ``C``), separated when a solution violates them.

The result is optimal among the DAGs whose parent sets have at most
``max_size`` variables; it is the global optimum only if no larger parent set
would be a candidate, which is not checked. It runs through
:func:`~causalbase.exact.exact_search` with ``max_parents``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit, prange

from causalbase.exact.dp import _priors, _qf, _refine, _score
from causalbase.graph import DAG
from causalbase.scores._compiled import engine_of
from causalbase.search._result import SearchResult

if TYPE_CHECKING:
    from causalbase.scores._compiled import CompiledBDeu
    from causalbase.scores.bdeu import BDeuScore

__all__: list[str] = []


@njit(cache=True, nogil=True, parallel=True)
def _limited_scores(
    cols: np.ndarray,
    cards: np.ndarray,
    child: np.ndarray,
    r: int,
    ess: float,
    priors: np.ndarray,
    kmax: int,
    starts: np.ndarray,
    out_masks: np.ndarray,
    out_scores: np.ndarray,
) -> None:
    """Score every non-empty parent set of at most ``kmax`` variables.

    Chunk ``i`` holds the sets whose smallest variable is ``i``, written from
    ``starts[i]`` in depth-first order.
    """
    m, n = cols.shape
    mc = 1
    for b in range(m):
        mc = max(mc, cards[b])
    for i in prange(m):
        gs = np.zeros((kmax + 1, n), dtype=np.int16)
        ids = np.full(n * mc, -1, dtype=np.int16)
        made = np.zeros(n, dtype=np.int32)
        cell = np.zeros(n * r, dtype=np.int16)
        hist = np.zeros((4, n + 1), dtype=np.int64)
        keys = np.zeros(256)
        n_keys = np.zeros(1, dtype=np.int64)
        table = np.full((256, min(n + 1, 512)), np.nan)
        masks = np.zeros(kmax + 1, dtype=np.int64)
        nxt = np.zeros(kmax + 1, dtype=np.int64)
        pos = starts[i]
        G = _refine(gs[0], cols[i], cards[i], gs[1], ids, made, child, r, cell)
        S = np.int64(1) << i
        s, K = _score(G, r, _qf(S, cards, m), ess, priors[1], cell, hist, keys, n_keys, table)
        out_masks[pos] = S
        out_scores[pos] = s
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
            Gn = _refine(gs[depth], cols[j], cards[j], gs[depth + 1], ids, made, child, r, cell)
            S = masks[depth] | (np.int64(1) << j)
            s, K = _score(
                Gn, r, _qf(S, cards, m), ess, priors[depth + 1], cell, hist, keys, n_keys, table
            )
            out_masks[pos] = S
            out_scores[pos] = s
            pos += 1
            depth += 1
            masks[depth] = S
            nxt[depth] = j + 1


@njit(cache=True)
def _sparse_candidates(
    masks: np.ndarray, scores: np.ndarray, sizes: np.ndarray, kmax: int
) -> np.ndarray:
    """Flag the sets scoring strictly better than all their subsets.

    ``masks`` is sorted and closed under subsets.
    """
    N = masks.shape[0]
    best = scores.copy()
    keep = np.zeros(N, dtype=np.bool_)
    for k in range(kmax + 1):
        for t in range(N):
            if sizes[t] != k:
                continue
            S = masks[t]
            strict = -np.inf
            x = S
            while x:
                low = x & -x
                x ^= low
                j = np.searchsorted(masks, S ^ low)
                if best[j] > strict:
                    strict = best[j]
            keep[t] = scores[t] > strict
            if strict > best[t]:
                best[t] = strict
    return keep


def limited_candidates(
    score: BDeuScore | CompiledBDeu, v: int, max_size: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Candidate parent sets of ``v`` among the sets of at most ``max_size`` variables.

    Returns the candidates as bitmasks over the other variables (bit ``b`` is
    the ``b``-th other variable in increasing order) with their scores, by
    decreasing score, and how many sets were scored. `score` is a
    :class:`~causalbase.scores.BDeuScore` or its compiled engine. The masks
    are int64, so the data may have at most 64 variables.
    """
    score = engine_of(score)
    data = score.data
    p = data.n_vars
    if p > 64:
        raise ValueError(
            "parent sets are 64-bit masks over the other variables, "
            f"so at most 64 variables; got {p}"
        )
    if data.n_rows >= 2**15 or max(data.cardinalities) > 255:
        raise ValueError("the kernel counts in int16 and stores values in uint8")
    others = np.array([u for u in range(p) if u != v], dtype=np.int64)
    m = len(others)
    k = min(max_size, m)
    values = np.asarray(data.values)
    cols = np.ascontiguousarray(values[:, others].T.astype(np.uint8))
    child = np.ascontiguousarray(values[:, v].astype(np.uint8))
    cards = np.asarray(data.cardinalities, dtype=np.int64)[others].copy()
    r = int(data.cardinalities[v])
    counts = np.array(
        [sum(math.comb(m - 1 - i, s) for s in range(k)) for i in range(m)], dtype=np.int64
    )
    starts = np.ones(m, dtype=np.int64)
    starts[1:] += np.cumsum(counts)[:-1]
    total = int(counts.sum()) + 1
    masks = np.zeros(total, dtype=np.int64)
    scores = np.empty(total)
    scores[0] = score.local_score(v, ())
    if k > 0:
        _limited_scores(
            cols, cards, child, r, score.ess, _priors(score, m), k, starts, masks, scores
        )
    order = np.argsort(masks)
    masks, scores = masks[order], scores[order]
    sizes = np.array([bin(int(x)).count("1") for x in masks], dtype=np.int64)
    keep = _sparse_candidates(masks, scores, sizes, k)
    cm, cs = masks[keep], scores[keep]
    o = np.lexsort((cm, -cs))
    return cm[o], cs[o], total


def _to_vars(mask: int, v: int) -> list[int]:
    mask = int(mask)
    return [b if b < v else b + 1 for b in range(mask.bit_length()) if (mask >> b) & 1]


def _ilp_search(
    score: CompiledBDeu,
    max_size: int,
    start_order: Sequence[int] | None,
    time_limit: float | None,
    verbose: bool,
) -> SearchResult:
    """Optimal DAG among those whose parent sets have at most ``max_size`` variables.

    ``start_order`` (a variable order) seeds SCIP with the best network
    consistent with it. See :func:`~causalbase.exact.exact_search` for the
    result.
    """
    data = score.data
    # PySCIPOpt is optional: typed when installed, Any (and these ignores unused) otherwise.
    from pyscipopt import (  # type: ignore[attr-defined,unused-ignore]
        SCIP_RESULT,
        Conshdlr,
        Model,
        quicksum,
    )

    p = data.n_vars
    t0 = time.perf_counter()
    cands: list[list[tuple[frozenset[int], float]]]
    n_scored: list[int]
    cands, n_scored = [], []
    for v in range(p):
        cm, cs, total = limited_candidates(score, v, max_size)
        cands.append([(frozenset(_to_vars(x, v)), float(s)) for x, s in zip(cm, cs, strict=True)])
        n_scored.append(total)
        if verbose:
            print(f"  var {v}: {total} sets, {len(cm)} candidates", flush=True)
    t1 = time.perf_counter()

    model = Model()
    if not verbose:
        model.hideOutput()
    if time_limit is not None:
        model.setParam("limits/time", time_limit)
    x = [[model.addVar(vtype="B", obj=s) for _, s in cands[v]] for v in range(p)]
    model.setMaximize()
    for v in range(p):
        model.addCons(quicksum(x[v]) == 1)

    def meeting(C: frozenset[int]) -> list[tuple[int, int]]:
        """Return the families of the members of ``C`` whose parent set meets ``C``."""
        return [(v, i) for v in C for i, (ps, _) in enumerate(cands[v]) if ps & C]

    def cluster_cons(C: frozenset[int], **kw: bool) -> None:
        # Every cluster C has a member with no parent in C; with exactly one family per variable
        # that is: at most |C| - 1 members take a parent set meeting C.
        model.addCons(quicksum(x[v][i] for v, i in meeting(C)) <= len(C) - 1, **kw)

    has_parent = [set().union(*[ps for ps, _ in cands[v]]) for v in range(p)]
    for u in range(p):  # 2-cycles as model rows, 3-cycles for SCIP to separate itself
        for v in range(u + 1, p):
            if u in has_parent[v] and v in has_parent[u]:
                cluster_cons(frozenset((u, v)))
                for w in range(v + 1, p):
                    C = frozenset((u, v, w))
                    if sum(bool(has_parent[a] & (C - {a})) for a in C) == 3:
                        cluster_cons(C, initial=False, separate=True)

    def violation(vals: list[list[float]], C: frozenset[int]) -> float:
        return sum(vals[v][i] for v, i in meeting(C)) - (len(C) - 1)

    def cycles(vals: list[list[float]]) -> list[frozenset[int]]:
        """Return the clusters violated by an integral solution.

        They are the variables left after peeling roots, and each cycle.
        """
        par = [
            next((cands[v][i][0] for i in range(len(vals[v])) if vals[v][i] > 0.5), frozenset())
            for v in range(p)
        ]
        left = set(range(p))
        while True:
            roots = {v for v in left if not par[v] & left}
            if not roots:
                break
            left -= roots
        if not left:
            return []
        found = {frozenset(left)}
        seen: set[int] = set()
        for s in sorted(left):
            if s in seen:
                continue
            path: list[int]
            path, v = [], s
            while v not in path:
                path.append(v)
                v = min(u for u in par[v] if u in left)
            cyc = frozenset(path[path.index(v) :])
            seen |= set(path)
            found.add(cyc)
        return list(found)

    def greedy(vals: list[list[float]]) -> list[frozenset[int]]:
        """Return clusters violated by a fractional solution, grown greedily from each variable."""
        w = np.zeros((p, p))  # w[u, v]: LP weight of u being a parent of v
        for v in range(p):
            for i, (ps, _) in enumerate(cands[v]):
                if vals[v][i] > 1e-9:
                    for u in ps:
                        w[u, v] += vals[v][i]
        found: set[frozenset[int]] = set()
        for s in range(p):
            C = [s]
            inside = np.zeros(p, dtype=bool)
            inside[s] = True
            while len(C) < p:
                gain = w[C, :].sum(axis=0) + w[:, C].sum(axis=1)
                gain[inside] = -1.0
                arg = int(np.argmax(gain))
                if gain[arg] <= 1e-6:
                    break
                C.append(arg)
                inside[arg] = True
                if violation(vals, frozenset(C)) > 1e-4:
                    found.add(frozenset(C))
                    break
        return list(found)

    def subip(
        vals: list[list[float]], limit: float = 5.0, max_cuts: int = 20
    ) -> list[frozenset[int]]:
        """Return the most violated clusters found by a sub-IP (as in GOBNILP).

        The sub-IP looks for the C maximising the LP mass of parent sets meeting C.

        A cluster is violated when the sum over its members of the LP mass of
        their parent sets that meet ``C`` exceeds ``|C| - 1``.
        """
        sub = Model()
        sub.hideOutput()
        sub.setParam("limits/time", limit)
        y = [sub.addVar(vtype="B", obj=-1.0) for _ in range(p)]
        for v in range(p):
            for i, (ps, _) in enumerate(cands[v]):
                if ps and vals[v][i] > 1e-6:
                    z = sub.addVar(lb=0.0, ub=1.0, obj=vals[v][i])
                    sub.addCons(z <= y[v])
                    sub.addCons(z <= quicksum(y[u] for u in ps))
        sub.addCons(quicksum(y) >= 2)
        sub.setMaximize()
        sub.setObjlimit(-1.0 + 1e-4)
        found: set[frozenset[int]] = set()
        for _ in range(max_cuts):
            sub.optimize()
            new: set[frozenset[int]] = set()
            for sol in sub.getSols():
                if sub.getSolObjVal(sol) > -1.0 + 1e-4:
                    new.add(frozenset(v for v in range(p) if sub.getSolVal(sol, y[v]) > 0.5))
            new -= found
            if not new:
                break
            found |= new
            sub.freeTransform()
            for C in new:  # look for other clusters next time
                sub.addCons(
                    quicksum(1 - y[v] for v in C) + quicksum(y[v] for v in range(p) if v not in C)
                    >= 1
                )
        return list(found)

    added: set[frozenset[int]] = set()

    class Acyclic(Conshdlr):  # type: ignore[misc,unused-ignore]
        def _vals(self, sol: Any = None) -> list[list[float]]:
            return [[self.model.getSolVal(sol, xi) for xi in x[v]] for v in range(p)]

        def _add(self, clusters: list[frozenset[int]], vals: list[list[float]], tol: float) -> int:
            n = 0
            for C in clusters:
                if C in added or violation(vals, C) <= tol:
                    continue
                added.add(C)
                cluster_cons(C, removable=False, separate=True)
                n += 1
            return n

        def conscheck(
            self,
            constraints: Any,
            solution: Any,
            checkintegrality: Any,
            checklprows: Any,
            printreason: Any,
            completely: Any,
        ) -> dict[str, Any]:
            return {
                "result": SCIP_RESULT.INFEASIBLE
                if cycles(self._vals(solution))
                else SCIP_RESULT.FEASIBLE
            }

        def consenfolp(
            self, constraints: Any, nusefulconss: Any, solinfeasible: Any
        ) -> dict[str, Any]:
            vals = self._vals()
            found = cycles(vals)
            if not found:
                return {"result": SCIP_RESULT.FEASIBLE}
            return {
                "result": SCIP_RESULT.CONSADDED
                if self._add(found, vals, 0.5)
                else SCIP_RESULT.INFEASIBLE
            }

        def consenfops(
            self, constraints: Any, nusefulconss: Any, solinfeasible: Any, objinfeasible: Any
        ) -> dict[str, Any]:
            return {
                "result": SCIP_RESULT.INFEASIBLE if cycles(self._vals()) else SCIP_RESULT.FEASIBLE
            }

        def conssepalp(self, constraints: Any, nusefulconss: Any) -> dict[str, Any]:
            vals = self._vals()
            n = self._add(greedy(vals), vals, 1e-4)
            n += self._add(subip(vals), vals, 1e-4)
            return {"result": SCIP_RESULT.CONSADDED if n else SCIP_RESULT.DIDNOTFIND}

        def conslock(self, constraint: Any, locktype: Any, nlockspos: int, nlocksneg: int) -> None:
            for row in x:
                for xi in row:
                    self.model.addVarLocks(xi, nlockspos + nlocksneg, nlockspos + nlocksneg)

    model.includeConshdlr(
        Acyclic(),
        "acyclic",
        "cluster constraints",
        sepapriority=1,
        enfopriority=-1,
        chckpriority=-1,
        sepafreq=1,
        needscons=False,
    )
    if start_order is not None:
        pos = {v: i for i, v in enumerate(start_order)}
        sol = model.createSol()
        for v in range(p):
            i = next(i for i, (ps, _) in enumerate(cands[v]) if all(pos[u] < pos[v] for u in ps))
            for j, xi in enumerate(x[v]):
                model.setSolVal(sol, xi, 1.0 if j == i else 0.0)
        model.addSol(sol)
    model.optimize()
    t2 = time.perf_counter()
    sol = model.getBestSol()
    chosen = [
        next(i for i in range(len(x[v])) if model.getSolVal(sol, x[v][i]) > 0.5) for v in range(p)
    ]
    parents = [sorted(cands[v][i][0]) for v, i in enumerate(chosen)]
    dag = DAG.from_edges(p, [(u, v) for v in range(p) for u in parents[v]])
    best = float(sum(cands[v][i][1] for v, i in enumerate(chosen)))
    info = {
        "parents": parents,
        "n_candidates": [len(c) for c in cands],
        "n_scored": n_scored,
        "time_scores": t1 - t0,
        "time_dp": t2 - t1,
        "max_candidate_size": [max(len(ps) for ps, _ in c) for c in cands],
        "status": model.getStatus(),
        "gap": model.getGap(),
        "dual_bound": model.getDualbound(),
        "n_clusters": len(added),
    }
    return SearchResult(dag, best, 0, [(t2 - t0, 0, best)], info)
