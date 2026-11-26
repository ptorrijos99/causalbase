"""Hill climbing in the space of DAGs (add, delete or reverse one arc).

The search runs on a :class:`~causalbase.scores.BDeuScore`. It is the
classic score-based search, as in bnlearn's ``hc`` and ``tabu``: from a
start DAG (empty by default), apply the valid move with the largest gain until
none improves. ``D[u, v]`` holds the gain of toggling the arc ``u -> v`` (adding
it when absent, deleting it when present), and reversing ``u -> v`` gains
``D[u, v] + D[v, u]``. A move changes at most two families, so after it only
their columns of ``D`` are scored again, with the batch kernel for additions.

Moves are numbered ``u * n + v`` (toggle) and ``n * n + u * n + v`` (reversal);
the best valid move wins and ties go to the smallest number, so the search is
deterministic. A move is valid when the graph stays acyclic: adding ``u -> v``
needs no directed path ``v ~> u``, reversing it needs no path ``u ~> v`` other
than the arc itself.

Options, with bnlearn's meaning (bnlearn's ``tabu``, ``max.tabu``, ``restart``
and ``perturb``):

* ``tabu_length``, ``max_tabu_moves``: tabu search. The best valid move whose
  resulting DAG is not among the last ``tabu_length`` visited is applied even
  if it loses score, and the search stops after ``max_tabu_moves`` moves
  without improving the best DAG.
* ``n_restarts``, ``perturbation``: after each climb, restart from the best DAG
  with ``perturbation`` random valid moves (add, delete or reverse) and climb
  again.
* ``time_limit``: wall-clock budget in seconds for the whole run.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Any

import numpy as np

from causalbase._validation import check_count, check_time_limit, resolve_seed
from causalbase.graph import DAG
from causalbase.scores._compiled import HAVE_NUMBA, CompiledBDeu, engine_of
from causalbase.scores.bdeu import BDeuScore
from causalbase.search._result import SearchResult

__all__ = ["hill_climbing"]

TOL = 1e-9

if HAVE_NUMBA:
    from numba import njit

    @njit(cache=True)
    def _best_moves(
        D: np.ndarray, adj: np.ndarray, desc: np.ndarray, mask_t: np.ndarray, mask_r: np.ndarray
    ) -> tuple[int, float, int, float]:  # pragma: no cover - compiled
        """Find the best valid toggle and best valid reversal in one pass, smallest number on ties.

        A toggle of ``u -> v`` is valid when it deletes an arc or ``v`` does not reach ``u``; a
        reversal of the arc ``u -> v`` when no other child of ``u`` reaches ``v``, checked only for
        reversals that beat the best so far. Masked moves (tabu) are skipped.
        """
        n = D.shape[0]
        bt, kt, br, kr = -np.inf, -1, -np.inf, -1
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                g = D[u, v]
                if g > bt and not mask_t[u, v] and (adj[u, v] or not desc[v, u]):
                    bt, kt = g, u * n + v
                if adj[u, v] and not mask_r[u, v]:
                    g = D[u, v] + D[v, u]
                    if g > br:
                        ok = True
                        for c in range(n):
                            if c != v and adj[u, c] and desc[c, v]:
                                ok = False
                                break
                        if ok:
                            br, kr = g, u * n + v
        return kt, bt, kr, br


class _State:
    def __init__(self, fast: CompiledBDeu, n: int, parents: list[set[int]]) -> None:
        self.fast, self.n = fast, n
        self.parents = parents
        self.children: list[set[int]] = [set() for _ in range(n)]
        for v, ps in enumerate(parents):
            for u in ps:
                self.children[u].add(v)
        self.adj = np.zeros((n, n), dtype=bool)
        for v, ps in enumerate(parents):
            self.adj[list(ps), v] = True
        self.desc = np.zeros((n, n), dtype=bool)  # desc[x, y]: a directed path x ~> y exists
        self.recompute_desc()
        self._no_mask = np.zeros((n, n), dtype=bool)
        self.fam = np.zeros(n)
        self.D = np.full((n, n), -math.inf)
        self._cache: list[dict[frozenset[int], float]] = [{} for _ in range(n)]
        for v in range(n):
            self.column(v)
        zob = np.random.default_rng(12345).integers(1, 2**62, size=(n, n))
        self.zob = zob
        self.hash = 0
        for v, ps in enumerate(parents):
            for u in ps:
                self.hash ^= int(zob[u, v])

    def ls(self, v: int, ps: frozenset[int]) -> float:
        s = self._cache[v].get(ps)
        if s is None:
            s = self.fast.local_score(v, ps)
            s = s if s == s else -math.inf
            self._cache[v][ps] = s
        return s

    def column(self, v: int) -> None:
        """Score the family of ``v`` again.

        Stores its score and the gain of toggling every other variable.
        """
        n, ps = self.n, self.parents[v]
        par = frozenset(ps)
        cur = self.ls(v, par)
        col = np.full(n, -math.inf)
        add = np.array([u for u in range(n) if u != v and u not in ps], dtype=np.int64)
        if add.size:
            vals = np.asarray(self.fast.local_scores_add(v, sorted(par), add), dtype=float)
            col[add] = np.where(np.isnan(vals), -math.inf, vals) - cur
        for u in ps:
            col[u] = self.ls(v, par - {u}) - cur
        self.D[:, v] = col
        self.fam[v] = cur

    def recompute_desc(self) -> None:
        """Rebuild the descendant matrix, children before parents (Kahn on the reversed graph)."""
        n = self.n
        indeg = [len(c) for c in self.children]
        stack = [x for x in range(n) if indeg[x] == 0]
        desc = np.zeros((n, n), dtype=bool)
        while stack:
            x = stack.pop()
            for c in self.children[x]:
                desc[x] |= desc[c]
                desc[x, c] = True
            for p in self.parents[x]:
                indeg[p] -= 1
                if indeg[p] == 0:
                    stack.append(p)
        self.desc = desc

    def valid(self, u: int, v: int, rev: bool) -> bool:
        if rev:  # no other path u ~> v: no child of u but v reaches v
            return not any(self.desc[c, v] for c in self.children[u] if c != v)
        if self.adj[u, v]:
            return True
        return not self.desc[v, u]

    def hash_after(self, u: int, v: int, rev: bool) -> int:
        h = self.hash ^ int(self.zob[u, v])
        return h ^ int(self.zob[v, u]) if rev else h

    def apply(self, u: int, v: int, rev: bool) -> None:
        self.hash = self.hash_after(u, v, rev)
        if rev or self.adj[u, v]:
            self.parents[v].discard(u)
            self.children[u].discard(v)
            self.adj[u, v] = False
            if rev:
                self.parents[u].add(v)
                self.children[v].add(u)
                self.adj[v, u] = True
                self.column(u)
        else:
            self.parents[v].add(u)
            self.children[u].add(v)
            self.adj[u, v] = True
            # Whatever reaches u (and u itself) now reaches v and all of v's descendants.
            rows = self.desc[:, u].copy()
            rows[u] = True
            reach_v = self.desc[v].copy()
            reach_v[v] = True
            self.desc[rows] |= reach_v
            self.column(v)
            return
        self.recompute_desc()
        self.column(v)

    def total(self) -> float:
        return float(sum(self.fam.tolist()))

    def snapshot(self) -> tuple[Any, ...]:
        return (
            [set(p) for p in self.parents],
            [set(c) for c in self.children],
            self.adj.copy(),
            self.D.copy(),
            self.fam.copy(),
            self.hash,
            self.desc.copy(),
        )

    def restore(self, snap: tuple[Any, ...]) -> None:
        parents, children, adj, D, fam, h, desc = snap
        self.parents, self.children = [set(p) for p in parents], [set(c) for c in children]
        self.adj, self.D, self.fam, self.hash, self.desc = (
            adj.copy(),
            D.copy(),
            fam.copy(),
            h,
            desc.copy(),
        )

    def pick(
        self, tabu_set: set[int] | None, kernel: bool = HAVE_NUMBA
    ) -> tuple[int, int, bool] | None:
        """Return the best valid move not leading to a tabu DAG.

        With ``tabu_set`` None, only improving moves are considered.
        """
        if not kernel:
            return self._pick_numpy(tabu_set)
        n = self.n
        mask_t = np.zeros((n, n), dtype=bool) if tabu_set is not None else self._no_mask
        mask_r = np.zeros((n, n), dtype=bool) if tabu_set is not None else self._no_mask
        while True:
            kt, gt, kr, gr = _best_moves(self.D, self.adj, self.desc, mask_t, mask_r)
            k, g, rev = (kt, gt, False) if gt >= gr else (kr, gr, True)
            if k < 0 or not math.isfinite(g) or (tabu_set is None and g <= TOL):
                return None
            u, v = divmod(int(k), n)
            if tabu_set is None or self.hash_after(u, v, rev) not in tabu_set:
                return u, v, rev
            (mask_r if rev else mask_t)[u, v] = True

    def _pick_numpy(self, tabu_set: set[int] | None) -> tuple[int, int, bool] | None:
        D = self.D
        R = np.where(self.adj, D + D.T, -math.inf)
        # Additions that would close a cycle (v already reaches u) are dropped at once;
        # deletions stay.
        Dm, Rm, masked = np.where(~self.adj & self.desc.T, -math.inf, D), R, True
        while True:
            kt = int(np.argmax(Dm))
            kr = int(np.argmax(Rm))
            gt, gr = Dm.flat[kt], Rm.flat[kr]
            k, g, rev = (kt, gt, False) if gt >= gr else (kr, gr, True)
            if not math.isfinite(g) or (tabu_set is None and g <= TOL):
                return None
            u, v = divmod(k, self.n)
            if self.valid(u, v, rev) and (
                tabu_set is None or self.hash_after(u, v, rev) not in tabu_set
            ):
                return u, v, rev
            if not masked:
                Dm, Rm, masked = D.copy(), R.copy(), True
            (Rm if rev else Dm).flat[k] = -math.inf

    def perturb(self, k: int, rng: random.Random) -> None:
        """Apply ``k`` random valid moves.

        Each is an addition, a deletion or a reversal with equal probability.
        """
        n = self.n
        for _ in range(k):
            for _ in range(100):
                kind = rng.randrange(3)
                if kind == 0:
                    u, v = rng.randrange(n), rng.randrange(n)
                    if (
                        u != v
                        and not self.adj[u, v]
                        and not self.adj[v, u]
                        and self.valid(u, v, False)
                    ):
                        self.apply(u, v, False)
                        break
                    continue
                arcs = np.flatnonzero(self.adj)
                if not arcs.size:
                    continue
                u, v = divmod(int(arcs[rng.randrange(arcs.size)]), n)
                if kind == 1:
                    self.apply(u, v, False)  # toggling an existing arc deletes it
                    break
                if self.valid(u, v, True):
                    self.apply(u, v, True)
                    break


def hill_climbing(
    score: BDeuScore,
    initial_graph: DAG | None = None,
    *,
    tabu_length: int = 0,
    max_tabu_moves: int = 0,
    n_restarts: int = 0,
    perturbation: int = 1,
    time_limit: float | None = None,
    random_state: int | None = None,
) -> SearchResult:
    """Run hill climbing, or tabu search, over DAGs.

    Each step applies the valid arc addition, deletion or reversal of largest
    gain, the smallest move number on ties (see the module docstring), until
    none improves.

    Parameters
    ----------
    score : BDeuScore
        The score to climb, with either prior; the search scores with its
        compiled engine.
    initial_graph : DAG, optional
        Where the first climb starts. The empty graph by default.
    tabu_length : int, default=0
        With a positive value, tabu search: the best valid move is applied even
        if it loses score, as long as its DAG is not among the last
        `tabu_length` visited.
    max_tabu_moves : int, default=0
        Tabu search stops after this many moves without improving its best DAG.
    n_restarts : int, default=0
        Climbs after the first, each from the best DAG so far perturbed by
        `perturbation` random valid moves.
    perturbation : int, default=1
        Random moves before each restart.
    time_limit : float or None, default=None
        Wall-clock budget in seconds for the whole run.
    random_state : int or None, default=None
        Seeds the perturbations; None draws a fresh seed.

    Returns
    -------
    SearchResult
        ``n_iter`` is the number of moves applied, perturbations excluded, and
        the iterations of ``trace`` count moves too. ``info`` holds
        ``"climbs"`` (climbs run, 1 + restarts done), ``"time"`` (seconds of the
        whole run), ``"time_first"`` and ``"score_first"`` (seconds to the end
        of the first climb, and its score).

    Raises
    ------
    ValueError
        If `initial_graph` has another number of nodes, or an option is out of
        range.

    Examples
    --------
    >>> from causalbase.datasets import load_asia
    >>> from causalbase.scores import BDeuScore
    >>> from causalbase.search import hill_climbing
    >>> result = hill_climbing(BDeuScore(load_asia().data), n_restarts=2, random_state=0)
    >>> result.info["climbs"]
    3
    """
    for name, value in (
        ("tabu_length", tabu_length),
        ("max_tabu_moves", max_tabu_moves),
        ("n_restarts", n_restarts),
        ("perturbation", perturbation),
    ):
        check_count(name, value)
    check_time_limit(time_limit)
    if initial_graph is not None and initial_graph.num_nodes != score.n_vars:
        raise ValueError(
            f"initial_graph has {initial_graph.num_nodes} nodes, expected {score.n_vars}"
        )
    seed = resolve_seed(random_state)
    n = score.n_vars
    t0 = time.perf_counter()
    deadline = None if time_limit is None else t0 + time_limit
    rng = random.Random(seed)
    parents = [
        set(initial_graph.parents(v)) if initial_graph is not None else set() for v in range(n)
    ]
    st = _State(engine_of(score), n, parents)
    best_score, best_snap = -math.inf, None
    moves, climbs = 0, 0
    trace: list[tuple[float, int, float]] = []
    time_first = score_first = 0.0

    def out_of_time() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    while True:
        climbs += 1
        recent: deque[int] = deque()
        tabu_set: set[int] | None = {st.hash} if tabu_length else None

        if tabu_length:
            recent.append(st.hash)
        climb_best, climb_snap, stale = st.total(), None, 0
        while not out_of_time():
            mv = st.pick(tabu_set)
            if mv is None:
                break
            st.apply(*mv)
            moves += 1
            if tabu_set is None:
                continue
            recent.append(st.hash)
            tabu_set.add(st.hash)
            if len(recent) > tabu_length:
                tabu_set.discard(recent.popleft())
            cur = st.total()
            if cur > climb_best + TOL:
                climb_best, climb_snap, stale = cur, st.snapshot(), 0
            else:
                stale += 1
                if stale >= max_tabu_moves:
                    break
        if tabu_set is None or climb_snap is None:
            climb_best, climb_snap = st.total(), st.snapshot()
        if climb_best > best_score + TOL:
            best_score, best_snap = climb_best, climb_snap
            trace.append((time.perf_counter() - t0, moves, best_score))
        if climbs == 1:
            time_first, score_first = time.perf_counter() - t0, best_score
        assert best_snap is not None  # the first climb always sets it (best_score starts at -inf)
        if climbs > n_restarts or out_of_time():
            break
        st.restore(best_snap)
        st.perturb(perturbation, rng)

    best_parents = best_snap[0]
    dag = DAG.from_edges(n, [(u, v) for v in range(n) for u in sorted(best_parents[v])])
    info = {
        "climbs": climbs,
        "time": time.perf_counter() - t0,
        "time_first": time_first,
        "score_first": score_first,
    }
    return SearchResult(dag, best_score, moves, trace, info)
