"""Local search over orders with insertion moves, on top of :class:`OrderScorer`.

A move takes one variable ``v`` and reinserts it at the position that gives
the best order score (the neighbourhood of OBS, INOBS and BOSS). All positions
are scored in one sweep: ``v`` jumps over its neighbours one at a time, and a
jump over ``u`` changes only the results of ``u`` and ``v``.

Almost no jump changes anything, so the sweep asks the evaluator only when it
can. Adding ``v`` to the predecessors of ``u`` changes ``u`` only if ``v``
would beat a step of ``u``'s greedy path or improve its final state
(:meth:`OrderScorer.breaks`, computed once per predecessor set for all
candidates). Removing ``v`` changes ``u`` only if ``v`` is toggled on that path
(:meth:`OrderScorer.path_moves`). Every other jump adds exactly zero, so the
gains, and hence the moves, are bit for bit those of a sweep that calls the
evaluator at every jump.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from causalbase.order.evaluator import OrderScorer

__all__ = ["InsertionSearch"]


class InsertionSearch:
    """First-improvement insertion local search over orders.

    Parameters
    ----------
    scorer : OrderScorer
        Order evaluator; its caches are shared by every move.
    """

    def __init__(self, scorer: OrderScorer) -> None:
        self.scorer = scorer
        n = scorer.n_vars
        # Row u: OrderScorer.breaks(u, preds) and the path moves of u as a mask,
        # for the predecessor set each row was computed with (None: not yet).
        self._brk = np.full((n, n), -1, dtype=np.int32)
        self._mov = np.zeros((n, n), dtype=bool)
        self._brk_preds: list[int | None] = [None] * n
        self._mov_preds: list[int | None] = [None] * n
        # Prefix masks and array of the order of the last call to best_move.
        self._order: list[int] | None = None
        self._pre: list[int] = []
        self._nodes = np.zeros(0, dtype=np.int64)

    def _moves_mask(self, node: int, preds: int) -> np.ndarray:
        mask = np.zeros(self.scorer.n_vars, dtype=bool)
        moves = self.scorer.path_moves(node, preds)
        if moves:
            mask[self.scorer._idx(moves)] = True
        return mask

    def _breaks_rows(self, nodes: Sequence[int], preds: Sequence[int]) -> np.ndarray:
        for u, p in zip(nodes, preds, strict=True):
            if self._brk_preds[u] != p:
                self._brk[u] = self.scorer.breaks(u, p)
                self._brk_preds[u] = p
        return self._brk

    def _moves_rows(self, nodes: Sequence[int], preds: Sequence[int]) -> np.ndarray:
        for u, p in zip(nodes, preds, strict=True):
            if self._mov_preds[u] != p:
                self._mov[u] = self._moves_mask(u, p)
                self._mov_preds[u] = p
        return self._mov

    def best_move(self, order: Sequence[int], v: int) -> tuple[float, int]:
        """Return the gain and new index of the best reinsertion of ``v``.

        The gain is 0 and the index its own if no reinsertion improves.

        Only the positions where a jump changes ``u`` or ``v`` are visited: the
        gain is constant in between, so the best position, found by the same
        rule as a sweep over every position, is always one of them.
        """
        sc = self.scorer
        n = len(order)
        i = order.index(v)
        if self._order != order:  # the prefix masks serve every move tried on the same order
            self._order = list(order)
            pre = [0] * (n + 1)  # pre[j]: bitmask of order[:j]
            for j, u in enumerate(order):
                pre[j + 1] = pre[j] | (1 << u)
            self._pre = pre
            self._nodes = np.asarray(order, dtype=np.int64)
        pre = self._pre
        nodes = self._nodes
        best_gain, best_j = 0.0, i
        bit_v = 1 << v

        # Leftwards: v jumps over order[i-1], order[i-2], ...; u gains v, v loses u.
        left = nodes[:i][::-1]  # left[e] = order[i - 1 - e]
        brk = self._breaks_rows(order[:i], pre[:i])
        u_changes = brk[left, v] >= 0
        pv = pre[i]
        sv = sc.local(v, pv)[0]
        v_changes = self._moves_mask(v, pv)[left]
        gain = 0.0
        e = 0
        while True:
            hits = np.flatnonzero(u_changes[e:] | v_changes[e:])
            if not hits.size:
                break
            e += int(hits[0])
            j = i - 1 - e
            u = order[j]
            pu = pre[j]
            du = sc.local(u, pu | bit_v)[0] - sc.local(u, pu)[0] if u_changes[e] else 0.0
            dv = 0.0
            if v_changes[e]:
                sv_new = sc.local(v, pu)[0]  # v's predecessors are now order[:j]
                dv = sv_new - sv
                sv = sv_new
                v_changes = v_changes.copy()
                v_changes[e + 1 :] = self._moves_mask(v, pu)[left[e + 1 :]]
            gain += dv + du
            if gain > best_gain + 1e-9:
                best_gain, best_j = gain, j
            e += 1

        # Rightwards: v jumps over order[i+1], order[i+2], ...; u loses v, v gains u.
        right = nodes[i + 1 :]  # right[e] = order[i + 1 + e]
        mov = self._moves_rows(order[i + 1 :], pre[i + 1 : n])
        u_changes = mov[right, v]
        pv = pre[i]
        sv = sc.local(v, pv)[0]
        v_changes = sc.breaks(v, pv)[right] >= 0
        gain = 0.0
        e = 0
        while True:
            hits = np.flatnonzero(u_changes[e:] | v_changes[e:])
            if not hits.size:
                break
            e += int(hits[0])
            j = i + 1 + e
            u = order[j]
            pu = pre[j]  # holds v
            du = sc.local(u, pu & ~bit_v)[0] - sc.local(u, pu)[0] if u_changes[e] else 0.0
            dv = 0.0
            if v_changes[e]:
                pv = pre[j + 1] & ~bit_v  # v's predecessors now include order[i+1..j]
                sv_new = sc.local(v, pv)[0]
                dv = sv_new - sv
                sv = sv_new
                v_changes = v_changes.copy()
                v_changes[e + 1 :] = sc.breaks(v, pv)[right[e + 1 :]] >= 0
            gain += dv + du
            if gain > best_gain + 1e-9:
                best_gain, best_j = gain, j
            e += 1
        return best_gain, best_j

    def run(
        self,
        order: Sequence[int],
        rng: np.random.Generator,
        max_passes: int | None = None,
        deadline: float | None = None,
    ) -> tuple[list[int], float]:
        """Apply improving moves, visiting the variables in a random order each pass.

        Stops after a pass without improvement, after ``max_passes`` passes or
        at ``deadline`` (a ``time.perf_counter`` value). Returns the order and
        its score.
        """
        order = [int(u) for u in order]
        passes = 0
        while True:
            improved = False
            for v in rng.permutation(len(order)):
                v = int(v)
                g, j = self.best_move(order, v)
                if g > 1e-9:
                    order.remove(v)
                    order.insert(j, v)
                    improved = True
                if deadline is not None and time.perf_counter() > deadline:
                    return order, self.scorer.score(order).total
            passes += 1
            if not improved or (max_passes is not None and passes >= max_passes):
                return order, self.scorer.score(order).total
