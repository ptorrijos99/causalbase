"""Order evaluators: the score of the best network consistent with an order.

:class:`OrderScorer` is the evaluator of MCTS-BN (IPMU 2024): for each node,
a greedy hill climbing that adds or removes one parent at a time among the
node's predecessors in the order, always taking the move of largest gain while
that gain is positive. Nodes are independent given the order, so the score of
an order is the sum of the per-node results.

The search is greedy and deterministic (ties go to the smallest candidate
index), so a node's result depends only on its set of predecessors. When that
set changes, :class:`OrderScorer` replays the stored greedy path instead of
climbing again: removing a candidate that was never chosen changes no argmax,
and a new candidate changes the path only if it beats the move that was taken
at some step, or improves on the final state. The result is identical to a
climb from scratch; ``incremental=False`` gives that climb, for testing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import numpy as np

if TYPE_CHECKING:
    from causalbase.scores import BDeuScore

__all__ = ["OrderResult", "OrderScorer"]


class OrderResult(NamedTuple):
    """Score of an order and the parent set chosen for every node."""

    total: float
    parents: tuple[frozenset[int], ...]


class _OrderEvaluator(Protocol):
    """Anything that scores a complete order of ``n_vars`` variables."""

    n_vars: int

    def score(self, order: Sequence[int]) -> OrderResult:
        """Score a complete order."""
        ...


def _bits(mask: int) -> list[int]:
    """Return the indices of the set bits of ``mask``, in increasing order."""
    out: list[int] = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


class _NodeMemo:
    __slots__ = ("final_parents", "final_score", "final_state", "path", "preds")

    def __init__(
        self, preds: int, path: list[list[Any]], final_state: int, final_score: float
    ) -> None:
        self.preds = preds
        # [state mask, gain, candidate, ok, beat] for every applied move; ``ok`` holds the
        # candidates already known not to beat that move and ``beat`` those known to beat
        # it. Entries are shared by every memo with the same path prefix, so what one
        # replay or one call to ``breaks`` learns serves all.
        self.path = path
        self.final_state = final_state
        self.final_score = final_score
        self.final_parents = frozenset(_bits(final_state))


class OrderScorer:
    """Hill climbing restricted to an order, with exact incremental replay.

    Parent sets are bitmasks internally, and local scores are cached here per
    node and parent mask, so each family is scored once, by the compiled
    engine of the :class:`~causalbase.scores.BDeuScore` (which does not cache
    them).

    Parameters
    ----------
    score : BDeuScore
        The score, kept as ``family_score``; its compiled engine scores the
        families. Any other object with ``n_vars`` and
        ``local_score(node, parents)`` (and, optionally, the batch
        ``local_scores_add(node, parents, candidates)``) is used as it is.
    max_steps : int, default=1000
        Maximum number of moves per node (1000 in the Java implementation).
    incremental : bool, default=True
        Replay the stored greedy path when a node's predecessors change,
        instead of climbing again from the empty parent set.
    memo_slots : int, default=8
        Greedy paths kept per node. A new predecessor set is replayed from the
        stored one closest to it (fewest predecessors added or removed), which
        matters when rollouts alternate between several base orders.
    exact_cap : int, default=1024
        Answers kept per node for predecessor sets met before, oldest dropped
        first; 0 keeps none. A node often meets a set again after it has left the slots
        (same base order, prefix variables already in front).
    n_restarts : int, default=0
        Extra greedy climbs per node. Climb ``i + 1`` runs on the predecessors of
        climb ``i`` minus the first parent climb ``i`` added (its best single
        parent), and the node keeps the best of all climbs (the earliest on
        ties). This gets past a parent that wins alone but blocks better sets,
        such as a variable with many states whose CPT leaves no room for more
        parents. Each climb is an ordinary greedy climb on a smaller predecessor
        set, so it reuses the memos and replays, and :meth:`breaks` and
        :meth:`path_moves` stay exact.
    """

    def __init__(
        self,
        score: BDeuScore | Any,
        max_steps: int = 1000,
        incremental: bool = True,
        memo_slots: int = 8,
        exact_cap: int = 1024,
        n_restarts: int = 0,
    ) -> None:
        self.family_score = score
        n_vars = self.n_vars = score.n_vars
        self.max_steps = max_steps
        self.incremental = incremental
        self.memo_slots = max(1, memo_slots)
        self.exact_cap = max(0, exact_cap)
        self.n_restarts = max(0, n_restarts)
        # Per node: predecessor mask -> best answer over the restart chain.
        self._best: list[dict[int, tuple[float, frozenset[int]]]] = [{} for _ in range(n_vars)]
        self._exact: list[dict[int, tuple[float, frozenset[int]]]] = [{} for _ in range(n_vars)]
        self._memo: list[list[_NodeMemo]] = [[] for _ in range(n_vars)]
        # Per node: predecessor mask -> its memo, for the memos in the slots.
        self._slot: list[dict[int, _NodeMemo]] = [{} for _ in range(n_vars)]
        self._cache: list[dict[int, float]] = [{} for _ in range(n_vars)]
        # Per node: parent mask -> candidates known not to improve on that parent set,
        # and those known to improve on it.
        self._final_ok: list[dict[int, int]] = [{} for _ in range(n_vars)]
        self._final_bad: list[dict[int, int]] = [{} for _ in range(n_vars)]
        # Per node: parent masks whose neighbour vector has every entry filled in.
        self._full: list[set[int]] = [set() for _ in range(n_vars)]
        # Per node: parent mask -> bitmask of the filled entries of its neighbour vector, and the
        # improving candidates among them with the bitmask they were listed for (see _positive).
        self._known: list[dict[int, int]] = [{} for _ in range(n_vars)]
        self._pos: list[dict[int, tuple[int, list[tuple[int, float]]]]] = [
            {} for _ in range(n_vars)
        ]
        self._all = (1 << n_vars) - 1
        # Per node: parent mask -> vector whose entry c is the score of the mask with c
        # toggled (NaN while unknown, -inf for a non-finite score). A climb step reads
        # all its candidates from one vector instead of one dictionary lookup each.
        self._vec: list[dict[int, np.ndarray]] = [{} for _ in range(n_vars)]
        self._nbytes = (n_vars + 7) // 8
        engine = getattr(score, "_engine", None)
        self._scoring = engine() if engine is not None else score
        self._batch = getattr(self._scoring, "local_scores_add", None)
        self.n_local_calls = 0
        self.n_reused = 0

    def _ls(self, node: int, mask: int) -> float:
        self.n_local_calls += 1
        cache = self._cache[node]
        value = cache.get(mask)
        if value is None:
            value = self._scoring.local_score(node, frozenset(_bits(mask)))
            cache[mask] = value
        return value

    def _idx(self, mask: int) -> np.ndarray:
        """Return the indices of the set bits of ``mask`` as an array."""
        raw = np.frombuffer(mask.to_bytes(self._nbytes, "little"), dtype=np.uint8)
        return np.flatnonzero(np.unpackbits(raw, bitorder="little"))

    def _lookup(self, node: int, state: int, cands: Sequence[int] | np.ndarray) -> np.ndarray:
        """Neighbour vector of ``state`` with the entries of ``cands`` filled in."""
        self.n_local_calls += len(cands)
        vecs = self._vec[node]
        vec = vecs.get(state)
        if vec is None:
            vec = np.full(self.n_vars, np.nan)
            vec[node] = -math.inf  # never asked for; without it no vector would ever be complete
            vecs[state] = vec
        elif state in self._full[node]:
            return vec
        if isinstance(cands, np.ndarray):
            missing = cands[np.isnan(vec[cands])]
        else:
            missing = np.array([c for c in cands if vec.item(c) != vec.item(c)], dtype=np.int64)
        if missing.size:
            if self._batch is not None:
                # Additions go straight to the batch kernel: going through the
                # family cache costs a dictionary probe and insert per candidate,
                # more than the few families reachable from two states save.
                inside = np.zeros(self.n_vars, dtype=bool)
                inside[self._idx(state)] = True
                add = missing[~inside[missing]]
                missing = missing[inside[missing]]
                if add.size:
                    vals = np.asarray(self._batch(node, _bits(state), add), dtype=float)
                    vec[add] = np.where(np.isnan(vals), -math.inf, vals)
            cache = self._cache[node]
            for c in missing.tolist():
                m = state ^ (1 << c)
                v = cache.get(m)
                if v is None:
                    v = self._scoring.local_score(node, frozenset(_bits(m)))
                    cache[m] = v
                vec[c] = v if v == v else -math.inf
            known = int.from_bytes(np.packbits(vec == vec, bitorder="little").tobytes(), "little")
            self._known[node][state] = known
            if known == self._all:
                self._full[node].add(state)
        return vec

    def _positive(self, node: int, state: int, need: int) -> list[tuple[int, float]] | None:
        """List the scored candidates whose toggle improves ``state``, with their gains.

        The list is best first (smallest index on ties). None unless every
        candidate of ``need`` has been scored against ``state``.
        Otherwise the best move among ``need`` is the first of the list that belongs
        to it, and only candidates of the list can beat a move, so a climb or a
        replay needs no pass over the predecessors.
        """
        known = self._known[node].get(state, 0)
        if need & ~known or not known:
            return None
        hit = self._pos[node].get(state)
        if hit is None or hit[0] != known:
            gains = self._vec[node][state] - self._ls(node, state)
            idx = np.flatnonzero(gains > 0)
            if idx.size > 1:
                idx = idx[np.lexsort((idx, -gains[idx]))]
            hit = (known, list(zip(idx.tolist(), gains[idx].tolist(), strict=True)))
            self._pos[node][state] = hit
        return hit[1]

    def _climb(
        self, node: int, preds: int, state: int, budget: int
    ) -> tuple[list[list[Any]], int, float]:
        path: list[list[Any]] = []
        cands = None
        cur = self._ls(node, state)
        while len(path) < budget:
            pos = self._positive(node, state, preds)
            if pos is not None:
                self.n_local_calls += preds.bit_count()  # the lookups this step stands for
                for best_c, best_gain in pos:  # noqa: B007 - best_gain is read after the loop
                    if (preds >> best_c) & 1:
                        break
                else:
                    break
                vec = self._vec[node][state]
            else:
                if cands is None:
                    cands = self._idx(preds)
                if not cands.size:
                    break
                vec = self._lookup(node, state, cands)
                gains = vec[cands] - cur
                j = int(gains.argmax())  # first maximum, i.e. the smallest index on ties
                best_gain = float(gains[j])
                if not best_gain > 0.0:
                    break
                best_c = int(cands[j])
            path.append([state, best_gain, best_c, preds, 0])
            state ^= 1 << best_c
            # Finite, as its gain is positive; the same float _ls would give.
            cur = vec.item(best_c)
            self._cache[node][state] = cur
        if len(path) < budget:
            fin = self._final_ok[node]
            fin[state] = fin.get(state, 0) | preds
        return path, state, cur

    def _chain(self, node: int, preds: int) -> list[int]:
        """Return the predecessor sets of the climbs.

        They are ``preds``, then each one minus the first parent its climb added.
        """
        chain = [preds]
        for _ in range(self.n_restarts):
            path = self._memo_of(node, chain[-1]).path
            if not path:
                break
            chain.append(chain[-1] & ~(1 << path[0][2]))
        return chain

    def local(self, node: int, preds: int) -> tuple[float, frozenset[int]]:
        """Best local score of ``node`` and its parents, given a predecessor bitmask."""
        if not self.n_restarts:
            return self._local0(node, preds)
        cache = self._best[node] if self.incremental and self.exact_cap else None
        if cache is not None:
            hit = cache.get(preds)
            if hit is not None:
                self.n_reused += 1
                return hit
        chain = self._chain(node, preds)
        best = self._local0(node, chain[0])
        for p in chain[1:]:
            r = self._local0(node, p)
            if r[0] > best[0]:
                best = r
        if cache is not None:
            cache[preds] = best
            if len(cache) > self.exact_cap:
                del cache[next(iter(cache))]
        return best

    def _local0(self, node: int, preds: int) -> tuple[float, frozenset[int]]:
        """Return the result of the single greedy climb of ``node`` given ``preds``."""
        if self.incremental:
            hit = self._exact[node].get(preds)
            if hit is not None:
                self.n_reused += 1
                return hit
        memo = self._memo_of(node, preds)
        return memo.final_score, memo.final_parents

    def path_moves(self, node: int, preds: int) -> int:
        """Return the bitmask of the candidates toggled along the greedy path of ``node``.

        The path is the one given ``preds``. Removing a predecessor outside this
        mask leaves the result of :meth:`local` unchanged. With restarts it is
        the union over the climbs of the chain: a predecessor off every path
        changes no climb, hence no first parent either.
        """
        mask = 0
        for p in self._chain(node, preds) if self.n_restarts else [preds]:
            for entry in self._memo_of(node, p).path:
                mask |= 1 << entry[2]
        return mask

    def breaks(self, node: int, preds: int) -> np.ndarray:
        """For each variable ``c`` outside ``preds``, where adding it would alter the greedy path.

        Entry ``c`` is the first step of the path of ``node`` that ``c`` would
        beat (the path length if it only improves on the final state), or -1
        if adding ``c`` to ``preds`` leaves the result of :meth:`local`
        unchanged. Entries for ``node`` and for ``preds`` are -1. The test is
        the one :meth:`_replay` applies, done for all candidates at once, so
        their neighbour vectors are filled in large batches. With restarts an
        entry is non-negative when ``c`` would alter any climb of the chain
        (the value is then only a flag): if it alters none, every climb and
        every first parent stay as they were.
        """
        brk = np.full(self.n_vars, -1, dtype=np.int64)
        if not self.n_restarts:
            for t, hit in self._breaks0(node, preds):
                brk[self._idx(hit)] = t
            return brk
        outside = self._all & ~preds & ~(1 << node)
        for p in self._chain(node, preds):
            for t, hit in self._breaks0(node, p):
                hit &= outside
                if hit:
                    idx = self._idx(hit)
                    brk[idx] = np.maximum(brk[idx], t)
        return brk

    def _mask(self, idx: np.ndarray) -> int:
        """Bitmask with the bits of ``idx`` set."""
        mask = 0
        for i in idx.tolist():
            mask |= 1 << i
        return mask

    def _breaks0(self, node: int, preds: int) -> list[tuple[int, int]]:
        """Return ``(step, candidates)`` pairs.

        The candidates are those outside ``preds`` that first alter the path at
        that step. What a step answers for a candidate depends only on the step,
        so it is kept with it (``ok`` and ``beat``, and their counterparts for
        the final state) and each candidate is scored against each step once.
        """
        memo = self._memo_of(node, preds)
        out: list[tuple[int, int]] = []
        alive = self._all & ~preds & ~(1 << node)
        for t, entry in enumerate(memo.path):
            if not alive:
                return out
            beat = entry[4]
            todo = alive & ~entry[3] & ~beat
            if todo:
                state, gain, cand = entry[0], entry[1], entry[2]
                idx = self._idx(todo)
                g = self._lookup(node, state, idx)[idx] - self._ls(node, state)
                hit = (g > gain) | ((g == gain) & (idx < cand))
                if hit.any():
                    new = self._mask(idx[hit])
                    beat |= new
                    entry[4] = beat
                    todo &= ~new
                entry[3] |= todo
            hit = alive & beat
            if hit:
                out.append((t, hit))
                alive &= ~hit
        if alive and len(memo.path) < self.max_steps:
            state = memo.final_state
            fin, bad = self._final_ok[node], self._final_bad[node]
            beat = bad.get(state, 0)
            todo = alive & ~fin.get(state, 0) & ~beat
            if todo:
                idx = self._idx(todo)
                hit = self._lookup(node, state, idx)[idx] - memo.final_score > 0
                if hit.any():
                    new = self._mask(idx[hit])
                    beat |= new
                    bad[state] = beat
                    todo &= ~new
                fin[state] = fin.get(state, 0) | todo
            hit = alive & beat
            if hit:
                out.append((len(memo.path), hit))
        return out

    def _memo_of(self, node: int, preds: int) -> _NodeMemo:
        """Return the greedy path of ``node`` given ``preds``.

        It is stored, replayed from the closest one, or climbed.
        """
        slots = self._memo[node] if self.incremental else []
        if not slots:
            memo = _NodeMemo(preds, *self._climb(node, preds, 0, self.max_steps))
        else:
            old = self._slot[node].get(preds)
            if old is not None:
                self.n_reused += 1
                if slots[0] is not old:
                    slots.remove(old)
                    slots.insert(0, old)
                self._remember(node, preds, old)
                return old
            k, best = 0, None
            for j, m in enumerate(slots):
                d = (m.preds ^ preds).bit_count()
                if best is None or d < best:
                    k, best = j, d
            memo = self._replay(node, slots[k], preds)
        if self.incremental:
            slots.insert(0, memo)
            index = self._slot[node]
            index[preds] = memo
            for m in slots[self.memo_slots :]:
                del index[m.preds]
            del slots[self.memo_slots :]
            self._remember(node, preds, memo)
        return memo

    def _remember(self, node: int, preds: int, memo: _NodeMemo) -> None:
        if not self.exact_cap:
            return
        exact = self._exact[node]
        exact[preds] = (memo.final_score, memo.final_parents)
        if len(exact) > self.exact_cap:
            del exact[next(iter(exact))]

    def _replay(self, node: int, memo: _NodeMemo, preds: int) -> _NodeMemo:
        added = preds & ~memo.preds
        removed = memo.preds & ~preds
        ls = self._ls
        diverge = None
        for s, entry in enumerate(memo.path):
            state, gain, cand, ok, beat = entry
            if (removed >> cand) & 1 or added & beat:
                diverge = s
                break
            todo = added & ~ok
            if todo:
                pos = self._positive(node, state, todo)
                if pos is not None:
                    self.n_local_calls += todo.bit_count()
                    for a, g in pos:
                        if g < gain:
                            break
                        if (todo >> a) & 1 and (g > gain or a < cand):
                            diverge = s
                            break
                else:
                    cur = ls(node, state)
                    cands = _bits(todo)
                    vec = self._lookup(node, state, cands)
                    for a in cands:
                        g = vec.item(a) - cur
                        if g > gain or (g == gain and a < cand):
                            diverge = s
                            break
                if diverge is not None:
                    break
                entry[3] = ok | todo
        if diverge is None:
            if added and len(memo.path) < self.max_steps:
                fin = self._final_ok[node]
                todo = added & ~fin.get(memo.final_state, 0)
                if todo & self._final_bad[node].get(memo.final_state, 0):
                    diverge = len(memo.path)
                elif todo:
                    pos = self._positive(node, memo.final_state, todo)
                    if pos is not None:
                        self.n_local_calls += todo.bit_count()
                        improves = any((todo >> a) & 1 for a, _ in pos)
                    else:
                        cur = memo.final_score
                        cands = _bits(todo)
                        vec = self._lookup(node, memo.final_state, cands)
                        improves = any(vec.item(a) - cur > 0 for a in cands)
                    if improves:
                        diverge = len(memo.path)
                    else:
                        fin[memo.final_state] = fin.get(memo.final_state, 0) | todo
            if diverge is None:
                self.n_reused += 1
                out = _NodeMemo.__new__(_NodeMemo)
                out.preds, out.path = preds, memo.path
                out.final_state, out.final_score = memo.final_state, memo.final_score
                out.final_parents = memo.final_parents
                return out
        start = memo.path[diverge][0] if diverge < len(memo.path) else memo.final_state
        tail, state, cur = self._climb(node, preds, start, self.max_steps - diverge)
        return _NodeMemo(preds, memo.path[:diverge] + tail, state, cur)

    def score(self, order: Sequence[int]) -> OrderResult:
        """Score a complete order: the sum of each node's best local score."""
        parents: list[frozenset[int]] = [frozenset()] * self.n_vars
        total = 0.0
        preds = 0
        exact = self._exact if self.incremental and not self.n_restarts else None
        for v in order:
            v = int(v)
            hit = exact[v].get(preds) if exact is not None else None
            if hit is None:
                s, ps = self.local(v, preds)
            else:
                self.n_reused += 1
                s, ps = hit
            parents[v] = ps
            total += s
            preds |= 1 << v
        return OrderResult(total, tuple(parents))
