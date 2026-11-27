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

Everything the scorer caches is a fact about the score (a family's score, the
improving toggles of a parent set, a node's answer for a predecessor set), so
dropping an entry only costs recomputing it. ``cache_mb`` uses that to keep
the caches within a memory budget.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from itertools import islice
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import numpy as np

from causalbase._validation import check_positive

if TYPE_CHECKING:
    from causalbase.scores import FamilyScore

__all__ = ["OrderResult", "OrderScorer"]

# Rough sizes in bytes on 64-bit CPython, for the memory budget: a dict entry with its
# share of the table, the header of a small numpy array, a pair, a float, and an answer
# (a pair holding a float and a small frozenset).
_SLOT = 100
_ARRAY = 112
_PAIR = 56
_FLOAT = 24
_ANSWER = 300
# Improving-toggle lists kept per node as Python lists, besides the arrays.
_RECENT = 16


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
    __slots__ = ("answer", "final_parents", "final_score", "final_state", "path", "preds")

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
        self.answer: tuple[float, frozenset[int]] | None = None  # shared, see _answer


class OrderScorer:
    """Hill climbing restricted to an order, with exact incremental replay.

    Parent sets are bitmasks internally, and local scores are cached here per
    node and parent mask, so each family is scored once: by the compiled
    engine of a :class:`~causalbase.scores.BDeuScore` (which does not cache
    them), by the score itself otherwise.

    Parameters
    ----------
    score : FamilyScore
        The score, kept as ``family_score``; a
        :class:`~causalbase.scores.BDeuScore` scores the families with its
        compiled engine. Any other object with ``n_vars`` and
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
    cache_mb : float or None, default=None
        Memory budget of the caches, in MiB; None leaves them unbounded. When a
        query leaves them above it, every cache drops its oldest quarter (least
        recently used, for the neighbour vectors) until they fit. Answers do
        not change, since dropped entries are recomputed when needed; a budget
        far below what a search visits only costs time.

    Attributes
    ----------
    cache_nbytes : int
        Estimated bytes held by the caches the budget covers: family scores,
        neighbour vectors, final-state verdicts and stored answers. The greedy
        paths in the slots (``memo_slots`` per node) are not counted.
    n_evicted : int
        Cache entries dropped to keep within `cache_mb`.
    """

    def __init__(
        self,
        score: FamilyScore | Any,
        max_steps: int = 1000,
        incremental: bool = True,
        memo_slots: int = 8,
        exact_cap: int = 1024,
        n_restarts: int = 0,
        cache_mb: float | None = None,
    ) -> None:
        if cache_mb is not None:
            check_positive("cache_mb", cache_mb)
        self.family_score = score
        n_vars = self.n_vars = score.n_vars
        self.max_steps = max_steps
        self.incremental = incremental
        self.memo_slots = max(1, memo_slots)
        self.exact_cap = max(0, exact_cap)
        self.n_restarts = max(0, n_restarts)
        self.cache_mb = cache_mb
        self._budget = math.inf if cache_mb is None else float(cache_mb) * 2**20
        self.cache_nbytes = 0
        self.n_evicted = 0
        # Per node: predecessor mask -> best answer over the restart chain.
        self._best: list[dict[int, tuple[float, frozenset[int]]]] = [{} for _ in range(n_vars)]
        self._exact: list[dict[int, tuple[float, frozenset[int]]]] = [{} for _ in range(n_vars)]
        # Per node: final parent mask -> its (score, parents) answer, so that every stored
        # answer with the same result is one object.
        self._answers: list[dict[int, tuple[float, frozenset[int]]]] = [{} for _ in range(n_vars)]
        self._memo: list[list[_NodeMemo]] = [[] for _ in range(n_vars)]
        # Per node: predecessor mask -> its memo, for the memos in the slots.
        self._slot: list[dict[int, _NodeMemo]] = [{} for _ in range(n_vars)]
        self._cache: list[dict[int, float]] = [{} for _ in range(n_vars)]
        # Per node: parent mask -> candidates known not to improve on that parent set,
        # and those known to improve on it.
        self._final_ok: list[dict[int, int]] = [{} for _ in range(n_vars)]
        self._final_bad: list[dict[int, int]] = [{} for _ in range(n_vars)]
        # Per node: parent masks whose every toggle has been scored.
        self._full: list[set[int]] = [set() for _ in range(n_vars)]
        # Per node: parent mask -> bitmask of the toggles scored against it.
        self._known: list[dict[int, int]] = [{} for _ in range(n_vars)]
        self._all = (1 << n_vars) - 1
        # Per node: parent mask -> the toggles that improve on it, best first (smallest
        # index on ties), and their scores. The other scored toggles are not kept: every
        # reader compares a gain with a positive threshold (the gain of a move, or zero),
        # which they never pass, so a vector of the improving entries and -inf elsewhere
        # answers as the full one. Most toggles improve on nothing, so this holds a
        # small fraction of the scores.
        self._vec: list[dict[int, tuple[np.ndarray, np.ndarray]]] = [{} for _ in range(n_vars)]
        # Per node: the last few of those, as the Python lists the climbs iterate over (a
        # climb or replay reads the same states again and again).
        self._recent: list[dict[int, tuple[Any, list[tuple[int, float, float]]]]] = [
            {} for _ in range(n_vars)
        ]
        self._nbytes = (n_vars + 7) // 8
        key = sys.getsizeof(self._all)  # an int as wide as a parent mask
        self._cost_vec = 2 * _SLOT + 2 * key + _PAIR + 2 * _ARRAY  # with its known mask
        self._cost_score = _SLOT + key + _FLOAT
        self._cost_mask = _SLOT + 2 * key
        self._cost_answer = _SLOT + key + _ANSWER
        self._cost_ref = _SLOT + key  # a key and a reference to a shared answer
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
            self.cache_nbytes += self._cost_score
        return value

    def _idx(self, mask: int) -> np.ndarray:
        """Return the indices of the set bits of ``mask`` as an array."""
        raw = np.frombuffer(mask.to_bytes(self._nbytes, "little"), dtype=np.uint8)
        return np.flatnonzero(np.unpackbits(raw, bitorder="little"))

    def _flags(self, mask: int) -> np.ndarray:
        """Return ``mask`` as a boolean array of length ``n_vars``."""
        raw = np.frombuffer(mask.to_bytes(self._nbytes, "little"), dtype=np.uint8)
        return np.unpackbits(raw, count=self.n_vars, bitorder="little").view(bool)

    def _entry(self, node: int, state: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the stored improving toggles of ``state``, marking them as just used."""
        vecs = self._vec[node]
        if self._budget == math.inf:
            return vecs.get(state)
        ent = vecs.pop(state, None)
        if ent is not None:
            vecs[state] = ent  # the newest entries are dropped last
        return ent

    def _lookup(self, node: int, state: int, cands: Sequence[int] | np.ndarray) -> np.ndarray:
        """Neighbour vector of ``state`` with the toggles of ``cands`` scored.

        Entry ``c`` is the score of ``state`` with ``c`` toggled if that
        improves on ``state``, and -inf otherwise (or not yet scored).
        """
        self.n_local_calls += len(cands)
        vec = np.full(self.n_vars, -math.inf)
        ent = self._vec[node].get(state) if self._budget == math.inf else self._entry(node, state)
        if ent is None:
            known = 1 << node  # never asked for; without it no state would ever be full
        else:
            vec[ent[0]] = ent[1]
            if state in self._full[node]:
                return vec
            known = self._known[node][state]
        if isinstance(cands, np.ndarray):
            missing = cands if ent is None else cands[~self._flags(known)[cands]]
        else:
            missing = np.array([c for c in cands if not (known >> c) & 1], dtype=np.int64)
        if not missing.size:
            return vec
        rest = missing
        if self._batch is not None:
            # Additions go straight to the batch kernel: going through the family
            # cache costs a dictionary probe and insert per candidate, more than
            # the few families reachable from two states save.
            inside = np.zeros(self.n_vars, dtype=bool)
            inside[self._idx(state)] = True
            add = missing[~inside[missing]]
            rest = missing[inside[missing]]
            if add.size:
                vals = np.asarray(self._batch(node, _bits(state), add), dtype=float)
                vec[add] = np.where(np.isnan(vals), -math.inf, vals)
        cache = self._cache[node]
        for c in rest.tolist():
            m = state ^ (1 << c)
            v = cache.get(m)
            if v is None:
                v = self._scoring.local_score(node, frozenset(_bits(m)))
                cache[m] = v
                self.cache_nbytes += self._cost_score
            vec[c] = v if v == v else -math.inf
        flags = self._flags(known).copy()
        flags[missing] = True
        known = int.from_bytes(np.packbits(flags, bitorder="little").tobytes(), "little")
        cur = cache.get(state)
        if cur is None:
            cur = self._ls(node, state)
        gains = vec - cur
        idx = np.flatnonzero(gains > 0)
        if idx.size > 1:
            idx = idx[np.lexsort((idx, -gains[idx]))]
        if ent is None:
            self.cache_nbytes += self._cost_vec + 16 * idx.size
        else:
            self.cache_nbytes += 16 * (idx.size - ent[0].size)
        self._vec[node][state] = (idx, vec[idx])
        self._known[node][state] = known
        if known == self._all:
            self._full[node].add(state)
        return vec

    def _positive(self, node: int, state: int, need: int) -> list[tuple[int, float, float]] | None:
        """List the toggles that improve ``state``: candidate, gain and score.

        Best first (smallest index on ties). None unless every candidate of
        ``need`` has been scored against ``state``. Otherwise the best move
        among ``need`` is the first one that belongs to it, and only these
        toggles can beat a move, so a climb or a replay needs no pass over the
        predecessors.
        """
        known = self._known[node].get(state, 0)
        if need & ~known or not known:
            return None
        ent = self._vec[node][state] if self._budget == math.inf else self._entry(node, state)
        recent = self._recent[node]
        hit = recent.get(state)
        if hit is not None and hit[0] is ent:
            return hit[1]
        idx, val = ent  # type: ignore[misc]  # known implies stored
        gains = val - self._ls(node, state)
        out = list(zip(idx.tolist(), gains.tolist(), val.tolist(), strict=True))
        recent[state] = (ent, out)
        if len(recent) > _RECENT:
            del recent[next(iter(recent))]
        return out

    def _note(self, pool: dict[int, int], state: int, bits: int) -> None:
        """Add ``bits`` to the candidate mask kept for ``state`` in ``pool``."""
        old = pool.get(state)
        if old is None:
            pool[state] = bits
            self.cache_nbytes += self._cost_mask
        else:
            pool[state] = old | bits

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
                for best_c, best_gain, best_score in pos:  # noqa: B007 - read after the loop
                    if (preds >> best_c) & 1:
                        break
                else:
                    break
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
                best_score = vec.item(best_c)
            path.append([state, best_gain, best_c, preds, 0])
            state ^= 1 << best_c
            # Finite, as its gain is positive; the same float _ls would give.
            cur = best_score
            cache = self._cache[node]
            if state not in cache:
                cache[state] = cur
                self.cache_nbytes += self._cost_score
        if len(path) < budget:
            self._note(self._final_ok[node], state, preds)
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
        out = self._local(node, preds)
        if self.cache_nbytes > self._budget:
            self._shrink()
        return out

    def _local(self, node: int, preds: int) -> tuple[float, frozenset[int]]:
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
            self.cache_nbytes += self._cost_ref
            if len(cache) > self.exact_cap:
                del cache[next(iter(cache))]
                self.cache_nbytes -= self._cost_ref
        return best

    def _local0(self, node: int, preds: int) -> tuple[float, frozenset[int]]:
        """Return the result of the single greedy climb of ``node`` given ``preds``."""
        if self.incremental:
            hit = self._exact[node].get(preds)
            if hit is not None:
                self.n_reused += 1
                return hit
        return self._answer(node, self._memo_of(node, preds))

    def _answer(self, node: int, memo: _NodeMemo) -> tuple[float, frozenset[int]]:
        """Return the ``(score, parents)`` answer of ``memo``, one object per final state."""
        out = memo.answer
        if out is None:
            answers = self._answers[node]
            out = answers.get(memo.final_state)
            if out is None:
                out = answers[memo.final_state] = (memo.final_score, memo.final_parents)
                self.cache_nbytes += self._cost_answer
            memo.answer = out
        return out

    def _shrink(self) -> None:
        """Drop the oldest quarter of every cache until the caches fit the budget."""
        while self.cache_nbytes > self._budget:
            for node in range(self.n_vars):
                self._recent[node].clear()
                vecs, known, full = self._vec[node], self._known[node], self._full[node]
                for state in list(islice(vecs, (len(vecs) + 3) // 4)):
                    del vecs[state], known[state]
                    full.discard(state)
                    self.n_evicted += 1
                for pool in (
                    self._cache[node],
                    self._final_ok[node],
                    self._final_bad[node],
                    self._answers[node],
                    self._exact[node],
                    self._best[node],
                ):
                    for key in list(islice(pool, (len(pool) + 3) // 4)):
                        del pool[key]
                        self.n_evicted += 1
            self.cache_nbytes = self._held()

    def _held(self) -> int:
        """Estimate the bytes the caches hold, entry by entry."""
        total = 0
        for node in range(self.n_vars):
            vecs = self._vec[node]
            total += len(vecs) * self._cost_vec + 16 * sum(e[0].size for e in vecs.values())
            total += len(self._cache[node]) * self._cost_score
            total += (len(self._final_ok[node]) + len(self._final_bad[node])) * self._cost_mask
            total += len(self._answers[node]) * self._cost_answer
            total += (len(self._exact[node]) + len(self._best[node])) * self._cost_ref
        return total

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
        if self.cache_nbytes > self._budget:
            self._shrink()
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
        else:
            outside = self._all & ~preds & ~(1 << node)
            for p in self._chain(node, preds):
                for t, hit in self._breaks0(node, p):
                    hit &= outside
                    if hit:
                        idx = self._idx(hit)
                        brk[idx] = np.maximum(brk[idx], t)
        if self.cache_nbytes > self._budget:
            self._shrink()
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
                    self._note(bad, state, new)
                    todo &= ~new
                self._note(fin, state, todo)
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
        if preds not in exact:
            self.cache_nbytes += self._cost_ref
        exact[preds] = self._answer(node, memo)
        if len(exact) > self.exact_cap:
            del exact[next(iter(exact))]
            self.cache_nbytes -= self._cost_ref

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
                    for a, g, _ in pos:
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
                        improves = any((todo >> a) & 1 for a, _, _ in pos)
                    else:
                        cur = memo.final_score
                        cands = _bits(todo)
                        vec = self._lookup(node, memo.final_state, cands)
                        improves = any(vec.item(a) - cur > 0 for a in cands)
                    if improves:
                        diverge = len(memo.path)
                    else:
                        self._note(fin, memo.final_state, todo)
            if diverge is None:
                self.n_reused += 1
                out = _NodeMemo.__new__(_NodeMemo)
                out.preds, out.path = preds, memo.path
                out.final_state, out.final_score = memo.final_state, memo.final_score
                out.final_parents = memo.final_parents
                out.answer = memo.answer
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
