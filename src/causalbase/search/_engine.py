"""The greedy hill climb that drives GES's phases and XGES.

The climb caches the best operator of each kind into every child. An operator
into `y` depends on the neighbourhood of `y` and on which neighbours of `y`
each `x` is adjacent to, so after a step only the children whose own
neighbourhood changed, or one of whose undirected neighbours changed, are
re-scored. A Reverse into `y` also depends on the parents of its `x`, a child
of `y`, so a climb with Reverses re-scores the parents of what changed too.

Whether a semi-directed path is blocked depends on more of the graph, though:
an Insert or a Reverse into `y` can become valid when a step cuts a path from
`y`, and every such path reaches a node the step changed. The cached best is
re-checked before it is applied, and the climb either re-scores every child
once more when the cache runs dry, or re-scores, after each step, the children
from which a changed node can be reached. Both end at a local optimum of the
full neighbourhood; the first is cheaper over a long climb, the second over a
short one.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from causalbase.search import _operators as ops

if TYPE_CHECKING:
    from collections.abc import Iterable

    from causalbase.graph import PDAG
    from causalbase.scores import Score
    from causalbase.search._knowledge import Knowledge

__all__ = [
    "backward_phase",
    "forward_phase",
    "marginal_sources",
    "sources_of",
    "xges0",
]

Pair = tuple[int, int]
Sources = Sequence["frozenset[int] | None"]
"""Per child `y`, the nodes `x` an operator into `y` may come from; ``None`` is any."""
_Finder = Callable[["PDAG", int, "frozenset[int] | None"], "ops.Best"]
_Applier = Callable[["PDAG", int, int, frozenset[int]], None]


@dataclass(frozen=True)
class _Kind:
    """One kind of operator: how to find the best ones into a child, and apply one."""

    find: _Finder
    apply: _Applier
    sources: Sources | None = None
    allowed: Callable[[int, int], bool] | None = None
    # Whether an edit at a node changes the operators into its parents too.
    parents_matter: bool = False
    # Whether validity depends on semi-directed paths, which an edit anywhere downstream can cut.
    uses_paths: bool = False


def sources_of(candidates: frozenset[Pair] | None, n: int) -> list[frozenset[int]] | None:
    """Return, per child `y`, the `x` with ``(x, y)`` in `candidates`; ``None`` for no limit."""
    if candidates is None:
        return None
    lists: list[set[int]] = [set() for _ in range(n)]
    for x, y in candidates:
        lists[y].add(x)
    return [frozenset(s) for s in lists]


def _intersect(a: Sources | None, b: Sources | None) -> Sources | None:
    if a is None:
        return b
    if b is None:
        return a
    return [u if v is None else v if u is None else u & v for u, v in zip(a, b, strict=True)]


def marginal_sources(score: Score, n: int) -> list[frozenset[int]]:
    """Return, per node `y`, the `x` whose edge with `y` alone improves the score.

    The first step of fGES under its one-edge faithfulness assumption (Ramsey
    et al., 2017): pairs that the score finds marginally independent never
    become adjacent. Both directions of a pair are kept if either improves,
    which for a score-equivalent score is the same as both.
    """
    found: list[set[int]] = [set() for _ in range(n)]
    nodes = np.arange(n, dtype=np.intp)
    for y in range(n):
        xs = nodes[nodes != y]
        if xs.size == 0:
            continue
        alone = float(score.local_score(y, frozenset()))
        with_x = np.asarray(score.local_scores_with(y, frozenset(), xs), dtype=np.float64)
        for x, s in zip(xs.tolist(), with_x.tolist(), strict=True):
            delta = s - alone
            if delta > 0 and not ops._is_noise(delta, s, alone):
                found[y].add(x)
                found[x].add(y)
    return [frozenset(s) for s in found]


def _climb(
    pdag: PDAG,
    kinds: Sequence[_Kind],
    *,
    changed: Iterable[int] | None = None,
    exhaustive: bool = True,
    in_place: bool = False,
) -> tuple[PDAG, float]:
    """Apply the best operator of the first kind that has one, until none improves.

    Returns the final graph and the sum of the improvements applied. The climb
    first scores every child, or, given the nodes `changed` by an edit of a
    local optimum, only the children that edit touched. With `exhaustive` it
    re-scores every child when the cache runs dry; without, it re-scores after
    each step the children upstream of the edit for the kinds that use paths,
    and stops when the cache runs dry. Either way it stops at a local optimum
    of the operators it climbs with. With `in_place` it edits `pdag` itself.
    """
    current = pdag if in_place else pdag.copy()
    n = current.num_nodes
    # Per kind, the best operator into each child: its parent, improvement and subset.
    caches: list[list[tuple[int, float, frozenset[int]] | None]] = [[None] * n for _ in kinds]
    # Per kind, a max-heap of those by improvement, then smallest pair. Each scoring
    # of a child stamps its entry; an entry with an older stamp is stale, and skipped.
    heaps: list[list[tuple[float, int, int, int]]] = [[] for _ in kinds]
    stamps = [[0] * n for _ in kinds]

    def score_child(k: int, y: int) -> ops.Best:
        """Score the operators of kind `k` into `y`, cache the best and return them all."""
        kind = kinds[k]
        xs, deltas, which, subsets = kind.find(
            current, y, kind.sources[y] if kind.sources is not None else None
        )
        if kind.allowed is not None and xs.size:
            keep = [i for i, x in enumerate(xs.tolist()) if kind.allowed(x, y)]
            xs, deltas, which = xs[keep], deltas[keep], which[keep]
        stamps[k][y] += 1
        if xs.size:
            i = int(np.argmax(deltas))  # the first of the largest: the smallest parent
            best = (int(xs[i]), float(deltas[i]), subsets[which[i]])
            caches[k][y] = best
            heapq.heappush(heaps[k], (-best[1], best[0], y, stamps[k][y]))
        else:
            caches[k][y] = None
        return xs, deltas, which, subsets

    def rescore(edited: Iterable[int] | None) -> None:
        """Re-score, per kind, the children whose operators an edit of `edited` may alter."""
        if edited is None:
            for y in range(n):
                for k in range(len(kinds)):
                    score_child(k, y)
            return
        edited = set(edited)
        near = set(edited)
        for v in edited:
            near |= current._undirected_set(v)
        wide: set[int] | None = None
        upstream: set[int] | None = None
        for k, kind in enumerate(kinds):
            children = near
            if kind.parents_matter:
                if wide is None:
                    wide = set(near)
                    for v in edited:
                        wide |= current._parents_set(v)
                children = wide
            if kind.uses_paths and not exhaustive:
                if upstream is None:
                    upstream = _upstream(current, edited)
                children = children | upstream
            for y in sorted(children):
                score_child(k, y)

    def pop_best(k: int) -> tuple[int, int, frozenset[int], float] | None:
        heap = heaps[k]
        while heap:
            _, x, y, stamped = heap[0]
            cached = caches[k][y]
            if stamped != stamps[k][y] or cached is None:
                heapq.heappop(heap)
                continue
            # Apply it only if scoring `y` again still gives this operator for `x`.
            xs, deltas, which, subsets = score_child(k, y)
            i = int(np.searchsorted(xs, x))
            if i < xs.size and xs[i] == x and (deltas[i], subsets[which[i]]) == cached[1:]:
                return x, y, cached[2], cached[1]
        return None

    def next_step() -> tuple[int, tuple[int, int, frozenset[int], float]] | None:
        for k in range(len(kinds)):
            found = pop_best(k)
            if found is not None:
                return k, found
        return None

    rescore(changed)
    gain = 0.0
    while True:
        step = next_step()
        if step is None:
            if not exhaustive:
                return current, gain
            rescore(None)
            step = next_step()
            if step is None:
                return current, gain
        k, (x, y, subset, delta) = step
        kinds[k].apply(current, x, y, subset)
        gain += delta
        # Every node whose marks may differ from before the step: those the operator
        # edited and those the completion changed. A node edited and then restored is
        # re-scored for nothing, which leaves its cached operators as they were.
        rescore({x, y, *subset, *current._complete_in_place().ravel().tolist()})


def _upstream(pdag: PDAG, nodes: set[int]) -> set[int]:
    """Return the nodes with a semi-directed path to one of `nodes`, `nodes` included."""
    seen = set(nodes)
    stack = list(nodes)
    while stack:
        v = stack.pop()
        for w in (*pdag._parents_set(v), *pdag._undirected_set(v)):
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def forward_phase(
    score: Score,
    pdag: PDAG,
    candidates: frozenset[Pair] | None = None,
    max_t_size: int | None = None,
    knowledge: Knowledge | None = None,
    max_parents: int | None = None,
    sources: Sources | None = None,
) -> PDAG:
    """Apply the best Insert until none improves the score; return the CPDAG.

    Parameters
    ----------
    score : Score
        Decomposable score.
    pdag : PDAG
        Starting CPDAG. It is not modified.
    candidates : frozenset of (int, int), optional
        If given, only these ordered pairs ``(x, y)`` are considered for
        ``x -> y``.
    max_t_size : int or None, default=None
        Largest `T` tried per pair; ``None`` tries every subset.
    knowledge : Knowledge, optional
        Forbidden adjacencies are never inserted.
    max_parents : int or None, optional
        No Insert gives its child more parents than this.
    sources : sequence of frozenset or None, optional
        Per child, the only nodes an Insert into it may come from, on top of
        `candidates`.
    """

    def find(g: PDAG, y: int, xs: frozenset[int] | None) -> ops.Best:
        return ops.best_inserts_into(score, g, y, max_t_size, max_parents, xs)

    kind = _Kind(
        find,
        ops.apply_insert,
        _intersect(sources_of(candidates, pdag.num_nodes), sources),
        knowledge.may_add if knowledge else None,
        uses_paths=True,
    )
    return _climb(pdag, [kind])[0]


def backward_phase(
    score: Score,
    pdag: PDAG,
    candidates: frozenset[Pair] | None = None,
    knowledge: Knowledge | None = None,
) -> PDAG:
    """Apply the best Delete until none improves the score; return the CPDAG.

    Parameters
    ----------
    score : Score
        Decomposable score.
    pdag : PDAG
        Starting CPDAG. It is not modified.
    candidates : frozenset of (int, int), optional
        If given, only the edges between these ordered pairs ``(x, y)`` can be
        deleted. An undirected edge is reachable from either ordering.
    knowledge : Knowledge, optional
        Required adjacencies are never deleted.
    """

    def find(g: PDAG, y: int, xs: frozenset[int] | None) -> ops.Best:
        return ops.best_deletes_into(score, g, y, xs)

    kind = _Kind(
        find,
        ops.apply_delete,
        sources_of(candidates, pdag.num_nodes),
        knowledge.may_remove if knowledge else None,
    )
    return _climb(pdag, [kind])[0]


def xges0(
    score: Score,
    pdag: PDAG,
    *,
    max_t_size: int | None = None,
    knowledge: Knowledge | None = None,
    max_parents: int | None = None,
    sources: Sources | None = None,
    banned: frozenset[Pair] = frozenset(),
    changed: Iterable[int] | None = None,
    exhaustive: bool = True,
    in_place: bool = False,
) -> tuple[PDAG, float]:
    """Run XGES-0 from `pdag`: the best Delete, else the best Reverse, else the best Insert.

    Returns the CPDAG it stops at and the improvement over `pdag`. `sources`
    limits where Inserts come from, as in :func:`forward_phase`; `banned` are
    ordered pairs ``(x, y)`` that may not become ``x -> y`` by an Insert or a
    Reverse. `changed`, `exhaustive` and `in_place` are :func:`_climb`'s.
    """

    def deletes(g: PDAG, y: int, xs: frozenset[int] | None) -> ops.Best:
        return ops.best_deletes_into(score, g, y, xs)

    def reverses(g: PDAG, y: int, xs: frozenset[int] | None) -> ops.Best:
        return ops.best_reverses_into(score, g, y, max_t_size, max_parents)

    def inserts(g: PDAG, y: int, xs: frozenset[int] | None) -> ops.Best:
        return ops.best_inserts_into(score, g, y, max_t_size, max_parents, xs)

    def may_point(x: int, y: int) -> bool:
        return (x, y) not in banned

    def may_add(x: int, y: int) -> bool:
        return (x, y) not in banned and (knowledge is None or knowledge.may_add(x, y))

    kinds = [
        _Kind(deletes, ops.apply_delete, None, knowledge.may_remove if knowledge else None),
        _Kind(
            reverses,
            ops.apply_reverse,
            None,
            may_point if banned else None,
            parents_matter=True,
            uses_paths=True,
        ),
        _Kind(
            inserts,
            ops.apply_insert,
            sources,
            may_add if banned or knowledge else None,
            uses_paths=True,
        ),
    ]
    return _climb(pdag, kinds, changed=changed, exhaustive=exhaustive, in_place=in_place)
