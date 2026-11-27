"""Extremely Greedy Equivalence Search (Nazaret and Blei, 2024)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from causalbase.search import _operators as ops
from causalbase.search._engine import marginal_sources, xges0
from causalbase.search._ges import _check_limit, _check_score, _start, value
from causalbase.search._result import SearchResult

if TYPE_CHECKING:
    from causalbase.graph import DAG, PDAG
    from causalbase.scores import Score
    from causalbase.search._engine import Pair
    from causalbase.search._knowledge import Knowledge

__all__ = ["xges"]

_Runner = Callable[["PDAG", "frozenset[Pair]", "set[int] | None", bool, bool], "tuple[PDAG, float]"]
"""XGES-0 from a graph, with banned pairs, the nodes an edit changed, whether to end
with a full pass, and whether it may edit the graph it is given."""


def xges(
    score: Score,
    initial_graph: DAG | PDAG | None = None,
    *,
    extended: bool = True,
    max_t_size: int | None = None,
    knowledge: Knowledge | None = None,
    max_parents: int | None = None,
    one_edge_faithfulness: bool = False,
) -> SearchResult:
    """Run XGES with `score` and return the learned equivalence class.

    XGES-0 applies, at every step, the best Delete that improves the score; if
    there is none, the best Reverse; if there is none, the best Insert. It
    stops where none of the three improves. With `extended`, XGES then forces
    out each edge in turn, bars the search from putting it back the same way,
    runs XGES-0 again and keeps the result if it scores higher, until no
    forced deletion pays. As in the authors' code, an accepted bar stays for
    the rest of the search; unlike it, the search ends with XGES-0 free of
    bars, and starts over if that improves, so the result is a local optimum
    of the whole neighbourhood.

    Parameters
    ----------
    score : Score
        Decomposable score; the graph has ``score.n_vars`` nodes. For
        ``BDeuScore(prior="tetrad")`` the search runs on the Bernoulli BDeu
        with the same differences between families (see
        :class:`~causalbase.scores.BDeuScore`), so it learns Tetrad's
        structures.
    initial_graph : DAG or PDAG, optional
        Where the search starts. The empty graph by default.
    extended : bool, default=True
        Run the forced deletions after XGES-0; ``False`` is XGES-0 alone.
    max_t_size, knowledge, max_parents, one_edge_faithfulness
        As in :func:`~causalbase.search.ges`. The screen of
        `one_edge_faithfulness` limits Inserts only.

    Returns
    -------
    SearchResult
        As :func:`~causalbase.search.ges` returns it, but ``n_iter`` counts
        every operator XGES-0 applied, those of the forced deletions it then
        rejected included; the ``trace`` has an entry at the start, after
        XGES-0 and after each round of forced deletions that improved the
        score, and ``info`` holds ``"rounds"``, the rounds of forced deletions.

    Raises
    ------
    ValueError
        If `initial_graph` has another number of nodes, or a limit is out of
        range.

    Examples
    --------
    >>> from causalbase.datasets import load_asia
    >>> from causalbase.scores import BDeuScore
    >>> from causalbase.search import xges
    >>> xges(BDeuScore(load_asia().data)).cpdag
    PDAG(num_nodes=8, directed=..., undirected=...)
    """
    original = score
    score, n_vars = _check_score(score)
    _check_limit("max_t_size", max_t_size)
    _check_limit("max_parents", max_parents)
    sources = marginal_sources(score, n_vars) if one_edge_faithfulness else None
    steps = 0

    def run(
        pdag: PDAG,
        banned: frozenset[Pair],
        changed: set[int] | None,
        exhaustive: bool,
        in_place: bool,
    ) -> tuple[PDAG, float]:
        nonlocal steps
        result, gain, done = xges0(
            score,
            pdag,
            max_t_size=max_t_size,
            knowledge=knowledge,
            max_parents=max_parents,
            sources=sources,
            banned=banned,
            changed=changed,
            exhaustive=exhaustive,
            in_place=in_place,
        )
        steps += done
        return result, gain

    t0 = time.perf_counter()
    cpdag = _start(n_vars, initial_graph, knowledge)
    trace = [(0.0, 0, value(original, cpdag))]
    cpdag, _ = run(cpdag, frozenset(), None, True, True)
    best = value(original, cpdag)
    if steps:
        trace.append((time.perf_counter() - t0, steps, best))
    rounds = 0
    if extended:
        scale = max(abs(value(score, cpdag)), 1.0)
        while True:
            cpdag = _force_deletions(score, cpdag, run, knowledge, ops._REL_TOL * scale)
            rounds += 1
            now = value(original, cpdag)
            if now > best:
                best = now
                trace.append((time.perf_counter() - t0, steps, best))
            # The bars may hold back an improving Insert or Reverse; free them and see.
            free, gain = run(cpdag, frozenset(), set(), True, False)
            if gain <= ops._REL_TOL * scale:
                break
            cpdag = free
    dag = cpdag.to_dag()
    return SearchResult(dag, value(original, dag), steps, trace, {"rounds": rounds})


def _force_deletions(
    score: Score, cpdag: PDAG, run: _Runner, knowledge: Knowledge | None, tol: float
) -> PDAG:
    """Try each Delete of `cpdag`, improving or not, followed by XGES-0; keep any gain.

    A trial starts from what the forced deletion changed, since the rest of the
    graph is at XGES-0's optimum, and tracks the paths it cuts instead of
    re-scoring every child at the end; an accepted one is completed by a full
    XGES-0 before the search goes on. Each pass tries the Deletes of the graph
    it started from, best first, re-derived on the current graph; a pass that
    accepts none ends the search. Accepting needs a gain above `tol`.
    """
    n = cpdag.num_nodes
    banned: frozenset[tuple[int, int]] = frozenset()
    accepted = True
    while accepted:
        accepted = False
        moves = sorted(
            (-delta, x, y)
            for y in range(n)
            for x, (delta, _) in ops.deletes_into(score, cpdag, y, improving=False).items()
            if knowledge is None or knowledge.may_remove(x, y)
        )
        for _, x, y in moves:
            current = ops.deletes_into(score, cpdag, y, frozenset({x}), improving=False)
            if x not in current:
                continue  # no longer an edge into `y`
            delta, h = current[x]
            trial = cpdag.copy()
            ops.apply_delete(trial, x, y, h)
            changed = {x, y, *h, *trial._complete_in_place().ravel().tolist()}
            tried = banned | {(x, y)}
            result, gain = run(trial, tried, changed, False, True)
            if result == cpdag or delta + gain <= tol:
                continue
            banned = tried
            # Nothing cached to start from: the climb re-scores every child once and goes on.
            cpdag, _ = run(result, banned, set(), True, True)
            accepted = True
    return cpdag
