"""Greedy Equivalence Search (Chickering, 2002)."""

from __future__ import annotations

import time
from numbers import Integral
from typing import TYPE_CHECKING

from causalbase.graph import DAG, PDAG
from causalbase.graph.cpdag import dag_to_cpdag
from causalbase.search._engine import backward_phase, forward_phase, marginal_sources
from causalbase.search._result import SearchResult

if TYPE_CHECKING:
    from causalbase.scores import Score
    from causalbase.search._engine import Pair
    from causalbase.search._knowledge import Knowledge

__all__ = ["ges"]


def _check_limit(name: str, value: object) -> None:
    """Raise unless `value` is None or a non-negative integer."""
    if value is None:
        return
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None, got {value!r}")


def _check_score(score: Score) -> tuple[Score, int]:
    """Return the score to search in place of `score`, and its number of variables.

    That is the score's ``_search_view()`` when it has one (see
    :class:`~causalbase.scores.FamilyScore`), the score itself otherwise.
    """
    view = getattr(score, "_search_view", None)
    searched: Score = view() if view is not None else score
    return searched, int(searched.n_vars)


def _start(n_vars: int, initial_graph: DAG | PDAG | None, knowledge: Knowledge | None) -> PDAG:
    """Return the CPDAG a search starts from, with `knowledge` imposed."""
    if initial_graph is not None and initial_graph.num_nodes != n_vars:
        raise ValueError(f"initial_graph has {initial_graph.num_nodes} nodes, expected {n_vars}")
    if initial_graph is None:
        start = DAG(n_vars)
    elif isinstance(initial_graph, DAG):
        start = initial_graph
    else:
        start = initial_graph.to_dag()
    return dag_to_cpdag(knowledge.impose(start) if knowledge else start)


def value(score: Score, graph: DAG | PDAG) -> float:
    """Return the score of `graph`, or of a DAG of its equivalence class."""
    dag = graph if isinstance(graph, DAG) else graph.to_dag()
    return float(sum(score.local_score(v, dag.parents(v)) for v in range(dag.num_nodes)))


def ges(
    score: Score,
    initial_graph: DAG | PDAG | None = None,
    *,
    candidates: frozenset[Pair] | None = None,
    max_t_size: int | None = None,
    knowledge: Knowledge | None = None,
    max_parents: int | None = None,
    forward: bool = True,
    backward: bool = True,
    one_edge_faithfulness: bool = False,
) -> SearchResult:
    """Run GES with `score` and return the learned equivalence class.

    Parameters
    ----------
    score : Score
        Decomposable score; the graph has ``score.n_vars`` nodes. For
        ``BDeuScore(prior="tetrad")`` the search runs on the Bernoulli BDeu
        with the same differences between families (see
        :class:`~causalbase.scores.BDeuScore`), so it learns Tetrad's
        structures.
    initial_graph : DAG or PDAG, optional
        Where the search starts: the equivalence class of a DAG, or of a
        consistent extension of a PDAG. The empty graph by default.
    candidates : frozenset of (int, int), optional
        Restrict both phases to these ordered pairs ``(x, y)``.
    max_t_size : int or None, default=None
        Largest conditioning subset tried by an Insert. ``None`` tries every
        one that can be valid, as Chickering defines the operator; only the
        subsets that keep a clique are visited, so this costs little more than
        a cap unless the neighbours of a node form a large clique.
    knowledge : Knowledge, optional
        Adjacencies the result must have or must not have. The search starts
        from `initial_graph` with them imposed.
    max_parents : int or None, optional
        No Insert gives its child more parents than this. ``None`` is no limit.
    forward, backward : bool, default=True
        Which phases to run.
    one_edge_faithfulness : bool, default=False
        Insert an edge only between variables that the score finds dependent
        on their own, as fGES does (Ramsey et al., 2017). Much faster on many
        variables; it misses an edge whose paths cancel exactly.

    Returns
    -------
    SearchResult
        ``cpdag`` is the learned equivalence class and ``dag`` one DAG of it,
        ``score`` the score of `dag` under `score` (Tetrad's value with its
        prior). ``n_iter`` counts the Inserts and Deletes applied, the
        ``trace`` has an entry at the start and after each phase that changed
        the graph, and ``info`` holds ``"forward_steps"`` and
        ``"backward_steps"``.

    Raises
    ------
    ValueError
        If `initial_graph` has another number of nodes, or a limit is out of
        range.

    Examples
    --------
    >>> from causalbase.datasets import load_asia
    >>> from causalbase.scores import BDeuScore
    >>> from causalbase.search import ges
    >>> ges(BDeuScore(load_asia().data)).cpdag
    PDAG(num_nodes=8, directed=..., undirected=...)
    """
    original = score
    score, n_vars = _check_score(score)
    _check_limit("max_t_size", max_t_size)
    _check_limit("max_parents", max_parents)
    t0 = time.perf_counter()
    cpdag = _start(n_vars, initial_graph, knowledge)
    trace = [(0.0, 0, value(original, cpdag))]
    forward_steps = backward_steps = 0
    if forward:
        screen = marginal_sources(score, n_vars) if one_edge_faithfulness else None
        cpdag, forward_steps = forward_phase(
            score, cpdag, candidates, max_t_size, knowledge, max_parents, sources=screen
        )
        if forward_steps:
            trace.append((time.perf_counter() - t0, forward_steps, value(original, cpdag)))
    if backward:
        cpdag, backward_steps = backward_phase(score, cpdag, candidates, knowledge)
        if backward_steps:
            steps = forward_steps + backward_steps
            trace.append((time.perf_counter() - t0, steps, value(original, cpdag)))
    dag = cpdag.to_dag()
    info = {"forward_steps": forward_steps, "backward_steps": backward_steps}
    return SearchResult(dag, value(original, dag), forward_steps + backward_steps, trace, info)
