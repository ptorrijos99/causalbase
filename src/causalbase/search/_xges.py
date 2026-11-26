"""Extremely Greedy Equivalence Search (Nazaret and Blei, 2024)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from causalbase.search import _operators as ops
from causalbase.search._engine import marginal_sources, xges0
from causalbase.search._ges import GES, _check_limit, _check_score, _start

if TYPE_CHECKING:
    from collections.abc import Sequence

    from causalbase.graph import DAG, PDAG
    from causalbase.scores import Prior, Score, ScoreName
    from causalbase.search._engine import Pair
    from causalbase.search._knowledge import Knowledge, PairLike

__all__ = ["XGES", "xges"]

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
) -> PDAG:
    """Run XGES with `score` and return the learned CPDAG.

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
    PDAG
        The CPDAG of the learned equivalence class.

    Raises
    ------
    ValueError
        If `initial_graph` has another number of nodes, or a limit is out of
        range.
    """
    score, n_vars = _check_score(score)
    _check_limit("max_t_size", max_t_size)
    _check_limit("max_parents", max_parents)
    sources = marginal_sources(score, n_vars) if one_edge_faithfulness else None

    def run(
        pdag: PDAG,
        banned: frozenset[Pair],
        changed: set[int] | None,
        exhaustive: bool,
        in_place: bool,
    ) -> tuple[PDAG, float]:
        return xges0(
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

    cpdag, _ = run(_start(n_vars, initial_graph, knowledge), frozenset(), None, True, True)
    if not extended:
        return cpdag
    dag = cpdag.to_dag()
    scale = max(abs(sum(score.local_score(v, dag.parents(v)) for v in range(n_vars))), 1.0)
    while True:
        cpdag = _force_deletions(score, cpdag, run, knowledge, ops._REL_TOL * scale)
        # The bars may hold back an improving Insert or Reverse; free them and see.
        free, gain = run(cpdag, frozenset(), set(), True, False)
        if gain <= ops._REL_TOL * scale:
            return cpdag
        cpdag = free


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


class XGES(GES):
    """Learn a causal structure with Extremely Greedy Equivalence Search.

    XGES (Nazaret and Blei, 2024) searches the same space as GES with the same
    score, but interleaves its operators: it applies an improving Delete
    whenever there is one, then a Reverse, and only then an Insert, so edges
    added early do not lock the search into a worse optimum. ``extended=True``
    then forces out each edge in turn and searches again from there.

    Parameters
    ----------
    criterion, ess, expected_parents, prior, penalty_discount, max_parents, max_t_size
        As in :class:`GES`.
    required, forbidden, one_edge_faithfulness
        As in :class:`GES`.
    extended : bool, default=True
        Run the forced deletions after XGES-0. ``False`` is XGES-0, several
        times cheaper.

    Attributes
    ----------
    cpdag_, dag_, score_, adjacency_, n_features_in_, feature_names_in_
        As in :class:`GES`.

    Examples
    --------
    >>> from causalbase import XGES
    >>> from causalbase.datasets import load_asia
    >>> model = XGES().fit(load_asia().data)
    >>> model.cpdag_
    PDAG(num_nodes=8, directed=..., undirected=...)
    """

    def __init__(
        self,
        criterion: ScoreName = "bdeu",
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        prior: Prior = "bernoulli",
        penalty_discount: float = 1.0,
        max_parents: int | None = None,
        max_t_size: int | None = None,
        required: Sequence[PairLike] | None = None,
        forbidden: Sequence[PairLike] | None = None,
        one_edge_faithfulness: bool = False,
        extended: bool = True,
    ) -> None:
        super().__init__(
            criterion=criterion,
            ess=ess,
            expected_parents=expected_parents,
            prior=prior,
            penalty_discount=penalty_discount,
            max_parents=max_parents,
            max_t_size=max_t_size,
            required=required,
            forbidden=forbidden,
            one_edge_faithfulness=one_edge_faithfulness,
        )
        self.extended = extended

    def _search(self, score: Score, initial_graph: DAG | None, knowledge: Knowledge) -> PDAG:
        return xges(
            score,
            initial_graph,
            extended=self.extended,
            max_t_size=self.max_t_size,
            knowledge=knowledge,
            max_parents=self.max_parents,
            one_edge_faithfulness=self.one_edge_faithfulness,
        )
