"""The estimators over graphs and equivalence classes: GES, XGES and HillClimbing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from causalbase._score_estimator import ScoreEstimator
from causalbase._validation import check_count, check_time_limit
from causalbase.search._ges import _check_limit, ges
from causalbase.search._knowledge import Knowledge
from causalbase.search._xges import xges
from causalbase.search.hill_climb import hill_climbing

if TYPE_CHECKING:
    from collections.abc import Sequence

    from causalbase.graph import DAG
    from causalbase.scores import FamilyScore, Prior, ScoreName
    from causalbase.search._knowledge import PairLike
    from causalbase.search._result import SearchResult

__all__ = ["GES", "XGES", "HillClimbing"]


class GES(ScoreEstimator):
    """Learn a causal structure with Greedy Equivalence Search.

    The search is :func:`~causalbase.search.ges` (Chickering, 2002) on the
    score `criterion` names: it adds edges while the score improves, then
    removes them while it improves, moving between Markov equivalence
    classes. It starts from the empty graph, or from `initial_graph` in
    :meth:`fit`.

    Parameters
    ----------
    criterion : {"bdeu", "bic", "gaussian_bic", "bge"}, default="bdeu"
        The score: BDeu or BIC over categorical data, or the BIC or BGe of a
        linear Gaussian model over continuous data.
    ess : float, default=1.0
        Equivalent sample size: of the BDeu prior, or of BGe's prior on the
        mean. Also the prior of the parameters behind :meth:`score` on
        categorical data.
    expected_parents : float or None, default=1.0
        Expected in-degree under the structure prior, for every criterion; see
        :class:`~causalbase.scores.FamilyScore`. ``None`` is a uniform prior.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior: the Bernoulli prior of
        :class:`~causalbase.scores.FamilyScore`, or, for BDeu only, Tetrad
        7.6.5's, with `expected_parents` as Tetrad's structure prior (see
        :class:`~causalbase.scores.BDeuScore`).
    penalty_discount : float, default=1.0
        Multiplier of the penalty of the two BIC criteria; larger values give
        sparser structures.
    alpha_w : float or None, default=None
        Degrees of freedom of BGe's Wishart prior, above ``n_vars + 1``;
        ``None`` is ``n_vars + 2``. See :class:`~causalbase.scores.BGeScore`.
    max_parents : int or None, default=None
        Most parents an Insert may give a node. ``None`` is no limit.
    max_t_size : int or None, default=None
        Largest conditioning subset tried by an Insert; ``None`` tries all.
    required, forbidden : sequence of pairs, optional
        Pairs of variables, by name or column index, that must be adjacent or
        must not be. The direction of a required edge is still learned.
    one_edge_faithfulness : bool, default=False
        Insert an edge only between variables that the score finds dependent
        on their own, as fGES does (Ramsey et al., 2017): faster on many
        variables, at the cost of edges whose paths cancel exactly.
    n_jobs : int, default=1
        Threads of the score's compiled kernels; -1 is every thread numba has.
        The result does not depend on it.

    Attributes
    ----------
    dag_ : DAG
        The learned network.
    cpdag_ : PDAG
        Its Markov equivalence class.
    score_ : float
        The score of `dag_` under `criterion`, with the structure prior chosen
        (with ``prior="tetrad"``, the value Tetrad reports).
    adjacency_ : pandas.DataFrame
        `cpdag_` with the variable names: ``A[i, j] = 1`` and ``A[j, i] = 0`` is
        ``i -> j``, and both set is an undirected edge.
    n_features_in_ : int
        Number of variables.
    feature_names_in_ : ndarray of str
        Variable names.
    n_iter_ : int
        Inserts and Deletes applied.
    trace_ : list of tuple
        ``(seconds, steps, score)`` at the start and after each phase that
        changed the graph.

    Examples
    --------
    >>> from causalbase import GES
    >>> from causalbase.datasets import load_asia
    >>> model = GES().fit(load_asia().data)
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
        alpha_w: float | None = None,
        max_parents: int | None = None,
        max_t_size: int | None = None,
        required: Sequence[PairLike] | None = None,
        forbidden: Sequence[PairLike] | None = None,
        one_edge_faithfulness: bool = False,
        n_jobs: int = 1,
    ) -> None:
        self.criterion = criterion
        self.ess = ess
        self.expected_parents = expected_parents
        self.prior = prior
        self.penalty_discount = penalty_discount
        self.alpha_w = alpha_w
        self.max_parents = max_parents
        self.max_t_size = max_t_size
        self.required = required
        self.forbidden = forbidden
        self.one_edge_faithfulness = one_edge_faithfulness
        self.n_jobs = n_jobs

    def _check(self) -> None:
        super()._check()
        _check_limit("max_parents", self.max_parents)
        _check_limit("max_t_size", self.max_t_size)

    def _knowledge(self, names: Sequence[str]) -> Knowledge:
        return Knowledge.from_pairs(self.required or (), self.forbidden or (), names)

    def _search(
        self, score: FamilyScore, initial_graph: DAG | None, names: Sequence[str]
    ) -> SearchResult:
        return ges(
            score,
            initial_graph,
            max_t_size=self.max_t_size,
            knowledge=self._knowledge(names),
            max_parents=self.max_parents,
            one_edge_faithfulness=self.one_edge_faithfulness,
        )


class XGES(GES):
    """Learn a causal structure with Extremely Greedy Equivalence Search.

    The search is :func:`~causalbase.search.xges` (Nazaret and Blei, 2024): the
    same space, score and operators as :class:`GES`, interleaved so that it
    applies an improving Delete whenever there is one, then a Reverse, and
    only then an Insert; edges added early then do not lock the search into
    a worse optimum. ``extended=True`` then forces out each edge in turn and
    searches again from there. It starts from the empty graph, or from
    `initial_graph` in :meth:`fit`.

    Parameters
    ----------
    criterion : {"bdeu", "bic", "gaussian_bic", "bge"}, default="bdeu"
        The score: BDeu or BIC over categorical data, or the BIC or BGe of a
        linear Gaussian model over continuous data.
    ess : float, default=1.0
        Equivalent sample size: of the BDeu prior, or of BGe's prior on the
        mean. Also the prior of the parameters behind :meth:`score` on
        categorical data.
    expected_parents : float or None, default=1.0
        Expected in-degree under the structure prior, for every criterion; see
        :class:`~causalbase.scores.FamilyScore`. ``None`` is a uniform prior.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior: the Bernoulli prior of
        :class:`~causalbase.scores.FamilyScore`, or, for BDeu only, Tetrad
        7.6.5's, with `expected_parents` as Tetrad's structure prior (see
        :class:`~causalbase.scores.BDeuScore`).
    penalty_discount : float, default=1.0
        Multiplier of the penalty of the two BIC criteria; larger values give
        sparser structures.
    alpha_w : float or None, default=None
        Degrees of freedom of BGe's Wishart prior, above ``n_vars + 1``;
        ``None`` is ``n_vars + 2``. See :class:`~causalbase.scores.BGeScore`.
    max_parents, max_t_size, required, forbidden, one_edge_faithfulness
        As in :class:`GES`.
    extended : bool, default=True
        Run the forced deletions after XGES-0. ``False`` is XGES-0, several
        times cheaper.
    n_jobs : int, default=1
        Threads of the score's compiled kernels; -1 is every thread numba has.
        The result does not depend on it.

    Attributes
    ----------
    dag_ : DAG
        The learned network.
    cpdag_ : PDAG
        Its Markov equivalence class.
    score_ : float
        The score of `dag_` under `criterion`, with the structure prior chosen
        (with ``prior="tetrad"``, the value Tetrad reports).
    adjacency_ : pandas.DataFrame
        `cpdag_` with the variable names: ``A[i, j] = 1`` and ``A[j, i] = 0`` is
        ``i -> j``, and both set is an undirected edge.
    n_features_in_ : int
        Number of variables.
    feature_names_in_ : ndarray of str
        Variable names.
    n_iter_ : int
        Operators applied, those of rejected forced deletions included.
    trace_ : list of tuple
        ``(seconds, steps, score)`` at the start, after XGES-0 and after each
        round of forced deletions that improved the score.

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
        alpha_w: float | None = None,
        max_parents: int | None = None,
        max_t_size: int | None = None,
        required: Sequence[PairLike] | None = None,
        forbidden: Sequence[PairLike] | None = None,
        one_edge_faithfulness: bool = False,
        extended: bool = True,
        n_jobs: int = 1,
    ) -> None:
        self.criterion = criterion
        self.ess = ess
        self.expected_parents = expected_parents
        self.prior = prior
        self.penalty_discount = penalty_discount
        self.alpha_w = alpha_w
        self.max_parents = max_parents
        self.max_t_size = max_t_size
        self.required = required
        self.forbidden = forbidden
        self.one_edge_faithfulness = one_edge_faithfulness
        self.extended = extended
        self.n_jobs = n_jobs

    def _search(
        self, score: FamilyScore, initial_graph: DAG | None, names: Sequence[str]
    ) -> SearchResult:
        return xges(
            score,
            initial_graph,
            extended=self.extended,
            max_t_size=self.max_t_size,
            knowledge=self._knowledge(names),
            max_parents=self.max_parents,
            one_edge_faithfulness=self.one_edge_faithfulness,
        )


class HillClimbing(ScoreEstimator):
    """Learn a Bayesian network by hill climbing, or tabu search, over DAGs.

    The search is :func:`~causalbase.search.hill_climbing` on the score
    `criterion` names: from the empty graph, or from `initial_graph` in
    :meth:`fit`, apply the arc addition, deletion or reversal of largest gain
    until none improves, optionally as a tabu search and with random
    restarts, as bnlearn's ``hc`` and ``tabu``.

    Parameters
    ----------
    criterion : {"bdeu", "bic", "gaussian_bic", "bge"}, default="bdeu"
        The score: BDeu or BIC over categorical data, or the BIC or BGe of a
        linear Gaussian model over continuous data.
    ess : float, default=1.0
        Equivalent sample size: of the BDeu prior, or of BGe's prior on the
        mean. Also the prior of the parameters behind :meth:`score` on
        categorical data.
    expected_parents : float or None, default=1.0
        Expected in-degree under the structure prior, for every criterion; see
        :class:`~causalbase.scores.FamilyScore`. ``None`` is a uniform prior.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior: the Bernoulli prior of
        :class:`~causalbase.scores.FamilyScore`, or, for BDeu only, Tetrad
        7.6.5's, with `expected_parents` as Tetrad's structure prior (see
        :class:`~causalbase.scores.BDeuScore`).
    penalty_discount : float, default=1.0
        Multiplier of the penalty of the two BIC criteria; larger values give
        sparser structures.
    alpha_w : float or None, default=None
        Degrees of freedom of BGe's Wishart prior, above ``n_vars + 1``;
        ``None`` is ``n_vars + 2``. See :class:`~causalbase.scores.BGeScore`.
    tabu_length : int, default=0
        With a positive value, tabu search: the best move is applied even if it
        loses score, unless its DAG is among the last `tabu_length` visited.
    max_tabu_moves : int, default=0
        Tabu search stops after this many moves without improving its best DAG.
    n_restarts : int, default=0
        Climbs after the first, each from the best DAG so far perturbed by
        `perturbation` random moves.
    perturbation : int, default=1
        Random moves before each restart.
    time_limit : float or None, default=None
        Wall-clock budget in seconds.
    random_state : int or None, default=None
        Seeds the perturbations; None draws a fresh seed.
    n_jobs : int, default=1
        Threads of the score's compiled kernels; -1 is every thread numba has.
        The result does not depend on it.

    Attributes
    ----------
    dag_ : DAG
        The learned network.
    cpdag_ : PDAG
        Its Markov equivalence class.
    score_ : float
        The score of `dag_` under `criterion`, with the structure prior chosen
        (with ``prior="tetrad"``, the value Tetrad reports).
    adjacency_ : pandas.DataFrame
        `cpdag_` with the variable names: ``A[i, j] = 1`` and ``A[j, i] = 0`` is
        ``i -> j``, and both set is an undirected edge.
    n_features_in_ : int
        Number of variables.
    feature_names_in_ : ndarray of str
        Variable names.
    n_iter_ : int
        Moves applied, perturbations excluded.
    trace_ : list of tuple
        ``(seconds, moves, score)`` each time the best score improved.

    Examples
    --------
    >>> from causalbase import HillClimbing
    >>> from causalbase.datasets import load_asia
    >>> model = HillClimbing(n_restarts=2, random_state=0).fit(load_asia().data)
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
        alpha_w: float | None = None,
        tabu_length: int = 0,
        max_tabu_moves: int = 0,
        n_restarts: int = 0,
        perturbation: int = 1,
        time_limit: float | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.criterion = criterion
        self.ess = ess
        self.expected_parents = expected_parents
        self.prior = prior
        self.penalty_discount = penalty_discount
        self.alpha_w = alpha_w
        self.tabu_length = tabu_length
        self.max_tabu_moves = max_tabu_moves
        self.n_restarts = n_restarts
        self.perturbation = perturbation
        self.time_limit = time_limit
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _check(self) -> None:
        super()._check()
        for name in ("tabu_length", "max_tabu_moves", "n_restarts", "perturbation"):
            check_count(name, getattr(self, name))
        check_time_limit(self.time_limit)

    def _search(
        self, score: FamilyScore, initial_graph: DAG | None, names: Sequence[str]
    ) -> SearchResult:
        return hill_climbing(
            score,
            initial_graph,
            tabu_length=self.tabu_length,
            max_tabu_moves=self.max_tabu_moves,
            n_restarts=self.n_restarts,
            perturbation=self.perturbation,
            time_limit=self.time_limit,
            random_state=self.random_state,
        )
