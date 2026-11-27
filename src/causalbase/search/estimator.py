"""Hill climbing over DAGs as a scikit-learn style estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from causalbase._score_estimator import ScoreEstimator
from causalbase._validation import check_count, check_time_limit
from causalbase.search.hill_climb import hill_climbing

if TYPE_CHECKING:
    from causalbase.scores import FamilyScore, Prior, ScoreName
    from causalbase.search._result import SearchResult

__all__ = ["HillClimbing"]


class HillClimbing(ScoreEstimator):
    """Learn a Bayesian network by hill climbing, or tabu search, over DAGs.

    The search is :func:`~causalbase.search.hill_climbing` on the score
    `criterion` names: from the empty graph, apply the arc addition,
    deletion or reversal of largest gain until none improves, optionally as a
    tabu search and with random restarts, as bnlearn's ``hc`` and ``tabu``.

    Parameters
    ----------
    criterion : {"bdeu", "bic", "gaussian_bic", "bge"}, default="bdeu"
        The score: BDeu or BIC over categorical data, or the BIC or BGe of a
        linear Gaussian model over continuous data.
    ess : float, default=1.0
        Equivalent sample size of the BDeu prior, or of BGe's prior on the
        mean. Also the prior of the parameters behind :meth:`score` on
        categorical data.
    expected_parents : float or None, default=1.0
        The parameter of the structure prior, as in
        :class:`~causalbase.scores.BDeuScore`; None leaves the prior out.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior: the Bernoulli prior of
        :class:`~causalbase.scores.FamilyScore`, or, for BDeu only, Tetrad
        7.6.5's, with `expected_parents` as Tetrad's structure prior.
    penalty_discount : float, default=1.0
        Multiplier of the penalty of the two BIC scores; larger values give
        sparser structures.
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
        Threads of the compiled BDeu engine for large batches of candidate
        parents; the result does not depend on it.

    Attributes
    ----------
    dag_, cpdag_, score_, adjacency_, n_features_in_, feature_names_in_
        As in :class:`~causalbase.order.OrderSearch`.
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

    def _search(self, score: FamilyScore) -> SearchResult:
        return hill_climbing(
            score,
            tabu_length=self.tabu_length,
            max_tabu_moves=self.max_tabu_moves,
            n_restarts=self.n_restarts,
            perturbation=self.perturbation,
            time_limit=self.time_limit,
            random_state=self.random_state,
        )
