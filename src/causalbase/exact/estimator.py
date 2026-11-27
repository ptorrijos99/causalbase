"""Exact structure learning as a scikit-learn style estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from causalbase._score_estimator import ScoreEstimator, initial_order
from causalbase._validation import check_count, check_time_limit
from causalbase.exact.dp import exact_search

if TYPE_CHECKING:
    from collections.abc import Sequence

    from causalbase.graph import DAG
    from causalbase.scores import FamilyScore, Prior, ScoreName
    from causalbase.search import SearchResult

__all__ = ["ExactSearch"]


class ExactSearch(ScoreEstimator):
    """Learn the Bayesian network of highest score, or the best one under a parent limit.

    The search is :func:`~causalbase.exact.exact_search` on the score
    `criterion` names: dynamic programming over variable subsets without a
    parent limit (some 25 variables at most), otherwise an integer program
    solved with SCIP, seeded with a topological order of the network GES
    learns, or of `initial_graph` in :meth:`fit`.

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
        Most parents of a node; None is no limit.
    time_limit : float or None, default=None
        With `max_parents`, SCIP's time limit in seconds; the result is then
        the best network SCIP found.
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
        Always 0.
    trace_ : list of tuple
        ``(seconds, 0, score)`` of the optimum.

    Examples
    --------
    >>> from causalbase import ExactSearch
    >>> from causalbase.datasets import load_asia
    >>> model = ExactSearch().fit(load_asia().data)
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
        time_limit: float | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.criterion = criterion
        self.ess = ess
        self.expected_parents = expected_parents
        self.prior = prior
        self.penalty_discount = penalty_discount
        self.alpha_w = alpha_w
        self.max_parents = max_parents
        self.time_limit = time_limit
        self.n_jobs = n_jobs

    def _check(self) -> None:
        super()._check()
        check_count("max_parents", self.max_parents, allow_none=True)
        check_time_limit(self.time_limit)

    def _search(
        self, score: FamilyScore, initial_graph: DAG | None, names: Sequence[str]
    ) -> SearchResult:
        if self.max_parents is None:
            return exact_search(score)
        start = (
            [int(v) for v in initial_graph.topological_order()]
            if initial_graph is not None
            else initial_order(score, "ges")
        )
        return exact_search(
            score,
            max_parents=self.max_parents,
            initial_order=start,
            time_limit=self.time_limit,
        )
