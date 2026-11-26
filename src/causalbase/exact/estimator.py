"""Exact structure learning as a scikit-learn style estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from causalbase._bdeu_estimator import BDeuEstimator, initial_order
from causalbase._validation import check_count, check_time_limit
from causalbase.exact.dp import exact_search

if TYPE_CHECKING:
    from causalbase.scores.bdeu import BDeuScore, Prior
    from causalbase.search import SearchResult

__all__ = ["ExactSearch"]


class ExactSearch(BDeuEstimator):
    """Learn the Bayesian network of highest score, or the best one under a parent limit.

    The search is :func:`~causalbase.exact.exact_search` on
    :class:`~causalbase.scores.BDeuScore`: dynamic programming over variable subsets without a
    parent limit (about 27 variables at most), otherwise an integer program
    solved with SCIP, seeded with a topological order of the network GES
    learns.

    Parameters
    ----------
    ess : float, default=1.0
        Equivalent sample size of the BDeu prior.
    expected_parents : float or None, default=1.0
        The parameter of the structure prior, as in
        :class:`~causalbase.scores.BDeuScore`; None leaves the prior out.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior: the Bernoulli prior of
        :class:`~causalbase.scores.FamilyScore`, or Tetrad 7.6.5's, with
        `expected_parents` as Tetrad's structure prior.
    max_parents : int or None, default=None
        Most parents of a node; None is no limit.
    time_limit : float or None, default=None
        With `max_parents`, SCIP's time limit in seconds; the result is then
        the best network SCIP found.
    n_jobs : int, default=1
        Threads of the compiled kernels; the result does not depend on it.

    Attributes
    ----------
    dag_, cpdag_, score_, adjacency_, n_features_in_, feature_names_in_
        As in :class:`~causalbase.order.OrderSearch`.
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
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        prior: Prior = "bernoulli",
        max_parents: int | None = None,
        time_limit: float | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.ess = ess
        self.expected_parents = expected_parents
        self.prior = prior
        self.max_parents = max_parents
        self.time_limit = time_limit
        self.n_jobs = n_jobs

    def _check(self) -> None:
        super()._check()
        check_count("max_parents", self.max_parents, allow_none=True)
        check_time_limit(self.time_limit)

    def _search(self, score: BDeuScore) -> SearchResult:
        if self.max_parents is None:
            return exact_search(score, n_jobs=self.n_jobs)
        return exact_search(
            score,
            max_parents=self.max_parents,
            initial_order=initial_order(score, "ges"),
            time_limit=self.time_limit,
            n_jobs=self.n_jobs,
        )
