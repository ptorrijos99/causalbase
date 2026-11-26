"""The base of the estimators on the compiled BDeu: OrderSearch, HillClimbing, FLOP, ExactSearch."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Literal

import numpy as np

from causalbase._base import Estimator, named_adjacency
from causalbase._validation import check_n_jobs, check_positive
from causalbase.data import DiscreteDataset, as_discrete
from causalbase.graph import dag_to_cpdag
from causalbase.scores.bdeu import PRIORS, BDeuScore, Prior
from causalbase.scores.likelihood import log_likelihood

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

    from causalbase.data.dataset import DiscreteLike
    from causalbase.search._result import SearchResult

__all__ = ["BDeuEstimator", "Init", "initial_order"]

Init = Literal["ges", "xges"]
"""The algorithms whose network gives an order search its first order."""


def initial_order(score: BDeuScore, init: Init) -> list[int]:
    """Return a topological order of the network that `init` learns on `score`."""
    from causalbase.order.polished import ges_order

    return ges_order(score, init)


class BDeuEstimator(Estimator):
    """Fit, score and results of the estimators on :class:`~causalbase.scores.BDeuScore`.

    A subclass stores its hyperparameters, including the score's ``ess``,
    ``expected_parents``, ``prior`` and ``n_jobs``, extends :meth:`_check`
    with its own, and implements :meth:`_search`, which runs the search on
    the score of the training data.
    """

    ess: float
    expected_parents: float | None
    prior: Prior
    n_jobs: int

    def _check(self) -> None:
        """Raise ValueError if a hyperparameter is out of range."""
        check_positive("ess", self.ess)
        if self.prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}, got {self.prior!r}")
        if self.expected_parents is not None:
            check_positive(
                "expected_parents", self.expected_parents, allow_zero=self.prior == "tetrad"
            )
        check_n_jobs(self.n_jobs)
        random_state = getattr(self, "random_state", None)
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral)
        ):
            raise ValueError(f"random_state must be an integer or None, got {random_state!r}")

    def _search(self, score: BDeuScore) -> SearchResult:
        """Run the search on `score` and return what it found."""
        raise NotImplementedError

    def fit(self, X: DiscreteLike, y: None = None) -> Self:
        """Learn the structure of `X`.

        Parameters
        ----------
        X : DiscreteDataset, pandas.DataFrame or array-like
            Of shape (n_rows, n_vars); every distinct value of a column is a
            level.
        y : None
            Ignored.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `X` has no rows, or a hyperparameter is out of range.
        """
        self._check()
        data = as_discrete(X)
        if data.n_rows == 0:
            raise ValueError("X has no rows")
        score = BDeuScore(
            data,
            ess=self.ess,
            expected_parents=self.expected_parents,
            prior=self.prior,
            n_jobs=self.n_jobs,
        )
        result = self._search(score)
        self.dag_ = result.dag
        self.cpdag_ = dag_to_cpdag(result.dag)
        self.score_ = score.score_dag(result.dag)
        self.n_iter_ = int(result.n_iter)
        self.trace_ = result.trace
        self.n_features_in_ = data.n_vars
        self.feature_names_in_ = np.asarray(data.names, dtype=object)
        self._data = data
        return self

    def score(self, X: DiscreteLike, y: None = None) -> float:
        """Return the mean log likelihood per row of `X` under the learned network.

        The parameters are learned from the training data with a BDeu prior of
        equivalent sample size `ess`. Higher is better, so a scikit-learn
        parameter search can select the hyperparameters on held-out rows.

        Parameters
        ----------
        X : DiscreteDataset, pandas.DataFrame or array-like
            Rows over the training variables, with no level the training data
            lacks.
        y : None
            Ignored.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If `X` has other variables, or a level the training data lacks.
        """
        self._check_fitted("dag_")
        test: DiscreteDataset = as_discrete(X, self._data.categories)
        return log_likelihood(self.dag_, self._data, test, ess=self.ess) / max(test.n_rows, 1)

    @property
    def adjacency_(self) -> pd.DataFrame:
        """pandas.DataFrame: The learned CPDAG as a named adjacency matrix."""
        self._check_fitted("cpdag_")
        return named_adjacency(self.cpdag_, self.feature_names_in_)
