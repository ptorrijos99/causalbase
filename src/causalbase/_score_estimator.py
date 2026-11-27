"""The base of the estimators on a local score: OrderSearch, HillClimbing, FLOP, ExactSearch."""

from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING, Literal

import numpy as np

from causalbase._base import Estimator, named_adjacency
from causalbase._validation import check_n_jobs, check_positive
from causalbase.data import ContinuousDataset, DiscreteDataset, as_continuous, as_discrete
from causalbase.graph import dag_to_cpdag
from causalbase.scores import CONTINUOUS_SCORES, PRIORS, SCORE_NAMES, make_score
from causalbase.scores.bdeu import BDeuScore
from causalbase.scores.likelihood import log_likelihood

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd
    from typing_extensions import Self

    from causalbase.data.dataset import DiscreteLike
    from causalbase.scores import FamilyScore, Prior, ScoreName
    from causalbase.search._result import SearchResult

__all__ = ["Init", "ScoreEstimator", "initial_order"]

Init = Literal["ges", "xges"]
"""The algorithms whose network gives an order search its first order."""


def initial_order(score: FamilyScore, init: Init) -> list[int]:
    """Return a topological order of the network that `init` learns on `score`."""
    from causalbase.order.polished import ges_order

    return ges_order(score, init)


class ScoreEstimator(Estimator):
    """Fit, score and results of the estimators that search one local score.

    A subclass stores its hyperparameters, including the score's
    ``criterion``, ``ess``, ``expected_parents``, ``prior``,
    ``penalty_discount`` and ``n_jobs``, extends :meth:`_check` with its own,
    and implements :meth:`_search`, which runs the search on the score of the
    training data.
    """

    criterion: ScoreName
    ess: float
    expected_parents: float | None
    prior: Prior
    penalty_discount: float
    n_jobs: int

    def _check(self) -> None:
        """Raise ValueError if a hyperparameter is out of range."""
        if self.criterion not in SCORE_NAMES:
            raise ValueError(f"criterion must be one of {SCORE_NAMES}, got {self.criterion!r}")
        check_positive("ess", self.ess)
        if self.prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}, got {self.prior!r}")
        if self.prior != "bernoulli" and self.criterion != "bdeu":
            raise ValueError(
                f"prior={self.prior!r} is a prior of the 'bdeu' criterion only, "
                f"not {self.criterion!r}"
            )
        if self.expected_parents is not None:
            check_positive(
                "expected_parents", self.expected_parents, allow_zero=self.prior == "tetrad"
            )
        check_positive("penalty_discount", self.penalty_discount)
        if not math.isfinite(self.penalty_discount):
            raise ValueError(f"penalty_discount must be finite, got {self.penalty_discount!r}")
        check_n_jobs(self.n_jobs)
        random_state = getattr(self, "random_state", None)
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral)
        ):
            raise ValueError(f"random_state must be an integer or None, got {random_state!r}")

    def _search(self, score: FamilyScore) -> SearchResult:
        """Run the search on `score` and return what it found."""
        raise NotImplementedError

    def _encode(
        self, X: DiscreteLike | ContinuousDataset | npt.ArrayLike
    ) -> DiscreteDataset | ContinuousDataset:
        if self.criterion in CONTINUOUS_SCORES:
            return as_continuous(X)  # type: ignore[arg-type]
        return as_discrete(X)  # type: ignore[arg-type]

    def _score_of(self, data: DiscreteDataset | ContinuousDataset) -> FamilyScore:
        if self.criterion == "bdeu" and isinstance(data, DiscreteDataset):
            return BDeuScore(
                data,
                ess=self.ess,
                expected_parents=self.expected_parents,
                prior=self.prior,
                n_jobs=self.n_jobs,
            )
        return make_score(
            self.criterion,
            data,
            ess=self.ess,
            expected_parents=self.expected_parents,
            penalty_discount=self.penalty_discount,
        )

    def fit(self, X: DiscreteLike | ContinuousDataset | npt.ArrayLike, y: None = None) -> Self:
        """Learn the structure of `X`.

        Parameters
        ----------
        X : DiscreteDataset, ContinuousDataset, pandas.DataFrame or array-like
            Of shape (n_rows, n_vars). For the discrete criteria every distinct
            value of a column is a level; for ``"gaussian_bic"`` and ``"bge"``
            the columns are real-valued.
        y : None
            Ignored.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `X` has no rows, or a hyperparameter is out of range (including
            ``prior="tetrad"`` with a criterion other than ``"bdeu"``).
        """
        self._check()
        data = self._encode(X)
        if data.n_rows == 0:
            raise ValueError("X has no rows")
        score = self._score_of(data)
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

    def score(self, X: DiscreteLike | ContinuousDataset | npt.ArrayLike, y: None = None) -> float:
        """Return the mean log likelihood per row of `X` under the learned network.

        The parameters are learned from the training data: by a BDeu prior of
        equivalent sample size `ess` for categorical data, by maximum
        likelihood for continuous data. Higher is better, so a scikit-learn
        parameter search can select the hyperparameters on held-out rows.

        Parameters
        ----------
        X : DiscreteDataset, ContinuousDataset, pandas.DataFrame or array-like
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
        if isinstance(self._data, DiscreteDataset):
            test: DiscreteDataset | ContinuousDataset = as_discrete(
                X,  # type: ignore[arg-type]
                self._data.categories,
            )
        else:
            test = as_continuous(X)  # type: ignore[arg-type]
        return log_likelihood(self.dag_, self._data, test, ess=self.ess) / max(test.n_rows, 1)

    @property
    def adjacency_(self) -> pd.DataFrame:
        """pandas.DataFrame: The learned CPDAG as a named adjacency matrix."""
        self._check_fitted("cpdag_")
        return named_adjacency(self.cpdag_, self.feature_names_in_)
