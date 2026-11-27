"""The base of every estimator: the score it searches, fit, and the results it keeps."""

from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING, Literal

import numpy as np

from causalbase._base import Estimator, named_adjacency
from causalbase._validation import check_n_jobs, check_positive
from causalbase.data import ContinuousDataset, DiscreteDataset, as_continuous, as_discrete
from causalbase.graph import DAG, PDAG
from causalbase.scores import CONTINUOUS_SCORES, PRIORS, SCORE_NAMES, make_score
from causalbase.scores.likelihood import log_likelihood

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    import pandas as pd
    from typing_extensions import Self

    from causalbase.data.dataset import DiscreteLike
    from causalbase.scores import FamilyScore, Prior, ScoreName
    from causalbase.search._result import SearchResult

__all__ = ["SCORE_PARAMETERS", "Init", "ScoreEstimator", "initial_order"]

Init = Literal["ges", "xges"]
"""The algorithms whose network gives an order search its first order."""

SCORE_PARAMETERS = """\
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
"""
"""The numpydoc entries of the score's hyperparameters, shared by every estimator."""

RESULT_ATTRIBUTES = """\
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
"""
"""The numpydoc entries of the results every estimator keeps after ``fit``."""


def initial_order(score: FamilyScore, init: Init) -> list[int]:
    """Return a topological order of the network that `init` learns on `score`."""
    from causalbase.order.polished import ges_order

    return ges_order(score, init)


class ScoreEstimator(Estimator):
    """Fit, score and results of every estimator: a search over one local score.

    A subclass stores its hyperparameters (the score's are those of
    :data:`SCORE_PARAMETERS`, plus ``n_jobs``), extends :meth:`_check` with
    its own, and implements :meth:`_search`, which runs the search on the
    score of the training data and returns its
    :class:`~causalbase.search.SearchResult`.
    """

    criterion: ScoreName
    ess: float
    expected_parents: float | None
    prior: Prior
    penalty_discount: float
    alpha_w: float | None
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
        if self.alpha_w is not None:
            check_positive("alpha_w", self.alpha_w)
        check_n_jobs(self.n_jobs)
        random_state = getattr(self, "random_state", None)
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral)
        ):
            raise ValueError(f"random_state must be an integer or None, got {random_state!r}")

    def _search(
        self, score: FamilyScore, initial_graph: DAG | None, names: Sequence[str]
    ) -> SearchResult:
        """Run the search on `score`, from `initial_graph` if given, and return what it found.

        `names` are the variables', for hyperparameters that name them.
        """
        raise NotImplementedError

    def _encode(
        self, X: DiscreteLike | ContinuousDataset | npt.ArrayLike
    ) -> DiscreteDataset | ContinuousDataset:
        if self.criterion in CONTINUOUS_SCORES:
            return as_continuous(X)  # type: ignore[arg-type]
        return as_discrete(X)  # type: ignore[arg-type]

    def fit(
        self,
        X: DiscreteLike | ContinuousDataset | npt.ArrayLike,
        y: None = None,
        initial_graph: DAG | PDAG | None = None,
    ) -> Self:
        """Learn the structure of `X`.

        Parameters
        ----------
        X : DiscreteDataset, ContinuousDataset, pandas.DataFrame or array-like
            Of shape (n_rows, n_vars). For the discrete criteria every distinct
            value of a column is a level; for ``"gaussian_bic"`` and ``"bge"``
            the columns are real-valued.
        y : None
            Ignored.
        initial_graph : DAG or PDAG, optional
            Where the search starts, over the columns of `X`: the graph itself
            for the searches over graphs and equivalence classes, a
            topological order of it for the searches over orders, and the
            first solution of the integer program for ``ExactSearch`` with a
            parent limit (the dynamic programming has no starting point). A
            PDAG stands for a DAG it extends to. By default each estimator's
            own start, which its docstring gives.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `X` has no rows, `initial_graph` has another number of nodes,
            or a hyperparameter is out of range (including ``prior="tetrad"``
            with a criterion other than ``"bdeu"``).
        """
        self._check()
        data = self._encode(X)
        if data.n_rows == 0:
            raise ValueError("X has no rows")
        start = None
        if initial_graph is not None:
            if initial_graph.num_nodes != data.n_vars:
                raise ValueError(
                    f"initial_graph has {initial_graph.num_nodes} nodes, expected {data.n_vars}"
                )
            start = initial_graph if isinstance(initial_graph, DAG) else initial_graph.to_dag()
        score = make_score(
            self.criterion,
            data,
            ess=self.ess,
            expected_parents=self.expected_parents,
            prior=self.prior,
            penalty_discount=self.penalty_discount,
            alpha_w=self.alpha_w,
            n_jobs=self.n_jobs,
        )
        result = self._search(score, start, data.names)
        self.dag_ = result.dag
        self.cpdag_ = result.cpdag
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
