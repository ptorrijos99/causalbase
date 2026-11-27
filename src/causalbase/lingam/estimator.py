"""The RCD estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from causalbase._base import Estimator, named_adjacency
from causalbase.data import as_continuous
from causalbase.lingam.cumulants import cumulants_from_sums, moment_sums
from causalbase.lingam.solvers import rcd, rcd_exact

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd
    from typing_extensions import Self

    from causalbase.data import ContinuousDataset
    from causalbase.lingam.solvers import LambdaSelection, Pruning, RCDResult

__all__ = ["RCD"]

_METHODS = ("deflation", "exact", "ratio", "sort")


class RCD(Estimator):
    """Learn a linear non-Gaussian causal model from higher-order cumulants.

    The third- and fourth-order cumulants of the data feed a tau
    discriminator that finds the most exogenous variable; what happens to it
    before the next one is sought depends on `method`. The coefficients are
    then estimated on the order from the covariance.

    Torrijos, Stella, Gámez & Puerta, "Federated Causal Discovery via
    Regression-Directed Cumulants", PGM 2026, of which this is the
    single-table case. Not to be confused with Repetitive Causal Discovery
    (Maeda and Shimizu, 2020).

    Parameters
    ----------
    method : {"deflation", "exact", "ratio", "sort"}, default="deflation"
        What follows each source. ``"deflation"`` removes it from the
        cumulants with regression coefficients and searches again, so the data
        are read once. ``"exact"`` removes it from the data and recomputes the
        cumulants, one pass per variable, so no estimation error carries from
        one step to the next. ``"ratio"`` removes it from the cumulants with
        cumulant-ratio coefficients, as FedISHC does. ``"sort"`` sorts once by the
        tau discriminator.
    order : {3, 4}, default=4
        Order of the cumulants behind the tau discriminator. The fourth-order
        one needs excess kurtosis, the third-order one skew. HC-LiNGAM is
        ``RCD(method="sort")`` and HC ``RCD(method="exact")`` (Chen et al.,
        IEEE TNNLS 2025).
    pruning : {"adaptive_lasso", None}, default="adaptive_lasso"
        How coefficients are estimated on the order. ``None`` is ordinary
        least squares, which keeps every predecessor as a parent.
    lambda_selection : {"bic", "fixed"}, default="bic"
        Penalty selection for the adaptive lasso.

    Attributes
    ----------
    causal_order_ : list of str
        The variables, most exogenous first.
    coef_ : ndarray of shape (n_vars, n_vars)
        ``coef_[child, parent]`` is the coefficient of `parent` in `child`'s
        structural equation.
    dag_ : DAG
        The non-zero coefficients as a DAG.
    adjacency_ : pandas.DataFrame
        `dag_` as a named adjacency matrix, ``A[i, j] = 1`` for ``i -> j``.
    n_features_in_ : int
    feature_names_in_ : ndarray of str

    Examples
    --------
    >>> import numpy as np
    >>> from causalbase.lingam import RCD
    >>> e = np.random.default_rng(0).laplace(size=(5000, 3))
    >>> x = np.column_stack([e[:, 0], 0.8 * e[:, 0] + e[:, 1], e[:, 2]])
    >>> x[:, 2] += -0.7 * x[:, 1]
    >>> RCD().fit(x).causal_order_
    ['x0', 'x1', 'x2']
    """

    def __init__(
        self,
        method: Literal["deflation", "exact", "ratio", "sort"] = "deflation",
        order: Literal[3, 4] = 4,
        pruning: Pruning = "adaptive_lasso",
        lambda_selection: LambdaSelection = "bic",
    ) -> None:
        self.method = method
        self.order = order
        self.pruning = pruning
        self.lambda_selection = lambda_selection

    def fit(self, X: ContinuousDataset | pd.DataFrame | npt.ArrayLike, y: None = None) -> Self:
        """Learn the model from `X`.

        Parameters
        ----------
        X : ContinuousDataset, pandas.DataFrame or array-like
            Real-valued data of shape (n_rows, n_vars).
        y : None
            Ignored.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `X` holds too few rows for cumulants of the chosen order, or a
            hyperparameter is out of range.
        """
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {self.method!r}")
        if self.order not in (3, 4):
            raise ValueError(f"order must be 3 or 4, got {self.order!r}")
        data = as_continuous(X)
        result: RCDResult
        if self.method == "exact":
            result = rcd_exact(
                data.values,
                order=self.order,
                pruning=self.pruning,
                lambda_selection=self.lambda_selection,
            )
        else:
            result = rcd(
                cumulants_from_sums([moment_sums(data.values, self.order)]),
                method=self.method,
                order=self.order,
                pruning=self.pruning,
                lambda_selection=self.lambda_selection,
            )

        names = np.asarray(data.names, dtype=object)
        self.causal_order_ = [str(names[i]) for i in result.causal_order]
        self.coef_ = result.coef
        self.dag_ = result.to_dag()
        self.n_features_in_ = len(names)
        self.feature_names_in_ = names
        return self

    @property
    def adjacency_(self) -> pd.DataFrame:
        """pandas.DataFrame: The learned DAG as a named adjacency matrix."""
        self._check_fitted("dag_")
        return named_adjacency(self.dag_, self.feature_names_in_)
