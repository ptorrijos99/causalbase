"""The BIC score of a linear Gaussian model, from the covariance matrix alone."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from causalbase.scores._family import FamilyScore

if TYPE_CHECKING:
    import numpy.typing as npt

    from causalbase.data import ContinuousDataset

__all__ = ["GaussianBICScore"]

# Residual variances are floored here, relative to the variable's own, so a
# deterministic family scores very well instead of infinitely well.
_MIN_VARIANCE = 1e-12


def _covariance(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the maximum likelihood covariance of the columns of `values`.

    ``np.cov(values, rowvar=False)`` reads a single row as a single variable,
    and no rows as NaN; both are all zeros here.
    """
    n_rows, n_vars = values.shape
    if n_rows <= 1 or n_vars == 0:
        return np.zeros((n_vars, n_vars))
    cov: npt.NDArray[np.float64] = np.atleast_2d(np.cov(values, rowvar=False, bias=True))
    return cov


class GaussianBICScore(FamilyScore):
    """BIC local score of a linear Gaussian model, memoised per family.

    The maximised log likelihood of regressing a node on its parents,
    ``-n/2 * (log(2 pi s2) + 1)`` with ``s2`` the residual variance, less
    ``penalty_discount * log(n) / 2`` per parameter (a coefficient per parent,
    the intercept and the variance), plus the structure prior of
    :class:`~causalbase.scores.FamilyScore`. It needs only the number of rows
    and the covariance matrix, computed once.

    Parameters
    ----------
    data : ContinuousDataset
        The data to score against.
    penalty_discount : float, default=1.0
        Multiplies the penalty; larger values give sparser structures.
    expected_parents : float or None, default=1.0
        Expected number of parents per node under the structure prior, as in
        :class:`~causalbase.scores.BDeuScore`. ``None`` is the plain BIC.

    Notes
    -----
    Not safe to share between threads, since it caches every family it
    computes; use one score per thread.
    """

    def __init__(
        self,
        data: ContinuousDataset,
        *,
        penalty_discount: float = 1.0,
        expected_parents: float | None = 1.0,
    ) -> None:
        if not 0 < penalty_discount < math.inf:
            raise ValueError(
                f"penalty_discount must be positive and finite, got {penalty_discount}"
            )
        super().__init__(data.n_vars, expected_parents)
        self.data = data
        self.penalty_discount = float(penalty_discount)
        self.n_rows = data.n_rows
        self.covariance = _covariance(data.values)
        self._per_parameter = penalty_discount * math.log(max(data.n_rows, 1)) / 2

    def _likelihood(
        self, node: int, variance: npt.NDArray[np.float64] | float
    ) -> npt.NDArray[np.float64]:
        floor = _MIN_VARIANCE * max(float(self.covariance[node, node]), _MIN_VARIANCE)
        s2 = np.maximum(variance, floor)
        out: npt.NDArray[np.float64] = -self.n_rows / 2 * (np.log(2 * math.pi * s2) + 1)
        return out

    def compute_family(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        """Return the BIC of regressing `node` on `parents`."""
        cov = self.covariance
        variance = float(cov[node, node])
        if parents.size:
            c = cov[parents, node]
            variance -= float(c @ np.linalg.lstsq(cov[np.ix_(parents, parents)], c, rcond=None)[0])
        k = parents.size
        return float(self._likelihood(node, variance)) - self._per_parameter * (k + 2)

    def compute_extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Return the BIC of every extension, from partial covariances."""
        # Partial covariances given the parents, then one more regressor each:
        # s2(y | P, x) = s2(y | P) - cov(y, x | P)^2 / var(x | P).
        cov = self.covariance
        cols = np.append(extra, node)
        block = cov[np.ix_(cols, cols)]
        diag, with_y = block.diagonal().copy(), block[:, -1].copy()
        if parents.size:
            cross = cov[np.ix_(parents, cols)]
            solved = np.linalg.lstsq(cov[np.ix_(parents, parents)], cross, rcond=None)[0]
            diag -= np.einsum("ij,ij->j", cross, solved)
            with_y -= cross.T @ solved[:, -1]
        s2_y = diag[-1]
        var_x, cov_xy = diag[:-1], with_y[:-1]
        safe = var_x > _MIN_VARIANCE * np.maximum(cov[extra, extra], _MIN_VARIANCE)
        gain = np.where(safe, cov_xy**2 / np.where(safe, var_x, 1.0), 0.0)
        k = parents.size + 1
        return self._likelihood(node, s2_y - gain) - self._per_parameter * (k + 2)
