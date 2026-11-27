"""The scores of a linear Gaussian model: what they share, and the BIC."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from causalbase.scores._family import FamilyScore
from causalbase.scores._gaussian_kernels import extension_parts, family_parts, gaussian_score

if TYPE_CHECKING:
    import numpy.typing as npt

    from causalbase.data import ContinuousDataset

__all__ = ["GaussianBICScore"]


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


class _GaussianScore(FamilyScore):
    """A linear Gaussian score, read from one matrix by the compiled kernels.

    The score of a node with ``k`` parents is
    ``a[k] + b[k] log(max(s, floor)) + c log |M_PP|`` (see
    :mod:`causalbase.scores._gaussian_kernels`), without the structure prior;
    a subclass passes the matrix ``M`` and the coefficients.
    """

    def __init__(
        self,
        data: ContinuousDataset,
        matrix: npt.NDArray[np.float64],
        coefficients: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float],
        floor: float,
        expected_parents: float | None,
        n_jobs: int,
    ) -> None:
        super().__init__(data.n_vars, expected_parents, n_jobs=n_jobs)
        self.data = data
        self.n_rows = data.n_rows
        self._matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        a, b, c = coefficients
        self._a = np.ascontiguousarray(a, dtype=np.float64)
        self._b = np.ascontiguousarray(b, dtype=np.float64)
        self._c = float(c)
        self._floor_rel = float(floor)
        self._full: dict[int, float] = {}

    def _floor(self, node: int) -> float:
        """Return the floor of the residual of `node`, relative to its diagonal entry."""
        return self._floor_rel * max(float(self._matrix[node, node]), self._floor_rel)

    def compute_family(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        """Return the score of `node` given `parents`, without the structure prior."""
        schur, log_det = family_parts(self._matrix, node, parents)
        return float(
            gaussian_score(
                schur, log_det, parents.size, self._a, self._b, self._c, self._floor(node)
            )
        )

    def compute_extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Return the score of every extension, from one factorisation of the parents."""
        schur, dets = extension_parts(self._matrix, node, parents, extra)
        k = parents.size + 1
        out: npt.NDArray[np.float64] = (
            self._a[k] + self._b[k] * np.log(np.maximum(schur, self._floor(node))) + self._c * dets
        )
        return out

    def _exact_parts(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Return the matrix and coefficients, with the structure prior, for the exact search."""
        k = np.arange(self.n_vars)
        return self._matrix, self._a + k * self._log_odds, self._b, self._c

    def _superset_bound(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        """Return an upper bound on the score of every strict superset of `parents`.

        Only for a score without the determinant term (``c = 0``) whose
        coefficient ``b`` is negative, the BIC: the residual cannot grow when a
        parent is added, so no family of `node` has a smaller one than the
        family of every other variable, and the bound is that residual with
        the largest ``a`` a superset can have. It gives way by a little more
        than the rounding of the residual.
        """
        k = parents.size + 1
        if self._c != 0.0 or np.any(self._b > 0):
            return math.inf
        if k > self.n_vars - 1:
            return -math.inf
        best = self._full.get(node)
        if best is None:
            others = np.array([u for u in range(self.n_vars) if u != node], dtype=np.int64)
            schur, _ = family_parts(self._matrix, node, others)
            best = float(np.max(self._b) * math.log(max(schur, self._floor(node))))
            best += 1e-9 * (abs(best) + 1.0)
            self._full[node] = best
        a = self._a[k:] + np.arange(k, self.n_vars) * self._log_odds
        return float(np.max(a)) + best


class GaussianBICScore(_GaussianScore):
    """BIC local score of a linear Gaussian model, memoised per family.

    The maximised log likelihood of regressing a node on its parents,
    ``-n/2 * (log(2 pi s2) + 1)`` with ``s2`` the residual variance, less
    ``penalty_discount * log(n) / 2`` per parameter (a coefficient per parent,
    the intercept and the variance), plus the structure prior of
    :class:`~causalbase.scores.FamilyScore`. It needs only the number of rows
    and the covariance matrix, computed once. A parent that is a linear
    combination of the others explains nothing more and costs only its
    penalty; the residual variance is floored at ``1e-12`` times the
    variable's own, so a deterministic family scores very well instead of
    infinitely well.

    Parameters
    ----------
    data : ContinuousDataset
        The data to score against.
    penalty_discount : float, default=1.0
        Multiplies the penalty; larger values give sparser structures.
    expected_parents : float or None, default=1.0
        Expected number of parents per node under the structure prior, as in
        :class:`~causalbase.scores.BDeuScore`. ``None`` is the plain BIC.
    n_jobs : int, default=1
        Threads of the exact search's compiled kernels; -1 is every thread
        numba has. Results do not depend on it.

    Raises
    ------
    ValueError
        If `penalty_discount` is not positive and finite, `expected_parents`
        is not positive, or `n_jobs` is neither a positive integer nor -1.

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
        n_jobs: int = 1,
    ) -> None:
        if not 0 < penalty_discount < math.inf:
            raise ValueError(
                f"penalty_discount must be positive and finite, got {penalty_discount}"
            )
        rows, n = data.n_rows, data.n_vars
        per_parameter = penalty_discount * math.log(max(rows, 1)) / 2
        k = np.arange(n)
        a = np.asarray(
            -rows / 2 * (math.log(2 * math.pi) + 1) - per_parameter * (k + 2), dtype=np.float64
        )
        b = np.full(n, -rows / 2, dtype=np.float64)
        covariance = _covariance(np.asarray(data.values, dtype=np.float64))
        super().__init__(data, covariance, (a, b, 0.0), 1e-12, expected_parents, n_jobs)
        self.penalty_discount = float(penalty_discount)
        self.covariance = covariance
