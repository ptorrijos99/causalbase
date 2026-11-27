"""Compiled routines of the coefficient solvers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

__all__ = ["solve_lasso_cd"]


@njit(cache=True)
def solve_lasso_cd(
    cxx: npt.NDArray[np.float64],
    cxy: npt.NDArray[np.float64],
    lmbda: float,
    weights: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    max_iter: int = 100,
    tol: float = 1e-5,
) -> npt.NDArray[np.float64]:
    """Solve the weighted Lasso by coordinate descent, from covariances alone.

    Minimises ``1/2 beta' Cxx beta - Cxy' beta + lambda |w * beta|``, starting
    from `start`, and stops when a sweep moves the coefficients by less than
    `tol` in Euclidean norm.
    """
    q = cxy.shape[0]
    beta = start.copy()
    for _ in range(max_iter):
        moved = 0.0
        for j in range(q):
            r_j = cxy[j]
            for k in range(q):
                if k != j:
                    r_j -= cxx[j, k] * beta[k]
            threshold = lmbda * weights[j]
            if r_j > threshold:
                new = (r_j - threshold) / cxx[j, j]
            elif r_j < -threshold:
                new = (r_j + threshold) / cxx[j, j]
            else:
                new = 0.0
            moved += (new - beta[j]) ** 2
            beta[j] = new
        if moved < tol * tol:
            break
    return beta
