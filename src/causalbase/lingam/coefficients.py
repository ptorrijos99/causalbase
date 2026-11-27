"""Coefficient estimates for a known causal order, from the covariance alone.

Each variable is regressed on the ones before it in the order, using only the
aggregated covariance, so no data is needed. OLS, or an adaptive Lasso that
also shrinks small coefficients to zero.

References
----------
[1] Zou, H. "The Adaptive Lasso and Its Oracle Properties." Journal of the
    American Statistical Association, 101(476), 2006.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from causalbase.lingam._kernels import solve_lasso_cd

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["estimate_b_from_cov", "estimate_b_lasso_from_cov"]


def estimate_b_from_cov(
    causal_order: Sequence[int],
    cov: npt.NDArray[np.float64],
    prune_thres: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Estimate the coefficient matrix B by OLS.

    Parameters
    ----------
    causal_order : Sequence[int]
        Variable indices from source to sink.
    cov : numpy.ndarray
        Covariance matrix, shape ``(p, p)``.
    prune_thres : float
        Coefficients smaller than this in absolute value become zero.

    Returns
    -------
    numpy.ndarray
        ``B[child, parent]``, shape ``(p, p)``.
    """
    p = cov.shape[0]
    b_est = np.zeros((p, p))
    for idx in range(1, len(causal_order)):
        child = causal_order[idx]
        parents = causal_order[:idx]
        sigma_pp = cov[np.ix_(parents, parents)]
        sigma_pc = cov[parents, child]
        try:
            beta = np.linalg.solve(sigma_pp, sigma_pc)
        except np.linalg.LinAlgError:
            # Singular covariance: fall back to least squares.
            beta = np.linalg.lstsq(sigma_pp, sigma_pc, rcond=None)[0]
        b_est[child, parents] = beta

    if prune_thres > 0:
        b_est[np.abs(b_est) < prune_thres] = 0.0
    return b_est


def estimate_b_lasso_from_cov(
    causal_order: Sequence[int],
    cov: npt.NDArray[np.float64],
    n_samples: int = 1000,
    lambda_selection: str = "fixed",
    prune_thres: float = 0.05,
    gamma: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Estimate the coefficient matrix B by adaptive Lasso.

    Parameters
    ----------
    causal_order : Sequence[int]
        Variable indices from source to sink.
    cov : numpy.ndarray
        Covariance matrix, shape ``(p, p)``.
    n_samples : int
        Total number of samples. Only ``"bic"`` uses it.
    lambda_selection : {"fixed", "bic"}
        ``"fixed"`` sets the penalty to 0.01 times the mean variance of the
        parents; ``"bic"`` picks it from a grid of 20 by BIC.
    prune_thres : float
        Coefficients smaller than this in absolute value become zero.
    gamma : float
        Exponent of the adaptive weights.

    Returns
    -------
    numpy.ndarray
        ``B[child, parent]``, shape ``(p, p)``.

    Raises
    ------
    ValueError
        If `lambda_selection` is not ``"fixed"`` or ``"bic"``.
    """
    p = cov.shape[0]
    b_est = np.zeros((p, p))
    for idx in range(1, len(causal_order)):
        child = causal_order[idx]
        parents = causal_order[:idx]
        sigma_pp = cov[np.ix_(parents, parents)]
        sigma_pc = cov[parents, child]
        sigma_cc = cov[child, child]

        # OLS gives the adaptive weights.
        try:
            beta_ols = np.linalg.solve(sigma_pp, sigma_pc)
        except np.linalg.LinAlgError:
            beta_ols = np.linalg.lstsq(sigma_pp, sigma_pc, rcond=None)[0]

        weights = 1.0 / (np.abs(beta_ols) ** gamma + 1e-10)

        if lambda_selection == "fixed":
            lmbda = float(0.01 * np.mean(np.diag(sigma_pp)))
            beta = solve_lasso_cd(sigma_pp, sigma_pc, lmbda, weights, np.zeros(len(parents)))
        elif lambda_selection == "bic":
            lmbda_max = np.max(np.abs(sigma_pc)) if len(sigma_pc) > 0 else 0
            if lmbda_max < 1e-12:
                beta = np.zeros(len(parents))
            else:
                # From the largest penalty down, each fit starting from the last.
                lambdas = np.logspace(np.log10(lmbda_max), np.log10(lmbda_max * 1e-3), 20)
                best_bic = np.inf
                beta = np.zeros(len(parents))
                log_n = np.log(n_samples)
                curr_beta = np.zeros(len(parents))

                for lam in lambdas:
                    curr_beta = solve_lasso_cd(sigma_pp, sigma_pc, float(lam), weights, curr_beta)
                    # RSS / n from the covariance alone.
                    rss_n = (
                        sigma_cc
                        - 2 * curr_beta.dot(sigma_pc)
                        + curr_beta.dot(sigma_pp).dot(curr_beta)
                    )
                    rss_n = max(rss_n, 1e-12)
                    k = np.count_nonzero(curr_beta)
                    bic = n_samples * np.log(rss_n) + k * log_n
                    if bic < best_bic:
                        best_bic = bic
                        beta = curr_beta
        else:
            raise ValueError(f"lambda_selection must be 'bic' or 'fixed', got {lambda_selection!r}")

        b_est[child, parents] = beta

    b_est[np.abs(b_est) < prune_thres] = 0.0
    return b_est
