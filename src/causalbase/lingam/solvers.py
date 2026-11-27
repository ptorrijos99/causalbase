"""RCD and the cumulant methods it generalises, as functions of the cumulants.

Every solver finds a causal order with the tau discriminator, third or fourth
order, and then estimates B on that order from the covariance. They differ
only in what happens after a source is found:

``method="deflation"`` (RCD)
    Removes the source from the cumulants with the OLS coefficients
    ``cov[j, s] / var[s]`` and searches again, without going back to the data.
``method="ratio"`` (FedISHC with ``order=3``)
    The same, with the cumulant-ratio coefficients ``Cum(s, .., s, j) / Cum(s, .., s)``.
``method="sort"`` (HC-LiNGAM)
    No deflation: sorts once by the tau row sums.
:func:`rcd_exact` (HC)
    Deflates the data itself and recomputes its cumulants after each source,
    so no estimation error carries from one step to the next.

RCD here stands for regression-directed cumulants. It is unrelated to the
Repetitive Causal Discovery of Maeda and Shimizu (2020), which shares the
acronym.

References
----------
[1] Torrijos, P., et al. "Federated Causal Discovery via Regression-Directed
    Cumulants." Probabilistic Graphical Models (PGM), 2026.
[2] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery." IEEE TNNLS, 2025.
[3] Chen, W., et al. "FedISHC: Federated Causal Discovery via Higher-Order
    Cumulants." AAAI / arXiv, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from causalbase.graph import DAG
from causalbase.lingam.coefficients import estimate_b_from_cov, estimate_b_lasso_from_cov
from causalbase.lingam.cumulants import (
    Cumulants,
    _n_samples,
    cumulants_from_sums,
    moment_sums,
)
from causalbase.lingam.tau import tau_matrix_3rd, tau_matrix_4th

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["LambdaSelection", "Pruning", "RCDResult", "exact_order", "rcd", "rcd_exact"]

Method = Literal["deflation", "ratio", "sort"]
_METHODS = ("deflation", "ratio", "sort")

Pruning = Literal["adaptive_lasso"] | None
"""How the coefficients are estimated on the order: adaptive lasso, or OLS (``None``)."""

LambdaSelection = Literal["bic", "fixed"]
"""How the adaptive lasso picks its penalty: by BIC over a grid, or a fixed fraction."""

# Below this, a source's own cumulant is treated as zero and the ratio as 0.
_RATIO_EPS = 1e-6
# Cumulant ratios are clipped to this magnitude, as in the FedISHC authors' code.
_RATIO_CLIP = 10.0


@dataclass(frozen=True)
class RCDResult:
    """A causal order and the coefficient matrix estimated for it.

    Attributes
    ----------
    causal_order : tuple of int
        Variable indices, source first. A topological order of the estimated
        structure.
    coef : numpy.ndarray
        Shape ``(p, p)``. ``coef[child, parent]`` is the linear coefficient of
        `parent` in `child`'s structural equation, and is zero whenever
        `parent` does not precede `child` in `causal_order`. ``RCD.coef_``
        is this matrix.
    """

    causal_order: tuple[int, ...]
    coef: npt.NDArray[np.float64]

    def to_dag(self, threshold: float = 0.0) -> DAG:
        """Return the DAG of coefficients whose magnitude exceeds `threshold`.

        Parameters
        ----------
        threshold : float, optional
            Keep an edge only when ``abs(coef[child, parent]) > threshold``. The
            default keeps every non-zero coefficient.

        Returns
        -------
        DAG
            Acyclic by construction: `coef` is lower-triangular with respect to
            `causal_order`, so every edge runs from an earlier variable to a
            later one.
        """
        children, parents = np.nonzero(np.abs(self.coef) > threshold)
        return DAG.from_edges(
            self.coef.shape[0], list(zip(parents.tolist(), children.tolist(), strict=True))
        )


def _resolve_order(stats: Cumulants, order: int | None) -> int:
    if order is None:
        return 4 if "c4" in stats else 3
    if order not in (3, 4):
        raise ValueError(f"order must be 3 or 4, got {order!r}")
    if order == 4 and "c4" not in stats:
        raise ValueError("order=4 needs fourth-order statistics; stats holds only third-order")
    return order


def _tau_row_sums(
    stats: Cumulants, order: int, idx: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    block = np.ix_(idx, idx)
    if order == 4:
        tau = tau_matrix_4th(
            stats["c4"][idx], stats["c13"][block], stats["c31"][block], stats["c22"][block]
        )
    else:
        tau = tau_matrix_3rd(stats["c3"][idx], stats["c12"][block], stats["c21"][block])
    sums: npt.NDArray[np.float64] = tau.sum(axis=1)
    return sums


def _check_estimation(pruning: object, lambda_selection: object) -> None:
    if pruning not in ("adaptive_lasso", None):
        raise ValueError(f"pruning must be 'adaptive_lasso' or None, got {pruning!r}")
    if lambda_selection not in ("bic", "fixed"):
        raise ValueError(f"lambda_selection must be 'bic' or 'fixed', got {lambda_selection!r}")


def _coefficients(
    causal_order: Sequence[int],
    cov: npt.NDArray[np.float64],
    n_samples: int,
    pruning: Pruning,
    lambda_selection: LambdaSelection,
) -> npt.NDArray[np.float64]:
    if pruning == "adaptive_lasso":
        return estimate_b_lasso_from_cov(
            causal_order, cov, n_samples=n_samples, lambda_selection=lambda_selection
        )
    return estimate_b_from_cov(causal_order, cov)


def _ratio(num: npt.NDArray[np.float64], den: float) -> npt.NDArray[np.float64]:
    if abs(den) < _RATIO_EPS:
        return np.zeros_like(num)
    return np.clip(num / den, -_RATIO_CLIP, _RATIO_CLIP)


def _deflated_order(cur: Cumulants, order: int, method: Method) -> list[int]:
    """Remove one source at a time from the cumulants in `cur`, in place."""
    active = list(range(len(cur["cov"])))
    causal_order: list[int] = []
    while len(active) > 1:
        s = active.pop(int(np.argmin(_tau_row_sums(cur, order, np.asarray(active)))))
        causal_order.append(s)
        r = np.asarray(active)
        rr = np.ix_(r, r)
        var_s = max(float(cur["cov"][s, s]), 1e-15)
        if method == "deflation":
            b = cur["cov"][r, s] / var_s
        elif order == 4:
            b = _ratio(cur["c13"][r, s], float(cur["c4"][s]))
        else:
            b = _ratio(cur["c12"][r, s], float(cur["c3"][s]))
        b2, b3 = b * b, b * b * b

        # x_i <- x_i - b_i x_s for every remaining i. Ratio deflation assumes, as
        # FedISHC does, that x_i - b_i x_s is independent of the source, so a
        # marginal cumulant loses only its b_i^k part.
        ratio = method == "ratio"
        cur["cov"][rr] += (
            np.outer(b, b) * var_s - np.outer(b, cur["cov"][r, s]) - np.outer(cur["cov"][r, s], b)
        )
        if order == 3:
            c3_s = cur["c3"][s]
            if ratio:
                cur["c3"][r] -= b3 * c3_s
            else:
                cur["c3"][r] += -b3 * c3_s - 3 * b * cur["c21"][r, s] + 3 * b2 * cur["c12"][r, s]
            cur["c12"][rr] -= np.outer(b, b2) * c3_s
            cur["c21"][rr] -= np.outer(b2, b) * c3_s
            cur["c12"][r, r] = cur["c3"][r]
            cur["c21"][r, r] = cur["c3"][r]
        else:
            c4_s = cur["c4"][s]
            if ratio:
                cur["c4"][r] -= b2 * b2 * c4_s
            else:
                cur["c4"][r] += (
                    -4 * b * cur["c31"][r, s]
                    + 6 * b2 * cur["c22"][r, s]
                    - 4 * b3 * cur["c13"][r, s]
                    + b2 * b2 * c4_s
                )
            cur["c13"][rr] -= np.outer(b, b3) * c4_s
            cur["c31"][rr] -= np.outer(b3, b) * c4_s
            cur["c22"][rr] -= np.outer(b2, b2) * c4_s
            for key in ("c13", "c31", "c22"):
                cur[key][r, r] = cur["c4"][r]
    causal_order.extend(active)
    return causal_order


def rcd(
    stats: Cumulants,
    *,
    method: Method = "deflation",
    order: Literal[3, 4] | None = None,
    pruning: Pruning = "adaptive_lasso",
    lambda_selection: LambdaSelection = "bic",
) -> RCDResult:
    """Find the causal order from the cumulants, then estimate the coefficients.

    Parameters
    ----------
    stats : Cumulants
        Output of :func:`~causalbase.lingam.cumulants_from_sums`.
    method : {"deflation", "ratio", "sort"}, optional
        What follows each source; see the module docstring.
    order : {3, 4} or None, optional
        Which tau discriminator to use. Defaults to 4 when `stats` has
        fourth-order cumulants, else 3.
    pruning : {"adaptive_lasso", None}, default="adaptive_lasso"
        How the coefficients are estimated on the order; ``None`` is OLS,
        which keeps every predecessor as a parent.
    lambda_selection : {"bic", "fixed"}, default="bic"
        Penalty selection for the adaptive lasso.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.

    Raises
    ------
    ValueError
        If `method`, `order`, `pruning` or `lambda_selection` is not one of the
        allowed values, or ``order=4`` is asked of third-order statistics.
    """
    _check_estimation(pruning, lambda_selection)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    k = _resolve_order(stats, order)
    cov = np.asarray(stats["cov"], dtype=np.float64)
    if method == "sort":
        causal_order = np.argsort(
            _tau_row_sums(stats, k, np.arange(len(cov))), kind="stable"
        ).tolist()
    else:
        keys = ("c3", "c12", "c21") if k == 3 else ("c4", "c13", "c31", "c22")
        cur = {key: np.array(stats[key], dtype=np.float64) for key in ("cov", *keys)}
        causal_order = _deflated_order(cur, k, method)
    coef = _coefficients(causal_order, cov, _n_samples(stats), pruning, lambda_selection)
    return RCDResult(tuple(causal_order), coef)


def exact_order(
    first: Cumulants,
    deflate: Callable[[int, npt.NDArray[np.float64]], Cumulants],
    order: Literal[3, 4] | None = None,
) -> list[int]:
    """Run the order search of :func:`rcd_exact` over data held anywhere.

    The search sees the data only through `deflate`, so the same loop serves a
    table in memory and data spread over several machines.

    Parameters
    ----------
    first : Cumulants
        The cumulants of the original data.
    deflate : callable
        ``deflate(source, betas)`` replaces each remaining column ``j`` of the
        data by ``x_j - betas[j] * x_source``, drops column `source` (both
        positions among the remaining columns), and returns the cumulants of
        the result.
    order : {3, 4} or None, optional
        Which tau discriminator to use; as in :func:`rcd`.

    Returns
    -------
    list of int
        The causal order, in the original column indices.
    """
    k = _resolve_order(first, order)
    active = list(range(len(first["cov"])))
    causal_order: list[int] = []
    stats = first
    while len(active) > 1:
        src = int(np.argmin(_tau_row_sums(stats, k, np.arange(len(active)))))
        causal_order.append(active.pop(src))
        cov = np.asarray(stats["cov"], dtype=np.float64)
        betas = cov[:, src] / max(float(cov[src, src]), 1e-15)
        if len(active) > 1:
            stats = deflate(src, betas)
    causal_order.extend(active)
    return causal_order


def _exact_from_blocks(
    blocks: Sequence[npt.ArrayLike],
    order: Literal[3, 4],
    pruning: Pruning,
    lambda_selection: LambdaSelection,
) -> RCDResult:
    """Run :func:`rcd_exact` on data split into blocks of rows, deflating each block."""
    _check_estimation(pruning, lambda_selection)
    local = [np.asarray(x, dtype=np.float64) for x in blocks]

    def pooled() -> Cumulants:
        return cumulants_from_sums([moment_sums(x, order) for x in local])

    def deflate(src: int, betas: npt.NDArray[np.float64]) -> Cumulants:
        keep = np.arange(len(betas)) != src
        for k, x in enumerate(local):
            local[k] = x[:, keep] - np.outer(x[:, src], betas[keep])
        return pooled()

    first = pooled()
    causal_order = exact_order(first, deflate, order)
    coef = _coefficients(
        causal_order, np.asarray(first["cov"]), int(first["n"]), pruning, lambda_selection
    )
    return RCDResult(tuple(causal_order), coef)


def rcd_exact(
    x: npt.ArrayLike,
    *,
    order: Literal[3, 4] = 4,
    pruning: Pruning = "adaptive_lasso",
    lambda_selection: LambdaSelection = "bic",
) -> RCDResult:
    """Find the causal order by deflating the data after each source, as HC does.

    Once the tau discriminator picks a source, every remaining column is
    replaced by its residual on the source (``x_j -= beta_j * x_s``, with the
    OLS coefficient), the source column is dropped, and the cumulants are
    computed afresh from what is left. This is HC (Chen et al., 2025).

    Parameters
    ----------
    x : array-like
        The data, shape ``(n, p)``.
    order : {3, 4}, optional
        Which tau discriminator to use.
    pruning, lambda_selection
        As in :func:`rcd`.

    Returns
    -------
    RCDResult
        The estimated causal order and coefficient matrix.

    Notes
    -----
    ``p - 1`` passes over the data, where :func:`rcd` makes one.
    """
    return _exact_from_blocks([x], order, pruning, lambda_selection)
