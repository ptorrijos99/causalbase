"""Unbiased third- and fourth-order cumulants, from moment sums that add up exactly.

Each block of rows is reduced to moment sums about its own mean
(:func:`moment_sums`), and :func:`cumulants_from_sums` merges the sums of any
number of blocks exactly into sums about the pooled mean before applying the
k-statistic bias corrections. The result equals what the pooled data would
give, so the blocks can be chunks of a table too large for memory or, as in
``fedcausal``, the data of separate sites.

References
----------
[1] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery." IEEE TNNLS, 2025.
[2] Schefczik, N. and Hägele, M. "Ready-to-use unbiased estimators for
    multivariate cumulants including one that outperforms x^3."
    arXiv:1904.12154, 2019.
[3] Chen, W., et al. "FedISHC: Federated Causal Discovery via Higher-Order
    Cumulants." AAAI / arXiv, 2025.
[4] Torrijos, P., et al. "Federated Causal Discovery via Regression-Directed
    Cumulants." Probabilistic Graphical Models (PGM), 2026.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Cumulants",
    "MomentSums",
    "cumulants_from_sums",
    "moment_sums",
]

#: The centred moment sums of a block of rows, produced by :func:`moment_sums`.
MomentSums = dict[str, Any]

#: Unbiased cumulant estimates produced by :func:`cumulants_from_sums`.
Cumulants = dict[str, Any]

_MIN_SAMPLES_3RD: int = 3
_MIN_SAMPLES_4TH: int = 4


def _n_samples(stats: Cumulants) -> int:
    """Return ``stats["n"]``, raising `KeyError` if it is missing rather than guessing."""
    if "n" not in stats:
        raise KeyError("stats is missing 'n'; the aggregator always sets it")
    return int(stats["n"])


# ═══════════════════════════════════════════════════════════════════════════
# Moment sums of one block of rows
# ═══════════════════════════════════════════════════════════════════════════

_KEYS_3RD = ("n", "mean", "m11", "m21")
_KEYS_4TH = (*_KEYS_3RD, "m31", "m22")


def moment_sums(x: npt.NDArray[np.float64], order: int = 4) -> MomentSums:
    """Return the moment sums of `x`, centred on its own mean.

    Sums centred on each block's own mean merge exactly, and they stay
    accurate wherever the data are located; raw power sums lose every
    significant digit once the mean is a few thousand standard deviations away
    from zero.

    Parameters
    ----------
    x : numpy.ndarray
        A block of rows, shape ``(n, p)``, with *n* samples and *p* variables.
    order : {3, 4}, default 4
        Maximum order of moments to compute.

    Returns
    -------
    dict
        With ``c = x - mean``:

        - ``n`` : int -- sample count.
        - ``mean`` : ``(p,)`` -- column means.
        - ``m11`` : ``(p, p)`` -- Σ cᵢ cⱼ.
        - ``m21`` : ``(p, p)`` -- Σ cᵢ² cⱼ; its diagonal is Σ cᵢ³.

        When ``order`` is 4, also:

        - ``m31`` : ``(p, p)`` -- Σ cᵢ³ cⱼ; its diagonal is Σ cᵢ⁴.
        - ``m22`` : ``(p, p)`` -- Σ cᵢ² cⱼ².
    """
    if order not in (3, 4):
        raise ValueError(f"order must be 3 or 4, got {order!r}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must be a 2-D array, got shape {x.shape}")
    n = x.shape[0]
    mean = x.mean(axis=0) if n else np.zeros(x.shape[1])
    c = x - mean
    c2 = c * c
    stats: MomentSums = {"n": n, "mean": mean, "m11": c.T @ c, "m21": c2.T @ c}
    if order == 4:
        stats["m31"] = (c2 * c).T @ c
        stats["m22"] = c2.T @ c2
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Merging the blocks into cumulants
# ═══════════════════════════════════════════════════════════════════════════


def _merge(client_stats: Sequence[MomentSums]) -> MomentSums:
    """Merge the blocks' centred moment sums into sums about the pooled mean.

    A block's sums move from its own mean to the pooled one by the binomial
    expansion of ``(cᵢ + dᵢ)^a (cⱼ + dⱼ)^b``, with ``d`` the shift.
    """
    keys = set(client_stats[0])
    if keys not in (set(_KEYS_3RD), set(_KEYS_4TH)):
        raise ValueError(f"unrecognised moment sums with keys {sorted(keys)}")
    if any(set(s) != keys for s in client_stats):
        raise ValueError("every block must have moment sums of the same order")
    p = len(client_stats[0]["mean"])
    if any(len(s["mean"]) != p for s in client_stats):
        raise ValueError("every block must have moment sums over the same variables")
    fourth = "m31" in keys

    n = sum(int(s["n"]) for s in client_stats)
    mean = np.zeros(p)
    for s in client_stats:
        mean += int(s["n"]) * np.asarray(s["mean"], dtype=np.float64)
    mean /= max(n, 1)
    m11 = np.zeros((p, p))
    m21 = np.zeros((p, p))
    m31 = np.zeros((p, p))
    m22 = np.zeros((p, p))
    for s in client_stats:
        k = int(s["n"])
        if k == 0:
            continue
        d = np.asarray(s["mean"]) - mean
        di, dj = d[:, None], d[None, :]
        a11 = np.asarray(s["m11"])
        a21 = np.asarray(s["m21"])
        s2i, s2j = np.diag(a11)[:, None], np.diag(a11)[None, :]
        m11 += a11 + k * di * dj
        m21 += a21 + 2 * di * a11 + dj * s2i + k * di**2 * dj
        if not fourth:
            continue
        a31 = np.asarray(s["m31"])
        s3i = np.diag(a21)[:, None]
        m31 += a31 + 3 * di * a21 + dj * s3i + 3 * di**2 * a11 + 3 * di * dj * s2i + k * di**3 * dj
        m22 += (
            np.asarray(s["m22"])
            + 2 * dj * a21
            + 2 * di * a21.T
            + dj**2 * s2i
            + di**2 * s2j
            + 4 * di * dj * a11
            + k * di**2 * dj**2
        )
    merged: MomentSums = {"n": n, "mean": mean, "m11": m11, "m21": m21}
    if fourth:
        merged.update({"m31": m31, "m22": m22})
    return merged


def cumulants_from_sums(client_stats: Sequence[MomentSums]) -> Cumulants:
    """Merge the blocks' moment sums and return the pooled covariance and cumulants.

    Parameters
    ----------
    client_stats : Sequence[dict]
        :func:`moment_sums` of each block of rows, all of the same order
        and over the same variables. One block gives the cumulants of a
        single table.

    Returns
    -------
    dict
        - ``n`` : int -- total sample count.
        - ``cov`` : ``(p, p)`` -- sample covariance matrix.
        - ``c3`` : ``(p,)`` -- C₃(xᵢ).
        - ``c12`` : ``(p, p)`` -- C₁,₂(xᵢ, xⱼ) = Cum(xᵢ, xⱼ, xⱼ).
        - ``c21`` : ``(p, p)`` -- C₂,₁(xᵢ, xⱼ) = Cum(xᵢ, xᵢ, xⱼ).

        When the sums are of order 4, also:

        - ``c4`` : ``(p,)`` -- C₄(xᵢ).
        - ``c13`` : ``(p, p)`` -- C₁,₃(xᵢ, xⱼ) = Cum(xᵢ, xⱼ, xⱼ, xⱼ).
        - ``c31`` : ``(p, p)`` -- C₃,₁(xᵢ, xⱼ) = Cum(xᵢ, xᵢ, xᵢ, xⱼ).
        - ``c22`` : ``(p, p)`` -- C₂,₂(xᵢ, xⱼ) = Cum(xᵢ, xᵢ, xⱼ, xⱼ).

        An empty dict when `client_stats` is empty.

    Raises
    ------
    ValueError
        If the blocks have sums of different orders or over different
        numbers of variables, or fewer rows in total than the order needs.

    Notes
    -----
    The bias corrections are the k-statistics of Schefczik and Hägele, with
    Mₖ the sum of k-th powers about the pooled mean:

    - 3rd order: k₃ = N / ((N-1)(N-2)) · M₃.
    - 4th order: k₄ = [N(N+1) M₄ - 3(N-1) M₂²] / [(N-1)(N-2)(N-3)].
    """
    if not client_stats:
        return {}
    merged = _merge(client_stats)
    n = merged["n"]
    fourth = "m31" in merged
    needed = _MIN_SAMPLES_4TH if fourth else _MIN_SAMPLES_3RD
    if n < needed:
        raise ValueError(
            f"the data hold {n} rows in total; cumulants of this order need at least {needed}"
        )

    m11, m21 = merged["m11"], merged["m21"]
    c21 = n / ((n - 1) * (n - 2)) * m21
    result: Cumulants = {
        "n": n,
        "cov": m11 / (n - 1),
        "c3": np.diag(c21).copy(),
        "c12": c21.T.copy(),
        "c21": c21,
    }
    if not fourth:
        return result

    d4 = (n - 1) * (n - 2) * (n - 3)
    alpha, beta = n * (n + 1), n - 1
    m2 = np.diag(m11)
    c31 = (alpha * merged["m31"] - 3 * beta * m11 * m2[:, None]) / d4
    c22 = (alpha * merged["m22"] - beta * (np.outer(m2, m2) + 2 * m11**2)) / d4
    result.update({"c4": np.diag(c31).copy(), "c13": c31.T.copy(), "c31": c31, "c22": c22})
    return result
