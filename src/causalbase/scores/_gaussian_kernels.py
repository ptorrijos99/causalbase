"""Compiled kernels behind the Gaussian scores.

Both Gaussian scores read one symmetric matrix ``M`` (the covariance matrix
for the BIC, the posterior scale matrix for BGe) and score the family of a
node ``y`` with parents ``P`` from two numbers: the Schur complement
``s = M_yy - M_yP M_PP^-1 M_Py`` (the residual variance of ``y`` given ``P``,
up to scale) and ``log |M_PP|``. A score is then

    a[k] + b[k] * log(max(s, floor)) + c * log |M_PP|,    k = |P|,

with coefficients that each score sets (:class:`_Coefficients`).

``M_PP`` is factored by a Cholesky decomposition that drops a parent whose
pivot falls below ``PIVOT`` times its diagonal entry: a parent that is a
linear combination of the others explains nothing more, which is what a
least-squares fit with a pseudo-inverse gives too, and the factorisation does
not break down on collinear or rank-deficient data. The extensions of ``P`` by
one more parent ``x`` share the factor of ``M_PP``, since
``s(y | P, x) = s(y | P) - s(x, y | P)^2 / s(x | P)`` and
``|M_(P+x)| = |M_PP| s(x | P)``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

__all__ = ["PIVOT", "family_parts", "extension_parts", "gaussian_score"]

PIVOT = 1e-12
"""A pivot below this fraction of its diagonal entry marks a linearly dependent variable."""


@njit(cache=True, nogil=True)
def _factor(M: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the Cholesky factor of ``M[parents, parents]``, dead rows zero, and its log det."""
    k = parents.shape[0]
    L = np.zeros((k, k))
    log_det = 0.0
    for i in range(k):
        pi = parents[i]
        for j in range(i + 1):
            pj = parents[j]
            s = M[pi, pj]
            for t in range(j):
                s -= L[i, t] * L[j, t]
            if i == j:
                if s > PIVOT * M[pi, pi]:
                    L[i, i] = math.sqrt(s)
                    log_det += math.log(s)
                # else: dependent on the parents before it, its row stays zero
            elif L[j, j] > 0.0:
                L[i, j] = s / L[j, j]
    return L, log_det


@njit(cache=True, nogil=True)
def _whiten(M: np.ndarray, parents: np.ndarray, L: np.ndarray, col: int) -> np.ndarray:
    """Return ``L^-1 M[parents, col]``, zero on the dead rows of ``L``."""
    k = parents.shape[0]
    w = np.zeros(k)
    for i in range(k):
        if L[i, i] > 0.0:
            s = M[parents[i], col]
            for t in range(i):
                s -= L[i, t] * w[t]
            w[i] = s / L[i, i]
    return w


@njit(cache=True, nogil=True)
def family_parts(M: np.ndarray, node: int, parents: np.ndarray) -> tuple[float, float]:
    """Return ``s(node | parents)`` and ``log |M_PP|``."""
    L, log_det = _factor(M, parents)
    w = _whiten(M, parents, L, node)
    return M[node, node] - np.dot(w, w), log_det


@njit(cache=True, nogil=True)
def extension_parts(
    M: np.ndarray, node: int, parents: np.ndarray, extra: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``s(node | parents, x)`` and ``log |M_(P+x)|`` for each `x` in `extra`.

    A candidate that is a linear combination of the parents adds nothing: its
    family scores as the parents alone, with the same determinant.
    """
    L, log_det = _factor(M, parents)
    w_node = _whiten(M, parents, L, node)
    base = M[node, node] - np.dot(w_node, w_node)
    n = extra.shape[0]
    schur = np.empty(n)
    dets = np.empty(n)
    for e in range(n):
        x = extra[e]
        w = _whiten(M, parents, L, x)
        pivot = M[x, x] - np.dot(w, w)
        if pivot > PIVOT * M[x, x]:
            cross = M[x, node] - np.dot(w, w_node)
            schur[e] = base - cross * cross / pivot
            dets[e] = log_det + math.log(pivot)
        else:
            schur[e] = base
            dets[e] = log_det
    return schur, dets


@njit(cache=True, nogil=True)
def gaussian_score(
    schur: float, log_det: float, k: int, a: np.ndarray, b: np.ndarray, c: float, floor: float
) -> float:
    """Return ``a[k] + b[k] log(max(schur, floor)) + c log_det``."""
    return a[k] + b[k] * math.log(max(schur, floor)) + c * log_det
