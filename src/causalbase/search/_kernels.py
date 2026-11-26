"""Compiled inner loops of the operator search."""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["keep_better"]


@njit(cache=True)
def keep_better(
    xs: np.ndarray,
    s_with: np.ndarray,
    s_without: float,
    rel_tol: float,
    best: np.ndarray,
    which: np.ndarray,
    index: int,
) -> bool:
    """Record in `best` and `which` the nodes of `xs` that subset `index` improves on.

    Node ``xs[i]`` improves by ``s_with[i] - s_without``, which counts when it
    is larger than `rel_tol` times the scores involved, so not rounding, and
    than what `best` holds for the node. Returns whether any node was recorded.
    """
    recorded = False
    for i in range(xs.shape[0]):
        delta = s_with[i] - s_without
        if delta > rel_tol * max(abs(s_with[i]), abs(s_without), 1.0) and delta > best[xs[i]]:
            best[xs[i]] = delta
            which[xs[i]] = index
            recorded = True
    return recorded
