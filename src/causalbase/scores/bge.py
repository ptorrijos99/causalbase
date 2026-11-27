"""The BGe score: the Bayesian marginal likelihood of a linear Gaussian model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from causalbase.scores.gaussian import _GaussianScore

if TYPE_CHECKING:
    from causalbase.data import ContinuousDataset

__all__ = ["BGeScore"]


class BGeScore(_GaussianScore):
    """BGe local score of a linear Gaussian model, memoised per family.

    The log marginal likelihood of Geiger and Heckerman (1994, 2002) under a
    normal-Wishart prior, the Gaussian counterpart of BDeu: Markov equivalent
    DAGs score the same. With the corrections of Kuipers, Moffa and Heckerman
    (2014), the local score of a node ``i`` with ``l`` parents ``P`` is ::

        - N/2 log(pi) + 1/2 log(a_mu / (N + a_mu))
        + lgamma((N + a_w - n + l + 1) / 2) - lgamma((a_w - n + l + 1) / 2)
        + (a_w - n + 2l + 1) / 2 * log(t)
        + (N + a_w - n + l) / 2 * log|R_P| - (N + a_w - n + l + 1) / 2 * log|R_(P+i)|

    for ``N`` rows and ``n`` variables, with ``R = t I + S``, ``S`` the
    matrix of sums of squares about the sample means and
    ``t = a_mu (a_w - n - 1) / (a_mu + 1)``. The prior mean is the vector of
    sample means, as bnlearn's ``bge`` takes it by default. The structure
    prior of :class:`~causalbase.scores.FamilyScore` is added.

    The prior scale ``t I`` does not scale with the data, so the score, unlike
    the BIC, changes when a variable is rescaled; standardise the columns to
    give every variable the same prior.

    Parameters
    ----------
    data : ContinuousDataset
        The data to score against.
    alpha_mu : float, default=1.0
        Equivalent sample size of the prior on the mean.
    alpha_w : float or None, default=None
        Degrees of freedom of the Wishart prior on the precision matrix; it
        must exceed ``n_vars + 1``. ``None`` is ``n_vars + 2``, the smallest
        integer allowed and bnlearn's default.
    expected_parents : float or None, default=1.0
        Expected number of parents per node under the structure prior, as in
        :class:`~causalbase.scores.BDeuScore`. ``None`` is the plain BGe.
    n_jobs : int, default=1
        Threads of the exact search's compiled kernels; -1 is every thread
        numba has. Results do not depend on it.

    Raises
    ------
    ValueError
        If `alpha_mu` is not positive and finite, `alpha_w` is not finite and
        above ``n_vars + 1``, `expected_parents` is not positive, or `n_jobs`
        is neither a positive integer nor -1.

    Notes
    -----
    Not safe to share between threads, since it caches every family it
    computes; use one score per thread.

    Examples
    --------
    >>> import numpy as np
    >>> from causalbase.data import ContinuousDataset
    >>> from causalbase.graph import DAG
    >>> from causalbase.scores import BGeScore
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=500)
    >>> data = ContinuousDataset(np.column_stack([x, x + rng.normal(size=500)]), ["x", "y"])
    >>> score = BGeScore(data)
    >>> forward, backward = DAG.from_edges(2, [(0, 1)]), DAG.from_edges(2, [(1, 0)])
    >>> bool(np.isclose(score.score_dag(forward), score.score_dag(backward)))
    True
    """

    def __init__(
        self,
        data: ContinuousDataset,
        *,
        alpha_mu: float = 1.0,
        alpha_w: float | None = None,
        expected_parents: float | None = 1.0,
        n_jobs: int = 1,
    ) -> None:
        n = data.n_vars
        if not 0 < alpha_mu < math.inf:
            raise ValueError(f"alpha_mu must be positive and finite, got {alpha_mu}")
        dof = n + 2.0 if alpha_w is None else float(alpha_w)
        if not n + 1 < dof < math.inf:
            raise ValueError(f"alpha_w must be finite and above n_vars + 1 = {n + 1}, got {dof}")
        rows = data.n_rows
        t = alpha_mu * (dof - n - 1) / (alpha_mu + 1)
        values = np.asarray(data.values, dtype=np.float64)
        centred = values - values.mean(axis=0) if rows else values
        scale = t * np.eye(n) + centred.T @ centred
        a_free = dof - n
        k = np.arange(n)
        a = np.array(
            [
                -rows / 2 * math.log(math.pi)
                + 0.5 * math.log(alpha_mu / (rows + alpha_mu))
                + math.lgamma((rows + a_free + j + 1) / 2)
                - math.lgamma((a_free + j + 1) / 2)
                + (a_free + 2 * j + 1) / 2 * math.log(t)
                for j in range(n)
            ]
        )
        b = -(rows + a_free + k + 1) / 2
        super().__init__(data, scale, (a, b, -0.5), 1e-15, expected_parents, n_jobs)
        self.alpha_mu = float(alpha_mu)
        self.alpha_w = dof
        self.t = t
        self.scale_matrix = scale
