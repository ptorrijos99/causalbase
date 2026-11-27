"""The BGe score: the Bayesian marginal likelihood of a linear Gaussian model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from causalbase.scores._family import FamilyScore

if TYPE_CHECKING:
    import numpy.typing as npt

    from causalbase.data import ContinuousDataset

__all__ = ["BGeScore"]

# Pivots of the posterior scale matrix are floored here, relative to the diagonal
# entry, so that rounding cannot take a logarithm of zero or of a negative number.
_MIN_PIVOT = 1e-15


class BGeScore(FamilyScore):
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

    Raises
    ------
    ValueError
        If `alpha_mu` is not positive and finite, or `alpha_w` is not finite
        and above ``n_vars + 1``.

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
    ) -> None:
        n = data.n_vars
        if not 0 < alpha_mu < math.inf:
            raise ValueError(f"alpha_mu must be positive and finite, got {alpha_mu}")
        dof = n + 2.0 if alpha_w is None else float(alpha_w)
        if not n + 1 < dof < math.inf:
            raise ValueError(f"alpha_w must be finite and above n_vars + 1 = {n + 1}, got {dof}")
        super().__init__(n, expected_parents)
        self.data = data
        self.alpha_mu = float(alpha_mu)
        self.alpha_w = dof
        self.n_rows = data.n_rows
        self.t = self.alpha_mu * (self.alpha_w - n - 1) / (self.alpha_mu + 1)
        values = np.asarray(data.values, dtype=np.float64)
        centred = values - values.mean(axis=0) if data.n_rows else values
        self.scale_matrix: npt.NDArray[np.float64] = self.t * np.eye(n) + centred.T @ centred
        rows = self.n_rows
        # Everything but the two determinants, per number of parents.
        a = self.alpha_w - n
        self._constant = np.array(
            [
                -rows / 2 * math.log(math.pi)
                + 0.5 * math.log(self.alpha_mu / (rows + self.alpha_mu))
                + math.lgamma((rows + a + k + 1) / 2)
                - math.lgamma((a + k + 1) / 2)
                + (a + 2 * k + 1) / 2 * math.log(self.t)
                for k in range(n)
            ]
        )

    def _parents_block(
        self, parents: npt.NDArray[np.int64], cols: npt.NDArray[np.int64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return ``log|R_P|`` and the partial scale of `cols` given `parents`.

        The second and third values are the diagonal of ``R_cc - R_cP R_PP^-1 R_Pc``
        and its last column, the partial scales of every column with the last one.
        """
        R = self.scale_matrix
        block = R[np.ix_(cols, cols)]
        diag, with_last = block.diagonal().copy(), block[:, -1].copy()
        if not parents.size:
            return 0.0, diag, with_last
        cholesky = np.linalg.cholesky(R[np.ix_(parents, parents)])
        log_det = 2.0 * float(np.log(cholesky.diagonal()).sum())
        whitened = np.linalg.solve(cholesky, R[np.ix_(parents, cols)])
        diag -= np.einsum("ij,ij->j", whitened, whitened)
        with_last -= whitened.T @ whitened[:, -1]
        return log_det, diag, with_last

    def _family(
        self,
        k: int,
        log_det_parents: float | npt.NDArray[np.float64],
        schur: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the score of `k` parents from ``log|R_P|`` and ``|R_(P+i)| / |R_P|``."""
        out: npt.NDArray[np.float64] = (
            self._constant[k]
            - 0.5 * log_det_parents
            - (self.n_rows + self.alpha_w - self.n_vars + k + 1) / 2 * np.log(schur)
        )
        return out

    def compute_family(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        """Return the BGe log marginal likelihood of `node` given `parents`."""
        log_det, diag, _ = self._parents_block(parents, np.array([node], dtype=np.int64))
        floor = _MIN_PIVOT * float(self.scale_matrix[node, node])
        return float(self._family(parents.size, log_det, np.maximum(diag, floor))[0])

    def compute_extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Return the BGe of every extension, from partial scales given the parents."""
        # |R_(P+x)| = |R_P| r(x | P) and r(i | P, x) = r(i | P) - r(i, x | P)^2 / r(x | P).
        cols = np.append(extra, node)
        log_det, diag, with_node = self._parents_block(parents, cols)
        R = self.scale_matrix
        pivots = np.maximum(diag[:-1], _MIN_PIVOT * R[extra, extra])
        schur = diag[-1] - with_node[:-1] ** 2 / pivots
        schur = np.maximum(schur, _MIN_PIVOT * float(R[node, node]))
        return self._family(parents.size + 1, log_det + np.log(pivots), schur)
