"""The tau discriminator that ranks variables by how exogenous they look.

For a linear non-Gaussian model, a variable that causes another leaves an
asymmetry in their joint higher-order cumulants, while the reverse direction
does not. ``tau[i, j]`` measures that asymmetry for the ordered pair ``(i, j)``,
and the row sums rank the variables: the smallest total belongs to the source.
This is the discriminator behind every solver in this subpackage.

References
----------
[1] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery." IEEE TNNLS, 2025.
[2] Torrijos, P., et al. "Federated Causal Discovery via Regression-Directed
    Cumulants." Probabilistic Graphical Models (PGM), 2026.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["tau_matrix_3rd", "tau_matrix_4th"]


def tau_matrix_3rd(
    c3: npt.NDArray[np.float64],
    c12: npt.NDArray[np.float64],
    c21: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the third-order tau matrix.

    Parameters
    ----------
    c3 : numpy.ndarray
        Marginal third cumulants, shape ``(p,)``.
    c12 : numpy.ndarray
        ``Cum(x_i, x_j, x_j)``, shape ``(p, p)``.
    c21 : numpy.ndarray
        ``Cum(x_i, x_i, x_j)``, shape ``(p, p)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(p, p)``, non-negative, zero on the diagonal. Entry
        ``(i, j)`` is ``|C3(x_i) * C12(x_i, x_j) - C21(x_i, x_j) * C12(x_j, x_i)|``.
    """
    tau = np.abs(c3[:, np.newaxis] * c12 - c21 * c12.T)
    np.fill_diagonal(tau, 0.0)
    return tau


def tau_matrix_4th(
    c4: npt.NDArray[np.float64],
    c13: npt.NDArray[np.float64],
    c31: npt.NDArray[np.float64],
    c22: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the fourth-order tau matrix.

    Parameters
    ----------
    c4 : numpy.ndarray
        Marginal fourth cumulants, shape ``(p,)``.
    c13 : numpy.ndarray
        ``Cum(x_i, x_j, x_j, x_j)``, shape ``(p, p)``.
    c31 : numpy.ndarray
        ``Cum(x_i, x_i, x_i, x_j)``, shape ``(p, p)``.
    c22 : numpy.ndarray
        ``Cum(x_i, x_i, x_j, x_j)``, shape ``(p, p)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(p, p)``, non-negative, zero on the diagonal. Entry
        ``(i, j)`` is ``|C4(x_i) * C13(x_i, x_j) - C22(x_i, x_j) * C31(x_i, x_j)|``.

    Notes
    -----
    The fourth-order discriminator is the stronger of the two and is the default
    everywhere in this subpackage. The third-order one matters when the data has
    skew but little excess kurtosis: it is the only one of the two that gets a
    signal from skew, while the fourth-order one is driven entirely by excess
    kurtosis and stays near zero on data that lacks it.
    """
    tau = np.abs(c4[:, np.newaxis] * c13 - c22 * c31)
    np.fill_diagonal(tau, 0.0)
    return tau
