"""Causal discovery over continuous data from higher-order cumulants.

The data are reduced to moment sums, which add up exactly over blocks of
rows, and from them to unbiased third- and fourth-order cumulants; a tau
discriminator on the cumulants recovers the causal order of a linear
non-Gaussian model. Every solver is a pure function of those statistics, so
``fedcausal`` runs the same ones on sums sent by separate sites.

:class:`RCD` is the estimator, and the published methods are its settings:

- RCD: ``RCD()``, which reads the data once.
- HC-LiNGAM: ``RCD(method="sort")``.
- HC: ``RCD(method="exact")``, one pass over the data per variable.

The functions behind it take the cumulants (:func:`rcd`) or the data
(:func:`rcd_exact`).

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

from causalbase.lingam.coefficients import estimate_b_from_cov, estimate_b_lasso_from_cov
from causalbase.lingam.cumulants import Cumulants, MomentSums, cumulants_from_sums, moment_sums
from causalbase.lingam.estimator import RCD
from causalbase.lingam.solvers import (
    LambdaSelection,
    Pruning,
    RCDResult,
    exact_order,
    rcd,
    rcd_exact,
)
from causalbase.lingam.tau import tau_matrix_3rd, tau_matrix_4th

__all__ = [
    "RCD",
    "Cumulants",
    "LambdaSelection",
    "MomentSums",
    "Pruning",
    "RCDResult",
    "cumulants_from_sums",
    "estimate_b_from_cov",
    "estimate_b_lasso_from_cov",
    "exact_order",
    "moment_sums",
    "rcd",
    "rcd_exact",
    "tau_matrix_3rd",
    "tau_matrix_4th",
]
