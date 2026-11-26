"""How well a structure, with parameters learned from one dataset, explains another."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from causalbase.data import ContinuousDataset, DiscreteDataset
from causalbase.scores._kernels import family_score
from causalbase.scores.gaussian import _covariance

if TYPE_CHECKING:
    from causalbase.graph import DAG

__all__ = ["log_likelihood"]


def log_likelihood(
    dag: DAG,
    train: DiscreteDataset | ContinuousDataset,
    test: DiscreteDataset | ContinuousDataset,
    *,
    ess: float = 1.0,
) -> float:
    """Return the log likelihood of `test`'s rows under `dag`, its parameters learned from `train`.

    Discrete data gets the Bayesian predictive likelihood under a BDeu prior
    of equivalent sample size `ess`: the marginal likelihood of `train` and
    `test` together divided by that of `train`, which is the same for every
    DAG of an equivalence class. Continuous data gets the likelihood of a
    linear Gaussian model fitted to `train` by maximum likelihood.

    Parameters
    ----------
    dag : DAG
        The structure.
    train, test : DiscreteDataset or ContinuousDataset
        Over the same variables, both discrete with the same levels or both
        continuous.
    ess : float, default=1.0
        Equivalent sample size of the BDeu prior, for discrete data.

    Returns
    -------
    float
        The total over `test`'s rows; divide by ``test.n_rows`` for a mean.

    Raises
    ------
    ValueError
        If the datasets do not share their variables, or their levels.
    """
    if train.names != test.names:
        raise ValueError(f"test has variables {test.names}, expected {train.names}")
    if isinstance(train, DiscreteDataset) and isinstance(test, DiscreteDataset):
        if train.categories != test.categories:
            raise ValueError("train and test must encode every variable with the same levels")
        return _discrete(dag, train, test, ess)
    if isinstance(train, ContinuousDataset) and isinstance(test, ContinuousDataset):
        return _gaussian(dag, train, test)
    raise ValueError("train and test must both be discrete or both continuous")


def _discrete(dag: DAG, train: DiscreteDataset, test: DiscreteDataset, ess: float) -> float:
    cards = train.cardinalities.astype(np.int64)
    both = np.asfortranarray(np.vstack([train.values, test.values]))
    alone = np.asfortranarray(train.values)
    scratch = np.empty(max(4 * both.shape[0], 64), dtype=np.int64)
    total = 0.0
    for v in range(dag.num_nodes):
        parents = np.array(sorted(dag.parents(v)), dtype=np.int64)
        total += family_score(both, cards, v, parents, ess, scratch)
        total -= family_score(alone, cards, v, parents, ess, scratch)
    return float(total)


def _gaussian(dag: DAG, train: ContinuousDataset, test: ContinuousDataset) -> float:
    if train.n_rows == 0:
        raise ValueError("the training data has no rows")
    mean = train.values.mean(axis=0)
    cov = _covariance(train.values)
    centred = test.values - mean
    total = 0.0
    for v in range(dag.num_nodes):
        parents = np.array(sorted(dag.parents(v)), dtype=np.int64)
        residual = centred[:, v]
        variance = float(cov[v, v])
        if parents.size:
            beta = np.linalg.lstsq(cov[np.ix_(parents, parents)], cov[parents, v], rcond=None)[0]
            residual = residual - centred[:, parents] @ beta
            variance -= float(cov[parents, v] @ beta)
        variance = max(variance, 1e-12 * max(float(cov[v, v]), 1e-12))
        total -= test.n_rows / 2 * math.log(2 * math.pi * variance)
        total -= float(residual @ residual) / (2 * variance)
    return total
