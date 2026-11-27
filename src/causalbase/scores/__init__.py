"""Scoring functions for candidate causal structures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from causalbase.scores._family import FamilyScore
from causalbase.scores.base import Score
from causalbase.scores.bdeu import PRIORS, BDeuScore, BICScore, Prior
from causalbase.scores.bge import BGeScore
from causalbase.scores.gaussian import GaussianBICScore
from causalbase.scores.likelihood import log_likelihood

if TYPE_CHECKING:
    from causalbase.data import ContinuousDataset, DiscreteDataset

__all__ = [
    "PRIORS",
    "CONTINUOUS_SCORES",
    "SCORE_NAMES",
    "BDeuScore",
    "BGeScore",
    "BICScore",
    "FamilyScore",
    "GaussianBICScore",
    "Prior",
    "Score",
    "ScoreName",
    "log_likelihood",
    "make_score",
]

ScoreName = Literal["bdeu", "bic", "gaussian_bic", "bge"]
"""The scores the estimators take by name: two discrete, two continuous."""

SCORE_NAMES: tuple[ScoreName, ...] = ("bdeu", "bic", "gaussian_bic", "bge")

CONTINUOUS_SCORES: tuple[ScoreName, ...] = ("gaussian_bic", "bge")
"""The scores among :data:`SCORE_NAMES` that take a ``ContinuousDataset``."""


def make_score(
    name: ScoreName,
    data: DiscreteDataset | ContinuousDataset,
    *,
    ess: float = 1.0,
    expected_parents: float | None = 1.0,
    prior: Prior = "bernoulli",
    penalty_discount: float = 1.0,
    alpha_w: float | None = None,
    n_jobs: int = 1,
) -> FamilyScore:
    """Return the score called `name` over `data`.

    Every estimator builds its score here, so each parameter means the same in
    all of them; one a score does not use is ignored.

    Parameters
    ----------
    name : {"bdeu", "bic", "gaussian_bic", "bge"}
        :class:`BDeuScore` and :class:`BICScore` take a ``DiscreteDataset``,
        :class:`GaussianBICScore` and :class:`BGeScore` a ``ContinuousDataset``.
    data : DiscreteDataset or ContinuousDataset
        The data to score against.
    ess : float, default=1.0
        Equivalent sample size: of the BDeu prior, and of BGe's prior on the
        mean (its `alpha_mu`).
    expected_parents : float or None, default=1.0
        Structure prior, for every score; see :class:`FamilyScore`.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        Which structure prior; ``"tetrad"`` (Tetrad 7.6.5's) for BDeu only,
        see :class:`BDeuScore`.
    penalty_discount : float, default=1.0
        Multiplier of the penalty of the two BIC scores.
    alpha_w : float or None, default=None
        Degrees of freedom of BGe's Wishart prior; see :class:`BGeScore`.
    n_jobs : int, default=1
        Threads of the score's compiled kernels; -1 is every thread numba has.

    Raises
    ------
    ValueError
        If `name` is not one of the four, `prior` is unknown or not one of
        the score's, or a parameter is out of the score's range.
    TypeError
        If `data` is not the kind of dataset the score takes.
    """
    from causalbase.data import ContinuousDataset, DiscreteDataset

    if name not in SCORE_NAMES:
        raise ValueError(f"score must be one of {SCORE_NAMES}, got {name!r}")
    if prior not in PRIORS:
        raise ValueError(f"prior must be one of {PRIORS}, got {prior!r}")
    if prior != "bernoulli" and name != "bdeu":
        raise ValueError(f"prior={prior!r} is a prior of the 'bdeu' score only, not {name!r}")
    wanted = ContinuousDataset if name in CONTINUOUS_SCORES else DiscreteDataset
    if not isinstance(data, wanted):
        raise TypeError(f"the {name!r} score takes a {wanted.__name__}, got {type(data).__name__}")
    if isinstance(data, ContinuousDataset):
        if name == "bge":
            return BGeScore(
                data,
                alpha_mu=ess,
                alpha_w=alpha_w,
                expected_parents=expected_parents,
                n_jobs=n_jobs,
            )
        return GaussianBICScore(
            data,
            penalty_discount=penalty_discount,
            expected_parents=expected_parents,
            n_jobs=n_jobs,
        )
    if name == "bdeu":
        return BDeuScore(
            data, ess=ess, expected_parents=expected_parents, prior=prior, n_jobs=n_jobs
        )
    return BICScore(
        data, penalty_discount=penalty_discount, expected_parents=expected_parents, n_jobs=n_jobs
    )
