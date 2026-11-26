"""Scoring functions for candidate causal structures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from causalbase.scores._family import FamilyScore
from causalbase.scores.base import Score
from causalbase.scores.bdeu import PRIORS, BDeuScore, BICScore, Prior
from causalbase.scores.gaussian import GaussianBICScore
from causalbase.scores.likelihood import log_likelihood

if TYPE_CHECKING:
    from causalbase.data import ContinuousDataset, DiscreteDataset

__all__ = [
    "PRIORS",
    "SCORE_NAMES",
    "BDeuScore",
    "BICScore",
    "FamilyScore",
    "GaussianBICScore",
    "Prior",
    "Score",
    "ScoreName",
    "log_likelihood",
    "make_score",
]

ScoreName = Literal["bdeu", "bic", "gaussian_bic"]
"""The scores the estimators take by name: two discrete, one continuous."""

SCORE_NAMES: tuple[ScoreName, ...] = ("bdeu", "bic", "gaussian_bic")


def make_score(
    name: ScoreName,
    data: DiscreteDataset | ContinuousDataset,
    *,
    ess: float = 1.0,
    expected_parents: float | None = 1.0,
    prior: Prior = "bernoulli",
    penalty_discount: float = 1.0,
) -> FamilyScore:
    """Return the score called `name` over `data`.

    Parameters
    ----------
    name : {"bdeu", "bic", "gaussian_bic"}
        :class:`BDeuScore` and :class:`BICScore` take a ``DiscreteDataset``,
        :class:`GaussianBICScore` a ``ContinuousDataset``.
    data : DiscreteDataset or ContinuousDataset
        The data to score against.
    ess : float, default=1.0
        Equivalent sample size, for BDeu only.
    expected_parents : float or None, default=1.0
        Structure prior, for every score; see :class:`FamilyScore`.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        Which structure prior; ``"tetrad"`` (Tetrad 7.6.5's) for BDeu only,
        see :class:`BDeuScore`.
    penalty_discount : float, default=1.0
        Multiplier of the BIC penalty, for the BIC scores only.

    Raises
    ------
    ValueError
        If `name` is not one of the three, or `prior` is unknown or not one of
        the score's.
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
    wanted = ContinuousDataset if name == "gaussian_bic" else DiscreteDataset
    if not isinstance(data, wanted):
        raise TypeError(f"the {name!r} score takes a {wanted.__name__}, got {type(data).__name__}")
    if isinstance(data, ContinuousDataset):
        return GaussianBICScore(
            data, penalty_discount=penalty_discount, expected_parents=expected_parents
        )
    if name == "bdeu":
        return BDeuScore(data, ess=ess, expected_parents=expected_parents, prior=prior)
    return BICScore(data, penalty_discount=penalty_discount, expected_parents=expected_parents)
