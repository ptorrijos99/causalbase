"""Datasets and their encoding, for one table or for several that share one encoding."""

from __future__ import annotations

from causalbase.data.continuous import ContinuousDataset, as_continuous, as_continuous_sites
from causalbase.data.dataset import DiscreteDataset, as_discrete, as_discrete_sites

__all__ = [
    "ContinuousDataset",
    "DiscreteDataset",
    "as_continuous",
    "as_continuous_sites",
    "as_discrete",
    "as_discrete_sites",
]
