"""Forward sampling from a discrete Bayesian network, vectorised over rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from causalbase.data import DiscreteDataset

if TYPE_CHECKING:
    import numpy.typing as npt

    from causalbase.io.xbif import DiscreteBN

__all__ = ["forward_sample"]


def _parent_row_indices(
    codes: npt.NDArray[np.integer[Any]],
    parents: tuple[int, ...],
    cardinalities: list[int],
) -> npt.NDArray[np.int64]:
    """Map each row's parent codes to a CPT row index.

    Parameters
    ----------
    codes : np.ndarray
        Sampled outcome codes so far, shape ``(n, num_nodes)``, in XBIF
        outcome order (column `p` holds variable `p`'s sampled code).
    parents : tuple[int, ...]
        Parent node indices, in `GIVEN` declaration order (matches
        `DiscreteBN.parent_order[j]`).
    cardinalities : list[int]
        Outcome counts for every node, indexed by node id.

    Returns
    -------
    np.ndarray
        Each sample's CPT row, shape ``(n,)``, with the first parent varying
        slowest, as in `load_xbif`.
    """
    n = codes.shape[0]
    if not parents:
        return np.zeros(n, dtype=np.int64)
    dims = tuple(cardinalities[p] for p in parents)
    multi_index = tuple(codes[:, p] for p in parents)
    return np.ravel_multi_index(multi_index, dims).astype(np.int64)


def forward_sample(bn: DiscreteBN, n: int, seed: int) -> DiscreteDataset:
    """Draw `n` i.i.d. samples from `bn` by ancestral (forward) sampling.

    Parameters
    ----------
    bn : DiscreteBN
        The network to sample from, as returned by
        :func:`causalbase.io.xbif.load_xbif`.
    n : int
        Number of rows to sample.
    seed : int
        Random seed. The same seed always gives the same data.

    Returns
    -------
    DiscreteDataset
        The samples, with variables in `bn`'s order and levels coded in
        sorted order, as `DiscreteDataset.from_csv` does.
    """
    rng = np.random.default_rng(seed)
    num_nodes = len(bn.names)
    cardinalities = [len(outcomes) for outcomes in bn.outcomes]

    # xbif-order codes: codes[:, j] indexes into bn.outcomes[j].
    codes = np.empty((n, num_nodes), dtype=np.int64)

    for j in bn.dag.topological_order():
        cpt = bn.cpts[j]
        row_idx = _parent_row_indices(codes, bn.parent_order[j], cardinalities)
        cum = np.cumsum(cpt, axis=1)
        cum_rows = cum[row_idx]
        # A row may sum to fractionally less than 1; keep the last level reachable.
        cum_rows[:, -1] = 1.0
        u = rng.random(n)
        codes[:, j] = np.sum(u[:, None] >= cum_rows[:, :-1], axis=1)

    # Recode from declaration order to sorted order.
    values = np.empty((n, num_nodes), dtype=np.int32)
    sorted_categories: list[list[str]] = []
    for j in range(num_nodes):
        xbif_outcomes = bn.outcomes[j]
        sorted_outcomes = sorted(xbif_outcomes)
        sorted_categories.append(sorted_outcomes)
        lut = np.array([sorted_outcomes.index(label) for label in xbif_outcomes], dtype=np.int32)
        values[:, j] = lut[codes[:, j]]

    return DiscreteDataset(
        values=values,
        cardinalities=np.array(cardinalities, dtype=np.int32),
        names=bn.names,
        categories=sorted_categories,
    )
