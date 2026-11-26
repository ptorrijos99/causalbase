"""BDeu with the structure prior of Tetrad 7.6.5, by plain counting: the reference implementation.

Tetrad's prior is ``k log p + (N - 1 - k) log(1 - p)`` for ``k`` parents with
``p = structure_prior / (N - 1)``, where ``N`` is the number of rows.
The Bernoulli prior of :class:`~causalbase.scores.BDeuScore` adds
``k log(p / (1 - p))`` with ``p = expected_parents / (n_vars - 1)``. Choosing
``expected_parents`` so that both ``p`` agree gives the same differences
between families of a node, so a search learns the same structures; adding the
constant ``(N - 1) log(1 - p)`` gives Tetrad's absolute values as well.

``BDeuScore(prior="tetrad")`` computes the same values with the compiled
engine; this class is the engine's fallback without numba, the source of the
Bernoulli BDeu that GES and XGES search for it (its ``_search_view``), and
the reference the tests compare it against.
"""

from __future__ import annotations

import math
from collections.abc import Collection

from causalbase.data import DiscreteDataset
from causalbase.graph import DAG
from causalbase.scores.bdeu import BDeuScore

__all__ = ["TetradBDeuReference"]


class TetradBDeuReference:
    """BDeu with Tetrad 7.6.5's structure prior, memoised per family.

    Parameters
    ----------
    data : DiscreteDataset
        Data to score; kept as ``data``, with its number of variables as ``n_vars``.
    ess : float
        Equivalent sample size of the BDeu prior.
    structure_prior : float
        Tetrad's structure prior ``e`` (expected parents among ``N - 1``); 0
        leaves the prior out.

    Attributes
    ----------
    bdeu : BDeuScore
        The same objective as a :class:`~causalbase.scores.BDeuScore`, without
        the constant offset, which is what GES and XGES take.
    """

    def __init__(
        self, data: DiscreteDataset, ess: float = 1.0, structure_prior: float = 1.0
    ) -> None:
        self.data = data
        self.n_vars = data.n_vars
        self.ess = float(ess)
        self.structure_prior = float(structure_prior)
        if structure_prior:
            vm = data.n_rows - 1
            p = structure_prior / vm
            expected = p * max(data.n_vars - 1, 1)
            self._offset = vm * math.log(1.0 - p)
        else:
            expected = None
            self._offset = 0.0
        self.bdeu = BDeuScore(data, ess=ess, expected_parents=expected)

    def local_score(self, node: int, parents: Collection[int]) -> float:
        """Local score of ``node`` given ``parents``, as Tetrad 7.6.5 computes it."""
        return self.bdeu.local_score(node, parents) + self._offset

    def score_dag(self, dag: DAG) -> float:
        """Return the score of `dag`: the sum of its families' local scores."""
        return float(sum(self.local_score(v, dag.parents(v)) for v in range(dag.num_nodes)))
