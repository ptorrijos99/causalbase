"""What every built-in score shares: the structure prior and the cache of families."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy.typing as npt

    from causalbase.graph import DAG

__all__ = ["FamilyScore"]


class FamilyScore:
    """A decomposable score whose families are computed once and cached.

    The base for a new score: a subclass implements :meth:`compute_family`,
    the score of one family, and may override :meth:`compute_extensions`, the
    families an Insert compares, with a faster batch. :meth:`local_score` and
    :meth:`local_scores_with` (also callable as :meth:`local_scores_add`),
    which the searches call, cache both and add a structure prior on the
    number of parents ``k``::

        k * log(p) + (m - k) * log(1 - p),    p = min(expected_parents / m, 1/2)

    a Bernoulli prior in which each of the ``m = n_vars - 1`` candidate parents
    is present with probability ``p``, so `expected_parents` is the expected
    in-degree under the prior. ``p`` is capped at one half, where the prior
    stops penalising edges, so on very few variables it has no effect. Only
    the ``k * log(p / (1 - p))`` part is added; the rest is the same for every
    family of a node.

    Parameters
    ----------
    n_vars : int
        Number of variables.
    expected_parents : float or None
        Expected in-degree under the structure prior; ``None`` is a uniform
        prior over structures.

    Examples
    --------
    A score that rewards each parent by a fixed amount, which GES then adds
    until the prior stops it:

    >>> import numpy as np
    >>> from causalbase.scores import FamilyScore
    >>> from causalbase.search import ges
    >>> class PerParent(FamilyScore):
    ...     def compute_family(self, node, parents):
    ...         return 0.5 * len(parents)
    >>> ges(PerParent(3, expected_parents=None))
    PDAG(num_nodes=3, directed=0, undirected=3)

    Notes
    -----
    A score caches every family it computes and keeps work buffers between
    calls, so it is not safe to use one score object from several threads at
    once: give each thread its own. Several processes are fine, each with its own copy.
    """

    def __init__(self, n_vars: int, expected_parents: float | None) -> None:
        if expected_parents is not None and not expected_parents > 0:
            raise ValueError(f"expected_parents must be positive, got {expected_parents}")
        self.n_vars = n_vars
        self.expected_parents = expected_parents
        if expected_parents is None:
            self._log_odds = 0.0
        else:
            p = min(expected_parents / max(n_vars - 1, 1), 0.5)
            self._log_odds = math.log(p) - math.log1p(-p)
        # A family is cached under one integer: a bit per parent, shifted past
        # the bits of the child's index. Far smaller than a frozenset key, which
        # is what bounds memory on dense searches.
        self._node_bits = max(1, n_vars.bit_length())
        self._cache: dict[int, float] = {}
        # What an Insert compares: every extension of one family by one more
        # parent, an array per family indexed by the new parent.
        self._rows: dict[int, npt.NDArray[np.float64]] = {}

    def compute_family(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        """Return the score of `node` given the sorted `parents`, without the prior.

        Called once per family; the result is cached. Subclasses must implement it.
        """
        raise NotImplementedError

    def compute_extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Return :meth:`compute_family` of `node` given ``parents + [x]``, for each `x` in `extra`.

        One call per family by default; override it to score the batch together.
        """
        return np.array([self.compute_family(node, np.sort(np.append(parents, x))) for x in extra])

    def _key(self, node: int, mask: int) -> int:
        return (mask << self._node_bits) | node

    def local_score(self, node: int, parents: Collection[int]) -> float:
        """Return the local score of `node` given `parents`.

        Parameters
        ----------
        node : int
            Index of the child.
        parents : Collection[int]
            Indices of the parents, in any order.

        Returns
        -------
        float
        """
        mask = 0
        for p in parents:
            mask |= 1 << p
        key = self._key(node, mask)
        value = self._cache.get(key)
        if value is None:
            ordered = np.fromiter(sorted(parents), dtype=np.int64, count=len(parents))
            value = float(self.compute_family(node, ordered)) + len(parents) * self._log_odds
            self._cache[key] = value
        return value

    def local_scores_with(
        self, node: int, parents: frozenset[int], extra: Sequence[int] | npt.NDArray[np.intp]
    ) -> npt.NDArray[np.float64]:
        """Return the local score of `node` given ``parents | {x}``, for each `x` in `extra`.

        Equivalent to calling :meth:`local_score` once per `x`, but the
        families are cached together, one array per `node` and `parents`, and
        the ones not yet cached are computed together.
        """
        mask = 0
        for p in parents:
            mask |= 1 << p
        key = self._key(node, mask)
        row = self._rows.get(key)
        if row is None:
            row = np.full(self.n_vars, np.nan)
            self._rows[key] = row
        extra_array = np.asarray(extra, dtype=np.int64)
        scores = row[extra_array]
        missing = np.isnan(scores)
        if missing.any():
            ordered = np.fromiter(sorted(parents), dtype=np.int64, count=len(parents))
            todo = extra_array[missing]
            row[todo] = (
                self.compute_extensions(node, ordered, todo) + (len(parents) + 1) * self._log_odds
            )
            scores = row[extra_array]
        return scores

    def local_scores_add(
        self, node: int, parents: Collection[int], candidates: Sequence[int] | npt.NDArray[np.intp]
    ) -> npt.NDArray[np.float64]:
        """Return the local score of `node` given `parents` plus each one of `candidates`.

        :meth:`local_scores_with` under the name the order searches, hill
        climbing and FLOP call, so that every score works with them.
        """
        return self.local_scores_with(node, frozenset(parents), candidates)

    def score_dag(self, dag: DAG) -> float:
        """Return the score of `dag`: the sum of its families' local scores."""
        return sum(self.local_score(v, dag.parents(v)) for v in range(dag.num_nodes))

    def _extensions_uncached(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Return what :meth:`local_scores_with` returns, without caching anything.

        The exact search scores every parent set of every node once, so
        caching them would only cost memory.
        """
        out: npt.NDArray[np.float64] = (
            self.compute_extensions(node, parents, extra) + (parents.size + 1) * self._log_odds
        )
        return out

    def _prior_bound(self, k: int) -> float:
        """Return the largest prior term of a family with at least `k` parents."""
        if self._log_odds <= 0:
            return k * self._log_odds
        return max(k, self.n_vars - 1) * self._log_odds

    def _superset_bound(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        """Return an upper bound on the local score of every strict superset of `parents`.

        The exact search skips the supersets of a set whose bound is below the
        score of one of its subsets. A score with no such bound returns
        infinity, and then every parent set is scored.
        """
        return math.inf

    def _search_view(self) -> FamilyScore:
        """Return the score that GES and XGES search in place of this one: itself.

        A score whose values are better searched through another one with the
        same differences between the families of a node, such as
        ``BDeuScore(prior="tetrad")``, overrides it.
        """
        return self
