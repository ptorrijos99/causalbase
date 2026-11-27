"""The interface every local score implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy as np
    import numpy.typing as npt


@runtime_checkable
class Score(Protocol):
    """A decomposable score: a DAG scores the sum of its families' local scores.

    This is what :func:`causalbase.search.ges` and
    :func:`causalbase.search.xges` need from a score, and the extension point
    for new ones. Any object with an ``n_vars`` attribute (the number of
    variables, which sets the size of the graph a search learns) and these
    two methods works with them; the other searches also call
    ``local_scores_add(node, parents, candidates)``. The easiest way to write
    a score is to subclass :class:`~causalbase.scores.FamilyScore` and
    implement its ``compute_family``, which adds caching, batching, the
    structure prior and ``local_scores_add``, so the score works with every
    search.

    Higher is better, and both methods must give the same value for the same
    family: the search compares their results directly.
    """

    n_vars: int

    def local_score(self, node: int, parents: Collection[int]) -> float:
        """Return the local score of `node` given `parents`."""
        ...

    def local_scores_with(
        self, node: int, parents: frozenset[int], extra: Sequence[int] | npt.NDArray[np.intp]
    ) -> Sequence[float] | npt.NDArray[np.float64]:
        """Return the local score of `node` given ``parents | {x}``, for each `x` in `extra`.

        The families an Insert compares, all at once; a loop over
        :meth:`local_score` is a correct implementation.
        """
        ...
