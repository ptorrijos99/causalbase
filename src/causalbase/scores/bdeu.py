"""The discrete scores: BDeu and BIC."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numba import get_num_threads

from causalbase._validation import check_n_jobs
from causalbase.scores._family import FamilyScore
from causalbase.scores._kernels import (
    extension_buffers,
    family_extensions,
    family_score,
    value_bits,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy.typing as npt

    from causalbase.data import DiscreteDataset
    from causalbase.graph import DAG
    from causalbase.scores._compiled import CompiledBDeu

__all__ = ["PRIORS", "BDeuScore", "BICScore", "Prior"]

Prior = Literal["bernoulli", "tetrad"]
"""The structure priors of :class:`BDeuScore`."""

PRIORS: tuple[Prior, ...] = ("bernoulli", "tetrad")


class _DiscreteScore(FamilyScore):
    """Counts families of a discrete dataset with the compiled kernels."""

    def __init__(self, data: DiscreteDataset, expected_parents: float | None) -> None:
        empty = [name for name, r in zip(data.names, data.cardinalities, strict=True) if r < 1]
        if empty:
            raise ValueError(f"variables with no levels cannot be scored: {empty}")
        super().__init__(data.n_vars, expected_parents)
        self.data = data
        self._values = np.asfortranarray(data.values)
        self._cards = data.cardinalities.astype(np.int64)
        self._scratch = np.empty(max(4 * data.n_rows, 64), dtype=np.int64)
        # Work buffers of the batch kernel, one row per thread, made on first use.
        self._buffers: tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]] | None = None
        # Row bitsets per value, for counting small tables by popcount; skipped when large.
        bit_bytes = int(self._cards.sum()) * ((data.n_rows + 63) // 64) * 8
        if bit_bytes <= 256 * 2**20:
            self._bits, self._offsets = value_bits(self._values, self._cards)
        else:
            self._bits, self._offsets = (
                np.zeros((0, 1), np.uint64),
                np.zeros(data.n_vars + 1, np.int64),
            )

    def _count(self, node: int, parents: npt.NDArray[np.int64], ess: float) -> float:
        return float(family_score(self._values, self._cards, node, parents, ess, self._scratch))

    def _count_extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64], ess: float
    ) -> npt.NDArray[np.float64]:
        threads = get_num_threads()
        if self._buffers is None or len(self._buffers[0]) < threads:
            self._buffers = extension_buffers(
                self._values.shape[0], self._cards, len(self._scratch), threads
            )
        ints, floats = self._buffers
        out: npt.NDArray[np.float64] = family_extensions(
            self._values,
            self._cards,
            node,
            parents,
            extra,
            ess,
            len(self._scratch),
            self._bits,
            self._offsets,
            ints[:threads],
            floats[:threads],
        )
        return out


class BDeuScore(_DiscreteScore):
    """BDeu local score over a discrete dataset, with a structure prior.

    The local score of a family is its log marginal likelihood under a BDeu
    prior with equivalent sample size `ess`, plus a structure prior on its
    number of parents ``k``, one of two:

    * ``prior="bernoulli"`` (the default): the prior of
      :class:`~causalbase.scores.FamilyScore`, in which each of the
      ``n_vars - 1`` other variables is a parent with probability
      ``p = min(expected_parents / (n_vars - 1), 1/2)``. Only the part that
      depends on ``k``, ``k * log(p / (1 - p))``, is added. Families are
      cached.
    * ``prior="tetrad"``: the prior of Tetrad 7.6.5's BDeu,
      ``k * log(p) + (m - k) * log(1 - p)`` with ``p = expected_parents / m``
      and ``m = n_rows - 1``: the Bernoulli prior with the number of rows in
      place of the number of variables, constant included, so that the values
      are Tetrad's (and those of MCTS-BN, IPMU 2024). `expected_parents` is
      Tetrad's structure prior. The values come from the compiled engine,
      which does not cache them.

    Every search takes either. GES and XGES search Tetrad's prior through the
    Bernoulli BDeu with the same ``p``, whose differences between the families
    of a node are the same (as long as ``p`` stays below the cap of one half,
    that is, ``expected_parents <= (n_rows - 1) / 2``); the order searches,
    hill climbing, FLOP and the exact searches score either prior with the
    compiled engine.

    Parameters
    ----------
    data : DiscreteDataset
        The data to score against.
    ess : float, default=1.0
        Equivalent sample size of the BDeu prior.
    expected_parents : float or None, default=1.0
        The parameter of the structure prior: the expected number of parents
        per node among the ``n_vars - 1`` candidates for ``"bernoulli"``, among
        ``n_rows - 1`` for ``"tetrad"``. Smaller values penalise edges more.
        ``None`` gives a uniform prior over structures, the plain BDeu score,
        as does 0 for ``"tetrad"``. One expected parent recovers markedly
        better structures than the uniform prior once networks reach a few
        dozen variables, and about the same below that.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior.
    n_jobs : int, default=1
        Threads of the compiled engine for large batches of candidate parents
        (the order searches, hill climbing, FLOP and, with ``"tetrad"``,
        :meth:`local_scores_with`). Results do not depend on it.

    Attributes
    ----------
    n_families : int
        Families the compiled engine has scored so far, single or batched: the
        work of the order searches, hill climbing, FLOP and the exact searches,
        and, with ``"tetrad"``, of :meth:`local_score`,
        :meth:`local_scores_with` and :meth:`score_dag`. GES and XGES are not
        counted.

    Raises
    ------
    ValueError
        If `ess` is not positive and finite, `prior` is unknown,
        `expected_parents` is out of range (positive for ``"bernoulli"``, at
        least 0 and below ``n_rows - 1`` for ``"tetrad"``), or `n_jobs` is not
        a positive integer.

    Notes
    -----
    Not safe to share between threads: the score keeps per-thread work
    buffers for its parallel kernel and a cache of families, and two threads
    calling it at once would overwrite each other's. Use one score per thread,
    or processes.

    Examples
    --------
    >>> from causalbase.datasets import load_asia
    >>> from causalbase.scores import BDeuScore
    >>> from causalbase.search import ges
    >>> score = BDeuScore(load_asia().data, prior="tetrad")
    >>> round(score.score_dag(ges(score).to_dag()), 2)
    -11347.46
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> BDeuScore:
        """Return a new score; ``prior="tetrad"`` makes it the private subclass for that prior.

        That subclass overrides the methods that give values, so the Bernoulli
        prior keeps FamilyScore's without a test per call.
        """
        if cls is BDeuScore and kwargs.get("prior") == "tetrad":
            cls = _TetradPriorBDeuScore
        return super().__new__(cls)

    def __init__(
        self,
        data: DiscreteDataset,
        *,
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        prior: Prior = "bernoulli",
        n_jobs: int = 1,
    ) -> None:
        if not 0 < ess < math.inf:
            raise ValueError(f"ess must be positive and finite, got {ess}")
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}, got {prior!r}")
        check_n_jobs(n_jobs)
        if prior == "tetrad":
            if not isinstance(self, _TetradPriorBDeuScore):
                raise TypeError(f"{type(self).__name__} takes prior='bernoulli' only")
            vm = data.n_rows - 1
            if expected_parents is not None and not 0 <= expected_parents < vm:
                raise ValueError(
                    "expected_parents must be at least 0 and below n_rows - 1 = "
                    f"{vm} for Tetrad's prior, got {expected_parents}"
                )
            super().__init__(data, None)
            self.expected_parents = expected_parents
        else:
            super().__init__(data, expected_parents)
        self.ess = float(ess)
        self.prior: Prior = prior
        self.n_jobs = int(n_jobs)
        self._compiled: CompiledBDeu | None = None

    @property
    def n_families(self) -> int:
        """int: Families the compiled engine has scored so far."""
        return 0 if self._compiled is None else self._compiled.n_families

    def _engine(self) -> CompiledBDeu:
        """Return the compiled engine with this score's prior, built on first use."""
        if self._compiled is None:
            from causalbase.scores._compiled import CompiledBDeu

            self._compiled = CompiledBDeu(
                self.data, self.ess, self.expected_parents, self.n_jobs, prior=self.prior
            )
        return self._compiled

    def compute_family(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        """Return the BDeu log marginal likelihood of the family."""
        return self._count(node, parents, self.ess)

    def compute_extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Return the BDeu of every extension, counted together."""
        return self._count_extensions(node, parents, extra, self.ess)


class _TetradPriorBDeuScore(BDeuScore):
    """``BDeuScore(prior="tetrad")``: Tetrad's absolute values, from the compiled engine."""

    def local_score(self, node: int, parents: Collection[int]) -> float:
        """Return the local score of `node` given `parents`, as Tetrad 7.6.5 computes it."""
        return self._engine().local_score(node, parents)

    def local_scores_with(
        self, node: int, parents: frozenset[int], extra: Sequence[int] | npt.NDArray[np.intp]
    ) -> npt.NDArray[np.float64]:
        """Return the local score of `node` given ``parents | {x}``, for each `x` in `extra`.

        Equal, bit for bit, to calling :meth:`local_score` once per `x`.
        """
        out: npt.NDArray[np.float64] = self._engine().local_scores_add(node, sorted(parents), extra)
        return out

    def score_dag(self, dag: DAG) -> float:
        """Return the score of `dag`: the sum of its families' local scores."""
        return self._engine().score_dag(dag)

    def _search_view(self) -> BDeuScore:
        """Return the Bernoulli BDeu with Tetrad's ``p``, which GES and XGES search.

        Its families differ from this score's by a constant per node, so the
        searches learn this score's structures. Each call builds a new score
        with an empty cache, so that no search inherits another one's cached
        families (and their memory); keep the result to share them.
        """
        from causalbase.scores._tetrad import TetradBDeuReference

        return TetradBDeuReference(self.data, self.ess, self.expected_parents or 0.0).bdeu


class BICScore(_DiscreteScore):
    """BIC local score over a discrete dataset, memoised per family.

    The maximised log likelihood of a family less ``penalty_discount * log(n) / 2``
    per free parameter, ``q * (r - 1)`` for a child of ``r`` levels whose
    parents take ``q`` joint configurations, plus the structure prior of
    :class:`~causalbase.scores.FamilyScore`.

    Parameters
    ----------
    data : DiscreteDataset
        The data to score against.
    penalty_discount : float, default=1.0
        Multiplies the penalty; larger values give sparser structures.
    expected_parents : float or None, default=1.0
        Expected number of parents per node under the structure prior, as in
        :class:`BDeuScore`. ``None`` is the plain BIC.

    Notes
    -----
    Not safe to share between threads, for the reasons given in
    :class:`BDeuScore`.
    """

    def __init__(
        self,
        data: DiscreteDataset,
        *,
        penalty_discount: float = 1.0,
        expected_parents: float | None = 1.0,
    ) -> None:
        if not 0 < penalty_discount < math.inf:
            raise ValueError(
                f"penalty_discount must be positive and finite, got {penalty_discount}"
            )
        super().__init__(data, expected_parents)
        self.penalty_discount = float(penalty_discount)
        self._per_parameter = penalty_discount * math.log(max(data.n_rows, 1)) / 2

    def compute_family(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        """Return the maximised log likelihood less the BIC penalty."""
        q = float(np.prod(self._cards[parents].astype(np.float64)))
        free = q * (self._cards[node] - 1)
        return self._count(node, parents, 0.0) - self._per_parameter * float(free)

    def compute_extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Return the BIC of every extension, counted together."""
        q = float(np.prod(self._cards[parents].astype(np.float64)))
        free = q * self._cards[extra].astype(np.float64) * float(self._cards[node] - 1)
        out: npt.NDArray[np.float64] = (
            self._count_extensions(node, parents, extra, 0.0) - self._per_parameter * free
        )
        return out
