"""Greedy Equivalence Search (Chickering, 2002)."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from causalbase._base import Estimator, named_adjacency
from causalbase.data import ContinuousDataset, DiscreteDataset, as_continuous, as_discrete
from causalbase.graph import DAG, PDAG
from causalbase.graph.cpdag import dag_to_cpdag
from causalbase.scores import SCORE_NAMES, log_likelihood, make_score
from causalbase.search._engine import backward_phase, forward_phase, marginal_sources
from causalbase.search._knowledge import Knowledge, PairLike

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    import pandas as pd
    from typing_extensions import Self

    from causalbase.data.dataset import DiscreteLike
    from causalbase.scores import Prior, Score, ScoreName

__all__ = ["GES", "ges"]


def _check_limit(name: str, value: object) -> None:
    """Raise unless `value` is None or a non-negative integer."""
    if value is None:
        return
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None, got {value!r}")


def _check_score(score: Score) -> tuple[Score, int]:
    """Return the score to search in place of `score`, and its number of variables.

    That is the score's ``_search_view()`` when it has one (see
    :class:`~causalbase.scores.FamilyScore`), the score itself otherwise.
    """
    view = getattr(score, "_search_view", None)
    searched: Score = view() if view is not None else score
    return searched, int(searched.n_vars)


def _start(n_vars: int, initial_graph: DAG | PDAG | None, knowledge: Knowledge | None) -> PDAG:
    """Return the CPDAG a search starts from, with `knowledge` imposed."""
    if initial_graph is not None and initial_graph.num_nodes != n_vars:
        raise ValueError(f"initial_graph has {initial_graph.num_nodes} nodes, expected {n_vars}")
    if initial_graph is None:
        start = DAG(n_vars)
    elif isinstance(initial_graph, DAG):
        start = initial_graph
    else:
        start = initial_graph.to_dag()
    return dag_to_cpdag(knowledge.impose(start) if knowledge else start)


def ges(
    score: Score,
    initial_graph: DAG | PDAG | None = None,
    *,
    candidates: frozenset[tuple[int, int]] | None = None,
    max_t_size: int | None = None,
    knowledge: Knowledge | None = None,
    max_parents: int | None = None,
    forward: bool = True,
    backward: bool = True,
    one_edge_faithfulness: bool = False,
) -> PDAG:
    """Run GES with `score` and return the learned CPDAG.

    Parameters
    ----------
    score : Score
        Decomposable score; the graph has ``score.n_vars`` nodes. For
        ``BDeuScore(prior="tetrad")`` the search runs on the Bernoulli BDeu
        with the same differences between families (see
        :class:`~causalbase.scores.BDeuScore`), so it learns Tetrad's
        structures.
    initial_graph : DAG or PDAG, optional
        Where the search starts: the equivalence class of a DAG, or of a
        consistent extension of a PDAG. The empty graph by default.
    candidates : frozenset of (int, int), optional
        Restrict both phases to these ordered pairs ``(x, y)``.
    max_t_size : int or None, default=None
        Largest conditioning subset tried by an Insert. ``None`` tries every
        one that can be valid, as Chickering defines the operator; only the
        subsets that keep a clique are visited, so this costs little more than
        a cap unless the neighbours of a node form a large clique.
    knowledge : Knowledge, optional
        Adjacencies the result must have or must not have. The search starts
        from `initial_graph` with them imposed.
    max_parents : int or None, optional
        No Insert gives its child more parents than this. ``None`` is no limit.
    forward, backward : bool, default=True
        Which phases to run.
    one_edge_faithfulness : bool, default=False
        Insert an edge only between variables that the score finds dependent
        on their own, as fGES does (Ramsey et al., 2017). Much faster on many
        variables; it misses an edge whose paths cancel exactly.

    Returns
    -------
    PDAG
        The CPDAG of the learned equivalence class.

    Raises
    ------
    ValueError
        If `initial_graph` has another number of nodes, or a limit is out of
        range.

    Examples
    --------
    >>> from causalbase.datasets import load_asia
    >>> from causalbase.scores import BDeuScore
    >>> from causalbase.search import ges
    >>> ges(BDeuScore(load_asia().data))
    PDAG(num_nodes=8, directed=..., undirected=...)
    """
    score, n_vars = _check_score(score)
    _check_limit("max_t_size", max_t_size)
    _check_limit("max_parents", max_parents)
    cpdag = _start(n_vars, initial_graph, knowledge)
    if forward:
        screen = marginal_sources(score, n_vars) if one_edge_faithfulness else None
        cpdag = forward_phase(
            score, cpdag, candidates, max_t_size, knowledge, max_parents, sources=screen
        )
    if backward:
        cpdag = backward_phase(score, cpdag, candidates, knowledge)
    return cpdag


class GES(Estimator):
    """Learn a causal structure with Greedy Equivalence Search.

    The search adds edges while the score improves, then removes them while it
    improves, moving between Markov equivalence classes.

    Parameters
    ----------
    criterion : {"bdeu", "bic", "gaussian_bic"}, default="bdeu"
        The scoring criterion: BDeu or BIC over categorical data, or the BIC of
        a linear Gaussian model over continuous data.
    ess : float, default=1.0
        Equivalent sample size of the BDeu score. Also the prior of the
        parameters behind :meth:`score` on categorical data.
    expected_parents : float or None, default=1.0
        Expected in-degree under the structure prior, for every score; see
        :class:`~causalbase.scores.FamilyScore`. ``None`` is a uniform prior.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior of BDeu: ``"tetrad"`` is Tetrad 7.6.5's, with
        `expected_parents` as Tetrad's structure prior (see
        :class:`~causalbase.scores.BDeuScore`). Only ``"bernoulli"`` for the
        BIC scores.
    penalty_discount : float, default=1.0
        Multiplier of the BIC penalty; larger values give sparser structures.
    max_parents : int or None, default=None
        Most parents an Insert may give a node. ``None`` is no limit.
    max_t_size : int or None, default=None
        Largest conditioning subset tried by an Insert; ``None`` tries all.
    required, forbidden : sequence of pairs, optional
        Pairs of variables, by name or column index, that must be adjacent or
        must not be. The direction of a required edge is still learned.
    one_edge_faithfulness : bool, default=False
        Insert an edge only between variables that the score finds dependent
        on their own, as fGES does (Ramsey et al., 2017): faster on many
        variables, at the cost of edges whose paths cancel exactly.

    Attributes
    ----------
    cpdag_ : PDAG
        The learned equivalence class.
    dag_ : DAG
        One DAG of that class.
    score_ : float
        The score of `dag_`; with ``prior="tetrad"``, Tetrad's value.
    adjacency_ : pandas.DataFrame
        The CPDAG with variable names: ``A[i, j] = 1`` and ``A[j, i] = 0`` is
        ``i -> j``, and both set is an undirected edge.
    n_features_in_ : int
        Number of variables.
    feature_names_in_ : ndarray of str
        Variable names.

    Examples
    --------
    >>> from causalbase import GES
    >>> from causalbase.datasets import load_asia
    >>> model = GES().fit(load_asia().data)
    >>> model.cpdag_
    PDAG(num_nodes=8, directed=..., undirected=...)
    """

    def __init__(
        self,
        criterion: ScoreName = "bdeu",
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        prior: Prior = "bernoulli",
        penalty_discount: float = 1.0,
        max_parents: int | None = None,
        max_t_size: int | None = None,
        required: Sequence[PairLike] | None = None,
        forbidden: Sequence[PairLike] | None = None,
        one_edge_faithfulness: bool = False,
    ) -> None:
        self.criterion = criterion
        self.ess = ess
        self.expected_parents = expected_parents
        self.prior = prior
        self.penalty_discount = penalty_discount
        self.max_parents = max_parents
        self.max_t_size = max_t_size
        self.required = required
        self.forbidden = forbidden
        self.one_edge_faithfulness = one_edge_faithfulness

    def _encode(
        self, X: DiscreteLike | ContinuousDataset | npt.ArrayLike
    ) -> DiscreteDataset | ContinuousDataset:
        if self.criterion not in SCORE_NAMES:
            raise ValueError(f"criterion must be one of {SCORE_NAMES}, got {self.criterion!r}")
        if self.criterion == "gaussian_bic":
            return as_continuous(X)  # type: ignore[arg-type]
        return as_discrete(X)  # type: ignore[arg-type]

    def _search(self, score: Score, initial_graph: DAG | None, knowledge: Knowledge) -> PDAG:
        return ges(
            score,
            initial_graph,
            max_t_size=self.max_t_size,
            knowledge=knowledge,
            max_parents=self.max_parents,
            one_edge_faithfulness=self.one_edge_faithfulness,
        )

    def fit(
        self,
        X: DiscreteLike | ContinuousDataset | npt.ArrayLike,
        y: None = None,
        initial_graph: DAG | None = None,
    ) -> Self:
        """Learn the structure of `X`.

        Parameters
        ----------
        X : DiscreteDataset, ContinuousDataset, pandas.DataFrame or array-like
            Of shape (n_rows, n_vars). For the discrete scores every distinct
            value of a column is a level; for ``"gaussian_bic"`` the columns
            are real-valued.
        y : None
            Ignored.
        initial_graph : DAG, optional
            Where the search starts. The empty graph by default.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `X` has no rows, or a hyperparameter is out of range (including
            ``prior="tetrad"`` with a BIC criterion).
        """
        data = self._encode(X)
        if data.n_rows == 0:
            raise ValueError("X has no rows")
        local = make_score(
            self.criterion,
            data,
            ess=self.ess,
            expected_parents=self.expected_parents,
            prior=self.prior,
            penalty_discount=self.penalty_discount,
        )
        knowledge = Knowledge.from_pairs(self.required or (), self.forbidden or (), data.names)
        self.cpdag_ = self._search(local, initial_graph, knowledge)
        self.dag_ = self.cpdag_.to_dag()
        self.score_ = local.score_dag(self.dag_)
        self.n_features_in_ = data.n_vars
        self.feature_names_in_ = np.asarray(data.names, dtype=object)
        self._data = data
        return self

    def score(self, X: DiscreteLike | ContinuousDataset | npt.ArrayLike, y: None = None) -> float:
        """Return the mean log likelihood per row of `X` under the learned structure.

        The parameters are learned from the training data: by a BDeu prior of
        equivalent sample size `ess` for categorical data, by maximum
        likelihood for continuous data. Higher is better, so a scikit-learn
        parameter search can select the hyperparameters on held-out rows.

        Parameters
        ----------
        X : DiscreteDataset, ContinuousDataset, pandas.DataFrame or array-like
            Rows over the training variables, with no level the training data
            lacks.
        y : None
            Ignored.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If `X` has other variables, or a level the training data lacks.
        """
        self._check_fitted("dag_")
        if isinstance(self._data, DiscreteDataset):
            test: DiscreteDataset | ContinuousDataset = as_discrete(
                X,  # type: ignore[arg-type]
                self._data.categories,
            )
        else:
            test = as_continuous(X)  # type: ignore[arg-type]
        return log_likelihood(self.dag_, self._data, test, ess=self.ess) / max(test.n_rows, 1)

    @property
    def adjacency_(self) -> pd.DataFrame:
        """pandas.DataFrame: The learned CPDAG as a named adjacency matrix."""
        self._check_fitted("cpdag_")
        return named_adjacency(self.cpdag_, self.feature_names_in_)
