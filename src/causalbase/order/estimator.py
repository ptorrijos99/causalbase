"""The estimators over variable orders: OrderSearch and FLOP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from causalbase._score_estimator import Init, ScoreEstimator, initial_order
from causalbase._validation import check_count, check_time_limit
from causalbase.order.flop import flop
from causalbase.order.polished import Polish, order_search

if TYPE_CHECKING:
    from collections.abc import Sequence

    from causalbase.graph import DAG
    from causalbase.scores import FamilyScore, Prior, ScoreName
    from causalbase.search import SearchResult

__all__ = ["FLOP", "OrderSearch"]


def _check_order_search(init: object, max_iter: object, time_limit: object) -> None:
    if init not in ("ges", "xges"):
        raise ValueError(f"init must be 'ges' or 'xges', got {init!r}")
    check_count("max_iter", max_iter, allow_none=True)
    check_time_limit(time_limit)
    if time_limit is None and max_iter is None:
        raise ValueError("set time_limit, max_iter or both, or the search never ends")


def _first_order(score: FamilyScore, initial_graph: DAG | None, init: Init) -> list[int]:
    """Return a topological order of `initial_graph`, or of the network `init` learns."""
    if initial_graph is not None:
        return [int(v) for v in initial_graph.topological_order()]
    return initial_order(score, init)


class OrderSearch(ScoreEstimator):
    """Learn a Bayesian network by iterated local search over variable orders.

    The search is :func:`~causalbase.order.order_search` on the score
    `criterion` names: it starts from a topological order of the network GES
    (or XGES) learns, or of `initial_graph` in :meth:`fit`, climbs over orders
    by insertion moves, scores each order by the best network consistent with
    it (a greedy climb per node, restarted `n_restarts` times), and ends each
    local search with GES or XGES started from that network, so that arcs the
    order forbids can still be added.

    Parameters
    ----------
    criterion : {"bdeu", "bic", "gaussian_bic", "bge"}, default="bdeu"
        The score: BDeu or BIC over categorical data, or the BIC or BGe of a
        linear Gaussian model over continuous data.
    ess : float, default=1.0
        Equivalent sample size: of the BDeu prior, or of BGe's prior on the
        mean. Also the prior of the parameters behind :meth:`score` on
        categorical data.
    expected_parents : float or None, default=1.0
        Expected in-degree under the structure prior, for every criterion; see
        :class:`~causalbase.scores.FamilyScore`. ``None`` is a uniform prior.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior: the Bernoulli prior of
        :class:`~causalbase.scores.FamilyScore`, or, for BDeu only, Tetrad
        7.6.5's, with `expected_parents` as Tetrad's structure prior (see
        :class:`~causalbase.scores.BDeuScore`).
    penalty_discount : float, default=1.0
        Multiplier of the penalty of the two BIC criteria; larger values give
        sparser structures.
    alpha_w : float or None, default=None
        Degrees of freedom of BGe's Wishart prior, above ``n_vars + 1``;
        ``None`` is ``n_vars + 2``. See :class:`~causalbase.scores.BGeScore`.
    n_restarts : int, default=2
        Extra greedy climbs per node, each without the first parent the
        previous climb chose (see :class:`~causalbase.order.OrderScorer`).
    polish : {"ges", "xges"}, default="ges"
        Search over equivalence classes that ends each local search.
    n_pair_candidates : int, default=0
        When positive, each local search also tries adding pairs of parents
        drawn from this many best single candidates
        (:func:`~causalbase.order.pair_polish`).
    init : {"ges", "xges"}, default="ges"
        Algorithm whose network gives the first order, without `initial_graph`.
    max_iter : int or None, default=None
        Kicks of the iterated local search.
    time_limit : float or None, default=60.0
        Seconds of search, after the starting network is learned. At least one
        of the two limits must be set.
    random_state : int or None, default=None
        Seeds every random choice; None draws a fresh seed. With an integer the
        result is reproducible for a fixed `max_iter` and no time limit.
    n_jobs : int, default=1
        Threads of the score's compiled kernels; -1 is every thread numba has.
        The result does not depend on it.

    Attributes
    ----------
    dag_ : DAG
        The learned network.
    cpdag_ : PDAG
        Its Markov equivalence class.
    score_ : float
        The score of `dag_` under `criterion`, with the structure prior chosen
        (with ``prior="tetrad"``, the value Tetrad reports).
    adjacency_ : pandas.DataFrame
        `cpdag_` with the variable names: ``A[i, j] = 1`` and ``A[j, i] = 0`` is
        ``i -> j``, and both set is an undirected edge.
    n_features_in_ : int
        Number of variables.
    feature_names_in_ : ndarray of str
        Variable names.
    n_iter_ : int
        Kicks performed.
    trace_ : list of tuple
        ``(seconds, iteration, score)`` at the start and each time the best
        score improved.

    Examples
    --------
    >>> from causalbase import OrderSearch
    >>> from causalbase.datasets import load_asia
    >>> model = OrderSearch(max_iter=5, time_limit=None, random_state=0).fit(load_asia().data)
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
        alpha_w: float | None = None,
        n_restarts: int = 2,
        polish: Polish = "ges",
        n_pair_candidates: int = 0,
        init: Init = "ges",
        max_iter: int | None = None,
        time_limit: float | None = 60.0,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.criterion = criterion
        self.ess = ess
        self.expected_parents = expected_parents
        self.prior = prior
        self.penalty_discount = penalty_discount
        self.alpha_w = alpha_w
        self.n_restarts = n_restarts
        self.polish = polish
        self.n_pair_candidates = n_pair_candidates
        self.init = init
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _check(self) -> None:
        super()._check()
        if self.polish not in ("ges", "xges"):
            raise ValueError(f"polish must be 'ges' or 'xges', got {self.polish!r}")
        check_count("n_restarts", self.n_restarts)
        check_count("n_pair_candidates", self.n_pair_candidates)
        _check_order_search(self.init, self.max_iter, self.time_limit)

    def _search(
        self, score: FamilyScore, initial_graph: DAG | None, names: Sequence[str]
    ) -> SearchResult:
        return order_search(
            score,
            _first_order(score, initial_graph, self.init),
            polish=self.polish,
            n_restarts=self.n_restarts,
            n_pair_candidates=self.n_pair_candidates,
            max_iter=self.max_iter,
            time_limit=self.time_limit,
            random_state=self.random_state,
        )


class FLOP(ScoreEstimator):
    """Learn a Bayesian network with FLOP (Wienöbst, Henckel and Weichwald, 2026).

    The search is :func:`~causalbase.order.flop` on the score `criterion`
    names: a local search over orders with FLOP's grow-shrink evaluator and
    reinsertion moves, from a topological order of the network GES (or XGES)
    learns, or of `initial_graph` in :meth:`fit`, then an iterated local
    search with random swaps.

    Parameters
    ----------
    criterion : {"bdeu", "bic", "gaussian_bic", "bge"}, default="bdeu"
        The score: BDeu or BIC over categorical data, or the BIC or BGe of a
        linear Gaussian model over continuous data.
    ess : float, default=1.0
        Equivalent sample size: of the BDeu prior, or of BGe's prior on the
        mean. Also the prior of the parameters behind :meth:`score` on
        categorical data.
    expected_parents : float or None, default=1.0
        Expected in-degree under the structure prior, for every criterion; see
        :class:`~causalbase.scores.FamilyScore`. ``None`` is a uniform prior.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior: the Bernoulli prior of
        :class:`~causalbase.scores.FamilyScore`, or, for BDeu only, Tetrad
        7.6.5's, with `expected_parents` as Tetrad's structure prior (see
        :class:`~causalbase.scores.BDeuScore`).
    penalty_discount : float, default=1.0
        Multiplier of the penalty of the two BIC criteria; larger values give
        sparser structures.
    alpha_w : float or None, default=None
        Degrees of freedom of BGe's Wishart prior, above ``n_vars + 1``;
        ``None`` is ``n_vars + 2``. See :class:`~causalbase.scores.BGeScore`.
    init : {"ges", "xges"}, default="ges"
        Algorithm whose network gives the first order, without `initial_graph`.
    max_iter : int or None, default=None
        Kicks of the iterated local search.
    time_limit : float or None, default=60.0
        Seconds of search, after the starting network is learned. At least one
        of the two limits must be set.
    random_state : int or None, default=None
        Seeds every random choice; None draws a fresh seed.
    n_jobs : int, default=1
        Threads of the score's compiled kernels; -1 is every thread numba has.
        The result does not depend on it.

    Attributes
    ----------
    dag_ : DAG
        The learned network.
    cpdag_ : PDAG
        Its Markov equivalence class.
    score_ : float
        The score of `dag_` under `criterion`, with the structure prior chosen
        (with ``prior="tetrad"``, the value Tetrad reports).
    adjacency_ : pandas.DataFrame
        `cpdag_` with the variable names: ``A[i, j] = 1`` and ``A[j, i] = 0`` is
        ``i -> j``, and both set is an undirected edge.
    n_features_in_ : int
        Number of variables.
    feature_names_in_ : ndarray of str
        Variable names.
    n_iter_ : int
        Kicks performed.
    trace_ : list of tuple
        ``(seconds, iteration, score)`` at the start and each time the best
        score improved.

    Examples
    --------
    >>> from causalbase import FLOP
    >>> from causalbase.datasets import load_asia
    >>> model = FLOP(max_iter=3, time_limit=None, random_state=0).fit(load_asia().data)
    >>> model.n_iter_
    3
    """

    def __init__(
        self,
        criterion: ScoreName = "bdeu",
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        prior: Prior = "bernoulli",
        penalty_discount: float = 1.0,
        alpha_w: float | None = None,
        init: Init = "ges",
        max_iter: int | None = None,
        time_limit: float | None = 60.0,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.criterion = criterion
        self.ess = ess
        self.expected_parents = expected_parents
        self.prior = prior
        self.penalty_discount = penalty_discount
        self.alpha_w = alpha_w
        self.init = init
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _check(self) -> None:
        super()._check()
        _check_order_search(self.init, self.max_iter, self.time_limit)

    def _search(
        self, score: FamilyScore, initial_graph: DAG | None, names: Sequence[str]
    ) -> SearchResult:
        return flop(
            score,
            _first_order(score, initial_graph, self.init),
            max_iter=self.max_iter,
            time_limit=self.time_limit,
            random_state=self.random_state,
        )
