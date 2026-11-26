"""Order-based structure learning as scikit-learn style estimators: OrderSearch and FLOP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from causalbase._bdeu_estimator import BDeuEstimator, Init, initial_order
from causalbase._validation import check_count, check_time_limit
from causalbase.order.flop import flop
from causalbase.order.polished import Polish, order_search

if TYPE_CHECKING:
    from causalbase.scores.bdeu import BDeuScore, Prior
    from causalbase.search import SearchResult

__all__ = ["FLOP", "OrderSearch"]


def _check_order_search(init: object, max_iter: object, time_limit: object) -> None:
    if init not in ("ges", "xges"):
        raise ValueError(f"init must be 'ges' or 'xges', got {init!r}")
    check_count("max_iter", max_iter, allow_none=True)
    check_time_limit(time_limit)
    if time_limit is None and max_iter is None:
        raise ValueError("set time_limit, max_iter or both, or the search never ends")


class OrderSearch(BDeuEstimator):
    """Learn a Bayesian network by iterated local search over variable orders.

    The search is :func:`~causalbase.order.order_search` on
    :class:`~causalbase.scores.BDeuScore`: it starts from a topological order
    of the network GES (or XGES) learns, climbs over orders by insertion
    moves, scores each order by the best network consistent with it (a
    greedy climb per node, restarted `n_restarts` times), and ends each local
    search with GES or XGES started from that network, so that arcs the order
    forbids can still be added.

    Parameters
    ----------
    ess : float, default=1.0
        Equivalent sample size of the BDeu prior.
    expected_parents : float or None, default=1.0
        The parameter of the structure prior, as in
        :class:`~causalbase.scores.BDeuScore`; None leaves the prior out.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior: the Bernoulli prior of
        :class:`~causalbase.scores.FamilyScore`, or Tetrad 7.6.5's, with
        `expected_parents` as Tetrad's structure prior.
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
        Algorithm whose network gives the first order.
    time_limit : float or None, default=60.0
        Seconds of search, after the starting network is learned.
    max_iter : int or None, default=None
        Kicks of the iterated local search. At least one of the two limits
        must be set.
    random_state : int or None, default=None
        Seeds every random choice; None draws a fresh seed. With an integer the
        result is reproducible for a fixed `max_iter` and no time limit.
    n_jobs : int, default=1
        Threads that score large batches of candidate parents; the result does
        not depend on it.

    Attributes
    ----------
    dag_ : DAG
        The learned network.
    cpdag_ : PDAG
        Its equivalence class.
    score_ : float
        The :class:`~causalbase.scores.BDeuScore` of `dag_`, with the prior chosen.
    n_iter_ : int
        Kicks performed.
    trace_ : list of tuple
        ``(seconds, iteration, score)`` at the start and each time the best
        score improved.
    adjacency_ : pandas.DataFrame
        The CPDAG with variable names: ``A[i, j] = 1`` and ``A[j, i] = 0`` is
        ``i -> j``, and both set is an undirected edge.
    n_features_in_ : int
        Number of variables.
    feature_names_in_ : ndarray of str
        Variable names.

    Examples
    --------
    >>> from causalbase import OrderSearch
    >>> from causalbase.datasets import load_asia
    >>> model = OrderSearch(time_limit=None, max_iter=5, random_state=0).fit(load_asia().data)
    >>> model.cpdag_
    PDAG(num_nodes=8, directed=..., undirected=...)
    """

    def __init__(
        self,
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        prior: Prior = "bernoulli",
        n_restarts: int = 2,
        polish: Polish = "ges",
        n_pair_candidates: int = 0,
        init: Init = "ges",
        time_limit: float | None = 60.0,
        max_iter: int | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.ess = ess
        self.expected_parents = expected_parents
        self.prior = prior
        self.n_restarts = n_restarts
        self.polish = polish
        self.n_pair_candidates = n_pair_candidates
        self.init = init
        self.time_limit = time_limit
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _check(self) -> None:
        super()._check()
        if self.polish not in ("ges", "xges"):
            raise ValueError(f"polish must be 'ges' or 'xges', got {self.polish!r}")
        check_count("n_restarts", self.n_restarts)
        check_count("n_pair_candidates", self.n_pair_candidates)
        _check_order_search(self.init, self.max_iter, self.time_limit)

    def _search(self, score: BDeuScore) -> SearchResult:
        return order_search(
            score,
            initial_order(score, self.init),
            polish=self.polish,
            n_restarts=self.n_restarts,
            n_pair_candidates=self.n_pair_candidates,
            max_iter=self.max_iter,
            time_limit=self.time_limit,
            random_state=self.random_state,
        )


class FLOP(BDeuEstimator):
    """Learn a Bayesian network with FLOP (Wienöbst, Henckel and Weichwald, 2026).

    The search is :func:`~causalbase.order.flop` on
    :class:`~causalbase.scores.BDeuScore`: a local search over orders with
    FLOP's grow-shrink evaluator and reinsertion moves, from a topological
    order of the network GES (or XGES) learns, then an iterated local search with random swaps.

    Parameters
    ----------
    ess : float, default=1.0
        Equivalent sample size of the BDeu prior.
    expected_parents : float or None, default=1.0
        The parameter of the structure prior, as in
        :class:`~causalbase.scores.BDeuScore`; None leaves the prior out.
    prior : {"bernoulli", "tetrad"}, default="bernoulli"
        The structure prior: the Bernoulli prior of
        :class:`~causalbase.scores.FamilyScore`, or Tetrad 7.6.5's, with
        `expected_parents` as Tetrad's structure prior.
    init : {"ges", "xges"}, default="ges"
        Algorithm whose network gives the first order.
    max_iter : int or None, default=None
        Kicks of the iterated local search.
    time_limit : float or None, default=60.0
        Seconds of search, after the starting network is learned. At least one
        of the two limits must be set.
    random_state : int or None, default=None
        Seeds every random choice; None draws a fresh seed.
    n_jobs : int, default=1
        Threads that score large batches of candidate parents; the result does
        not depend on it.

    Attributes
    ----------
    dag_, cpdag_, score_, adjacency_, n_features_in_, feature_names_in_
        As in :class:`OrderSearch`.
    n_iter_ : int
        Kicks performed.
    trace_ : list of tuple
        ``(seconds, iteration, score)`` at the start and each time the best
        score improved.

    Examples
    --------
    >>> from causalbase import FLOP
    >>> from causalbase.datasets import load_asia
    >>> model = FLOP(time_limit=None, max_iter=3, random_state=0).fit(load_asia().data)
    >>> model.n_iter_
    3
    """

    def __init__(
        self,
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        prior: Prior = "bernoulli",
        init: Init = "ges",
        max_iter: int | None = None,
        time_limit: float | None = 60.0,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.ess = ess
        self.expected_parents = expected_parents
        self.prior = prior
        self.init = init
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _check(self) -> None:
        super()._check()
        _check_order_search(self.init, self.max_iter, self.time_limit)

    def _search(self, score: BDeuScore) -> SearchResult:
        return flop(
            score,
            initial_order(score, self.init),
            max_iter=self.max_iter,
            time_limit=self.time_limit,
            random_state=self.random_state,
        )
