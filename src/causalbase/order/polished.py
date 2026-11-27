"""Order search followed by GES in the space of equivalence classes, and an ILS built on both.

The network an order yields is a local optimum among the networks consistent with
that order, but often not among all networks: a node may need a parent that the
order places after it, and reaching that order takes several coordinated
insertions. GES started from the order's network (forward and backward phases)
adds those arcs directly. :func:`order_search` runs that pair as the local search
of an iterated local search: insertion to a local optimum over orders, then GES
from the resulting network, and it accepts the polished network when it scores at
least as well as the best. Its kicks start from the best network: half the time a
random topological order of it, otherwise that order with ``n_swaps`` random
transpositions. :class:`PolishedILS` is the implementation behind it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from causalbase._validation import check_count, check_time_limit, resolve_seed
from causalbase.data import ContinuousDataset, DiscreteDataset
from causalbase.graph import DAG
from causalbase.order.evaluator import OrderScorer
from causalbase.order.insertion import InsertionSearch
from causalbase.order.pairs import pair_polish
from causalbase.scores._compiled import engine_of
from causalbase.search import SearchResult, ges, xges

if TYPE_CHECKING:
    from causalbase.scores import FamilyScore

__all__ = ["Polish", "order_search"]

Polish = Literal["ges", "xges"]
"""The searches over equivalence classes that can end a local search."""


def _polish(kind: Polish, score: FamilyScore, dag: DAG) -> DAG:
    if kind == "ges":
        return ges(score, initial_graph=dag).to_dag()
    return xges(score, initial_graph=dag, extended=True).to_dag()


def ges_order(score: FamilyScore, kind: Polish = "ges") -> list[int]:
    """Return a topological order of the network GES (or XGES) learns with `score`."""
    search = ges if kind == "ges" else xges
    return [int(v) for v in search(score).to_dag().topological_order()]


def start_order(score: FamilyScore, initial_order: Sequence[int] | None) -> list[int]:
    """Return `initial_order` as a list, or GES's order when it is None.

    Raises ValueError unless the order holds every variable once.
    """
    if initial_order is None:
        return ges_order(score)
    out = [int(v) for v in initial_order]
    if sorted(out) != list(range(score.n_vars)):
        raise ValueError(f"initial_order must be a permutation of range({score.n_vars})")
    return out


def random_topological(parents: Sequence[Sequence[int]], rng: np.random.Generator) -> list[int]:
    """Kahn's algorithm taking a uniformly drawn ready node at each step."""
    n = len(parents)
    children: list[list[int]] = [[] for _ in range(n)]
    missing = [len(ps) for ps in parents]
    for v, ps in enumerate(parents):
        for u in ps:
            children[u].append(v)
    ready = [v for v in range(n) if missing[v] == 0]
    out = []
    while ready:
        v = ready.pop(int(rng.integers(len(ready))))
        out.append(v)
        for w in children[v]:
            missing[w] -= 1
            if missing[w] == 0:
                ready.append(w)
    return out


class PolishedILS:
    """Iterated local search whose local search is insertion over orders followed by GES.

    The implementation behind :func:`order_search`, which most code should call.

    Parameters
    ----------
    scorer : OrderScorer
        Order evaluator. Its score (``scorer.family_score``) is the one the
        polish searches and the networks are scored with (through its
        compiled engine for a :class:`~causalbase.scores.BDeuScore`).
    random_state : int or None, default=None
        Seeds every random choice; None draws a fresh seed.
    n_swaps : int or None, default=None
        Transpositions of a transposition kick; ``ceil(ln p)`` when None.
    n_pair_candidates : int, default=0
        When positive, each local search ends with
        :func:`~causalbase.order.pair_polish` over this many candidates (and
        GES again if it added anything), so families whose parents only help
        together are found. 0 turns it off.
    polish : {"ges", "xges"}, default="ges"
        Search in the space of equivalence classes that ends each local search,
        started from the order's DAG.
    """

    def __init__(
        self,
        scorer: OrderScorer,
        *,
        random_state: int | None = None,
        n_swaps: int | None = None,
        n_pair_candidates: int = 0,
        polish: Polish = "ges",
    ) -> None:
        if polish not in ("ges", "xges"):
            raise ValueError(f"polish must be 'ges' or 'xges', not {polish!r}")
        self.polish: Polish = polish
        self.scorer = scorer
        family = scorer.family_score
        self.data: DiscreteDataset | ContinuousDataset | None = getattr(family, "data", None)
        ess = getattr(family, "ess", None)
        self.ess = None if ess is None else float(ess)
        # GES and XGES search the score's view (itself, or the Bernoulli BDeu with Tetrad's p),
        # one for the whole search so that the polishes share its cache; the networks are
        # scored with the engine the evaluator scores with.
        self._family = family
        view = getattr(family, "_search_view", None)
        self._view = view() if view is not None else family
        self._engine = engine_of(family)
        self.search = InsertionSearch(scorer)
        self.n = scorer.n_vars
        self.n_swaps = n_swaps if n_swaps is not None else max(1, math.ceil(math.log(self.n)))
        self.n_pair_candidates = n_pair_candidates
        self.rng = np.random.default_rng(resolve_seed(random_state))

    def _score(self, dag: DAG) -> float:
        engine = self._engine
        return float(sum(engine.local_score(v, sorted(dag.parents(v))) for v in range(self.n)))

    def _local(self, order: Sequence[int], deadline: float | None) -> tuple[DAG, float]:
        order, _ = self.search.run(order, self.rng, deadline=deadline)
        ps = self.scorer.score(order).parents
        dag = DAG.from_edges(self.n, [(u, v) for v in range(self.n) for u in ps[v]])
        dag = _polish(self.polish, self._view, dag)
        if self.n_pair_candidates > 0:
            dag, gain, _ = pair_polish(self._family, dag, self.n_pair_candidates)
            if gain > 1e-9:
                dag = _polish(self.polish, self._view, dag)
        return dag, self._score(dag)

    def kick(self, dag: DAG) -> list[int]:
        """Return a random topological order of ``dag``; half the time, ``n_swaps`` swaps of it."""
        order = random_topological([sorted(dag.parents(v)) for v in range(self.n)], self.rng)
        if self.rng.random() >= 0.5:
            for _ in range(self.n_swaps):
                i, j = self.rng.choice(self.n, 2, replace=False)
                order[i], order[j] = order[j], order[i]
        return order

    def run(
        self,
        start: Sequence[int],
        time_limit: float | None = None,
        max_iterations: int | None = None,
    ) -> SearchResult:
        """Search from ``start`` until ``time_limit`` seconds or ``max_iterations`` kicks.

        Returns
        -------
        SearchResult
            As :func:`order_search` returns it.
        """
        t0 = time.perf_counter()
        deadline = None if time_limit is None else t0 + time_limit
        best, best_score = self._local(start, deadline)
        trace = [(time.perf_counter() - t0, 0, best_score)]
        iterations = accepted = 0
        while (max_iterations is None or iterations < max_iterations) and (
            deadline is None or time.perf_counter() < deadline
        ):
            cand, score = self._local(self.kick(best), deadline)
            iterations += 1
            if score >= best_score - 1e-9:
                accepted += 1
                if score > best_score + 1e-9:
                    trace.append((time.perf_counter() - t0, iterations, score))
                best, best_score = cand, score
        return SearchResult(best, best_score, iterations, trace, {"accepted": accepted})


def order_search(
    score: FamilyScore,
    initial_order: Sequence[int] | None = None,
    *,
    polish: Polish = "ges",
    n_restarts: int = 2,
    n_pair_candidates: int = 0,
    n_swaps: int | None = None,
    max_iter: int | None = None,
    time_limit: float | None = None,
    random_state: int | None = None,
) -> SearchResult:
    """Learn a network by iterated local search over orders, each local search polished by GES.

    A local search climbs over orders by insertion moves, scoring each order
    by the best network consistent with it (:class:`OrderScorer`, a greedy
    climb per node), and ends with GES (or XGES) started from that network,
    so that arcs the order forbids can still be added. Each iteration kicks
    the best network found so far (a random topological order of it, half the
    time with `n_swaps` random transpositions), runs a local search from
    there, and keeps the result if it scores at least as well.

    Parameters
    ----------
    score : FamilyScore
        The score: BDeu with either prior, BIC, Gaussian BIC or BGe. Orders
        and networks are scored with it (with its compiled engine for BDeu);
        GES and XGES search it as :func:`~causalbase.search.ges` does.
    initial_order : sequence of int, optional
        Order the first local search starts from. By default a topological
        order of the network GES learns on the same score, learned before the
        time limit starts.
    polish : {"ges", "xges"}, default="ges"
        Search over equivalence classes that ends each local search.
    n_restarts : int, default=2
        Extra greedy climbs per node in the order evaluator, each without the
        first parent the previous climb chose (see :class:`OrderScorer`).
    n_pair_candidates : int, default=0
        When positive, each local search also tries adding pairs of parents
        drawn from this many best single candidates (:func:`pair_polish`), and
        polishes again if that added any.
    n_swaps : int or None, default=None
        Transpositions of a kick; ``ceil(ln p)`` for ``p`` variables when None.
    max_iter : int or None, default=None
        Kicks, each followed by a local search, after the first local search.
    time_limit : float or None, default=None
        Seconds of search, checked inside the local searches too. At least one
        of `max_iter` and `time_limit` must be set.
    random_state : int or None, default=None
        Seeds every random choice; None draws a fresh seed. With an integer and
        `max_iter` (and no time limit) the result is reproducible.

    Returns
    -------
    SearchResult
        ``n_iter`` counts the kicks; ``info["accepted"]`` those whose local
        search scored at least as well as the best, which then moved there.

    Raises
    ------
    ValueError
        If neither limit is set, or an option is out of range.

    Examples
    --------
    >>> from causalbase.datasets import load_asia
    >>> from causalbase.order import order_search
    >>> from causalbase.scores import BDeuScore
    >>> result = order_search(BDeuScore(load_asia().data), max_iter=3, random_state=0)
    >>> result.n_iter
    3
    """
    if polish not in ("ges", "xges"):
        raise ValueError(f"polish must be 'ges' or 'xges', got {polish!r}")
    check_count("n_restarts", n_restarts)
    check_count("n_pair_candidates", n_pair_candidates)
    check_count("n_swaps", n_swaps, allow_none=True)
    check_count("max_iter", max_iter, allow_none=True)
    check_time_limit(time_limit)
    if max_iter is None and time_limit is None:
        raise ValueError("set max_iter, time_limit or both, or the search never ends")
    seed = resolve_seed(random_state)
    start = start_order(score, initial_order)
    search = PolishedILS(
        OrderScorer(score, n_restarts=n_restarts),
        random_state=seed,
        n_swaps=n_swaps,
        n_pair_candidates=n_pair_candidates,
        polish=polish,
    )
    return search.run(start, time_limit=time_limit, max_iterations=max_iter)
