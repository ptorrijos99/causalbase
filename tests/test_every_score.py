"""Every search on every score: BIC, Gaussian BIC and BGe through the order searches,
hill climbing, FLOP and the exact searches, which BDeu's own tests cover."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from _problems import best_dag_score, random_discrete, random_gaussian
from causalbase import FLOP, GES, ExactSearch, HillClimbing, OrderSearch
from causalbase.data import ContinuousDataset
from causalbase.exact import exact_search
from causalbase.exact.dp import parent_set_scores, prune_candidates
from causalbase.graph import DAG
from causalbase.order import OrderScorer, flop, order_search, pair_polish
from causalbase.order.polished import PolishedILS
from causalbase.scores import (
    BDeuScore,
    BGeScore,
    BICScore,
    FamilyScore,
    GaussianBICScore,
    make_score,
)
from causalbase.search import ges, hill_climbing

SCORES: dict[str, Callable[[int, int, int], FamilyScore]] = {
    "bic": lambda p, n, seed: BICScore(random_discrete(p, n, seed)),
    "gaussian_bic": lambda p, n, seed: GaussianBICScore(random_gaussian(p, n, seed)),
    "bge": lambda p, n, seed: BGeScore(random_gaussian(p, n, seed)),
}
"""The scores other than BDeu, each over a random problem of ``p`` variables and ``n`` rows."""

ESTIMATORS = (OrderSearch, HillClimbing, FLOP, ExactSearch)


def _acyclic(parents: list[set[int]]) -> bool:
    left = set(range(len(parents)))
    while left:
        roots = {v for v in left if not parents[v] & left}
        if not roots:
            return False
        left -= roots
    return True


def _best_single_move(score: Any, dag: DAG) -> float:
    """Return the largest gain of adding, deleting or reversing one arc of `dag`, keeping a DAG."""
    n = dag.num_nodes
    parents = [set(dag.parents(v)) for v in range(n)]
    local = [score.local_score(v, parents[v]) for v in range(n)]
    best = -np.inf
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            moved = [set(ps) for ps in parents]
            if u in parents[v]:
                moved[v].discard(u)
                best = max(best, score.local_score(v, moved[v]) - local[v])
                moved[u].add(v)  # the reversal
            else:
                moved[v].add(u)
            if _acyclic(moved):
                gain = sum(score.local_score(w, moved[w]) - local[w] for w in (u, v))
                best = max(best, gain)
    return best


@pytest.mark.parametrize("name", SCORES)
@pytest.mark.parametrize("p,seed", [(3, 0), (4, 1), (4, 2), (5, 3)])
def test_exact_search_finds_the_best_dag(name, p, seed):
    score = SCORES[name](p, 200, seed)
    best = best_dag_score(score)
    for prune in (True, False):
        result = exact_search(score, prune=prune)
        assert result.score == pytest.approx(best, rel=1e-9)
        assert score.score_dag(result.dag) == pytest.approx(result.score, rel=1e-9)


@pytest.mark.parametrize("name", SCORES)
def test_superset_bounds_keep_the_same_candidates(name):
    """Pruning skips only sets that no subset-dominance check would keep."""
    skipped = 0
    for seed in range(3):
        # Twelve variables, so that the walk goes deeper than the chunks it is split into.
        score = SCORES[name](12, 300, seed)
        for v in range(12):
            full, n_full = parent_set_scores(score, v, prune=False)
            pruned, n_pruned = parent_set_scores(score, v, prune=True)
            assert n_full == 1 << 11 and n_pruned <= n_full
            (a, sa), (b, sb) = prune_candidates(full, 11), prune_candidates(pruned, 11)
            assert np.array_equal(a, b)
            np.testing.assert_allclose(sa, sb, rtol=1e-12)
            skipped += n_full - n_pruned
    if name != "bge":
        assert skipped > 0  # the bounds do prune; BGe has none


@pytest.mark.parametrize("name", SCORES)
def test_the_integer_program_agrees_with_dynamic_programming(name):
    pytest.importorskip("pyscipopt")
    for seed in range(2):
        score = SCORES[name](5, 200, seed)
        limited = exact_search(score, max_parents=4)
        assert limited.score == pytest.approx(exact_search(score).score, rel=1e-9)
        assert limited.info["status"] == "optimal"


@pytest.mark.parametrize("name", SCORES)
def test_hill_climbing_stops_where_no_arc_move_improves(name):
    for seed in range(3):
        score = SCORES[name](6, 300, seed)
        result = hill_climbing(score)
        assert score.score_dag(result.dag) == pytest.approx(result.score, rel=1e-9)
        assert _best_single_move(score, result.dag) < 1e-6


@pytest.mark.parametrize("name", SCORES)
def test_order_searches_return_the_score_of_their_network(name):
    for seed in range(2):
        score = SCORES[name](6, 300, seed)
        empty = score.score_dag(DAG(6))
        for result in (
            order_search(score, max_iter=2, random_state=seed),
            order_search(score, polish="xges", n_pair_candidates=3, max_iter=2, random_state=seed),
            flop(score, max_iter=2, random_state=seed),
        ):
            assert score.score_dag(result.dag) == pytest.approx(result.score, rel=1e-9)
            assert result.score >= empty


@pytest.mark.parametrize("name", SCORES)
def test_pair_polish_gains_what_it_reports(name):
    score = SCORES[name](6, 300, 4)
    start = DAG(6)
    dag, gain, _ = pair_polish(score, start, n_candidates=4)
    assert gain >= 0
    assert score.score_dag(dag) - score.score_dag(start) == pytest.approx(gain, rel=1e-9)


def test_polished_ils_takes_a_score_without_an_equivalent_sample_size():
    score = SCORES["gaussian_bic"](5, 200, 0)
    ils = PolishedILS(OrderScorer(score), random_state=0)
    assert ils.ess is None and ils.data is score.data
    result = ils.run(list(range(5)), max_iterations=2)
    assert score.score_dag(result.dag) == pytest.approx(result.score, rel=1e-9)


def _fast(cls, criterion):
    if cls is OrderSearch:
        return OrderSearch(criterion, max_iter=2, time_limit=None, random_state=0)
    if cls is FLOP:
        return FLOP(criterion, max_iter=2, time_limit=None, random_state=0)
    if cls is HillClimbing:
        return HillClimbing(criterion, n_restarts=1, random_state=0)
    return ExactSearch(criterion)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("criterion", ["bic", "gaussian_bic", "bge"])
def test_the_estimators_take_every_criterion(cls, criterion):
    data = random_discrete(6, 400, 2) if criterion == "bic" else random_gaussian(6, 400, 2)
    model = _fast(cls, criterion).set_params(expected_parents=1.5, penalty_discount=1.5)
    model.fit(data)
    score = make_score(criterion, data, expected_parents=1.5, penalty_discount=1.5)
    assert model.score_ == pytest.approx(score.score_dag(model.dag_), rel=1e-12)
    assert model.trace_[-1][2] == pytest.approx(model.score_, rel=1e-9)
    frame = data.to_pandas() if criterion == "bic" else data.values
    assert np.isfinite(model.score(frame[:100]))


def test_the_estimators_run_the_functions_on_every_criterion():
    data = random_gaussian(6, 400, 5)
    for criterion in ("gaussian_bic", "bge"):
        score = make_score(criterion, data)
        cases = [
            (
                HillClimbing(criterion, n_restarts=2, random_state=3),
                hill_climbing(score, n_restarts=2, random_state=3).dag,
            ),
            (
                FLOP(criterion, max_iter=3, time_limit=None, random_state=2),
                flop(score, max_iter=3, random_state=2).dag,
            ),
            (
                OrderSearch(criterion, max_iter=3, time_limit=None, random_state=2),
                order_search(score, max_iter=3, random_state=2).dag,
            ),
            (ExactSearch(criterion), exact_search(score).dag),
            (GES(criterion), ges(score).dag),
        ]
        for model, dag in cases:
            model.fit(data)
            assert score.score_dag(model.dag_) == pytest.approx(score.score_dag(dag), rel=1e-12)


def test_criteria_other_than_bdeu_refuse_bdeu_only_settings():
    data = random_discrete(4, 100, 0)
    for cls in ESTIMATORS:
        for bad in (
            {"criterion": "nope"},
            {"criterion": "bic", "prior": "tetrad"},
            {"penalty_discount": 0.0},
            {"penalty_discount": np.inf},
        ):
            with pytest.raises(ValueError):
                _fast(cls, "bdeu").set_params(**bad).fit(data)


@pytest.mark.parametrize("name", SCORES)
def test_threads_do_not_change_what_the_exact_search_finds(name):
    serial = SCORES[name](9, 300, 6)
    threaded = type(serial)(serial.data, n_jobs=-1)
    a, b = exact_search(serial), exact_search(threaded)
    assert a.dag == b.dag
    assert a.score == pytest.approx(b.score, rel=1e-12)


@pytest.mark.parametrize("make", [GaussianBICScore, BGeScore])
def test_collinear_columns_leave_the_gaussian_exact_search_exact(make):
    data = random_gaussian(4, 200, 8)
    values = np.column_stack([data.values, data.values[:, 0] - 2 * data.values[:, 2]])
    score = make(ContinuousDataset(values, [*data.names, "sum"]))
    result = exact_search(score)
    assert result.score == pytest.approx(best_dag_score(score), rel=1e-9)
    assert score.score_dag(result.dag) == pytest.approx(result.score, rel=1e-9)


def test_discrete_data_too_large_for_the_counting_kernels_still_gets_the_optimum():
    data = random_discrete(4, 2**15 + 5, 9)  # counts no longer fit in int16
    for score in (BDeuScore(data), BICScore(data)):
        result = exact_search(score)
        assert result.score == pytest.approx(best_dag_score(score), rel=1e-9)
