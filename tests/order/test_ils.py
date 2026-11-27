import math
import time

import numpy as np
import pytest

from _samples import load_cell
from causalbase.order import ILSResult, IteratedLocalSearch, OrderScorer
from causalbase.scores import BDeuScore
from causalbase.search import ges


@pytest.fixture(scope="module")
def alarm_small():
    full, _ = load_cell("alarm", 0)
    return full.subset(np.arange(500))


def _scorer(ds):
    return OrderScorer(BDeuScore(ds, prior="tetrad"))


def test_default_perturbation_is_log_p_transpositions(alarm_small):
    ils = IteratedLocalSearch(_scorer(alarm_small))
    assert ils.n_swaps == math.ceil(math.log(alarm_small.n_vars))


def test_the_default_kick_mixes_transpositions_and_topological_orders(alarm_small):
    assert IteratedLocalSearch(_scorer(alarm_small)).kick == "mixed"


def test_never_worse_than_the_start_and_the_trace_only_improves(alarm_small):
    sc = _scorer(alarm_small)
    start = list(np.random.default_rng(1).permutation(alarm_small.n_vars))
    start_score = sc.score(start).total
    res = IteratedLocalSearch(sc, random_state=2).run(start, max_iterations=15)
    assert res.score >= start_score
    assert res.score == sc.score(res.order).total
    times = [t for t, _, _ in res.trace]
    scores = [s for _, _, s in res.trace]
    assert times == sorted(times)
    assert all(b > a for a, b in zip(scores, scores[1:], strict=False))
    assert res.iterations == 15


def test_same_seed_same_result(alarm_small):
    start = list(range(alarm_small.n_vars))
    a = IteratedLocalSearch(_scorer(alarm_small), random_state=5).run(start, max_iterations=10)
    b = IteratedLocalSearch(_scorer(alarm_small), random_state=5).run(start, max_iterations=10)
    assert a.order == b.order and a.score == b.score


def test_the_time_limit_is_a_wall_clock_cap(alarm_small):
    t0 = time.perf_counter()
    res = IteratedLocalSearch(_scorer(alarm_small), random_state=0).run(
        list(range(alarm_small.n_vars)), time_limit=2.0
    )
    assert time.perf_counter() - t0 < 2.0 + 1.0
    assert res.iterations >= 1


def test_each_improvement_records_what_changed(alarm_small):
    sc = _scorer(alarm_small)
    res = IteratedLocalSearch(sc, random_state=3, kick="swaps").run(
        list(np.random.default_rng(4).permutation(alarm_small.n_vars)), max_iterations=40
    )
    assert len(res.improvements) == len(res.trace) - 1
    for imp in res.improvements:
        assert imp["gain"] > 0
        assert len(imp["swaps"]) == IteratedLocalSearch(sc).n_swaps
        assert imp["changed"]  # the nodes whose parent set differs from the previous best
        assert all(0 <= v < alarm_small.n_vars for v in imp["changed"])


def test_the_topological_kick_reorders_within_the_best_network(alarm_small):
    sc = _scorer(alarm_small)
    ils = IteratedLocalSearch(sc, random_state=6, kick="topo")
    order = list(np.random.default_rng(7).permutation(alarm_small.n_vars))
    parents = sc.score(order).parents
    kicked = ils.perturb(order)
    pos = {v: i for i, v in enumerate(kicked)}
    assert sorted(kicked) == sorted(order)
    assert all(pos[u] < pos[v] for v, ps in enumerate(parents) for u in ps)
    res = ils.run(order, max_iterations=10)
    assert res.score >= sc.score(order).total


def test_random_state_reproduces_the_old_seed(alarm_small):
    # Recorded with the retired IteratedLocalSearch(seed=4) from the GES order, 5 kicks.
    score = BDeuScore(alarm_small, prior="tetrad")
    start = list(ges(score).dag.topological_order())
    res = IteratedLocalSearch(OrderScorer(score), random_state=4).run(start, max_iterations=5)
    assert isinstance(res, ILSResult)
    assert res.score == pytest.approx(-6064.86145288258, rel=1e-12)
    assert res.accepted == 4
    assert [i for _, i, _ in res.trace] == [0]


def test_n_swaps_sets_the_strength_of_a_kick(alarm_small):
    sc = _scorer(alarm_small)
    ils = IteratedLocalSearch(sc, n_swaps=3, random_state=0, kick="swaps")
    assert ils.n_swaps == 3
    ils.perturb(list(range(alarm_small.n_vars)))
    assert len(ils.last_swaps) == 3
