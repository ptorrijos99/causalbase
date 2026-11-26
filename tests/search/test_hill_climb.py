import math

import numpy as np
import pytest

from _samples import load_cell
from causalbase.graph import DAG
from causalbase.scores import BDeuScore
from causalbase.search import SearchResult, ges, hill_climbing


@pytest.fixture(scope="module", params=["asia", "child"])
def small(request):
    ds, _ = load_cell(request.param, 0)
    return ds, BDeuScore(ds, prior="tetrad")


@pytest.fixture(scope="module")
def alarm():
    ds, _ = load_cell("alarm", 0)
    return ds, BDeuScore(ds, prior="tetrad")


def _ls(fast, v, ps):
    s = fast.local_score(v, frozenset(ps))
    return s if s == s else -math.inf


def _reaches(children, src, dst, skip=None):
    """Whether dst is reachable from src, optionally ignoring the arc ``skip``."""
    stack, seen = [src], {src}
    while stack:
        x = stack.pop()
        for y in children[x]:
            if (x, y) == skip:
                continue
            if y == dst:
                return True
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


def naive_hill_climb(fast, n):
    """Best-improvement add/delete/reverse from the empty graph, every gain recomputed from scratch.

    Moves are numbered as in ``hill_climbing``: ``u * n + v`` toggles the arc u -> v,
    ``n * n + u * n + v`` reverses the existing arc u -> v; the best valid move wins, the
    smallest number on ties.
    """
    parents = [set() for _ in range(n)]
    children = [set() for _ in range(n)]
    moves = 0
    while True:
        cands = []  # (gain, move number) of every valid move
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                cur_v = _ls(fast, v, parents[v])
                if u in parents[v]:
                    g = _ls(fast, v, parents[v] - {u}) - cur_v
                    cands.append((g, u * n + v))
                    if not _reaches(children, u, v, skip=(u, v)):
                        gu = _ls(fast, u, parents[u] | {v}) - _ls(fast, u, parents[u])
                        cands.append((g + gu, n * n + u * n + v))
                elif not _reaches(children, v, u):
                    cands.append((_ls(fast, v, parents[v] | {u}) - cur_v, u * n + v))
        best = max(g for g, _ in cands)
        best_k = min(k for g, k in cands if g == best) if best > 1e-9 else None
        if best_k is None:
            return parents, moves
        k = best_k
        rev, k = k >= n * n, k % (n * n)
        u, v = divmod(k, n)
        delete = rev or u in parents[v]
        if delete:
            parents[v].discard(u)
            children[u].discard(v)
        if rev:
            parents[u].add(v)
            children[v].add(u)
        elif not delete:
            parents[v].add(u)
            children[u].add(v)
        moves += 1


def _parents_of(dag, n):
    return [set(dag.parents(v)) for v in range(n)]


def test_climb_equals_the_naive_reference_move_for_move(small):
    ds, fast = small
    res = hill_climbing(fast)
    parents, moves = naive_hill_climb(fast, ds.n_vars)
    assert _parents_of(res.dag, ds.n_vars) == parents
    assert res.n_iter == moves


def test_reported_score_is_the_score_of_the_returned_dag(alarm):
    ds, fast = alarm
    res = hill_climbing(fast)
    assert res.score == pytest.approx(fast.score_dag(res.dag), rel=1e-12)
    assert res.info["score_first"] == res.score
    assert res.info["climbs"] == 1


def test_the_result_is_a_local_optimum(alarm):
    ds, fast = alarm
    n = ds.n_vars
    res = hill_climbing(fast)
    parents = _parents_of(res.dag, n)
    children = [set(res.dag.children(v)) for v in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            cur = _ls(fast, v, parents[v])
            if u in parents[v]:
                assert _ls(fast, v, parents[v] - {u}) - cur <= 1e-9
                if not _reaches(children, u, v, skip=(u, v)):
                    gain = (_ls(fast, v, parents[v] - {u}) - cur) + (
                        _ls(fast, u, parents[u] | {v}) - _ls(fast, u, parents[u])
                    )
                    assert gain <= 1e-9
            elif not _reaches(children, v, u):
                assert _ls(fast, v, parents[v] | {u}) - cur <= 1e-9


def test_restarts_keep_the_best_and_the_trace_only_improves(alarm):
    ds, fast = alarm
    plain = hill_climbing(fast)
    res = hill_climbing(fast, n_restarts=5, perturbation=4, random_state=1)
    assert res.info["climbs"] == 6
    assert res.score >= plain.score
    assert res.score == pytest.approx(fast.score_dag(res.dag), rel=1e-12)
    scores = [row[-1] for row in res.trace]
    assert scores == sorted(scores) and scores[-1] == res.score


def test_tabu_search_never_ends_below_the_plain_climb(alarm):
    ds, fast = alarm
    plain = hill_climbing(fast)
    res = hill_climbing(fast, tabu_length=10, max_tabu_moves=10)
    assert res.score >= plain.score
    assert res.score == pytest.approx(fast.score_dag(res.dag), rel=1e-12)


def test_a_start_dag_is_climbed_from(alarm):
    ds, fast = alarm
    start = ges(fast).to_dag()
    res = hill_climbing(fast, start)
    assert res.score >= fast.score_dag(start) - 1e-9


def test_same_seed_same_result(alarm):
    ds, fast = alarm
    a = hill_climbing(fast, n_restarts=3, perturbation=3, random_state=7)
    b = hill_climbing(fast, n_restarts=3, perturbation=3, random_state=7)
    assert a.dag.edges() == b.dag.edges() and a.score == b.score


def test_time_limit_stops_the_restarts(alarm):
    ds, fast = alarm
    res = hill_climbing(fast, n_restarts=10**9, perturbation=3, time_limit=2.0)
    assert res.info["climbs"] >= 1 and res.info["time"] < 10.0
    assert isinstance(res.dag, DAG)
    assert np.isfinite(res.score)


@pytest.mark.skipif(
    not __import__("causalbase.scores._compiled", fromlist=["HAVE_NUMBA"]).HAVE_NUMBA,
    reason="needs numba",
)
def test_the_compiled_move_choice_equals_the_numpy_one(alarm):
    import random

    from causalbase.search.hill_climb import _State

    ds, fast = alarm
    n = ds.n_vars
    st = _State(fast._engine(), n, [set() for _ in range(n)])
    rng = random.Random(3)
    checked = 0
    for step in range(120):
        tabu = (
            None if step % 3 else {st.hash_after(*mv) for mv in [st.pick(None, kernel=False)] if mv}
        )
        a, b = st.pick(tabu, kernel=True), st.pick(tabu, kernel=False)
        assert a == b
        checked += a is not None
        if a is not None and rng.random() < 0.7:
            st.apply(*a)
        else:
            st.perturb(2, rng)
    assert checked > 50


def test_the_result_is_a_search_result(alarm):
    ds, fast = alarm
    res = hill_climbing(fast, n_restarts=1, perturbation=2, random_state=0)
    assert isinstance(res, SearchResult)
    assert set(res.info) == {"climbs", "time", "time_first", "score_first"}
    assert res.n_iter >= res.trace[-1][1]
    assert [s for _, _, s in res.trace][-1] == res.score


def test_random_state_reproduces_the_old_seed(alarm):
    # Recorded with the retired dag_hill_climb(restarts=2, perturb=3, seed=2).
    ds, fast = alarm
    res = hill_climbing(fast, n_restarts=2, perturbation=3, random_state=2)
    assert res.score == pytest.approx(-54628.393303628705, rel=1e-12)
    assert res.n_iter == 59 and res.info["climbs"] == 3
    assert [i for _, i, _ in res.trace] == [53]


def test_without_a_random_state_each_run_draws_its_own(alarm):
    ds, fast = alarm
    res = hill_climbing(fast, n_restarts=1, perturbation=2)
    assert res.info["climbs"] == 2 and np.isfinite(res.score)


def test_hill_climbing_refuses_a_wrong_start(alarm):
    ds, fast = alarm
    with pytest.raises(ValueError, match="nodes"):
        hill_climbing(fast, DAG(3))
