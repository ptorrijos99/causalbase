import numpy as np
import pytest

from _samples import load_cell
from causalbase.graph import DAG
from causalbase.order import OrderScorer, SearchResult, order_search
from causalbase.order.insertion import InsertionSearch
from causalbase.order.polished import PolishedILS
from causalbase.scores import BDeuScore
from causalbase.scores._compiled import CompiledBDeu
from causalbase.search import ges, xges


def _with_kernel(data, kernel):
    """BDeuScore(prior="tetrad") whose compiled engine counts with `kernel`."""
    score = BDeuScore(data, prior="tetrad")
    score._compiled = CompiledBDeu(data, prior="tetrad", kernel=kernel)
    return score


@pytest.fixture(scope="module")
def alarm_small():
    full, _ = load_cell("alarm", 0)
    return full.subset(np.arange(500))


def _dag(parents):
    return DAG.from_edges(len(parents), [(u, v) for v, ps in enumerate(parents) for u in ps])


def _ges_order(score):
    return list(ges(score).dag.topological_order())


def test_polishing_an_order_result_never_lowers_its_score(alarm_small):
    fast = BDeuScore(alarm_small, prior="tetrad")
    sc = OrderScorer(fast)
    order, _ = InsertionSearch(sc).run(
        list(np.random.default_rng(0).permutation(alarm_small.n_vars)),
        np.random.default_rng(1),
        max_passes=1,
    )
    dag = _dag(sc.score(order).parents)
    polished = ges(fast, dag).dag
    assert fast.score_dag(polished) >= fast.score_dag(dag) - 1e-6


def test_order_search_reports_the_score_of_its_dag_and_beats_polished_ges(alarm_small):
    fast = BDeuScore(alarm_small, prior="tetrad")
    start = ges(fast).dag
    res = order_search(fast, list(start.topological_order()), max_iter=8, random_state=3)
    assert isinstance(res, SearchResult)
    assert res.n_iter == 8
    assert set(res.info) == {"accepted"}
    assert abs(res.score - fast.score_dag(res.dag)) < 1e-6
    assert res.score >= fast.score_dag(ges(fast, start).dag) - 1e-6
    scores = [s for _, _, s in res.trace]
    assert all(b > a for a, b in zip(scores, scores[1:], strict=False))


def test_order_search_starts_from_the_ges_order_by_default(alarm_small):
    a = order_search(BDeuScore(alarm_small, prior="tetrad"), max_iter=3, random_state=2)
    score = BDeuScore(alarm_small, prior="tetrad")
    b = order_search(score, _ges_order(score), max_iter=3, random_state=2)
    assert a.score == b.score and sorted(a.dag.edges()) == sorted(b.dag.edges())
    assert [(i, s) for _, i, s in a.trace] == [(i, s) for _, i, s in b.trace]


def test_order_search_same_random_state_same_result(alarm_small):
    def go():
        fast = BDeuScore(alarm_small, prior="tetrad")
        return order_search(fast, list(range(alarm_small.n_vars)), max_iter=5, random_state=5)

    a, b = go(), go()
    assert a.score == b.score and set(a.dag.edges()) == set(b.dag.edges())


def test_order_search_needs_a_limit_and_a_known_polish(alarm_small):
    score = BDeuScore(alarm_small, prior="tetrad")
    start = list(range(alarm_small.n_vars))
    with pytest.raises(ValueError, match="max_iter"):
        order_search(score, start)
    with pytest.raises(ValueError, match="polish"):
        order_search(score, start, polish="bes", max_iter=1)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def xor():
    """C = A xor B (5 % noise), D independent: no single parent of C helps, the pair does."""
    import pandas as pd

    from causalbase.data import DiscreteDataset

    rng = np.random.default_rng(0)
    n = 3000
    a, b, d = rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 2, n)
    c = (a ^ b) ^ (rng.random(n) < 0.05)
    return DiscreteDataset.from_pandas(
        pd.DataFrame({"A": a, "B": b, "C": c.astype(int), "D": d}).astype(str)
    )


def test_order_search_without_pairs_misses_the_xor_family(xor):
    res = order_search(BDeuScore(xor, prior="tetrad"), [0, 1, 2, 3], max_iter=3, random_state=0)
    assert len(res.dag.edges()) == 0


def test_order_search_with_pairs_finds_the_xor_family(xor):
    fast = BDeuScore(xor, prior="tetrad")
    res = order_search(fast, [0, 1, 2, 3], n_pair_candidates=10, max_iter=3, random_state=0)
    triple = {0, 1, 2}
    assert any(set(res.dag.parents(x)) == triple - {x} for x in triple)
    assert abs(res.score - fast.score_dag(res.dag)) < 1e-6


def test_xges_from_an_order_result_never_lowers_the_score(alarm_small):
    fast = BDeuScore(alarm_small, prior="tetrad")
    sc = OrderScorer(fast)
    order, _ = InsertionSearch(sc).run(
        list(np.random.default_rng(0).permutation(alarm_small.n_vars)),
        np.random.default_rng(1),
        max_passes=1,
    )
    dag = _dag(sc.score(order).parents)
    assert fast.score_dag(xges(fast, dag).dag) >= fast.score_dag(dag) - 1e-6


def test_order_search_with_xges_polish_reports_the_score_of_its_dag(alarm_small):
    fast = BDeuScore(alarm_small, prior="tetrad")
    res = order_search(
        fast, list(range(alarm_small.n_vars)), polish="xges", max_iter=3, random_state=3
    )
    assert abs(res.score - fast.score_dag(res.dag)) < 1e-6


def test_bdeu_score_keeps_the_equivalent_sample_size(alarm_small):
    fast10 = BDeuScore(alarm_small, ess=10.0, prior="tetrad")
    empty = DAG.from_edges(alarm_small.n_vars, [])
    polished = ges(fast10, empty).dag
    assert fast10.score_dag(polished) == pytest.approx(fast10.score_dag(ges(fast10).dag))


def test_polished_ils_takes_the_equivalent_sample_size_from_its_scorer(alarm_small):
    ils = PolishedILS(OrderScorer(BDeuScore(alarm_small, ess=10.0, prior="tetrad")))
    assert ils.ess == 10.0


def test_order_search_gives_what_it_gave_before(alarm_small):
    # Before the evaluator and the counting kernel were optimised (restarts=2, seed 5, 6 kicks).
    score = BDeuScore(alarm_small, prior="tetrad")
    res = order_search(
        score, list(range(alarm_small.n_vars)), n_restarts=2, max_iter=6, random_state=5
    )
    # Recorded on Linux; the last bit of a sum of lgamma terms differs between platforms' libm.
    assert res.score == pytest.approx(-6080.819658808304, rel=1e-13)
    assert res.info["accepted"] == 6
    assert [i for _, i, _ in res.trace] == [0, 1]
    assert [s for _, _, s in res.trace] == pytest.approx(
        [-6090.443358848632, -6080.819658808304], rel=1e-13
    )


def test_order_search_does_not_depend_on_the_kernel_or_on_the_caches(alarm_small):
    def go(kernel="cells", **kw):
        sc = OrderScorer(_with_kernel(alarm_small, kernel), n_restarts=2, **kw)
        res = PolishedILS(sc, random_state=9).run(list(range(alarm_small.n_vars)), max_iterations=4)
        return res.score, sorted(res.dag.edges()), [(i, s) for _, i, s in res.trace]

    expected = go()
    assert go(kernel="legacy") == expected
    assert go(incremental=False) == expected
    score = BDeuScore(alarm_small, prior="tetrad")
    res = order_search(score, list(range(alarm_small.n_vars)), max_iter=4, random_state=9)
    assert (res.score, sorted(res.dag.edges()), [(i, s) for _, i, s in res.trace]) == expected


def test_order_search_under_a_tight_cache_budget_finds_the_same_network(alarm_small):
    score = BDeuScore(alarm_small, prior="tetrad")
    free = order_search(score, max_iter=3, random_state=4)
    tight = order_search(score, max_iter=3, random_state=4, cache_mb=0.05)
    assert tight.score == free.score
    assert sorted(tight.dag.edges()) == sorted(free.dag.edges())
