import numpy as np
import pytest

from _problems import best_dag_score, random_discrete
from _samples import load_cell
from causalbase.data import DiscreteDataset
from causalbase.exact import SearchResult, exact_search
from causalbase.exact.dp import parent_set_scores, prune_candidates
from causalbase.scores import BDeuScore
from causalbase.search import ges


@pytest.mark.parametrize("p,seed", [(3, 0), (4, 1), (4, 2), (5, 3), (5, 4), (5, 5)])
def test_agrees_with_brute_force_over_all_dags(p, seed):
    ds = random_discrete(p, 300, seed)
    fast = BDeuScore(ds, prior="tetrad")
    res = exact_search(fast)
    assert res.score == pytest.approx(best_dag_score(fast), abs=1e-9)
    assert fast.score_dag(res.dag) == pytest.approx(res.score, abs=1e-9)
    assert sorted(res.dag.topological_order()) == list(range(p))


def test_scores_of_every_parent_set_equal_fastbdeu_bit_for_bit():
    ds, _ = load_cell("child", 0)
    ds = ds.subset(np.arange(2000))
    fast = BDeuScore(ds, prior="tetrad")
    rng = np.random.default_rng(0)
    p = ds.n_vars
    for v in (0, 3, 12):
        scores, _ = parent_set_scores(fast, v, prune=False)
        others = [u for u in range(p) if u != v]
        for _ in range(200):
            k = int(rng.integers(0, 7))
            S = int(sum(1 << int(b) for b in rng.choice(p - 1, size=k, replace=False)))
            assert scores[S] == fast.local_score(
                v, [others[b] for b in range(p - 1) if (S >> b) & 1]
            )


def test_superset_bound_keeps_the_same_candidates():
    ds, _ = load_cell("child", 0)
    ds = ds.subset(np.arange(1000))
    fast = BDeuScore(ds, prior="tetrad")
    m = ds.n_vars - 1
    for v in (0, 3, 7):
        full, n_full = parent_set_scores(fast, v, prune=False)
        pruned, n_pruned = parent_set_scores(fast, v, prune=True)
        assert n_full == 1 << m and n_pruned <= n_full
        a, b = prune_candidates(full, m), prune_candidates(pruned, m)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_candidates_beat_all_their_subsets():
    ds = random_discrete(5, 300, 7)
    fast = BDeuScore(ds, prior="tetrad")
    m = 4
    scores, _ = parent_set_scores(fast, 2, prune=False)
    masks, sc = prune_candidates(scores, m)
    assert masks[np.argmax(sc)] == np.argmax(scores)
    for mask, s in zip(masks, sc, strict=True):
        for sub in range(1 << m):
            if sub != mask and sub & mask == sub:
                assert scores[sub] < s
    assert list(sc) == sorted(sc, reverse=True)


def test_optimum_is_at_least_the_true_network_and_ges():
    ds, true = load_cell("asia", 0)
    fast = BDeuScore(ds, prior="tetrad")
    res = exact_search(fast)
    assert res.score >= fast.score_dag(true) - 1e-9
    assert res.score >= fast.score_dag(ges(fast).dag) - 1e-9
    assert fast.score_dag(res.dag) == pytest.approx(res.score, abs=1e-9)


def test_limited_candidates_equal_the_full_enumeration_without_a_limit():
    from causalbase.exact.ilp import limited_candidates

    ds, _ = load_cell("child", 0)
    ds = ds.subset(np.arange(1000))
    fast = BDeuScore(ds, prior="tetrad")
    m = ds.n_vars - 1
    for v in (0, 3):
        full, _ = parent_set_scores(fast, v, prune=False)
        masks, sc = prune_candidates(full, m)
        lm, ls, total = limited_candidates(fast, v, m)
        assert total == 1 << m
        assert np.array_equal(lm, masks) and np.array_equal(ls, sc)
        # A limit keeps exactly the candidates of the full lattice that fit in it.
        lm3, ls3, _ = limited_candidates(fast, v, 3)
        small = np.array([bin(int(x)).count("1") <= 3 for x in masks])
        assert np.array_equal(lm3, masks[small]) and np.array_equal(ls3, sc[small])


def _wide_data(p, n, seed):
    """Many variables, each a noisy copy of the previous one, so single parents matter."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, p), dtype=np.int64)
    X[:, 0] = rng.integers(0, 2, n)
    for v in range(1, p):
        X[:, v] = np.where(rng.random(n) < 0.9, X[:, v - 1], 1 - X[:, v - 1])
    return DiscreteDataset(X, np.full(p, 2), [f"x{i}" for i in range(p)], [["0", "1"]] * p)


def test_limited_candidates_refuse_more_than_64_variables():
    # Parent sets are bitmasks over the other variables in an int64: from 65 variables on the
    # high bits wrap and the candidates of the later variables would be wrong, silently.
    from causalbase.exact.ilp import limited_candidates

    fast = BDeuScore(_wide_data(66, 300, 1), prior="tetrad")
    with pytest.raises(ValueError, match="64 variables"):
        limited_candidates(fast, 65, 1)
    with pytest.raises(ValueError, match="64 variables"):
        exact_search(fast, max_parents=1)


def test_limited_candidates_are_right_with_64_variables():
    from causalbase.exact.ilp import _to_vars, limited_candidates

    fast = BDeuScore(_wide_data(64, 300, 1), prior="tetrad")
    for v in (0, 62, 63):
        masks, scores, total = limited_candidates(fast, v, 1)
        assert total == 64 and (masks >= 0).all()
        for m, s in zip(masks, scores, strict=True):
            assert fast.local_score(v, _to_vars(m, v)) == pytest.approx(s, abs=1e-9)
        # Candidates come by decreasing score; the best single parent is a neighbour in the chain.
        assert tuple(_to_vars(masks[0], v)) in {(v - 1,), (v + 1,)}


@pytest.mark.parametrize("p,seed", [(5, 3), (6, 8), (7, 9)])
def test_integer_program_agrees_with_dynamic_programming(p, seed):
    pytest.importorskip("pyscipopt")
    ds = random_discrete(p, 400, seed)
    fast = BDeuScore(ds, prior="tetrad")
    dp = exact_search(fast)
    ilp = exact_search(fast, max_parents=p - 1)
    assert ilp.info["status"] == "optimal"
    assert ilp.score == pytest.approx(dp.score, abs=1e-9)
    assert fast.score_dag(ilp.dag) == pytest.approx(ilp.score, abs=1e-9)


DP_INFO = {"parents", "n_candidates", "n_scored", "time_scores", "time_dp", "max_candidate_size"}


def test_exact_search_returns_a_search_result_and_the_old_optimum():
    # Recorded with the retired exact_search on the asia sample.
    ds, _ = load_cell("asia", 0)
    res = exact_search(BDeuScore(ds, prior="tetrad"))
    assert isinstance(res, SearchResult)
    assert set(res.info) == DP_INFO
    assert res.score == pytest.approx(-11261.721669491984, rel=1e-12)
    assert sorted(res.dag.edges()) == [(1, 5), (2, 4), (3, 2), (3, 5), (4, 7), (5, 6), (5, 7)]
    assert res.n_iter == 0 and res.trace[-1][2] == res.score
    assert res.info["parents"] == [sorted(res.dag.parents(v)) for v in range(ds.n_vars)]


def test_a_parent_limit_runs_the_integer_program():
    pytest.importorskip("pyscipopt")
    ds, _ = load_cell("asia", 0)
    score = BDeuScore(ds, prior="tetrad")
    start = list(ges(score).dag.topological_order())
    res = exact_search(score, max_parents=2, initial_order=start, time_limit=60.0)
    assert set(res.info) == DP_INFO | {"status", "gap", "dual_bound", "n_clusters"}
    assert res.info["max_candidate_size"] and max(res.info["max_candidate_size"]) <= 2
    assert res.score <= exact_search(score).score + 1e-9


def test_exact_search_refuses_a_bad_parent_limit():
    ds, _ = load_cell("asia", 0)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="max_parents"):
            exact_search(BDeuScore(ds, prior="tetrad"), max_parents=bad)
