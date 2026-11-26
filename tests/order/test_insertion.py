import numpy as np
import pytest

from _samples import load_cell
from causalbase.order import OrderScorer
from causalbase.order.insertion import InsertionSearch
from causalbase.scores import BDeuScore
from causalbase.scores._compiled import CompiledBDeu


def _with_kernel(data, kernel):
    """BDeuScore(prior="tetrad") whose compiled engine counts with `kernel`."""
    score = BDeuScore(data, prior="tetrad")
    score._compiled = CompiledBDeu(data, prior="tetrad", kernel=kernel)
    return score


def _mask(nodes):
    m = 0
    for u in nodes:
        m |= 1 << int(u)
    return m


def reference_best_insertion(scorer, order, v):
    """Best insertion of v by sweeping adjacent swaps, calling the evaluator at every jump."""
    i = order.index(v)
    pred = _mask(order[:i])
    best_gain, best_j = 0.0, i
    pv, gain = pred, 0.0
    sv = scorer.local(v, pv)[0]
    for j in range(i - 1, -1, -1):
        u = order[j]
        pu = pv & ~(1 << u)
        su_old, su_new = scorer.local(u, pu)[0], scorer.local(u, pu | (1 << v))[0]
        pv = pu
        sv_new = scorer.local(v, pv)[0]
        gain += (sv_new - sv) + (su_new - su_old)
        sv = sv_new
        if gain > best_gain + 1e-9:
            best_gain, best_j = gain, j
    pv, gain = pred, 0.0
    sv = scorer.local(v, pv)[0]
    for j in range(i + 1, len(order)):
        u = order[j]
        su_old, su_new = scorer.local(u, pv | (1 << v))[0], scorer.local(u, pv)[0]
        pv |= 1 << u
        sv_new = scorer.local(v, pv)[0]
        gain += (sv_new - sv) + (su_new - su_old)
        sv = sv_new
        if gain > best_gain + 1e-9:
            best_gain, best_j = gain, j
    return best_gain, best_j


def reference_search(scorer, order, rng):
    order = list(order)
    while True:
        improved = False
        for v in rng.permutation(len(order)):
            g, j = reference_best_insertion(scorer, order, int(v))
            if g > 1e-9:
                order.remove(int(v))
                order.insert(j, int(v))
                improved = True
        if not improved:
            return order


@pytest.fixture(scope="module")
def alarm():
    ds, _ = load_cell("alarm", 0)
    return ds


@pytest.mark.parametrize("restarts", [0, 2])
def test_breaks_mark_every_candidate_that_can_change_the_result(alarm, restarts):
    ds = alarm
    sc = OrderScorer(BDeuScore(ds, prior="tetrad"), n_restarts=restarts)
    oracle = OrderScorer(BDeuScore(ds, prior="tetrad"), incremental=False, n_restarts=restarts)
    rng = np.random.default_rng(0)
    changed = 0
    for _ in range(25):
        order = [int(x) for x in rng.permutation(ds.n_vars)]
        k = int(rng.integers(0, ds.n_vars))
        node, preds = order[k], _mask(order[:k])
        brk = sc.breaks(node, preds)
        base = oracle.local(node, preds)
        for c in order[k + 1 :]:
            new = oracle.local(node, preds | (1 << c))
            if brk[c] < 0:
                assert new == base
            changed += new != base
    assert changed > 0  # the check above is not vacuous


@pytest.mark.parametrize("restarts", [0, 2])
def test_removing_a_candidate_off_the_greedy_path_changes_nothing(alarm, restarts):
    ds = alarm
    sc = OrderScorer(BDeuScore(ds, prior="tetrad"), n_restarts=restarts)
    oracle = OrderScorer(BDeuScore(ds, prior="tetrad"), incremental=False, n_restarts=restarts)
    rng = np.random.default_rng(1)
    for _ in range(25):
        order = [int(x) for x in rng.permutation(ds.n_vars)]
        k = int(rng.integers(1, ds.n_vars))
        node, preds = order[k], _mask(order[:k])
        moves = sc.path_moves(node, preds)
        base = oracle.local(node, preds)
        for c in order[:k]:
            if not (moves >> c) & 1:
                assert oracle.local(node, preds & ~(1 << c)) == base


@pytest.mark.parametrize("restarts", [0, 2])
@pytest.mark.parametrize("net", ["alarm", "hepar2"])
def test_best_move_equals_reference_sweep_bit_for_bit(net, restarts):
    ds, _ = load_cell(net, 0)
    fast = BDeuScore(ds, prior="tetrad")
    search = InsertionSearch(OrderScorer(fast, n_restarts=restarts))
    ref = OrderScorer(fast, n_restarts=restarts)
    order = [int(x) for x in np.random.default_rng(2).permutation(ds.n_vars)]
    for v in order[:: max(1, ds.n_vars // 25)]:
        assert search.best_move(order, v) == reference_best_insertion(ref, order, v)


@pytest.mark.parametrize("restarts", [0, 2])
def test_search_reaches_the_reference_local_optimum(alarm, restarts):
    ds = alarm
    fast = BDeuScore(ds, prior="tetrad")
    start = [int(x) for x in np.random.default_rng(3).permutation(ds.n_vars)]
    expected = reference_search(
        OrderScorer(fast, n_restarts=restarts), start, np.random.default_rng(4)
    )
    sc = OrderScorer(fast, n_restarts=restarts)
    order, score = InsertionSearch(sc).run(start, np.random.default_rng(4))
    assert order == expected
    assert score == pytest.approx(sc.score(order).total, rel=1e-12)


# What the search gave before the evaluator and the counting kernel were optimised
# (start: default_rng(7).permutation; visiting order: default_rng(11)).
BEFORE = {
    ("alarm", 0): (
        [
            0,
            10,
            11,
            14,
            35,
            20,
            33,
            4,
            26,
            27,
            34,
            29,
            18,
            12,
            19,
            8,
            3,
            22,
            32,
            13,
            9,
            17,
            31,
            36,
            30,
            7,
            1,
            15,
            6,
            16,
            23,
            24,
            5,
            28,
            25,
            2,
            21,
        ],
        -53941.60046330815,
    ),
    ("alarm", 2): (
        [
            0,
            10,
            11,
            14,
            35,
            20,
            33,
            4,
            26,
            27,
            34,
            29,
            18,
            12,
            19,
            8,
            22,
            3,
            32,
            13,
            9,
            17,
            31,
            36,
            30,
            7,
            1,
            15,
            6,
            16,
            23,
            24,
            5,
            28,
            25,
            2,
            21,
        ],
        -53941.60046330815,
    ),
    ("child", 0): (
        [9, 13, 4, 12, 18, 3, 14, 17, 16, 2, 6, 15, 1, 0, 10, 7, 11, 5, 19, 8],
        -61999.82802742705,
    ),
    ("child", 2): (
        [9, 13, 4, 12, 18, 3, 14, 17, 16, 2, 6, 15, 1, 0, 10, 7, 11, 5, 19, 8],
        -61999.82802742705,
    ),
}


@pytest.mark.parametrize("net,restarts", list(BEFORE))
def test_search_gives_the_orders_and_scores_it_gave_before(net, restarts):
    ds, _ = load_cell(net, 0)
    sc = OrderScorer(BDeuScore(ds, prior="tetrad"), n_restarts=restarts)
    start = [int(v) for v in np.random.default_rng(7).permutation(ds.n_vars)]
    order, score = InsertionSearch(sc).run(start, np.random.default_rng(11))
    assert (order, score) == BEFORE[(net, restarts)]


@pytest.mark.parametrize("restarts", [0, 2])
def test_search_does_not_depend_on_the_kernel_or_on_the_caches(alarm, restarts):
    start = [int(v) for v in np.random.default_rng(3).permutation(alarm.n_vars)]

    def go(**kw):
        sc = OrderScorer(_with_kernel(alarm, kw.pop("kernel", "cells")), n_restarts=restarts, **kw)
        return InsertionSearch(sc).run(start, np.random.default_rng(4), max_passes=2)

    expected = go()
    assert go(kernel="legacy") == expected
    assert go(memo_slots=1, exact_cap=0) == expected
    assert go(incremental=False) == expected
