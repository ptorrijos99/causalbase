import numpy as np
import pytest

from _samples import load_cell
from causalbase.order import OrderResult, OrderScorer
from causalbase.scores import BDeuScore
from causalbase.scores._tetrad import TetradBDeuReference
from causalbase.search import ges


@pytest.fixture(scope="module")
def alarm():
    ds, true = load_cell("alarm", 0)
    return ds, true, TetradBDeuReference(ds)


@pytest.fixture(scope="module")
def child():
    ds, true = load_cell("child", 0)
    return ds, true, TetradBDeuReference(ds)


def _random_order(n, seed):
    return list(np.random.default_rng(seed).permutation(n))


def test_total_is_sum_of_local_scores(alarm):
    ds, _, score = alarm
    res = OrderScorer(score).score(_random_order(ds.n_vars, 1))
    assert isinstance(res, OrderResult)
    expected = sum(score.local_score(v, res.parents[v]) for v in range(ds.n_vars))
    assert res.total == pytest.approx(expected, abs=1e-9)


def test_parents_are_predecessors(alarm):
    ds, _, score = alarm
    order = _random_order(ds.n_vars, 2)
    res = OrderScorer(score).score(order)
    pos = {v: k for k, v in enumerate(order)}
    assert any(res.parents)
    for v, ps in enumerate(res.parents):
        assert all(pos[p] < pos[v] for p in ps)


def _order_chain(n, length, seed):
    """Orders as MCTS produces them: moves to the front, swaps, prefix + projection."""
    rng = np.random.default_rng(seed)
    base = list(rng.permutation(n))
    order = list(base)
    for _ in range(length):
        kind = rng.integers(3)
        if kind == 0:
            v = order.pop(int(rng.integers(n)))
            order.insert(0, v)
        elif kind == 1:
            i, j = rng.integers(n, size=2)
            order[i], order[j] = order[j], order[i]
        else:
            k = int(rng.integers(n))
            prefix = order[:k]
            rest = set(order[k:])
            order = prefix + [v for v in base if v in rest]
        yield list(order)


@pytest.mark.parametrize("restarts", [0, 2])
@pytest.mark.parametrize("net", ["alarm", "child"])
def test_incremental_equals_fresh(net, restarts, request):
    ds, _, score = request.getfixturevalue(net)
    inc = OrderScorer(score, incremental=True, n_restarts=restarts)
    fresh = OrderScorer(score, incremental=False, n_restarts=restarts)
    for order in _order_chain(ds.n_vars, 200, seed=3):
        a, b = inc.score(order), fresh.score(order)
        assert a.parents == b.parents
        assert a.total == b.total
    assert inc.n_reused > 0


def test_added_and_removed_predecessors_same_node(alarm):
    ds, true, score = alarm
    n = ds.n_vars
    order = true.topological_order()
    inc = OrderScorer(score, incremental=True)
    fresh = OrderScorer(score, incremental=False)
    inc.score(order)
    # Node in the middle gains the node right after it and loses the node right before it.
    k = n // 2
    new = list(order)
    new[k - 1], new[k + 1] = new[k + 1], new[k - 1]
    node = new[k]
    preds_old = set(order[:k])
    preds_new = set(new[:k])
    assert preds_new - preds_old and preds_old - preds_new
    assert inc.score(new).parents[node] == fresh.score(new).parents[node]
    assert inc.score(new) == fresh.score(new)


def test_true_order_beats_ges(alarm):
    ds, true, score = alarm
    ges_dag = ges(BDeuScore(ds, prior="tetrad")).dag
    res = OrderScorer(score).score(true.topological_order())
    assert res.total >= score.score_dag(ges_dag)


def test_multi_slot_memo_avoids_recomputation_when_alternating(alarm):
    # Rollouts alternate between two base orders, as with pool_size > 1.
    ds, _, score = alarm
    a = _random_order(ds.n_vars, 10)
    b = _random_order(ds.n_vars, 11)
    # exact_cap=0: exact answers would hide the slots in this alternation.
    one = OrderScorer(score, memo_slots=1, exact_cap=0)
    many = OrderScorer(score, memo_slots=4, exact_cap=0)
    fresh = OrderScorer(score, incremental=False)
    for order in [a, b] * 5:
        assert many.score(order) == fresh.score(order) == one.score(order)
    calls_one, calls_many = one.n_local_calls, many.n_local_calls
    assert calls_many < calls_one / 3


class _CountingScore:
    """Wraps a score and counts the requests that reach it."""

    def __init__(self, score):
        self.score = score
        self.n_vars = score.n_vars
        self.requests = {}

    def local_score(self, node, parents):
        key = (node, frozenset(parents))
        self.requests[key] = self.requests.get(key, 0) + 1
        return self.score.local_score(node, parents)


def test_each_family_reaches_the_score_once(alarm):
    ds, _, score = alarm
    counting = _CountingScore(score)
    scorer = OrderScorer(counting)
    for order in _order_chain(ds.n_vars, 50, seed=7):
        scorer.score(order)
    assert max(counting.requests.values()) == 1


def test_moving_an_unchosen_predecessor_back_costs_no_lookups(child):
    # Candidates already compared against a stored path are remembered, so a
    # predecessor that no node picks can come and go without new comparisons.
    ds, true, score = child
    order1 = true.topological_order()
    scorer = OrderScorer(score, memo_slots=1)
    chosen = set().union(*scorer.score(order1).parents)
    x = next(v for v in order1 if v not in chosen)
    order2 = [v for v in order1 if v != x] + [x]
    for order in (order2, order1, order2):
        scorer.score(order)
    counts = {}
    real = scorer._ls

    def counting(node, mask):
        counts[node] = counts.get(node, 0) + 1
        return real(node, mask)

    def counting_lookup(node, state, cands):
        counts[node] = counts.get(node, 0) + len(cands)
        return real_lookup(node, state, cands)

    real_lookup = scorer._lookup
    scorer._ls = counting
    scorer._lookup = counting_lookup
    scorer.score(order1)
    assert sum(c for node, c in counts.items() if node != x) == 0


class _BatchCountingScore:
    """The compiled engine with both entry points counted per family."""

    def __init__(self, fast):
        self.fast = fast
        self.n_vars = fast.n_vars
        self.requests = {}
        self.pairs = {}
        self.batches = 0

    def _count(self, node, parents):
        key = (node, frozenset(parents))
        self.requests[key] = self.requests.get(key, 0) + 1

    def local_score(self, node, parents):
        self._count(node, parents)
        return self.fast.local_score(node, parents)

    def local_scores_add(self, node, parents, cands):
        self.batches += 1
        for c in cands:
            self._count(node, set(parents) | {c})
            key = (node, frozenset(parents), c)
            self.pairs[key] = self.pairs.get(key, 0) + 1
        return self.fast.local_scores_add(node, parents, cands)


class _SingleOnly:
    def __init__(self, fast):
        self.fast = fast
        self.n_vars = fast.n_vars

    def local_score(self, node, parents):
        return self.fast.local_score(node, parents)


def test_batched_scorer_matches_single_family_scorer(child):
    ds, _, _ = child
    fast = BDeuScore(ds, prior="tetrad")._engine()
    counting = _BatchCountingScore(fast)
    batched = OrderScorer(counting)
    single = OrderScorer(_SingleOnly(fast))
    fresh = OrderScorer(_SingleOnly(fast), incremental=False)
    for order in _order_chain(ds.n_vars, 60, seed=5):
        a, b, c = batched.score(order), single.score(order), fresh.score(order)
        assert a == b == c
    assert counting.batches > 0
    # Each (state, candidate) pair is scored once. Additions skip the family cache,
    # so a family reachable from two states ({a} + b and {b} + a) may be scored twice.
    assert max(counting.pairs.values()) == 1
    repeated = sum(k - 1 for k in counting.requests.values())
    assert repeated <= 0.05 * len(counting.requests)


def test_a_predecessor_set_seen_before_costs_no_lookups(alarm):
    # With pool_size > 1 a node often meets a predecessor set again after it has
    # left the replay slots; the exact answer is kept apart from the slots.
    ds, _, score = alarm
    a = _random_order(ds.n_vars, 20)
    b = _random_order(ds.n_vars, 21)
    scorer = OrderScorer(score, memo_slots=1)
    first = scorer.score(a)
    scorer.score(b)
    calls = scorer.n_local_calls
    assert scorer.score(a) == first
    assert scorer.n_local_calls == calls


def test_exact_answers_are_bounded_per_node(alarm):
    ds, _, score = alarm
    scorer = OrderScorer(score, exact_cap=3)
    fresh = OrderScorer(score, incremental=False)
    for order in _order_chain(ds.n_vars, 40, seed=9):
        assert scorer.score(order) == fresh.score(order)
    assert max(len(d) for d in scorer._exact) <= 3


def _greedy_first_parent(score, node, preds):
    """First move of the greedy climb from the empty set.

    It is the best single parent, the smallest index on ties.
    """
    base = score.local_score(node, frozenset())
    best, first = 0.0, None
    for c in sorted(preds):
        g = score.local_score(node, frozenset([c])) - base
        if g > best:
            best, first = g, c
    return first


@pytest.mark.parametrize("net", ["alarm", "child"])
def test_restarts_take_the_best_climb_after_forbidding_first_parents(net, request):
    ds, _, score = request.getfixturevalue(net)
    plain = OrderScorer(score, incremental=False)
    sc = OrderScorer(score, n_restarts=2)
    rng = np.random.default_rng(5)
    for _ in range(60):
        order = [int(x) for x in rng.permutation(ds.n_vars)]
        k = int(rng.integers(1, ds.n_vars))
        node, preds = order[k], set(order[:k])
        expected = plain.local(node, sum(1 << u for u in preds))
        for _ in range(2):
            f = _greedy_first_parent(score, node, preds)
            if f is None:
                break
            preds.discard(f)
            r = plain.local(node, sum(1 << u for u in preds))
            if r[0] > expected[0]:
                expected = r
        assert sc.local(node, sum(1 << u for u in order[:k])) == expected


def test_restarts_never_lower_an_order_and_lift_some_on_win95pts():
    ds, _ = load_cell("win95pts", 0)
    fast = BDeuScore(ds, prior="tetrad")
    plain, sc = OrderScorer(fast), OrderScorer(fast, n_restarts=2)
    lifted = 0
    for seed in range(10):
        order = _random_order(ds.n_vars, seed)
        a, b = plain.score(order), sc.score(order)
        assert b.total >= a.total
        assert b.total == pytest.approx(
            sum(fast.local_score(v, b.parents[v]) for v in range(ds.n_vars)), rel=1e-12
        )
        lifted += b.total > a.total + 1e-9
    assert lifted > 0


@pytest.mark.parametrize("restarts", [0, 2])
def test_answers_do_not_depend_on_what_was_asked_before(alarm, restarts):
    # A scorer that has replayed many paths (its steps remember which candidates beat them) answers
    # as one that climbs from scratch every time.
    ds = alarm[0]
    fast = BDeuScore(ds, prior="tetrad")
    warm = OrderScorer(fast, n_restarts=restarts)
    cold = OrderScorer(fast, n_restarts=restarts, incremental=False)
    rng = np.random.default_rng(8)
    nodes = [int(v) for v in rng.choice(ds.n_vars, size=6, replace=False)]
    for _ in range(150):
        node = nodes[int(rng.integers(len(nodes)))]
        preds = 0
        for u in rng.choice(ds.n_vars, size=int(rng.integers(0, ds.n_vars)), replace=False):
            if int(u) != node:
                preds |= 1 << int(u)
        kind = int(rng.integers(3))
        if kind == 0:
            assert warm.local(node, preds) == cold.local(node, preds)
        elif kind == 1:
            assert np.array_equal(warm.breaks(node, preds), cold.breaks(node, preds))
        else:
            assert warm.path_moves(node, preds) == cold.path_moves(node, preds)
