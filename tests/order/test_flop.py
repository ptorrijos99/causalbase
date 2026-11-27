import time

import numpy as np
import pytest

from _samples import load_cell
from causalbase.order import SearchResult, flop
from causalbase.scores import BDeuScore
from causalbase.search import ges


@pytest.fixture(scope="module")
def alarm_small():
    full, _ = load_cell("alarm", 0)
    return full.subset(np.arange(500))


def test_flop_returns_a_dag_that_follows_its_order_and_scores_what_it_claims(alarm_small):
    ds = alarm_small
    fast = BDeuScore(ds, prior="tetrad")
    t0 = time.perf_counter()
    res = flop(fast, list(range(ds.n_vars)), random_state=0, time_limit=3.0)
    assert time.perf_counter() - t0 < 3.0 + 2.0
    assert isinstance(res, SearchResult)
    assert set(res.info) == {"order", "parents", "restarts"}
    pos = {v: i for i, v in enumerate(res.info["order"])}
    for v, ps in enumerate(res.info["parents"]):
        assert all(pos[u] < pos[v] for u in ps)
    assert abs(res.score - fast.score_dag(res.dag)) < 1e-6
    assert res.info["restarts"] >= 1
    assert res.n_iter == res.info["restarts"] - 1


def test_max_iter_counts_the_kicks_after_the_first_local_search(alarm_small):
    fast = BDeuScore(alarm_small, prior="tetrad")
    res = flop(fast, list(range(alarm_small.n_vars)), max_iter=2, random_state=3)
    assert res.n_iter == 2 and res.info["restarts"] == 3
    res = flop(fast, list(range(alarm_small.n_vars)), max_iter=0, random_state=3)
    assert res.n_iter == 0 and res.info["restarts"] == 1


def test_random_state_reproduces_the_old_seed(alarm_small):
    # Recorded with the retired flop_search(seed=1, max_restarts=3) from the GES order.
    score = BDeuScore(alarm_small, prior="tetrad")
    res = flop(score, max_iter=2, random_state=1)
    assert res.score == pytest.approx(-6063.186205815752, rel=1e-12)
    assert [i for _, i, _ in res.trace] == [0, 1]
    assert len(res.dag.edges()) == 39
    start = list(ges(score).dag.topological_order())
    again = flop(BDeuScore(alarm_small, prior="tetrad"), start, max_iter=2, random_state=1)
    assert again.score == res.score and sorted(again.dag.edges()) == sorted(res.dag.edges())


def test_flop_needs_a_limit(alarm_small):
    with pytest.raises(ValueError, match="max_iter"):
        flop(BDeuScore(alarm_small, prior="tetrad"), list(range(alarm_small.n_vars)))
