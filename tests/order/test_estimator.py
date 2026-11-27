import numpy as np
import pandas as pd
import pytest

from _samples import load_cell
from causalbase import FLOP, OrderSearch
from causalbase.exceptions import NotFittedError
from causalbase.graph import dag_to_cpdag
from causalbase.order import order_search
from causalbase.scores import BDeuScore
from causalbase.scores._tetrad import TetradBDeuReference
from causalbase.search import ges, xges


@pytest.fixture(scope="module")
def alarm_small():
    full, _ = load_cell("alarm", 0)
    return full.subset(np.arange(1000))


def test_order_search_is_order_search_from_the_ges_order(alarm_small):
    model = OrderSearch(
        prior="tetrad", n_restarts=2, max_iter=3, time_limit=None, random_state=4
    ).fit(alarm_small)
    expected = order_search(
        BDeuScore(alarm_small, prior="tetrad"), n_restarts=2, max_iter=3, random_state=4
    )
    assert sorted(model.dag_.edges()) == sorted(expected.dag.edges())
    assert model.score_ == pytest.approx(expected.score, rel=1e-12)
    assert model.n_iter_ == 3
    assert [(i, s) for _, i, s in model.trace_] == [(i, s) for _, i, s in expected.trace]


def test_order_search_can_start_from_xges(alarm_small):
    model = OrderSearch(
        prior="tetrad", init="xges", max_iter=2, time_limit=None, random_state=1
    ).fit(alarm_small)
    score = BDeuScore(alarm_small, prior="tetrad")
    start = list(xges(score).to_dag().topological_order())
    expected = order_search(score, start, max_iter=2, random_state=1)
    assert sorted(model.dag_.edges()) == sorted(expected.dag.edges())


def test_order_search_reports_its_network_in_every_form(alarm_small):
    model = OrderSearch(prior="tetrad", max_iter=1, time_limit=None).fit(alarm_small.to_pandas())
    assert model.score_ == BDeuScore(alarm_small, prior="tetrad").score_dag(model.dag_)
    assert model.score_ == pytest.approx(
        TetradBDeuReference(alarm_small).score_dag(model.dag_), rel=1e-12
    )
    cpdag = dag_to_cpdag(model.dag_)
    assert model.cpdag_.directed_edges() == cpdag.directed_edges()
    assert model.cpdag_.undirected_edges() == cpdag.undirected_edges()
    assert isinstance(model.adjacency_, pd.DataFrame)
    assert list(model.adjacency_.columns) == list(alarm_small.names)
    assert list(model.feature_names_in_) == list(alarm_small.names)
    assert model.n_features_in_ == alarm_small.n_vars
    assert model.trace_[0][1] == 0


def test_order_search_takes_the_polish_and_the_prior(alarm_small):
    model = OrderSearch(polish="xges", ess=10.0, prior="tetrad", max_iter=1, time_limit=None).fit(
        alarm_small
    )
    assert model.score_ == pytest.approx(
        TetradBDeuReference(alarm_small, ess=10.0).score_dag(model.dag_), rel=1e-12
    )
    model = OrderSearch(ess=10.0, expected_parents=2.0, max_iter=1, time_limit=None).fit(
        alarm_small
    )
    assert model.score_ == BDeuScore(alarm_small, ess=10.0, expected_parents=2.0).score_dag(
        model.dag_
    )


def test_order_search_follows_the_estimator_protocol(alarm_small):
    model = OrderSearch()
    with pytest.raises(NotFittedError):
        _ = model.adjacency_
    assert repr(OrderSearch(polish="xges")) == "OrderSearch(polish='xges')"
    assert repr(OrderSearch(n_restarts=0, random_state=3)) == (
        "OrderSearch(n_restarts=0, random_state=3)"
    )
    assert model.set_params(**model.get_params()) is model
    for bad in (
        {"polish": "nope"},
        {"n_restarts": -1},
        {"n_pair_candidates": 1.5},
        {"time_limit": None, "max_iter": None},
        {"init": "nope"},
        {"max_iter": -1},
        {"time_limit": 0.0},
        {"random_state": "seed"},
        {"n_jobs": 0},
        {"ess": -1.0},
        {"expected_parents": -1.0},
        {"prior": "uniform"},
    ):
        with pytest.raises(ValueError):
            OrderSearch(**bad).fit(alarm_small)


def test_the_scores_share_one_interface(alarm_small):
    fast, tetrad = (
        BDeuScore(alarm_small, ess=5.0, prior="tetrad"),
        TetradBDeuReference(alarm_small, ess=5.0),
    )
    dag = ges(fast).to_dag()
    assert fast.n_vars == tetrad.n_vars == alarm_small.n_vars
    assert fast.data is alarm_small and tetrad.data is alarm_small
    assert fast.ess == tetrad.ess == 5.0
    assert fast.score_dag(dag) == pytest.approx(tetrad.score_dag(dag), rel=1e-12)
    assert fast.score_dag(dag) == pytest.approx(
        sum(fast.local_score(v, dag.parents(v)) for v in range(dag.num_nodes)), rel=1e-15
    )


@pytest.mark.parametrize("estimator", [OrderSearch, FLOP])
def test_a_cache_budget_leaves_the_learned_network_as_it_is(alarm_small, estimator):
    kwargs = {"prior": "tetrad", "max_iter": 2, "time_limit": None, "random_state": 5}
    free = estimator(**kwargs).fit(alarm_small)
    tight = estimator(**kwargs, cache_mb=0.05).fit(alarm_small)
    assert sorted(tight.dag_.edges()) == sorted(free.dag_.edges())
    assert tight.score_ == free.score_
    with pytest.raises(ValueError, match="cache_mb"):
        estimator(**kwargs, cache_mb=-1).fit(alarm_small)
