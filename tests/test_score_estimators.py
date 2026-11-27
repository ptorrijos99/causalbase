"""The estimators on one local score: OrderSearch, HillClimbing, FLOP and ExactSearch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalbase import FLOP, GES, XGES, ExactSearch, HillClimbing, OrderSearch
from causalbase.datasets import load_asia
from causalbase.exact import exact_search
from causalbase.exceptions import NotFittedError
from causalbase.graph import DAG, dag_to_cpdag
from causalbase.order import flop, order_search
from causalbase.scores import BDeuScore
from causalbase.search import ges, hill_climbing, xges

ESTIMATORS = (OrderSearch, HillClimbing, FLOP, ExactSearch)


@pytest.fixture(scope="module")
def asia():
    return load_asia().data


def _fast(cls):
    """An instance that fits asia in well under a second."""
    if cls is OrderSearch:
        return OrderSearch(max_iter=2, time_limit=None, random_state=0)
    if cls is FLOP:
        return FLOP(max_iter=2, time_limit=None, random_state=0)
    if cls is HillClimbing:
        return HillClimbing(n_restarts=1, random_state=0)
    return ExactSearch()


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_fit_sets_every_result(cls, asia):
    model = _fast(cls).fit(asia)
    assert isinstance(model.dag_, DAG)
    assert model.cpdag_ == dag_to_cpdag(model.dag_)
    assert model.score_ == BDeuScore(asia).score_dag(model.dag_)
    assert isinstance(model.n_iter_, int) and model.n_iter_ >= 0
    assert model.trace_ and model.trace_[-1][1] <= model.n_iter_
    assert model.trace_[-1][2] == pytest.approx(model.score_, rel=1e-9)
    assert model.n_features_in_ == asia.n_vars
    assert list(model.feature_names_in_) == list(asia.names)
    adjacency = model.adjacency_
    assert isinstance(adjacency, pd.DataFrame)
    assert list(adjacency.columns) == list(asia.names)
    assert adjacency.to_numpy().sum() == 2 * len(model.cpdag_.undirected_edges()) + len(
        model.cpdag_.directed_edges()
    )


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_score_is_the_mean_held_out_log_likelihood(cls, asia):
    frame = asia.to_pandas()
    model = _fast(cls).fit(frame.iloc[:4000])
    held_out = model.score(frame.iloc[4000:])
    assert np.isfinite(held_out) and held_out < 0


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_the_parameter_protocol(cls, asia):
    model = cls()
    assert repr(model) == f"{cls.__name__}()"
    assert model.set_params(**model.get_params()) is model
    with pytest.raises(ValueError, match="no parameter"):
        model.set_params(not_a_parameter=1)
    with pytest.raises(NotFittedError):
        _ = model.adjacency_
    with pytest.raises(NotFittedError):
        model.score(asia)
    with pytest.raises(ValueError, match="no rows"):
        _fast(cls).fit(np.zeros((0, 3), dtype=int))
    for bad in (
        {"ess": 0.0},
        {"expected_parents": -1.0},
        {"expected_parents": 0.0},  # positive for the Bernoulli prior
        {"prior": "uniform"},
        {"n_jobs": 0},
        {"n_jobs": 1.5},
    ):
        with pytest.raises(ValueError):
            cls(**bad).fit(asia)


def test_repr_shows_what_differs_from_the_defaults():
    assert repr(HillClimbing(tabu_length=10, max_tabu_moves=10)) == (
        "HillClimbing(tabu_length=10, max_tabu_moves=10)"
    )
    assert repr(FLOP(init="xges")) == "FLOP(init='xges')"
    assert repr(ExactSearch(max_parents=2)) == "ExactSearch(max_parents=2)"


def test_each_estimator_refuses_its_own_bad_settings(asia):
    refusals = [
        HillClimbing(tabu_length=-1),
        HillClimbing(max_tabu_moves=1.5),
        HillClimbing(n_restarts=-2),
        HillClimbing(perturbation=True),
        HillClimbing(time_limit=-1.0),
        HillClimbing(random_state=1.5),
        FLOP(init="nope"),
        FLOP(max_iter=None, time_limit=None),
        FLOP(max_iter=-1),
        ExactSearch(max_parents=-1),
        ExactSearch(time_limit=0.0),
    ]
    for model in refusals:
        with pytest.raises(ValueError):
            model.fit(asia)


@pytest.mark.parametrize("prior", ["bernoulli", "tetrad"])
def test_the_estimators_run_the_functions(asia, prior):
    score = BDeuScore(asia, expected_parents=1.5, prior=prior)
    params = {"expected_parents": 1.5, "prior": prior}
    start = None  # the functions start from GES's order, as the estimators do by default
    cases = [
        (
            HillClimbing(tabu_length=5, max_tabu_moves=5, n_restarts=2, random_state=3, **params),
            hill_climbing(score, tabu_length=5, max_tabu_moves=5, n_restarts=2, random_state=3),
        ),
        (
            FLOP(max_iter=3, time_limit=None, random_state=2, **params),
            flop(score, start, max_iter=3, random_state=2),
        ),
        (
            OrderSearch(max_iter=3, time_limit=None, random_state=2, **params),
            order_search(score, start, max_iter=3, random_state=2),
        ),
        (ExactSearch(**params), exact_search(score)),
    ]
    for model, result in cases:
        model.fit(asia)
        assert sorted(model.dag_.edges()) == sorted(result.dag.edges()), type(model).__name__
        assert model.n_iter_ == result.n_iter
        assert model.score_ == score.score_dag(model.dag_)
        assert model.score_ == pytest.approx(result.score, rel=1e-12)


def test_ges_and_xges_take_the_same_score_parameters(asia):
    for prior in ("bernoulli", "tetrad"):
        score = BDeuScore(asia, ess=2.0, expected_parents=1.5, prior=prior)
        for cls, search in ((GES, ges), (XGES, xges)):
            model = cls(ess=2.0, expected_parents=1.5, prior=prior).fit(asia)
            assert model.cpdag_ == search(score)
            assert model.score_ == score.score_dag(model.dag_)
    assert repr(GES(prior="tetrad")) == "GES(prior='tetrad')"
    for model in (GES(prior="uniform"), GES(criterion="bic", prior="tetrad"), XGES(prior="")):
        with pytest.raises(ValueError, match="prior"):
            model.fit(asia)


def test_a_fixed_random_state_reproduces_the_fit(asia):
    for cls in (OrderSearch, FLOP, HillClimbing):
        kwargs = {"random_state": 7}
        if cls is HillClimbing:
            kwargs |= {"n_restarts": 3, "perturbation": 3}
        else:
            kwargs |= {"max_iter": 3, "time_limit": None}
        a, b = cls(**kwargs).fit(asia), cls(**kwargs).fit(asia)
        assert sorted(a.dag_.edges()) == sorted(b.dag_.edges()) and a.score_ == b.score_


def test_exact_search_with_a_parent_limit_uses_the_integer_program(asia):
    pytest.importorskip("pyscipopt")
    model = ExactSearch(max_parents=2).fit(asia)
    assert max(len(model.dag_.parents(v)) for v in range(asia.n_vars)) <= 2
    assert model.score_ <= ExactSearch().fit(asia).score_ + 1e-9


def test_clone_keeps_the_parameters():
    sklearn_base = pytest.importorskip("sklearn.base")
    for model in (
        OrderSearch(n_restarts=1, random_state=3),
        HillClimbing(tabu_length=4),
        FLOP(max_iter=5),
        ExactSearch(max_parents=3),
    ):
        assert sklearn_base.clone(model).get_params() == model.get_params()
