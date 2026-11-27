"""BGe against the normal-Wishart model behind it, and its score equivalence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from _problems import random_dag, random_discrete, random_gaussian
from causalbase.data import ContinuousDataset
from causalbase.graph import DAG
from causalbase.scores import BGeScore, make_score


def _sequential_marginal(X: np.ndarray, alpha_mu: float, alpha_w: float, t: float) -> float:
    """Return log p(X) under a normal-Wishart prior, one row at a time.

    Each row's density is the multivariate t predictive of the posterior after
    the rows before it (Geiger and Heckerman, 1994), and the posterior is then
    updated with that row; no closed form of the marginal likelihood is used.
    The prior mean is the sample mean of `X` and the prior scale ``t I``.
    """
    dim = X.shape[1]
    mean, scale = X.mean(axis=0), t * np.eye(dim)
    total = 0.0
    for x in X:
        dof = alpha_w - dim + 1
        sigma = scale * (alpha_mu + 1) / (alpha_mu * dof)
        d = x - mean
        _, log_det = np.linalg.slogdet(sigma)
        total += (
            math.lgamma((dof + dim) / 2)
            - math.lgamma(dof / 2)
            - dim / 2 * math.log(dof * math.pi)
            - log_det / 2
            - (dof + dim) / 2 * math.log1p(d @ np.linalg.solve(sigma, d) / dof)
        )
        scale = scale + alpha_mu / (alpha_mu + 1) * np.outer(d, d)
        mean = (alpha_mu * mean + x) / (alpha_mu + 1)
        alpha_mu, alpha_w = alpha_mu + 1, alpha_w + 1
    return total


def _marginal(score: BGeScore, variables: list[int]) -> float:
    """Return log p of the columns `variables` under the marginal of the score's prior on them."""
    if not variables:
        return 0.0
    X = score.data.values[:, sorted(variables)]
    # A Wishart marginal on k of n variables has n - k fewer degrees of freedom.
    dof = score.alpha_w - score.n_vars + len(variables)
    return _sequential_marginal(X, score.alpha_mu, dof, score.t)


def test_local_scores_are_ratios_of_normal_wishart_marginal_likelihoods() -> None:
    for seed in range(4):
        rng = np.random.default_rng(seed)
        data = random_gaussian(5, 40, seed)
        for alpha_mu, alpha_w in [(1.0, None), (rng.uniform(0.5, 5), rng.uniform(6.5, 12))]:
            score = BGeScore(data, alpha_mu=alpha_mu, alpha_w=alpha_w, expected_parents=None)
            for _ in range(6):
                node = int(rng.integers(5))
                others = [u for u in range(5) if u != node]
                parents = [int(u) for u in rng.choice(others, rng.integers(0, 5), replace=False)]
                expected = _marginal(score, [*parents, node]) - _marginal(score, parents)
                assert score.local_score(node, parents) == pytest.approx(expected, rel=1e-9)


def test_a_batch_of_extensions_equals_each_family_alone() -> None:
    for seed in range(4):
        rng = np.random.default_rng(seed)
        data = random_gaussian(7, 200, seed)
        for _ in range(5):
            node = int(rng.integers(7))
            others = [u for u in range(7) if u != node]
            parents = frozenset(int(u) for u in rng.choice(others, rng.integers(0, 4), False))
            extra = [u for u in others if u not in parents]
            together = BGeScore(data).local_scores_with(node, parents, extra)
            alone = [BGeScore(data).local_score(node, parents | {x}) for x in extra]
            np.testing.assert_allclose(together, alone, rtol=1e-10)


def _covered_reversals(dag: DAG) -> list[DAG]:
    """Return the DAGs reached by reversing one covered edge, each Markov equivalent to `dag`.

    An edge ``u -> v`` is covered when the parents of ``v`` are those of ``u``
    and ``u`` (Chickering, 1995).
    """
    out = []
    edges = list(dag.edges())
    for u, v in edges:
        if set(dag.parents(v)) == set(dag.parents(u)) | {u}:
            rest = [e for e in edges if e != (u, v)]
            out.append(DAG.from_edges(dag.num_nodes, [*rest, (v, u)]))
    return out


def test_markov_equivalent_dags_score_the_same() -> None:
    checked = 0
    for seed in range(12):
        data = random_gaussian(6, 150, seed)
        score = BGeScore(data, alpha_mu=float(seed % 3 + 1))
        dag = random_dag(6, 0.5, seed)
        for other in _covered_reversals(dag):
            assert score.score_dag(other) == pytest.approx(score.score_dag(dag), rel=1e-10)
            checked += 1
    assert checked > 10


def test_every_complete_dag_scores_the_marginal_likelihood_of_the_data() -> None:
    data = random_gaussian(4, 60, 3)
    score = BGeScore(data, alpha_w=8.0, expected_parents=None)
    expected = _marginal(score, list(range(4)))
    rng = np.random.default_rng(0)
    for _ in range(5):
        order = rng.permutation(4)
        complete = DAG.from_edges(
            4, [(int(order[i]), int(order[j])) for j in range(4) for i in range(j)]
        )
        assert score.score_dag(complete) == pytest.approx(expected, rel=1e-9)


def test_the_estimators_name_it_and_take_ess_as_its_prior_on_the_mean() -> None:
    data = random_gaussian(4, 100, 1)
    score = make_score("bge", data, ess=2.5, expected_parents=None)
    assert isinstance(score, BGeScore) and score.alpha_mu == 2.5
    assert score.alpha_w == data.n_vars + 2
    with pytest.raises(TypeError, match="ContinuousDataset"):
        make_score("bge", random_discrete(3, 50, 0))


def test_bge_refuses_bad_settings() -> None:
    data = random_gaussian(3, 30, 0)
    for bad in ({"alpha_mu": 0.0}, {"alpha_mu": math.inf}, {"alpha_w": 4.0}, {"alpha_w": math.inf}):
        with pytest.raises(ValueError):
            BGeScore(data, **bad)
    with pytest.raises(ValueError, match="expected_parents"):
        BGeScore(data, expected_parents=0.0)


def test_a_deterministic_column_scores_finitely() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 2))
    data = ContinuousDataset(np.column_stack([x, x[:, 0] - x[:, 1]]), ["a", "b", "c"])
    score = BGeScore(data)
    assert np.isfinite(score.local_score(2, {0, 1}))
    assert score.local_score(2, {0, 1}) > score.local_score(2, {0})
