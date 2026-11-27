"""RCD and its settings: cumulants from moment sums, the tau discriminator, the solvers."""

import numpy as np
import pandas as pd
import pytest

from causalbase.graph import DAG
from causalbase.lingam import (
    RCD,
    RCDResult,
    cumulants_from_sums,
    estimate_b_from_cov,
    estimate_b_lasso_from_cov,
    moment_sums,
    rcd,
    rcd_exact,
    tau_matrix_3rd,
    tau_matrix_4th,
)

KEYS = ("cov", "c3", "c12", "c21", "c4", "c13", "c31", "c22")


def _chain(seed: int, n: int = 6000, skewed: bool = False) -> np.ndarray:
    """x0 -> x1 -> x2 -> x3 with Laplace (or skewed) noise. The order is 0, 1, 2, 3."""
    rng = np.random.default_rng(seed)
    e = rng.exponential(1.0, size=(n, 4)) - 1.0 if skewed else rng.laplace(size=(n, 4))
    x = np.zeros((n, 4))
    x[:, 0] = e[:, 0]
    for j in range(1, 4):
        x[:, j] = 0.8 * x[:, j - 1] + e[:, j]
    return x


def _cumulants(x: np.ndarray, order: int = 4) -> dict:
    return cumulants_from_sums([moment_sums(x, order)])


def test_blocks_merge_into_the_k_statistics_of_the_whole_table() -> None:
    x = np.random.default_rng(3).laplace(size=(2000, 3))
    whole = _cumulants(x)
    # Blocks with their own means, far from zero, merge exactly into the whole-table values.
    blocks = [x[:700] + 5.0, x[700:1500] - 2.0, x[1500:] + 1e6]
    shifted = cumulants_from_sums([moment_sums(b) for b in blocks])
    unshifted = cumulants_from_sums([moment_sums(b) for b in (x[:700], x[700:1500], x[1500:])])
    for key in KEYS:
        np.testing.assert_allclose(unshifted[key], whole[key], rtol=1e-9, atol=1e-9)
    for key in ("c4", "c13", "c22"):  # still accurate a million deviations away
        np.testing.assert_allclose(
            cumulants_from_sums([moment_sums(b - np.mean(b, 0) + 1e6) for b in blocks])[key],
            cumulants_from_sums([moment_sums(b - np.mean(b, 0)) for b in blocks])[key],
            rtol=1e-4,
            atol=1e-6,
        )
    assert shifted["n"] == 2000
    n, c = len(x), x - x.mean(axis=0)
    m2, m3, m4 = (c**2).sum(0), (c**3).sum(0), (c**4).sum(0)
    k3 = n * m3 / ((n - 1) * (n - 2))
    k4 = (n * (n + 1) * m4 - 3 * (n - 1) * m2**2) / ((n - 1) * (n - 2) * (n - 3))
    np.testing.assert_allclose(whole["c3"], k3, rtol=1e-9)
    np.testing.assert_allclose(whole["c4"], k4, rtol=1e-9)


def test_sums_of_different_orders_or_too_few_rows_are_refused() -> None:
    x = np.random.default_rng(6).laplace(size=(100, 3))
    third = moment_sums(x, order=3)
    assert "m31" not in third
    assert "c4" not in cumulants_from_sums([third])
    assert cumulants_from_sums([]) == {}
    with pytest.raises(ValueError, match="same order"):
        cumulants_from_sums([moment_sums(x), third])
    with pytest.raises(ValueError, match="same variables"):
        cumulants_from_sums([moment_sums(x), moment_sums(x[:, :2])])
    with pytest.raises(ValueError, match="rows in total"):
        cumulants_from_sums([moment_sums(x[:3])])
    with pytest.raises(ValueError, match="order"):
        moment_sums(x, order=5)


def test_tau_ranks_the_source_lowest() -> None:
    """The exogenous variable has the smallest total tau. This is the whole method."""
    cum = _cumulants(_chain(1))
    tau = tau_matrix_4th(cum["c4"], cum["c13"], cum["c31"], cum["c22"])
    assert np.all(tau >= 0) and not np.diagonal(tau).any()
    assert int(np.argmin(tau.sum(axis=1))) == 0
    cum = _cumulants(_chain(2, skewed=True), order=3)
    assert int(np.argmin(tau_matrix_3rd(cum["c3"], cum["c12"], cum["c21"]).sum(axis=1))) == 0


def test_every_method_recovers_the_chain() -> None:
    stats, skewed = _cumulants(_chain(0)), _cumulants(_chain(8, skewed=True), order=3)
    for method in ("deflation", "ratio", "sort"):
        assert rcd(stats, method=method).causal_order == (0, 1, 2, 3)  # type: ignore[arg-type]
        assert rcd(skewed, method=method).causal_order == (0, 1, 2, 3)  # type: ignore[arg-type]
    assert rcd_exact(_chain(1)).causal_order == (0, 1, 2, 3)
    assert rcd_exact(_chain(8, skewed=True), order=3).causal_order == (0, 1, 2, 3)
    result = rcd(stats)
    assert isinstance(result, RCDResult)
    assert {(0, 1), (1, 2), (2, 3)} <= set(result.to_dag(threshold=0.3).edges())
    assert result.to_dag(threshold=10.0).edges() == []
    with pytest.raises(AttributeError):
        result.causal_order = (1, 0, 2, 3)  # type: ignore[misc]


def test_ratio_deflation_removes_the_source_with_its_coefficient() -> None:
    """Cum(s, s, j) / Cum(s, s, s) is the coefficient of s in j; Cum(j, j, s) gives its square."""
    from causalbase.lingam.solvers import _deflated_order

    e = np.random.default_rng(7).exponential(1.0, size=(200_000, 2)) - 1.0
    stats = _cumulants(np.column_stack([e[:, 0], 1.5 * e[:, 0] + e[:, 1]]), 3)
    cur = {k: np.array(stats[k]) for k in ("cov", "c3", "c12", "c21")}
    assert _deflated_order(cur, 3, "ratio") == [0, 1]
    assert cur["cov"][1, 1] == pytest.approx(1.0, abs=0.05)


def test_coefficients_on_a_known_order() -> None:
    rng = np.random.default_rng(0)
    e = rng.laplace(size=(20000, 3))
    x1 = 0.8 * e[:, 0] + e[:, 1]
    x = np.column_stack([e[:, 0], x1, -0.6 * x1 + e[:, 2]])
    cov = np.cov(x, rowvar=False)
    ols = estimate_b_from_cov([0, 1, 2], cov)
    np.testing.assert_allclose(ols, [[0, 0, 0], [0.8, 0, 0], [0, -0.6, 0]], atol=0.05)
    # No variable takes a parent that comes after it in the order.
    reordered = estimate_b_from_cov([2, 0, 1], cov)
    assert reordered[2].sum() == 0 and reordered[0, 1] == 0
    # The lasso keeps the true edges and shrinks: a "lasso" returning OLS would fail.
    for selection in ("bic", "fixed"):
        lasso = estimate_b_lasso_from_cov(
            [0, 1, 2], cov, n_samples=len(x), lambda_selection=selection
        )
        assert lasso[1, 0] != 0 and lasso[2, 1] != 0
        assert np.abs(lasso).sum() < np.abs(ols).sum()


def test_solvers_refuse_bad_settings() -> None:
    stats = _cumulants(_chain(9))
    for kwargs, match in [
        ({"method": "bogus"}, "method"),
        ({"order": 5}, "order"),
        ({"pruning": "lasso"}, "pruning"),
        ({"lambda_selection": "aic"}, "lambda_selection"),
    ]:
        with pytest.raises(ValueError, match=match):
            rcd(stats, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fourth-order"):
        rcd(_cumulants(_chain(9), order=3), order=4)
    with pytest.raises(ValueError, match="pruning"):
        rcd_exact(_chain(9), pruning="lasso")  # type: ignore[arg-type]
    without_n = dict(stats)
    del without_n["n"]  # a silent default would mis-scale the lasso penalty
    with pytest.raises(KeyError, match="missing 'n'"):
        rcd(without_n)
    with pytest.raises(ValueError, match="lambda_selection"):
        estimate_b_lasso_from_cov([0, 1], np.eye(2), n_samples=10, lambda_selection="x")
    assert isinstance(rcd(stats).to_dag(), DAG)


def test_the_estimator_is_the_solvers_on_one_table() -> None:
    x = _chain(4)
    frame = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    for method in ("deflation", "exact", "ratio", "sort"):
        for order in (3, 4):
            model = RCD(method=method, order=order, pruning=None).fit(frame)  # type: ignore[arg-type]
            if method == "exact":
                expected = rcd_exact(x, order=order, pruning=None)  # type: ignore[arg-type]
            else:
                expected = rcd(_cumulants(x, order), method=method, pruning=None)  # type: ignore[arg-type]
            assert model.causal_order_ == ["abcd"[i] for i in expected.causal_order]
            np.testing.assert_allclose(model.coef_, expected.coef, rtol=1e-9, atol=1e-12)
    model = RCD().fit(frame)
    assert list(model.adjacency_.columns) == ["a", "b", "c", "d"]
    assert model.adjacency_.loc["a", "b"] == 1 and model.adjacency_.loc["b", "a"] == 0
    assert model.n_features_in_ == 4 and list(model.feature_names_in_) == ["a", "b", "c", "d"]
    children, parents = np.nonzero(model.coef_)
    assert set(model.dag_.edges()) == set(zip(parents.tolist(), children.tolist(), strict=True))


def test_the_estimator_refuses_bad_settings_and_data() -> None:
    x = _chain(5, n=200)
    for model, match in [
        (RCD(method="bogus"), "method"),  # type: ignore[arg-type]
        (RCD(order=2), "order"),  # type: ignore[arg-type]
        (RCD(pruning="lasso"), "pruning"),  # type: ignore[arg-type]
        (RCD(method="exact", lambda_selection="aic"), "lambda_selection"),  # type: ignore[arg-type]
    ]:
        with pytest.raises(ValueError, match=match):
            model.fit(x)
    with pytest.raises(ValueError, match="rows in total"):
        RCD().fit(x[:3])
    with pytest.raises(ValueError, match="non-finite"):
        RCD().fit(np.where(np.eye(200, 4, dtype=bool), np.nan, x))
