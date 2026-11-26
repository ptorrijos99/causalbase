"""The estimators' shared contract: parameters, input checks, fitted attributes."""

from __future__ import annotations

import numpy as np
import pytest

from causalbase import GES, XGES
from causalbase.exceptions import CausalBaseError, NotFittedError


def _discrete(n_rows: int = 300, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 3, size=(n_rows, 4))


def test_the_scikit_learn_parameter_protocol() -> None:
    sklearn_base = pytest.importorskip("sklearn.base")
    for cls in (GES, XGES):
        model = cls()
        assert repr(model) == f"{cls.__name__}()"
        assert model.set_params(**model.get_params()) is model
        with pytest.raises(ValueError, match="no parameter"):
            model.set_params(not_a_parameter=1)
        # Before fit, a result is missing, as scikit-learn expects.
        with pytest.raises(NotFittedError):
            _ = model.adjacency_
        assert not hasattr(model, "adjacency_")
    assert issubclass(NotFittedError, CausalBaseError)
    for model in (GES(ess=5.0), XGES(extended=False)):
        assert sklearn_base.clone(model).get_params() == model.get_params()


def test_estimators_refuse_bad_input_and_settings() -> None:
    x = _discrete()
    refusals = [
        (lambda: GES().fit(np.zeros((0, 3), dtype=int)), "no rows"),
        (lambda: GES(criterion="gaussian_bic").fit(np.zeros((0, 3))), "no rows"),
        (lambda: GES(ess=float("nan")).fit(x), "ess"),
    ]
    for name in ("max_parents", "max_t_size"):
        for value in (-1, 1.5, True, "2"):
            refusals.append((lambda n=name, v=value: GES(**{n: v}).fit(x), name))  # type: ignore[misc]
    for fit, match in refusals:
        with pytest.raises(ValueError, match=match):
            fit()


def test_degenerate_but_valid_input_is_learned_from() -> None:
    model = GES(criterion="gaussian_bic").fit(np.array([[0.1, 0.2, 0.3]]))
    assert model.dag_.num_edges == 0
    assert np.isfinite(model.score_)
