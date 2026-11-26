"""The estimator base class: scikit-learn's parameter protocol without the dependency."""

from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, Any

from causalbase.exceptions import NotFittedError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    from typing_extensions import Self

    from causalbase.graph import DAG, PDAG

__all__ = ["Estimator", "NotFittedError", "is_frame", "named_adjacency"]


def is_frame(obj: object) -> bool:
    """Return whether `obj` is a pandas DataFrame, without importing pandas.

    pandas is imported only where a DataFrame is built; if it has not been
    imported, nothing can be one.
    """
    pandas = sys.modules.get("pandas")
    return pandas is not None and isinstance(obj, pandas.DataFrame)


class Estimator:
    """Base class giving `get_params`, `set_params` and a readable ``repr``.

    Subclasses take their hyperparameters as keyword arguments of ``__init__``
    and store each under the same name, unchanged, as scikit-learn requires;
    validation happens in ``fit``. Estimators therefore work with
    ``sklearn.base.clone`` and with scikit-learn's parameter searches.
    """

    @classmethod
    def _param_names(cls) -> list[str]:
        signature = inspect.signature(cls.__init__)
        return [p for p in signature.parameters if p != "self"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the estimator's hyperparameters by name."""
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> Self:
        """Set hyperparameters by name and return the estimator."""
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"{type(self).__name__} has no parameter {name!r}")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        defaults = inspect.signature(type(self).__init__).parameters
        changed = [
            f"{name}={value!r}"
            for name, value in self.get_params().items()
            if value is not defaults[name].default
            and not (isinstance(value, (int, float, str)) and value == defaults[name].default)
        ]
        return f"{type(self).__name__}({', '.join(changed)})"

    def __sklearn_tags__(self) -> Any:
        """Describe the estimator to scikit-learn 1.6 and later, which only call this.

        Unsupervised, and the tables may hold strings and categories.
        """
        from sklearn.utils import InputTags, Tags, TargetTags

        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            input_tags=InputTags(categorical=True, string=True),
        )

    def _check_fitted(self, attribute: str) -> None:
        if not hasattr(self, attribute):
            raise NotFittedError(f"this {type(self).__name__} is not fitted yet; call fit first")


def named_adjacency(graph: DAG | PDAG, names: npt.NDArray[np.object_]) -> pd.DataFrame:
    """Return `graph`'s adjacency matrix as a DataFrame indexed by `names`.

    ``A[i, j] = 1`` and ``A[j, i] = 0`` is ``i -> j``; both set is an undirected
    edge.
    """
    import pandas as pd

    return pd.DataFrame(graph.adjacency().astype(int), index=names, columns=names)
