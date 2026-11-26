"""Fully observed real-valued data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from causalbase._base import is_frame
from causalbase.data.dataset import _check_unique_names

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd

__all__ = ["ContinuousDataset", "as_continuous", "as_continuous_sites"]


class ContinuousDataset:
    """A fully-observed dataset over continuous (real-valued) variables.

    Parameters
    ----------
    values : np.ndarray
        Real-valued data, shape ``(n_rows, n_vars)``. Cast to ``float64``.
    names : Sequence[str]
        Variable names, length ``n_vars``. Coerced to an immutable
        ``tuple[str, ...]``.

    Attributes
    ----------
    values : np.ndarray
        See Parameters.
    names : tuple[str, ...]
        See Parameters.
    n_rows : int
        Number of rows (observations).
    n_vars : int
        Number of variables (columns).

    Raises
    ------
    ValueError
        If `values` is not 2-D, if `names` does not have exactly `n_vars`
        entries or repeats a name, or if `values` contains any non-finite entry (``NaN`` or
        infinity).

    Notes
    -----
    Non-finite values are rejected: the cumulant estimators have no notion of
    a missing entry.
    """

    values: npt.NDArray[np.float64]
    names: tuple[str, ...]

    def __init__(self, values: npt.NDArray[np.float64], names: Sequence[str]) -> None:
        arr = np.asarray(values, dtype=np.float64)

        if arr.ndim != 2:
            raise ValueError(f"values must be 2-D (n_rows, n_vars), got shape {arr.shape}")
        n_vars = arr.shape[1]
        if len(names) != n_vars:
            raise ValueError(f"names must have {n_vars} entries, got {len(names)}")
        _check_unique_names(names)
        if arr.size and not bool(np.isfinite(arr).all()):
            raise ValueError(
                "values contains a non-finite entry (NaN or infinity); "
                "ContinuousDataset requires fully-observed finite data"
            )

        self.values = np.ascontiguousarray(arr)
        self.names = tuple(names)

    @property
    def n_rows(self) -> int:
        """int: Number of rows (observations) in the dataset."""
        return int(self.values.shape[0])

    @property
    def n_vars(self) -> int:
        """int: Number of variables (columns) in the dataset."""
        return int(self.values.shape[1])

    def subset(self, rows: npt.NDArray[np.intp] | npt.NDArray[np.bool_]) -> ContinuousDataset:
        """Select a subset of rows, keeping the parent's variable names.

        Parameters
        ----------
        rows : numpy.ndarray
            Row indices (integer array) or a boolean mask of length `n_rows`.

        Returns
        -------
        ContinuousDataset
            The selected rows over the same variables.
        """
        subset_ds: ContinuousDataset = object.__new__(ContinuousDataset)
        subset_ds.values = np.ascontiguousarray(self.values[np.asarray(rows)])
        subset_ds.names = self.names
        return subset_ds

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> ContinuousDataset:
        """Build a dataset from a pandas DataFrame of numeric columns.

        Parameters
        ----------
        df : pandas.DataFrame
            One column per variable, one row per observation. Column names
            become `names`.

        Returns
        -------
        ContinuousDataset
            The dataset.

        Raises
        ------
        ValueError
            If two columns share a name, or a column is non-numeric or
            contains a non-finite value.
        """
        import pandas as pd

        names = [str(c) for c in df.columns]
        _check_unique_names(names)
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"column {col!r} is not numeric; ContinuousDataset requires real-valued columns"
                )
        return cls(df.to_numpy(dtype=np.float64), names)

    @classmethod
    def from_csv(cls, path: str | Path) -> ContinuousDataset:
        """Build a dataset from a CSV file of numeric columns.

        The file must have a header row of variable names followed by
        comma-separated numeric values.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the CSV file.

        Returns
        -------
        ContinuousDataset
            The dataset.

        Raises
        ------
        ValueError
            If any column is non-numeric, or contains a non-finite value.
        """
        import pandas as pd

        return cls.from_pandas(pd.read_csv(path))

    def __repr__(self) -> str:
        return (
            f"ContinuousDataset(n_rows={self.n_rows}, n_vars={self.n_vars}, names={self.names!r})"
        )


def as_continuous(X: ContinuousDataset | pd.DataFrame | npt.ArrayLike) -> ContinuousDataset:
    """Return `X` as a :class:`ContinuousDataset`; array columns are named ``x0, x1, ...``."""
    return as_continuous_sites([X])[0]


def as_continuous_sites(
    sites: Sequence[ContinuousDataset | pd.DataFrame | npt.ArrayLike],
) -> list[ContinuousDataset]:
    """Return each site's table as a :class:`ContinuousDataset`, checking they share variables.

    Parameters
    ----------
    sites : sequence of ContinuousDataset, pandas.DataFrame or array-like
        One table per site. Array columns are named ``x0, x1, ...``.

    Returns
    -------
    list of ContinuousDataset

    Raises
    ------
    ValueError
        If there are no sites, or their variables differ.
    """
    if len(sites) == 0:
        raise ValueError("expected at least one site")
    out = []
    for site in sites:
        if isinstance(site, ContinuousDataset):
            out.append(site)
        elif is_frame(site):
            out.append(ContinuousDataset.from_pandas(site))  # type: ignore[arg-type]
        else:
            array = np.asarray(site, dtype=float)
            names = [f"x{j}" for j in range(array.shape[1])] if array.ndim == 2 else []
            out.append(ContinuousDataset(array, names))
    for k, site in enumerate(out[1:], start=1):
        if site.names != out[0].names:
            raise ValueError(f"site {k} has variables {site.names}, expected {out[0].names}")
    return out
