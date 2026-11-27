"""Fully observed categorical data, the input to the BDeu score."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from causalbase._base import is_frame

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import IO, TypeAlias

    import numpy.typing as npt
    import pandas as pd

    DiscreteLike: TypeAlias = "DiscreteDataset | pd.DataFrame | npt.ArrayLike"
    """Anything the discrete estimators accept as a table of observations."""

__all__ = ["DiscreteDataset", "as_discrete", "as_discrete_sites"]


def _check_unique_names(names: Sequence[str]) -> None:
    """Raise a ValueError naming the variables that occur more than once in `names`."""
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(f"variable names must be unique; repeated: {repeated}")


class DiscreteDataset:
    """A fully-observed dataset over discrete (categorical) variables.

    Parameters
    ----------
    values : np.ndarray
        Integer-coded data, shape ``(n_rows, n_vars)``. Cast to
        ``int32``. Entry ``values[i, j]`` is the code of row `i`'s level
        of variable `j`, an index into ``categories[j]``.
    cardinalities : np.ndarray
        Number of levels per variable, shape ``(n_vars,)``. Cast to
        ``int32``.
    names : Sequence[str]
        Variable names, length ``n_vars``. Coerced to an immutable
        ``tuple[str, ...]``.
    categories : Sequence[Sequence[str]]
        Level labels per variable; ``categories[j][i]`` is the label of
        level `i` of variable `j`, so ``len(categories[j]) ==
        cardinalities[j]``. Coerced to an immutable
        ``tuple[tuple[str, ...], ...]``.

    Attributes
    ----------
    values : np.ndarray
        See Parameters.
    cardinalities : np.ndarray
        See Parameters.
    names : tuple[str, ...]
        See Parameters.
    categories : tuple[tuple[str, ...], ...]
        See Parameters.
    n_rows : int
        Number of rows (observations).
    n_vars : int
        Number of variables (columns).

    Raises
    ------
    ValueError
        If `values` is not 2-D, if `cardinalities`, `names`, or
        `categories` do not each have exactly `n_vars` entries, if two
        variables share a name, if some
        ``categories[j]`` does not have exactly ``cardinalities[j]``
        entries, or if any entry of `values` falls outside
        ``[0, cardinalities[j])`` for its column.

    Notes
    -----
    The loaders code each column's levels in sorted order, whatever order they
    appear in, so the same data always gets the same codes.

    `values` is stored column-major, because the BDeu kernel reads one column
    at a time.

    `names`, `cardinalities` and `categories` are immutable and shared by every
    `subset`, so all sites split from one dataset agree on the variables.
    """

    values: npt.NDArray[np.int32]
    cardinalities: npt.NDArray[np.int32]
    names: tuple[str, ...]
    categories: tuple[tuple[str, ...], ...]

    def __init__(
        self,
        values: npt.ArrayLike,
        cardinalities: npt.ArrayLike,
        names: Sequence[str],
        categories: Sequence[Sequence[str]],
    ) -> None:
        values = np.asarray(values, dtype=np.int32)
        # A copy, because it is made read-only below.
        cardinalities = np.array(cardinalities, dtype=np.int32)

        if values.ndim != 2:
            raise ValueError(f"values must be 2-D (n_rows, n_vars), got shape {values.shape}")
        n_rows, n_vars = values.shape

        if cardinalities.shape != (n_vars,):
            raise ValueError(
                f"cardinalities must have shape ({n_vars},) to match values' "
                f"{n_vars} columns, got shape {cardinalities.shape}"
            )
        if len(names) != n_vars:
            raise ValueError(f"names must have {n_vars} entries, got {len(names)}")
        _check_unique_names(names)
        if len(categories) != n_vars:
            raise ValueError(f"categories must have {n_vars} entries, got {len(categories)}")
        for j, cats in enumerate(categories):
            if len(cats) != int(cardinalities[j]):
                raise ValueError(
                    f"categories[{j}] has {len(cats)} levels but "
                    f"cardinalities[{j}] = {int(cardinalities[j])}"
                )
        if n_rows > 0 and n_vars > 0:
            out_of_range = bool(values.min() < 0 or np.any(values >= cardinalities[np.newaxis, :]))
            if out_of_range:
                raise ValueError(
                    "values contains a category code outside [0, cardinalities[j]) "
                    "for at least one column"
                )

        cardinalities.setflags(write=False)
        self.values = np.asfortranarray(values)
        self.cardinalities = cardinalities
        self.names = tuple(names)
        self.categories = tuple(tuple(cats) for cats in categories)

    @property
    def n_rows(self) -> int:
        """int: Number of rows (observations) in the dataset."""
        return int(self.values.shape[0])

    @property
    def n_vars(self) -> int:
        """int: Number of variables (columns) in the dataset."""
        return int(self.values.shape[1])

    def subset(self, rows: npt.NDArray[np.integer[Any]]) -> DiscreteDataset:
        """Select a subset of rows, keeping the full parent variable domains.

        Parameters
        ----------
        rows : np.ndarray
            Row indices (integer array) or a boolean mask of length
            `n_rows`, selecting the rows to keep.

        Returns
        -------
        DiscreteDataset
            The selected rows over the same variables. A level missing from
            those rows still counts toward its variable's cardinality, so
            every site agrees on the domains.
        """
        row_index = np.asarray(rows)
        subset_ds: DiscreteDataset = object.__new__(DiscreteDataset)
        subset_ds.values = np.asfortranarray(self.values[row_index])
        subset_ds.cardinalities = self.cardinalities
        subset_ds.names = self.names
        subset_ds.categories = self.categories
        return subset_ds

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, categories: Sequence[Sequence[str]] | None = None
    ) -> DiscreteDataset:
        """Build a dataset from a DataFrame of categorical columns.

        Values are read as strings, so ``1`` and ``"1"`` are the same level.

        Parameters
        ----------
        df : pandas.DataFrame
            One column per variable, one row per observation.
        categories : sequence of sequences of str, optional
            The levels of each column. By default, the distinct values of each
            column, sorted. Pass them to give several datasets one encoding.

        Returns
        -------
        DiscreteDataset

        Raises
        ------
        ValueError
            If two columns share a name, a column has missing values, or a
            value is outside `categories`.
        """
        import pandas as pd

        names = [str(c) for c in df.columns]
        _check_unique_names(names)
        values = np.empty((len(df), len(names)), dtype=np.int32)
        levels: list[list[str]] = []
        for j, col in enumerate(df.columns):
            series = df[col]
            if bool(series.isna().any()):
                raise ValueError(f"column {col!r} has missing values")
            labels = series.astype(str)
            col_levels = (
                sorted(labels.unique().tolist()) if categories is None else list(categories[j])
            )
            codes = pd.Categorical(labels, categories=col_levels).codes
            if bool((codes < 0).any()):
                unknown = sorted(set(labels) - set(col_levels))
                raise ValueError(f"column {col!r} has values outside its levels: {unknown}")
            values[:, j] = codes
            levels.append(col_levels)
        return cls(values, np.array([len(lv) for lv in levels]), names, levels)

    @classmethod
    def from_csv(cls, path: str | Path | IO[bytes]) -> DiscreteDataset:
        """Build a dataset from a CSV file of categorical columns.

        A header row of variable names, then one row per observation. Every
        value is read as a string label, including ``"None"`` and ``"NA"``,
        which some networks use as real levels. Only an empty or blank field
        counts as missing, and missing data is rejected.

        Parameters
        ----------
        path : str, pathlib.Path or binary file object
            Path to the CSV file (``.csv.gz`` is decompressed), or the file
            itself, opened in binary mode.

        Returns
        -------
        DiscreteDataset
            The encoded dataset.

        Raises
        ------
        ValueError
            If any column contains an empty or whitespace-only field.
        """
        import pandas as pd

        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
        for col in df.columns:
            if bool((df[col].str.strip() == "").any()):
                raise ValueError(
                    f"column {col!r} contains missing values (an empty or "
                    "whitespace-only field); DiscreteDataset requires "
                    "fully-observed discrete data"
                )
        return cls.from_pandas(df)

    def to_pandas(self) -> pd.DataFrame:
        """Return the data as a DataFrame of level labels."""
        import pandas as pd

        return pd.DataFrame(
            {
                name: np.asarray(levels, dtype=object)[self.values[:, j]]
                for j, (name, levels) in enumerate(zip(self.names, self.categories, strict=True))
            }
        )

    def __repr__(self) -> str:
        return f"DiscreteDataset(n_rows={self.n_rows}, n_vars={self.n_vars}, names={self.names!r})"


def _as_frame(X: pd.DataFrame | npt.ArrayLike) -> pd.DataFrame:
    import pandas as pd

    if is_frame(X):
        return cast("pd.DataFrame", X)
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D table, got an array of shape {array.shape}")
    return pd.DataFrame(array, columns=[f"x{j}" for j in range(array.shape[1])])


def as_discrete(
    X: DiscreteLike, categories: Sequence[Sequence[str]] | None = None
) -> DiscreteDataset:
    """Return `X` as a :class:`DiscreteDataset`, encoding it if needed.

    Parameters
    ----------
    X : DiscreteDataset, pandas.DataFrame or array-like of shape (n_rows, n_vars)
        Every distinct value of a column is one level. Array columns are
        named ``x0, x1, ...``.
    categories : sequence of sequences of str, optional
        The levels to encode each column with, to match another dataset.

    Returns
    -------
    DiscreteDataset

    Raises
    ------
    ValueError
        If a value is not among the given `categories`.
    """
    if isinstance(X, DiscreteDataset):
        if categories is None or X.categories == tuple(tuple(c) for c in categories):
            return X
        X = X.to_pandas()
    return DiscreteDataset.from_pandas(_as_frame(X), categories)


def as_discrete_sites(sites: Sequence[DiscreteLike]) -> list[DiscreteDataset]:
    """Encode several sites' tables with one shared set of levels per variable.

    A level that occurs at only one site still counts towards the variable's
    cardinality everywhere, so every site scores over the same domains.

    Parameters
    ----------
    sites : sequence of DiscreteDataset, pandas.DataFrame or array-like
        One table per site, all with the same columns in the same order.

    Returns
    -------
    list of DiscreteDataset

    Raises
    ------
    ValueError
        If there are no sites, or their columns differ.
    """
    if len(sites) == 0:
        raise ValueError("expected at least one site")
    first = sites[0]
    if isinstance(first, DiscreteDataset) and all(
        isinstance(s, DiscreteDataset)
        and s.names == first.names
        and s.categories == first.categories
        for s in sites
    ):
        return list(sites)  # type: ignore[arg-type]
    frames = [
        _as_frame(site) if not isinstance(site, DiscreteDataset) else site.to_pandas()
        for site in sites
    ]
    columns = list(frames[0].columns)
    for k, frame in enumerate(frames[1:], start=1):
        if list(frame.columns) != columns:
            raise ValueError(f"site {k} has columns {list(frame.columns)}, expected {columns}")
    levels = [
        sorted(set().union(*(frame[col].astype(str).unique() for frame in frames)))
        for col in columns
    ]
    return [DiscreteDataset.from_pandas(frame, levels) for frame in frames]
