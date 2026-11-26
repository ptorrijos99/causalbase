"""Datasets: how tables are read and encoded, and what is refused."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalbase.data import (
    ContinuousDataset,
    DiscreteDataset,
    as_discrete_sites,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_csv_levels_are_coded_in_sorted_order_and_none_is_a_level() -> None:
    ds = DiscreteDataset.from_csv(FIXTURES / "toy.csv")
    assert ds.names == ("A", "B", "C")
    assert ds.categories == (("a", "b", "c"), ("hi", "lo"), ("p", "q", "r"))
    assert ds.values[:, 0].tolist() == [1, 0, 2, 0, 1, 0, 2, 1]
    assert ds.cardinalities.tolist() == [3, 2, 3]
    with pytest.raises(ValueError, match="read-only"):
        ds.cardinalities[0] = 9
    # The same table through pandas, and back.
    frame = pd.read_csv(FIXTURES / "toy.csv", dtype=str)
    assert np.array_equal(DiscreteDataset.from_pandas(frame).values, ds.values)
    assert ds.to_pandas().equals(frame)
    # "None" is a real level in some networks (child's DuctFlow, insurance's ThisCarDam).
    none = DiscreteDataset.from_csv(FIXTURES / "none_category.csv")
    assert none.categories[0].count("None") == 1


def test_a_subset_keeps_every_level_and_its_own_values() -> None:
    ds = DiscreteDataset.from_csv(FIXTURES / "toy.csv")
    sub = ds.subset(ds.values[:, 0] != 2)  # no row with A == "c"
    assert sub.n_rows == 6
    assert sub.categories == ds.categories
    assert np.array_equal(sub.cardinalities, ds.cardinalities)
    sub.values[0, 0] = 2
    assert ds.values[0, 0] == 1


def test_sites_share_one_encoding_whatever_levels_each_has() -> None:
    a = pd.DataFrame({"x": ["p", "q"], "y": ["1", "1"]})
    b = pd.DataFrame({"x": ["r", "p"], "y": ["2", "1"]})
    sites = as_discrete_sites([a, b])
    assert sites[0].categories == sites[1].categories == (("p", "q", "r"), ("1", "2"))
    with pytest.raises(ValueError, match="columns"):
        as_discrete_sites([a, b.rename(columns={"y": "z"})])


MALFORMED = [
    (lambda: DiscreteDataset(np.zeros(3), [1], ["a"], [["x"]]), "2-D"),
    (lambda: DiscreteDataset(np.zeros((3, 2)), [1], ["a", "b"], [["x"], ["x"]]), "shape"),
    (lambda: DiscreteDataset(np.zeros((3, 1)), [2], ["a"], [["x"]]), "levels"),
    (lambda: DiscreteDataset(np.zeros((3, 2)), [1, 1], ["a"], [["x"], ["x"]]), "names"),
    (lambda: DiscreteDataset([[0], [2]], [2], ["a"], [["x", "y"]]), "outside"),
    (lambda: DiscreteDataset(np.zeros((2, 2)), [1, 1], ["a", "a"], [["x"], ["x"]]), "repeat"),
    # 1 and "1" are one name once read as strings.
    (lambda: DiscreteDataset.from_pandas(pd.DataFrame([["x", "y"]], columns=[1, "1"])), "rep"),
    (lambda: DiscreteDataset.from_pandas(pd.DataFrame({"a": ["x", None]})), "missing"),
    (lambda: ContinuousDataset(np.zeros(5), ["a"]), "2-D"),
    (lambda: ContinuousDataset(np.zeros((4, 3)), ["a", "b"]), "names"),
    (lambda: ContinuousDataset([[1.0, np.nan]], ["a", "b"]), "non-finite"),
    (lambda: ContinuousDataset([[1.0, -np.inf]], ["a", "b"]), "non-finite"),
    (lambda: ContinuousDataset(np.zeros((2, 2)), ["a", "a"]), "repeated"),
    (lambda: ContinuousDataset.from_pandas(pd.DataFrame({"x": [1.0], "l": ["a"]})), "numeric"),
]


def test_malformed_tables_are_refused() -> None:
    for build, match in MALFORMED:
        with pytest.raises(ValueError, match=match):
            build()


def test_a_blank_csv_field_is_missing_data(tmp_path: Path) -> None:
    for text in ("A,B\nx,y\n,z\n", "A,B\nx,y\n  ,z\n"):
        (tmp_path / "bad.csv").write_text(text)
        with pytest.raises(ValueError, match="missing"):
            DiscreteDataset.from_csv(tmp_path / "bad.csv")
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [0.5, 0.25]})
    frame.to_csv(tmp_path / "c.csv", index=False)
    assert np.array_equal(ContinuousDataset.from_csv(tmp_path / "c.csv").values, frame.values)
