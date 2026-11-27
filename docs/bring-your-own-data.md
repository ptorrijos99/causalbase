# Bring your own data

Every estimator takes fully observed categorical data, and continuous data with `criterion="gaussian_bic"` or `criterion="bge"`. None of them handles missing values.

## Tables the estimators accept

A pandas DataFrame, a 2-D array or a `DiscreteDataset` all work:

```python
import pandas as pd

from causalbase import GES

GES().fit(pd.read_csv("cohort.csv", dtype=str))
GES().fit(array)  # columns are named x0, x1, ...
```

Every distinct value of a column is one level, so `1, 2, 3` are three categories, not a quantity. A DataFrame with missing values is rejected, naming the column; impute or drop those rows first.

## From a CSV, without guessing

```python
from causalbase.data import DiscreteDataset

data = DiscreteDataset.from_csv("cohort.csv")  # .csv.gz works too
```

`from_csv` reads every value as a label, including `None`, `NA` and `NaN`, which several standard benchmark networks use as real levels. Only an empty or blank field counts as missing, and that raises. Pandas' own reader, by contrast, turns those strings into missing values unless told otherwise (`keep_default_na=False`).

## Continuous data

```python
GES(criterion="gaussian_bic").fit(frame)  # every column numeric
OrderSearch(criterion="bge", time_limit=30).fit(frame)
```

Both scores model each node as a linear Gaussian function of its parents and read only the covariance matrix, so the columns are read as numbers and nothing is discretised. The Gaussian BIC is the penalised likelihood; BGe the Bayesian marginal likelihood, which gives the same score to every DAG of an equivalence class. BGe's prior does not rescale with the data, so standardise the columns if their units differ widely. `causalbase.data.ContinuousDataset.from_csv` and `as_continuous` build the dataset explicitly.

## Encoding several tables at once

Several tables over the same variables can share one encoding, so that a value seen in only one table is a level in all of them and every table is scored over the same domains:

```python
from causalbase.data import as_discrete_sites

tables = as_discrete_sites([frame_a, frame_b, frame_c])
```

The tables must share their columns, in the same order.

## How levels are encoded

Within each column, levels are numbered in sorted order of their label, not in order of first appearance. So `["yes", "no"]` encodes `no` as 0 and `yes` as 1, whatever order they show up in the file.

This is a reproducibility choice. Two tables with the same variables but different row orders must agree on the encoding, or graphs learned from them mean different things when compared. Sorting the labels is the simplest rule that guarantees it.

The consequence is that level indices need not match another tool's. Anything computed from the contingency tables (counts, BDeu scores, learned structures) is unaffected, since those are invariant to which integer names which level.

## Loading a network definition

```python
from causalbase.io import load_xbif, forward_sample

bn = load_xbif("alarm.xbif")
data = forward_sample(bn, n=5000, seed=0)
```

`load_xbif` reads XMLBIF, the format the [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/) publishes. `forward_sample` draws from the network, which is how the recovery tests generate data with a known answer.

Watch the variable order here: `bn.names` comes from the file and need not match a CSV of the same network. Re-index before comparing, as [Quickstart](quickstart.md) shows.
